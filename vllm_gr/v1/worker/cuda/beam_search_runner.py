# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Independent, fixed-width CUDA Beam postprocessing for one active session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton

from vllm_gr.ops.cuda import beam_state as ops
from vllm_gr.ops.cuda.beam_kv_reorder import (
    _select_kv_gather_kernel,
    _select_kv_scatter_kernel,
)
from vllm_gr.ops.cuda.beam_selection import (
    _group_beam_outputs_kernel,
    _hierarchical_topk_out,
    bootstrap_beam_outputs_out,
)
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import BeamPostprocessWorkspace

try:
    from torch._C._dynamo.guards import TensorGuards as _TensorGuards
except ImportError:
    _TensorGuards = None  # type: ignore[assignment,misc]


def _bound_resource_tensors(context: Any, workspace: BeamPostprocessWorkspace) -> tuple[Any, ...]:
    return (
        context.decision.sequence,
        context.decision.beam_scores,
        context.decision.group_token_num,
        context.aux_key_buf,
        context.aux_value_buf,
        context.pool_key_ptrs,
        context.pool_value_ptrs,
        workspace.selected_token_ids,
        workspace.selected_parent_ids,
        workspace.selected_scores,
        workspace.selected_flat_indices,
        workspace.next_sequence,
    )


def _live_resource_tensors(context: Any, workspace: BeamPostprocessWorkspace) -> tuple[Any, ...]:
    return (
        *(tensor for pool in context.pools for tensor in (pool.unshared_key, pool.unshared_value)),
        *_bound_resource_tensors(context, workspace),
        workspace.topk_token_ids,
    )


@dataclass(frozen=True, slots=True)
class _BeamResourceValidation:
    workspace: BeamPostprocessWorkspace
    parameters: tuple[int, ...]
    guard: Any
    addresses: tuple[int, ...]
    key: tuple[Any, ...]


class BeamSearchRunner:
    """Enqueue Beam decisions and same-step KV reorder on the current stream.

    The caller has already written this round's Decode KV at the device
    ``state.num_decode_steps`` position. This runner does not run a forward,
    read scheduling metadata, encode results, or allocate a KV pool. Only one
    runner may execute against the shared workspace/context at a time.

    Capture binds this session's KV slot, state row, workspace and fixed
    configuration. Candidate addresses must remain fixed across replay;
    eager callers may supply other tensors with the same validated layout.
    Consumers on another stream must wait on an event recorded after advance.
    """

    def __init__(
        self,
        session: BeamSearchSession,
        state: BeamSearchState,
        workspace: BeamPostprocessWorkspace,
        *,
        state_index: int = 0,
        workspace_index: int = 0,
    ) -> None:
        if isinstance(state_index, bool) or not isinstance(state_index, int):
            raise TypeError("state_index must be an integer")
        if not 0 <= state_index < state.max_batch_size:
            raise ValueError("state_index exceeds state capacity")
        if isinstance(workspace_index, bool) or not isinstance(workspace_index, int):
            raise TypeError("workspace_index must be an integer")
        if not 0 <= workspace_index < workspace.max_batch:
            raise ValueError("workspace_index exceeds workspace capacity")
        context = session.context
        width = state.beam_width
        if state.device.type != "cuda":
            raise ValueError("BeamSearchRunner requires CUDA state")
        if state.decision is not context.decision:
            raise ValueError("state must alias this session's decision buffers")
        if not session.is_active or session.beam_width != width:
            raise ValueError("session must be active with the configured beam width")
        self._resource_key = self.validate_resources(
            context,
            workspace,
            beam_width=width,
            max_new_tokens=state.max_new_tokens,
            kv_slot=session.slot,
        )

        self.session = session
        first = context.pools[0].unshared_key
        layers, aux = len(context.pools), context.aux_key_buf
        self.state = state
        self.workspace = workspace
        self.state_index = state_index
        self.workspace_index = workspace_index
        self._closed = False
        # Construct device views only once. State and workspace rows need not
        # equal the KV slot. Selection/KV scratch still requires exclusive use.
        self._state = {
            name: value[state_index : state_index + 1].view(*value.shape[1:])
            if value.ndim > 1
            else value[state_index : state_index + 1]
            for name, value in vars(state).items()
            if isinstance(value, torch.Tensor)
        }
        if workspace.state_scratch is None:
            workspace.state_scratch = ops.BeamStateScratch.allocate(
                workspace.max_batch,
                width,
                state.device,
                candidate_width=workspace.key.top_k,
            )
        scratch = workspace.state_scratch
        self._scratch = {
            name: getattr(scratch, name)[workspace_index]
            if getattr(scratch, name).ndim > 1
            else getattr(scratch, name)[workspace_index : workspace_index + 1]
            for name in scratch.__dataclass_fields__
        }
        self._selected_tokens = workspace.selected_token_ids[workspace_index]
        self._selected_parents = workspace.selected_parent_ids[workspace_index]
        self._selected_scores = workspace.selected_scores[workspace_index]
        self._selected_flat = workspace.selected_flat_indices[workspace_index]
        self._next_sequence = workspace.next_sequence[workspace_index]
        self._kv_capacity = first.shape[3]
        self._kv_grid = (layers * width * first.shape[2],)
        self._kv_args = (
            context.pool_key_ptrs,
            context.pool_value_ptrs,
            aux.data_ptr(),
            context.aux_value_buf.data_ptr(),
            session.slot * first.stride(0) * first.element_size(),
            width,
            self._state["num_decode_steps"],
        )
        self._kv_kwargs = dict(
            H=first.shape[2],
            BLOCK_D=triton.next_power_of_2(first.shape[4]),
            D=first.shape[4],
            stride_pw=first.stride(1),
            stride_ph=first.stride(2),
            stride_ps=first.stride(3),
            stride_al=aux.stride(0),
            stride_aw=aux.stride(1),
            stride_ah=aux.stride(2),
            stride_as=aux.stride(3),
            active_mask=self._state["active_mask"],
            execute=self._scratch["reorder"],
            DEVICE_STEPS=True,
        )
        self._session_reset: tuple[Any, ...] | None = None
        self._scratch_owner = scratch

    def rebind_session(self, session: BeamSearchSession) -> None:
        """Bind an admitted session after release, retaining fixed tensor views.

        The caller must validate resources and compare the returned key before
        any mutation. Registry allocation initializes all host session fields.
        """
        if not self._closed or session.context is not self.session.context:
            raise RuntimeError("Beam rebind requires a released runner on the same context")
        if not session.is_active or session.beam_width != self.state.beam_width:
            raise ValueError("Beam rebind requires an active matching session")
        if session.slot != self.session.slot:
            first = session.context.pools[0].unshared_key
            args = self._kv_args
            self._kv_args = (
                *args[:4],
                session.slot * first.stride(0) * first.element_size(),
                *args[5:],
            )
        self.session = session
        self._closed = False

    def reset_for_session(self) -> None:
        """Match state.reset + decision.reset(row) after all consumers drain.

        Keep full initialization, including the -inf completed-score sentinel
        needed by final selection over the entire completed capacity. Decision
        sequence/scores are shared with state and are written only once.
        """
        if not self._closed:
            raise RuntimeError("Session reset requires a released Beam runner")
        plan = self._session_reset
        if plan is None:
            excluded = {
                "sequence",
                "beam_scores",
                "max_new_tokens_tensor",
                "stop_token_ids",
                "num_stop_tokens",
                "ignore_stop_tokens",
            }
            rows = [
                (value, name == "completed_scores")
                for name, value in self._state.items()
                if name not in excluded
            ]
            decision = self.state.decision
            for value in (decision.sequence, decision.beam_scores, decision.group_token_num):
                rows.append((value[self.state_index], False))
            tensors = tuple(value for value, _ in rows)
            sizes = tuple(value.numel() for value in tensors)
            starts = [0]
            for size in sizes:
                starts.append(starts[-1] + triton.cdiv(size, 256))
            plan = (tensors, sizes, tuple(starts), tuple(fill for _, fill in rows))
            self._session_reset = plan
        tensors, sizes, starts, negative_infinity = plan
        ops._reset_session_state_kernel[(starts[-1],)](
            tensors, sizes, starts, negative_infinity, BLOCK=256, num_warps=4
        )

    def initialize_from_prefill(
        self, token_ids: torch.Tensor, logprobs: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        """Initialize once from one GPU parent row; generated values stay on GPU."""
        self._advance(token_ids, logprobs, valid_mask, prefill=True)

    def advance(
        self, token_ids: torch.Tensor, logprobs: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        """Advance one Decode round; all runtime decisions are device-side."""
        self._advance(token_ids, logprobs, valid_mask, prefill=False)

    def reset(self) -> None:
        """Reuse this row after the caller has ordered all prior consumers.

        Same-stream consumers need no wait. For another stream, enqueue its
        completion event with current_stream().wait_event(...) before reset.
        Configuration and resource addresses remain unchanged.
        """
        if self._closed:
            raise RuntimeError("Beam runner's session has been released")
        self.state.reset(self.state_index)

    @staticmethod
    def validate_resources(
        context: Any,
        workspace: BeamPostprocessWorkspace,
        *,
        beam_width: int,
        max_new_tokens: int,
        kv_slot: int,
    ) -> tuple[Any, ...]:
        """Check live layouts in C++ and live addresses before reusing a key.

        TensorGuards does not check storage addresses. Inference tensors also
        lack version counters, so neither identity nor versions can replace
        the explicit data_ptr checks. A miss runs the complete validator.
        """
        if _TensorGuards is None:
            return BeamSearchRunner._validate_resources_full(
                context,
                workspace,
                beam_width=beam_width,
                max_new_tokens=max_new_tokens,
                kv_slot=kv_slot,
            )
        parameters = (
            beam_width,
            max_new_tokens,
            kv_slot,
            workspace.key.beam_width,
            workspace.key.top_k,
            workspace.max_batch,
            workspace.max_decode_steps,
        )
        cached: _BeamResourceValidation | None = getattr(context, "_gr_resource_validation", None)
        if cached is not None and cached.workspace is workspace and cached.parameters == parameters:
            tensors = _live_resource_tensors(context, workspace)
            if (
                len(tensors) == len(cached.addresses)
                and cached.guard.check(*tensors)
                and all(t.data_ptr() == address for t, address in zip(tensors, cached.addresses))
            ):
                return cached.key

        key = BeamSearchRunner._validate_resources_full(
            context,
            workspace,
            beam_width=beam_width,
            max_new_tokens=max_new_tokens,
            kv_slot=kv_slot,
        )
        tensors = _live_resource_tensors(context, workspace)
        if all(type(t) is torch.Tensor for t in tensors):
            # Guard constructor signatures vary across PyTorch versions. An
            # unsupported API leaves the full validator as the sole path.
            try:
                guard = _TensorGuards(
                    *tensors,
                    dynamic_dims_sizes=[list(t.shape) for t in tensors],
                    dynamic_dims_strides=[list(t.stride()) for t in tensors],
                )
            except TypeError:
                context._gr_resource_validation = None
            else:
                context._gr_resource_validation = _BeamResourceValidation(
                    workspace, parameters, guard, tuple(t.data_ptr() for t in tensors), key
                )
        return key

    @staticmethod
    def _validate_resources_full(
        context: Any,
        workspace: BeamPostprocessWorkspace,
        *,
        beam_width: int,
        max_new_tokens: int,
        kv_slot: int,
    ) -> tuple[Any, ...]:
        """Validate live storage and return the binding key for cached views.

        This is deliberately a live check: tensors and attention configuration
        have no generation/freeze API. Never treat object identity alone as a
        certificate for captured addresses or tensor layouts.
        """
        width, device = beam_width, context.decision.sequence.device
        # Candidate width K is independent of the beam width W: the workspace
        # key owns it, so require only that the allocated column extent agrees
        # with the key instead of asserting K == W.
        if (
            workspace.key.beam_width != width
            or workspace.key.top_k < 1
            or workspace.topk_token_ids.shape[2] != workspace.key.top_k
            or workspace.max_batch < 1
            or workspace.topk_token_ids.device != device
            or workspace.max_decode_steps < context.decision.sequence.shape[2]
        ):
            raise ValueError("workspace geometry/device does not match Beam state")
        if not context.pools:
            raise ValueError("session must provide at least one KV layer")
        first = context.pools[0].unshared_key
        shape, stride, dtype, pool_device = first.shape, first.stride(), first.dtype, first.device
        if (
            first.ndim != 5
            or first.dtype not in (torch.float16, torch.bfloat16)
            or first.device != device
            or first.shape[1] < width
            or first.shape[3] < max_new_tokens - 1
            or not first.is_contiguous()
            or not 0 <= kv_slot < first.shape[0]
        ):
            raise ValueError("KV pool capacity, dtype, device, or layout is incompatible")
        addresses = []
        for pool in context.pools:
            for tensor in (pool.unshared_key, pool.unshared_value):
                if (
                    tensor.shape != shape
                    or tensor.stride() != stride
                    or tensor.dtype != dtype
                    or tensor.device != pool_device
                ):
                    raise ValueError("all KV layers must have identical layouts")
                addresses.append(tensor.data_ptr())
        layers = len(context.pools)
        aux = context.aux_key_buf
        expected_aux = (layers, first.shape[1], *first.shape[2:])
        for tensor in (aux, context.aux_value_buf):
            if (
                tuple(tensor.shape) != expected_aux
                or not tensor.is_contiguous()
                or tensor.dtype != first.dtype
                or tensor.device != first.device
            ):
                raise ValueError("KV reorder requires compatible per-layer scratch")
        for ptrs in (context.pool_key_ptrs, context.pool_value_ptrs):
            if (
                ptrs is None
                or ptrs.shape != (layers,)
                or ptrs.dtype != torch.int64
                or ptrs.device != first.device
                or not ptrs.is_contiguous()
            ):
                raise ValueError("KV reorder requires persistent layer pointer tables")

        # These tensors back cached views/launch arguments. Include layout and
        # address so a valid replacement rebuilds views instead of using stale
        # ones; malformed replacements above are rejected before state reset.
        bound = _bound_resource_tensors(context, workspace)
        return (
            tuple(addresses),
            shape,
            stride,
            dtype,
            pool_device,
            tuple((t.data_ptr(), t.shape, t.stride(), t.dtype, t.device) for t in bound),
        )

    def end_session(self, consumer_done: torch.Event | None = None) -> None:
        """Release the existing slot only after device work and final consumers.

        This lifecycle boundary is outside graph capture and the steady path.
        Pass the final consumer's event when it ran on another stream.
        """
        if self._closed:
            return
        stream = torch.cuda.current_stream(self.state.device)
        if consumer_done is not None:
            stream.wait_event(consumer_done)
        done = torch.Event()
        done.record(stream)
        done.synchronize()
        self.session.end_session()
        self._closed = True

    def _advance(
        self,
        token_ids: torch.Tensor,
        logprobs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        prefill: bool,
    ) -> None:
        if self._closed:
            raise RuntimeError("Beam runner's session has been released")
        for owner in (
            getattr(self.session.context, "_gr_execution_owner", None),
            self.workspace.execution_owner,
        ):
            owner_beam = getattr(owner, "beam", None)
            if (
                owner is not None
                and owner_beam is not self
                and self not in getattr(owner_beam, "runners", ())
            ):
                raise RuntimeError("another runner owns the shared Beam execution scratch")
        width = self.state.beam_width
        # Candidate geometry is (rows, K): K is the workspace's column extent.
        candidate_width = self.workspace.key.top_k
        rows = 1 if prefill else width
        for name, tensor, dtype in (
            ("token_ids", token_ids, torch.int32),
            ("logprobs", logprobs, torch.float32),
            ("valid_mask", valid_mask, torch.bool),
        ):
            if (
                tensor.shape != (rows, candidate_width)
                or tensor.dtype != dtype
                or tensor.device != self.state.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA {dtype} [{rows}, {candidate_width}]"
                )
        metrics = getattr(self, "_gr_metrics", None)
        if metrics is not None:
            metrics.start("beam_select")
        s, w = self._state, self._scratch
        block = triton.next_power_of_2(width)
        candidate_block = triton.next_power_of_2(candidate_width)
        ops._prepare_candidates_kernel[(rows,)](
            token_ids,
            logprobs,
            valid_mask,
            s["beam_scores"],
            s["active_mask"],
            s["num_generated_tokens"],
            s["max_new_tokens_tensor"],
            s["stop_token_ids"],
            s["num_stop_tokens"],
            s["ignore_stop_tokens"],
            w["parent_scores"],
            w["selection_mask"],
            w["terminal_ranks"],
            w["finish_reasons"],
            w["cumulative_scores"],
            w["row_counts"],
            w["row_errors"],
            w["row_valid_counts"],
            CANDIDATE_WIDTH=candidate_width,
            BLOCK=candidate_block,
            PREFILL=prefill,
        )
        ops._validate_step_kernel[(1,)](
            s["is_initialized"],
            s["is_finished"],
            s["error_code"],
            s["num_generated_tokens"],
            s["num_decode_steps"],
            s["max_new_tokens_tensor"],
            s["num_completed"],
            w["row_counts"],
            w["row_errors"],
            w["row_valid_counts"],
            w["row_offsets"],
            w["execute"],
            w["reorder"],
            w["next_completed"],
            ROWS=rows,
            BLOCK=block,
            CAPACITY=self.state.completed_capacity,
            KV_CAPACITY=self._kv_capacity,
            PREFILL=prefill,
        )
        if prefill and candidate_width == width:
            bootstrap_beam_outputs_out(
                w["parent_scores"],
                token_ids,
                logprobs,
                w["selection_mask"],
                self._selected_tokens,
                self._selected_parents,
                self._selected_scores,
                self._selected_flat,
                batch_size=1,
                beam_width=width,
                candidate_width=candidate_width,
                compact_input=True,
                reuse_ordered_candidates=True,
            )
        else:
            selected_scores, selected_flat, stride = _hierarchical_topk_out(
                w["parent_scores"],
                logprobs,
                w["selection_mask"],
                1,
                rows,
                candidate_width,
                width,
                self.workspace.selection_scratch,
                mask_invalid_indices=True,
            )
            _group_beam_outputs_kernel[(1,)](
                selected_scores,
                selected_flat,
                token_ids,
                self._selected_tokens,
                self._selected_parents,
                self._selected_scores,
                self._selected_flat,
                width,
                candidate_width,
                stride,
                WRITE_FLAT_INDICES=True,
                BLOCK=block,
            )
        ops._append_completed_kernel[(rows,)](
            token_ids,
            w["terminal_ranks"],
            w["finish_reasons"],
            w["cumulative_scores"],
            w["row_offsets"],
            s["num_generated_tokens"],
            w["execute"],
            s["completed_token_ids"],
            s["completed_scores"],
            s["completed_parent_ids"],
            s["completed_lengths"],
            s["completed_finish_reasons"],
            s["completed_tie_break_indices"],
            # This kernel walks the per-parent candidate columns, so its
            # stride is the candidate width K, not the beam width W.
            WIDTH=candidate_width,
            BLOCK=candidate_block,
            PREFILL=prefill,
        )
        if metrics is not None:
            metrics.stop("beam_select")
            metrics.start("lineage_kv")
        extent = self.state.sequence.shape[2]
        ops._stage_sequence_kernel[(width,)](
            s["sequence"],
            self._selected_tokens,
            self._selected_parents,
            self._selected_flat,
            self._next_sequence,
            s["num_generated_tokens"],
            w["execute"],
            WIDTH=width,
            SEQUENCE_STRIDE=extent,
            OUT_STRIDE=self.workspace.max_decode_steps,
            BLOCK=triton.next_power_of_2(extent),
            PREFILL=prefill,
        )
        ops._commit_beams_kernel[(width,)](
            self._selected_tokens,
            self._selected_parents,
            self._selected_scores,
            self._selected_flat,
            self._next_sequence,
            s["sequence"],
            s["beam_scores"],
            s["input_ids"],
            s["parent_ids"],
            s["active_mask"],
            s["sequence_lengths"],
            s["history_token_ids"],
            s["history_parent_ids"],
            s["num_generated_tokens"],
            w["execute"],
            WIDTH=width,
            SEQUENCE_STRIDE=extent,
            NEXT_STRIDE=self.workspace.max_decode_steps,
            BLOCK=triton.next_power_of_2(extent),
            PREFILL=prefill,
        )
        if not prefill:
            _select_kv_gather_kernel[self._kv_grid](
                *self._kv_args, s["parent_ids"], **self._kv_kwargs
            )
            _select_kv_scatter_kernel[self._kv_grid](*self._kv_args, **self._kv_kwargs)
        if metrics is not None:
            metrics.stop("lineage_kv")
            metrics.start("beam_select")
        ops._commit_step_kernel[(1,)](
            s["active_mask"],
            s["num_generated_tokens"],
            s["num_decode_steps"],
            s["num_active_beams"],
            s["num_completed"],
            s["is_initialized"],
            s["is_finished"],
            w["next_completed"],
            w["execute"],
            WIDTH=width,
            BLOCK=block,
            PREFILL=prefill,
        )
        if metrics is not None:
            metrics.stop("beam_select")


@dataclass(frozen=True, slots=True)
class BeamCandidateBatch:
    """Device candidate tensors with immutable request-major row ownership."""

    layout: BeamBatchExecutionView
    token_ids: torch.Tensor
    logprobs: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self, *, prefill: bool, device: torch.device, candidate_width: int) -> None:
        # Rows are Beam rows (batch size, or one row per active Beam); columns
        # are the candidate width K, which is independent of the Beam width.
        rows = self.layout.batch_size if prefill else self.layout.num_execution_rows
        width = candidate_width
        if prefill != (self.layout.kind == "prefill"):
            raise ValueError("candidate stage does not match its batch layout")
        for name, tensor, dtype in (
            ("token_ids", self.token_ids, torch.int32),
            ("logprobs", self.logprobs, torch.float32),
            ("valid_mask", self.valid_mask, torch.bool),
        ):
            if (
                tensor.shape != (rows, width)
                or tensor.dtype != dtype
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous {dtype} [{rows}, {width}] on {device}")


class BeamBatchSearchRunner:
    """Atomic request-major adapter over existing request-local GPU runners."""

    def __init__(
        self,
        runners: tuple[BeamSearchRunner, ...],
        layout: BeamBatchExecutionView,
    ) -> None:
        if not runners or len(runners) != layout.batch_size:
            raise ValueError("Beam batch runner count must match the execution layout")
        first = runners[0]
        if (
            tuple(runner.state_index for runner in runners) != tuple(range(layout.batch_size))
            or tuple(runner.workspace_index for runner in runners)
            != tuple(range(layout.batch_size))
            or tuple(runner.session.slot for runner in runners) != tuple(range(layout.batch_size))
            or any(runner.state is not first.state for runner in runners)
            or any(runner.workspace is not first.workspace for runner in runners)
            or any(runner.session.context is not first.session.context for runner in runners)
            or first.state.beam_width != layout.beam_width
        ):
            raise ValueError("Beam batch resources do not match request-major row ownership")
        self.runners = runners
        self.layout = layout
        self.state = first.state
        self.workspace = first.workspace
        self.state_index = 0
        self.workspace_index = 0

    @property
    def session(self) -> BeamSearchSession:
        # Rebind replaces sessions; state/workspace retain their validated storage.
        return self.runners[0].session

    @property
    def sessions(self) -> tuple[BeamSearchSession, ...]:
        return tuple(runner.session for runner in self.runners)

    @property
    def closed(self) -> bool:
        return all(runner._closed for runner in self.runners)

    def assert_layout(self, layout: BeamBatchExecutionView) -> None:
        self.layout.assert_same_logical_batch(layout)
        if layout.batch_size != len(self.runners):
            raise ValueError("Beam batch size changed after resource binding")

    def reset_for_session(self) -> None:
        for runner in self.runners:
            runner.reset_for_session()

    def initialize_from_prefill(self, candidates: BeamCandidateBatch) -> None:
        candidates.validate(
            prefill=True,
            device=self.state.device,
            candidate_width=self.workspace.key.top_k,
        )
        self.assert_layout(candidates.layout)
        for index, runner in enumerate(self.runners):
            row = slice(index, index + 1)
            runner.initialize_from_prefill(
                candidates.token_ids[row],
                candidates.logprobs[row],
                candidates.valid_mask[row],
            )

    def advance(self, candidates: BeamCandidateBatch) -> None:
        candidates.validate(
            prefill=False,
            device=self.state.device,
            candidate_width=self.workspace.key.top_k,
        )
        self.assert_layout(candidates.layout)
        width = self.state.beam_width
        for index, runner in enumerate(self.runners):
            rows = slice(index * width, (index + 1) * width)
            runner.advance(
                candidates.token_ids[rows],
                candidates.logprobs[rows],
                candidates.valid_mask[rows],
            )

    def end_session(self, consumer_done: torch.Event | None = None) -> None:
        if self.closed:
            return
        stream = torch.cuda.current_stream(self.state.device)
        if consumer_done is not None:
            stream.wait_event(consumer_done)
        done = torch.Event()
        done.record(stream)
        done.synchronize()
        for runner in self.runners:
            if not runner._closed:
                runner.session.end_session()
                runner._closed = True
