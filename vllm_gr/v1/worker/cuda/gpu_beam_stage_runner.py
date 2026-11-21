# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared CUDA core for asynchronous scheduling and persistent Beam state.

Bindings are views of the existing Worker session registry. Runner-specific
operations are supplied explicitly. Resource rows and physical model rows are explicit
so expanding storage capacity does not implicitly enable concurrent scratch use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Condition, Lock
from typing import Any, cast

import torch
from torch.cuda.streams import Event as CUDAEvent
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_gr.config import resolve_candidate_width
from vllm_gr.config.models import deployment_for
from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend
from vllm_gr.ops.cuda.unconstrained_topk_mcap import (
    LOCAL_TILE_SIZE,
)
from vllm_gr.ops.cuda.unconstrained_topk_mcap import (
    MCarryTopKScratch as UnconstrainedTopKScratch,
)
from vllm_gr.ops.cuda.unconstrained_topk_mcap import (
    allocate_mcap_topk_scratch as allocate_unconstrained_topk_scratch,
)
from vllm_gr.ops.cuda.unconstrained_topk_mcap import (
    repair_emit,
    stream_slice,
)
from vllm_gr.ops.cuda.unconstrained_topk_mcap import (
    unconstrained_topk_mcap_out as unconstrained_topk_out,
)
from vllm_gr.v1.attention.backends.beam_prefix_policy import configured_fa_version
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    SUPPORTED_BEAM_BATCH_SIZES,
    BeamBatchFailure,
    BeamBatchResult,
)
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState
from vllm_gr.v1.beam.dispatch import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRDispatchSession,
    GRExecutionCapacity,
    GRRetireMetadata,
    GRRetireResult,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
    validate_dispatch,
    validate_dispatch_session,
)
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.metrics.timer import WallTimer
from vllm_gr.v1.worker.cuda.beam_final_output import GPUBeamFinalOutput
from vllm_gr.v1.worker.cuda.beam_search_runner import (
    BeamBatchSearchRunner,
    BeamCandidateBatch,
    BeamSearchRunner,
)
from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import BeamPostprocessWorkspaceKey
from vllm_gr.v1.worker.cuda.gpu_stage_hooks import (
    CudaStageHooks,
    CudaStageHooksType,
    validate_stage_hooks_type,
)
from vllm_gr.v1.worker.cuda.gr_inputs import (
    GRInputBuffers,
    PreparedGRInputs,
    bind_device_inputs,
)

METADATA_ABI = 1
_UNCONSTRAINED_HEAD_LAYOUT_LOCK = Lock()
CONTROL_CAPACITY = 2
SUPPORTED_BATCH_SIZES = SUPPORTED_BEAM_BATCH_SIZES


def _batch_finished(is_finished: torch.Tensor, error_code: torch.Tensor) -> torch.Tensor:
    """Finish a logical batch only when all members finish or any member errors."""
    return is_finished.all() | (error_code != 0).any()


@dataclass(frozen=True)
class BeamRequestBinding:
    session_id: str
    item_index: int
    native_request_id: str
    state_row: int
    kv_slot: int
    workspace_row: int


@dataclass(frozen=True)
class BeamDispatchEntry:
    binding: BeamRequestBinding
    metadata: GRStageMetadata
    row_offset: int
    row_count: int


@dataclass(frozen=True)
class BeamDispatchExecution:
    entries: tuple[BeamDispatchEntry, ...]
    num_model_rows: int
    metadata_abi: int = METADATA_ABI


# The upstream base is intentionally outside the explicit mypy boundary.
class AsyncGPUBeamOutput(AsyncModelRunnerOutput):  # type: ignore[misc]
    """Only the executor's output consumer waits and constructs the IPC value."""

    def __init__(
        self,
        owner: GPUBeamStageRunner,
        execution: BeamDispatchExecution,
        slot: int,
        terminal_payload: bool,
    ):
        self.owner = owner
        self.execution = execution
        self.slot = slot
        self.terminal_payload = terminal_payload
        self._output: ModelRunnerOutput | None = None
        self._lock = Lock()

    def get_output(self) -> ModelRunnerOutput:
        with self._lock:
            if self._output is not None:
                return self._output
            # CUDA/D2H failures are fatal and must reach the native Executor
            # failure channel. Never publish a successful release after one.
            metrics = None
            try:
                if self.terminal_payload:
                    controls = [self.owner.final_output.wait_embedded_control()]
                else:
                    self.owner.copy_done[self.slot].synchronize()
                    controls = self.owner.host_control[self.slot].tolist()
                timer = getattr(self.owner, "metrics", None)
                metrics = timer.consume(self.slot) if timer is not None else None
                materialize = (
                    WallTimer()
                    if metrics is not None and metrics.get("sampled") and self.terminal_payload
                    else None
                )
                if materialize is not None:
                    materialize.start()
                self._output = self._convert(controls, final_ready=self.terminal_payload)
                if materialize is not None and metrics is not None:
                    metrics["timings"]["output_materialize"] = materialize.stop()
            except (ValueError, TypeError, IndexError) as error:
                self._output = self._conversion_failure(error)
            finally:
                with self.owner.control_drained:
                    self.owner.control_busy[self.slot] = False
                    self.owner.control_drained.notify_all()
            results = getattr(self._output, GR_WORKER_RESULT_ATTR)
            for session_id, result in results.items():
                results[session_id] = replace(result, metrics=metrics)
            # The Scheduler looks ahead, so a terminal receipt from an earlier
            # stage can still be followed by an already-issued successor. Only
            # the declared last stage proves that nothing is still in transit;
            # early termination keeps the explicit retirement round trip.
            last_stage = self.execution.entries[-1].metadata.is_last_stage
            if last_stage and any(result.finished for result in results.values()):
                self.owner.request_terminal_release()
            if self.owner.release_terminal_if_drained():
                session_id = self.owner.binding.session_id
                result = results[session_id]
                results[session_id] = replace(result, resources_released=True)
            return self._output

    def _conversion_failure(self, error: Exception) -> ModelRunnerOutput:
        out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
        results = {}
        for entry in self.execution.entries:
            m = entry.metadata
            failure = BeamBatchResult(
                BEAM_BATCH_PROTOCOL_VERSION,
                m.session_id,
                None,
                BeamBatchFailure("worker_output_error", str(error), entry.binding.item_index),
            )
            terminal = self.owner.final_output.failure(self.owner, failure)
            results[m.session_id] = GRWorkerResult(
                m.session_id,
                m.dispatch_id,
                m.stage_index,
                0,
                m.output_device_state,
                True,
                terminal,
                owner_generation=m.owner_generation,
            )
        out.req_ids = [binding.native_request_id for binding in self.owner.bindings]
        out.req_id_to_index = {request_id: i for i, request_id in enumerate(out.req_ids)}
        out.sampled_token_ids = [[] for _ in out.req_ids]
        setattr(out, GR_WORKER_RESULT_ATTR, results)
        return out

    def _convert(self, controls: list[list[int]], *, final_ready: bool) -> ModelRunnerOutput:
        if len(controls) != len(self.execution.entries):
            raise ValueError("Asynchronous scheduling control snapshot has the wrong row count")
        req_ids, sampled, results = [], [], {}
        for entry, (produced, finished, error) in zip(self.execution.entries, controls):
            m = entry.metadata
            terminal = None
            final_output = self.owner.final_output
            if error:
                failure = BeamBatchResult(
                    protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
                    batch_id=m.session_id,
                    output=None,
                    failure=BeamBatchFailure(
                        "beam_device_error",
                        f"Beam device error {error}",
                        entry.binding.item_index,
                    ),
                )
                terminal = final_output.failure(self.owner, failure)
            elif finished and not self.owner.cancelled:
                try:
                    terminal = final_output.consume(self.owner, ready=final_ready)
                except (ValueError, TypeError, IndexError) as exc:
                    failure = BeamBatchResult(
                        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
                        batch_id=m.session_id,
                        output=None,
                        failure=BeamBatchFailure(
                            "beam_final_output_error",
                            f"Failed to materialize final Beam output: {exc}",
                            entry.binding.item_index,
                        ),
                    )
                    terminal = final_output.failure(self.owner, failure)
            # These are scheduler placeholders. They must never be used as
            # model inputs or published as generated Beam sequences.
            results[m.session_id] = GRWorkerResult(
                m.session_id,
                m.dispatch_id,
                m.stage_index,
                produced,
                m.output_device_state,
                bool(finished),
                terminal,
                owner_generation=m.owner_generation,
            )
        req_ids = [binding.native_request_id for binding in self.owner.bindings]
        sampled = [([0] if produced else []) for _ in req_ids]
        out = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
        )
        setattr(out, GR_WORKER_RESULT_ATTR, results)
        return out


def should_replay_decode_graph(
    worker: Any, graph: Any, *, capture_failed: bool | None = None
) -> bool:
    """Resolve decode graph replay, preserving unexpected-missing invariants."""
    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    requested = not worker.model_config.enforce_eager and _get_gr_config(worker).beam_graph_enabled
    if not requested:
        return False
    if capture_failed is None:
        capture_failed = getattr(worker, "_gr_async_decode_graph_capture_failed", False)
    if capture_failed:
        if graph is not None:
            raise RuntimeError("Decode capture failed but a graph is still published")
        return False
    if graph is not None:
        if not graph.ready:
            raise RuntimeError("Published Decode graph is not ready")
        return True
    raise RuntimeError("Asynchronous scheduling graph mode requires successful startup capture")


def _get_unconstrained_head_layout(
    worker: Any,
) -> tuple[torch.Tensor, torch.Event] | None:
    """Cache a read-only, vocabulary-contiguous Qwen3 head for Decode."""
    settings = deployment_for(getattr(worker, "vllm_config", None))
    if settings.unconstrained_head_layout != "transposed":
        return None
    if (
        worker.parallel_config.tensor_parallel_size != 1
        or worker.parallel_config.pipeline_parallel_size != 1
        or getattr(worker, "lora_config", None) is not None
        or getattr(worker.model_config, "quantization", None) is not None
        or getattr(worker.model_config.hf_config, "architectures", None) != ["Qwen3ForCausalLM"]
    ):
        return None
    from vllm import envs
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    model = worker.model
    head = getattr(model, "lm_head", None)
    processor = getattr(model, "logits_processor", None)
    if (
        envs.VLLM_BATCH_INVARIANT
        or not isinstance(head, VocabParallelEmbedding)
        or type(getattr(head, "quant_method", None)) is not UnquantizedEmbeddingMethod
        or type(processor) is not LogitsProcessor
        or processor.logits_as_input
        or processor.soft_cap is not None
        or processor.scale != 1.0
        or getattr(head, "bias", None) is not None
    ):
        return None
    weight = head.weight
    if (
        not weight.is_cuda
        or not weight.is_contiguous()
        or weight.ndim != 2
        or weight.dtype not in (torch.bfloat16, torch.float16)
        or weight.shape[0] != processor.org_vocab_size
    ):
        return None
    # Weight-only reload preserves shape/dtype. Refresh the head-owned cache in
    # the post-load hooks installed at model load instead of a tensor-version key:
    # existing stage runners retain this storage and must see the new weights.
    with _UNCONSTRAINED_HEAD_LAYOUT_LOCK:
        cached = getattr(head, "_gr_unconstrained_head_layout", None)
        if cached is not None:
            return cast(tuple[Any, Any], cached)
        transposed = weight.detach().T.contiguous()
        # CUDA-specific external events preserve this dependency as a wait
        # node in captured heads; torch.Event does not expose that option.
        ready = CUDAEvent(external=True)
        ready.record(torch.cuda.current_stream(weight.device))
        head._gr_unconstrained_head_layout = (transposed, ready)
        return transposed, ready


# ``gr.head_slice`` values: 0 disables slicing, -1 decides per stage.
SLICE_AUTO = -1
# Measured on L20 (V=176,256, K=256): blocks narrower than this only add
# launches, and wider than ``HEAD_SLICE_MAX_COLS`` only add memory.
HEAD_SLICE_MIN_COLS = 16384
HEAD_SLICE_MAX_COLS = 32768


def head_slice_plan(
    rows: int,
    vocab_size: int,
    itemsize: int,
    cols: int,
    min_rows: int,
    max_bytes: int,
) -> tuple[bool, int]:
    """Return ``(use_slicing, cols)`` for one selection of ``rows`` rows.

    Slicing removes the ``rows x vocab`` logits round trip but pays a fixed
    per-block cost of roughly 90 us ``vocab / cols`` times, so it only pays for
    wide selections: 1.22x at rows=1024 and 0.95-0.99x at rows=192..768 for
    V=176,256 / K=256 / bf16 on L20.  ``cols=SLICE_AUTO`` derives the width from
    the vocabulary and gates on ``min_rows``; an explicit width forces slicing
    for a manual A/B.  ``max_bytes`` shrinks the block down to
    ``HEAD_SLICE_MIN_COLS`` and no further.
    """

    if cols == 0 or rows <= 0 or vocab_size <= 0:
        return False, 0
    forced = cols != SLICE_AUTO
    if cols == SLICE_AUTO:
        cols = -(-(vocab_size // 6) // LOCAL_TILE_SIZE) * LOCAL_TILE_SIZE
    cols = min(
        max(cols - cols % LOCAL_TILE_SIZE, HEAD_SLICE_MIN_COLS),
        HEAD_SLICE_MAX_COLS,
        vocab_size,
    )
    if cols >= vocab_size:
        # A single block would only add launches to the monolithic path.
        return False, 0
    while max_bytes > 0 and cols > HEAD_SLICE_MIN_COLS and rows * cols * itemsize > max_bytes:
        # The caller addresses blocks by tile, so a reduced width must remain a
        # whole number of tiles; never shrink below the measured minimum.
        cols = max(cols // 2, HEAD_SLICE_MIN_COLS)
    cols -= cols % LOCAL_TILE_SIZE
    return forced or rows >= min_rows, cols


class _HeadLogits:
    """LM-head logits buffers, allocated only for the mode in use.

    A stage switch releases the inactive mode, so an ``auto`` session never
    keeps the full-vocabulary buffer alive beside the slice buffer.
    """

    def __init__(
        self,
        layout: tuple[torch.Tensor, torch.Event] | None,
        device: torch.device,
    ) -> None:
        self._layout = layout
        self._device = device
        self.full: torch.Tensor | None = None
        self.slice: torch.Tensor | None = None

    def full_buffer(self, rows: int) -> torch.Tensor | None:
        """``[rows, vocab]`` logits for the monolithic path."""

        if self._layout is None or rows <= 0:
            return None
        if self.full is None or self.full.shape[0] < rows:
            self.full = torch.empty(
                (rows, self._layout[0].shape[1]),
                dtype=self._layout[0].dtype,
                device=self._device,
            )
        return self.full[:rows]

    def slice_buffer(self, rows: int, cols: int) -> torch.Tensor | None:
        """``[rows, cols]`` logits for one vocabulary block of the sliced path."""

        if self._layout is None or rows <= 0 or cols <= 0:
            return None
        if self.slice is None or self.slice.shape[0] < rows or self.slice.shape[1] != cols:
            self.slice = torch.empty((rows, cols), dtype=self._layout[0].dtype, device=self._device)
        return self.slice[:rows]

    def release_full(self) -> None:
        """Drop the monolithic logits buffer after its last reader."""
        self.full = None

    def release_slice(self) -> None:
        """Drop the slice logits buffer after its last reader."""
        self.slice = None


def _stage_hidden(owner: GPUBeamStageRunner, hidden: torch.Tensor) -> torch.Tensor:
    """Copy Decode hidden states out of the graph's shared replay buffer.

    ``BeamDecodeGraph.replay()`` returns the graph's static output, which the
    next replay of the same geometry overwrites.  Decode keeps working storage
    for its logits, so the sliced path stages hidden states in a private buffer
    instead of aliasing that output across ``execute`` and ``sample``.
    """

    rows, width = hidden.shape
    staged = owner._hidden_staging
    if (
        staged is None
        or staged.shape[0] < rows
        or staged.shape[1] != width
        or staged.dtype != hidden.dtype
        or staged.device != hidden.device
    ):
        staged = torch.empty((rows, width), dtype=hidden.dtype, device=hidden.device)
        owner._hidden_staging = staged
    target = staged[:rows]
    target.copy_(hidden)
    return target


def _hidden_logits(owner: GPUBeamStageRunner, hidden: torch.Tensor) -> torch.Tensor | None:
    """Turn Decode hidden states into logits, or publish them for slicing."""

    if owner._head_slice_request:
        # The row count decides whether slicing pays for this stage.
        cols = owner.head_slice_cols(hidden.shape[0])
        if cols:
            # A mixed ``auto`` session must not keep the full-vocabulary buffer
            # alive beside the slice buffer.  The previous stage's ``sample``
            # already consumed it, and no logits buffer is graph-captured.
            owner._head_buffers.release_full()
            # Blocks are reduced before the next one is produced, so only the
            # hidden states travel forward from here.
            owner.hidden = _stage_hidden(owner, hidden)
            owner._active_slice_cols = cols
            return None
    owner._head_buffers.release_slice()
    owner.hidden = None
    return owner._decode_logits(hidden)


class GPUBeamStageRunner:
    """One exclusive execution owner, using the existing session/state/workspace."""

    def __init__(
        self,
        worker: Any,
        beam: BeamBatchSearchRunner,
        bindings: tuple[BeamRequestBinding, ...],
        backend: CUDAConstraintBackend | None,
        hooks: CudaStageHooks,
        trie_ids: tuple[int, ...] | None = None,
    ):
        expected = tuple(range(len(bindings)))
        if not bindings or (
            tuple(binding.item_index for binding in bindings) != expected
            or tuple(binding.kv_slot for binding in bindings) != expected
            or tuple(binding.state_row for binding in bindings) != expected
            or tuple(binding.workspace_row for binding in bindings) != expected
            or tuple(binding.native_request_id for binding in bindings) != beam.layout.request_ids
            or any(binding.session_id != beam.layout.session_id for binding in bindings)
        ):
            raise ValueError("GR binding disagrees with existing session resources")
        context = beam.session.context
        if (
            getattr(context, "_gr_execution_owner", None) is not None
            or beam.workspace.execution_owner is not None
        ):
            raise RuntimeError("Beam context already has an execution owner")
        self.worker, self.beam, self.bindings, self.backend = worker, beam, bindings, backend
        self.binding = bindings[0]
        self.batch_view = beam.layout
        # Device indexing is part of Decode binding and must not upload a CPU
        # index tensor while a CUDA graph is being captured.
        self.state_rows = torch.tensor(
            tuple(binding.state_row for binding in bindings),
            dtype=torch.long,
            device=beam.state.device,
        )
        self.hooks: CudaStageHooks = hooks
        self.previous: GRStageMetadata | None = None
        self.pending: tuple[BeamDispatchExecution, int] | None = None
        # Facts proven once per session by ``get_stage_runner``; every Decode
        # admission of this session re-uses them instead of re-deriving them.
        self._dispatch_session: GRDispatchSession | None = None
        self._validated_dispatch: (
            tuple[Any, tuple[GRStageMetadata, ...], GRDispatchSession] | None
        ) = None
        self.control_lock = Lock()
        self.control_drained = Condition(self.control_lock)
        self.control_busy = [False] * CONTROL_CAPACITY
        self.device_control = torch.empty(
            (CONTROL_CAPACITY, 1, 3), device=beam.state.device, dtype=torch.int32
        )
        self.host_control = torch.empty(
            (CONTROL_CAPACITY, 1, 3), pin_memory=True, dtype=torch.int32
        )
        self.produced_ready = [torch.Event() for _ in range(CONTROL_CAPACITY)]
        self.copy_done = [torch.Event() for _ in range(CONTROL_CAPACITY)]
        self.copy_stream = torch.cuda.Stream(device=beam.state.device)
        from vllm_gr.v1.metrics.gpu import GPUStageMetrics

        self.metrics = GPUStageMetrics(self, CONTROL_CAPACITY)
        self.last_work = torch.Event()
        self.closed = False
        self.cancelled = False
        self.terminal_release_requested = False
        self.terminal_cleanup_pending = False
        self.execution_error: Exception | None = None
        self.dynamic: Any = None
        self.unconstrained_topk_scratch: UnconstrainedTopKScratch | None = None
        self.inputs: GRInputBuffers | None = None
        self.prepared: PreparedGRInputs | None = None
        self.final_output: GPUBeamFinalOutput
        self.metadata: Any = None
        self.graph: Any = None
        self.logits: torch.Tensor | None = None
        batch_size = len(bindings)
        rows = batch_size * beam.state.beam_width
        host = getattr(worker, "vllm_config", None)
        self.gr_settings = deployment_for(host)
        self.explicit_fa_version = configured_fa_version(host)
        settings = self.gr_settings
        slice_request = 0
        if backend is None and self.hooks.supports_transposed_head:
            slice_request = SLICE_AUTO if settings.head_slice == "auto" else settings.head_slice
        # The transposed copy is only consumed by the opt-in sliced-head path.
        # Gate before creating it: the worker cache retains this allocation.
        self._head_layout = _get_unconstrained_head_layout(worker) if slice_request else None
        # Opt-in vocabulary slicing: drive the LM head one column block at a time
        # and reduce each block before moving on.  ``auto`` slices only the
        # stages whose row count makes it pay; switching modes releases the
        # inactive logits buffer, so a mixed session keeps at most one of the
        # full and slice allocations at a time.
        self.hidden: torch.Tensor | None = None
        self._hidden_staging: torch.Tensor | None = None
        self._head_buffers = _HeadLogits(self._head_layout, beam.state.device)
        self._head_slice_request = slice_request if self._head_layout is not None else 0
        self._head_slice_min_rows = settings.head_slice_min_rows
        self._head_slice_max_bytes = int(settings.head_slice_max_mb * 2**20)
        self._active_slice_cols = 0
        self.input_ids: torch.Tensor
        self.positions: torch.Tensor
        self.slots: torch.Tensor
        self.execution_mask: torch.Tensor
        self.hooks.initialize_inputs(self, rows)
        self.constraint_steps = beam.state.num_generated_tokens[:batch_size]
        w = beam.workspace
        # Candidate geometry is (rows, K): the row count is the beam width and
        # the column count is the independent candidate width.
        candidate_width = w.key.top_k
        self.prefixes = w.prefix_token_ids[:batch_size].reshape(rows, -1)
        self.temperatures = w.temperatures[:batch_size].reshape(rows)
        self.tokens = w.topk_token_ids[:batch_size].reshape(rows, candidate_width)
        self.scores = w.topk_scores[:batch_size].reshape(rows, candidate_width)
        self.valid = w.topk_valid_mask[:batch_size].reshape(rows, candidate_width)
        self.counts = w.topk_valid_counts[:batch_size].reshape(rows)
        self.trie_ids_by_width: dict[int, torch.Tensor] = {}
        self.configure_trie_ids(trie_ids)
        context._gr_execution_owner = self
        beam.workspace.execution_owner = self

    def configure_trie_ids(self, trie_ids: tuple[int, ...] | None) -> None:
        """Keep per-width request-major selectors stable across adapter reuse."""
        if self.backend is None or self.backend.info.table_metadata.trie_count == 1:
            return
        if trie_ids is None or len(trie_ids) != len(self.bindings):
            raise ValueError("multi-trie execution requires one trie_id per request")
        source = torch.tensor(trie_ids, dtype=torch.int32, device=self.beam.state.device)
        widths = {1, self.beam.state.beam_width, *(self.beam.layout.dynamic_beam_widths or ())}
        for width in widths:
            target = self.trie_ids_by_width.get(width)
            if target is None:
                target = torch.empty(
                    (len(trie_ids) * width,), dtype=torch.int32, device=source.device
                )
                self.trie_ids_by_width[width] = target
            target.view(len(trie_ids), width).copy_(source[:, None])

    def admit(self, scheduler_output: Any) -> BeamDispatchExecution:
        if self.closed:
            raise RuntimeError("GR session is closed")
        if (
            self.backend is not None
            and not self.backend.is_ready
            or self.beam.workspace.execution_owner is not self
            or self.beam.session.context._gr_execution_owner is not self
        ):
            raise RuntimeError("GR backend or shared execution ownership changed")
        if self.pending is not None:
            raise RuntimeError("sample_tokens must consume the previous GR execute state")
        validated = self._validated_dispatch
        self._validated_dispatch = None
        if validated is not None and validated[0] is scheduler_output:
            stages, self._dispatch_session = validated[1], validated[2]
        else:
            stages = validate_dispatch(
                scheduler_output,
                previous=self.previous,
                bindings=self.bindings,
                session=self._dispatch_session if self.previous is not None else None,
            )
        for m in stages:
            state, params = self.beam.state, m.beam_params
            if (
                state.beam_width != params.beam_width
                or state.max_new_tokens != params.max_tokens
                or state.stop_token_ids_config != params.stop_token_ids
                or state.ignore_stop_tokens_config != params.ignore_stop_tokens
            ):
                raise ValueError("GR metadata configuration does not match persistent Beam state")
        if not hasattr(self, "final_output"):
            first = stages[0]
            self.final_output = GPUBeamFinalOutput(
                self, self.bindings, first.beam_params, first.output_options
            )
        elif self.previous is None or self.final_output.bindings != self.bindings:
            first = stages[0]
            if self.final_output.is_compatible(self):
                self.final_output.reset(
                    self, self.bindings, first.beam_params, first.output_options
                )
            else:
                self.final_output = GPUBeamFinalOutput(
                    self, self.bindings, first.beam_params, first.output_options
                )
        entries, offset = [], 0
        for m in stages:
            view = BeamBatchExecutionView.from_stage_metadata(m)
            count = view.num_execution_rows if m.kind is GRStageKind.DECODE else view.batch_size
            entries.append(BeamDispatchEntry(self.binding, m, offset, count))
            offset += count
        with self.control_lock:
            slot = next((i for i, busy in enumerate(self.control_busy) if not busy), None)
            if slot is None:
                raise RuntimeError("both GR dispatch control slots await output consumption")
            self.control_busy[slot] = True
        execution = BeamDispatchExecution(tuple(entries), offset)
        self.pending = execution, slot
        self.previous = stages[0]
        torch.cuda.current_stream(self.beam.state.device).wait_event(self.last_work)
        self.device_control[slot, 0, :1].copy_(
            self.beam.state.num_generated_tokens[: len(self.bindings)].amax().reshape(1)
        )
        return execution

    def prepare_cpu(self, m: GRStageMetadata, slot: int) -> PreparedGRInputs:
        worker = self.worker
        if self.dynamic is not None:
            self.dynamic.activate(BeamBatchExecutionView.from_stage_metadata(m).input_width)
        groups = worker.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise RuntimeError("Asynchronous scheduling requires one uniform KV cache group")
        if self.inputs is None:
            self.inputs = GRInputBuffers(
                self.beam.state.beam_width,
                worker.model_config.max_model_len,
                groups[0].kv_cache_spec.block_size,
                self.beam.state.device,
                max_batch=len(self.bindings),
            )
        block_ids = (
            tuple([list(row)] for row in m.kv_blocks)
            if m.kv_blocks is not None
            else tuple(
                self.hooks.decode_block_ids(binding.native_request_id) for binding in self.bindings
            )
        )
        return self.inputs.prepare_cpu_batch(
            m, self.bindings, block_ids_by_request=block_ids, slot=slot
        )

    def _bind_attention(self, m: GRStageMetadata) -> None:
        from vllm_gr.v1.attention.backends.beam_attn_gpu import (
            GPUBeamAttentionBatch,
        )
        from vllm_gr.v1.attention.backends.beam_attn_metadata import (
            build_async,
        )

        if self.prepared is None or self.prepared.metadata != m or self.inputs is None:
            raise RuntimeError(
                "Asynchronous scheduling attention requires this dispatch's CPU preparation"
            )
        worker = self.worker
        views = self.inputs.upload(self.prepared)
        self.device_positions = views["positions"]
        self.metadata = build_async(
            prefix_policy=self.gr_settings.prefix_policy,
            prefix_min_chain_us=self.gr_settings.prefix_min_chain_us,
            explicit_fa_version=self.explicit_fa_version,
            prefix_calibrations=getattr(self.worker, "_gr_prefix_calibrations", None),
            layout=self.prepared.layout,
            views=views,
            session=GPUBeamAttentionBatch(
                self.beam.sessions,
                width=BeamBatchExecutionView.from_stage_metadata(m).input_width,
            ),
            mask=self.execution_mask,
            slots=self.slots,
            block_size=self.inputs.block_size,
            max_model_len=self.inputs.max_model_len,
            num_heads=worker.model_config.get_num_attention_heads(worker.parallel_config),
            num_kv_heads=worker.model_config.get_num_kv_heads(worker.parallel_config),
        )
        names = tuple(
            name for group in worker.kv_cache_config.kv_cache_groups for name in group.layer_names
        )
        self.attention = dict.fromkeys(names, self.metadata)
        self.slot_mappings = dict.fromkeys(names, self.slots)

    def prepare_decode(self, entry: BeamDispatchEntry) -> None:
        # Device binding has no dependency on Host output consumption.
        prepared = self.prepared
        if prepared is None:
            raise RuntimeError(
                "Asynchronous scheduling device binding does not match the prepared dispatch"
            )
        prepared_bindings = (
            prepared.binding if isinstance(prepared.binding, tuple) else (prepared.binding,)
        )
        if prepared.metadata != entry.metadata or prepared_bindings != self.bindings:
            raise RuntimeError(
                "Asynchronous scheduling device binding does not match the prepared dispatch"
            )
        bind_device_inputs(prepared, self)

    def capture_forward(self) -> Any:
        """Return a captured candidate; startup publishes it after cleanup."""
        graph = self.hooks.decode_graph(
            len(self.bindings),
            self.inputs.width if self.inputs is not None else self.beam.state.beam_width,
        )
        bucket = getattr(getattr(self, "metadata", None), "prefix_length_bucket", 0)
        if bucket:
            assert self.inputs is not None
            graph.select(len(self.bindings), self.inputs.width, bucket)
        graph.capture(self._forward_hidden, persistent_install=self.hooks.persistent_graph_install)
        return graph

    def _forward(self) -> torch.Tensor | None:
        """Eager Decode entry point; publishes hidden when the head is sliced."""

        return _hidden_logits(self, self._forward_hidden(False))

    def head_slice_cols(self, rows: int) -> int:
        """Vocabulary block width for one selection; 0 means "do not slice"."""

        layout = self._head_layout
        if layout is None:
            return 0
        use_slicing, cols = head_slice_plan(
            rows,
            int(layout[0].shape[1]),
            layout[0].element_size(),
            self._head_slice_request,
            self._head_slice_min_rows,
            self._head_slice_max_bytes,
        )
        return cols if use_slicing else 0

    def _decode_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self._try_transposed_head_logits(hidden)
        return logits if logits is not None else self.worker.model.compute_logits(hidden)

    def _select_from_hidden_slices(
        self,
        hidden: torch.Tensor,
        rows: int,
        top_k: int,
        scratch: UnconstrainedTopKScratch,
        candidates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        counts: torch.Tensor,
    ) -> None:
        """Reduce the LM head one vocabulary block at a time.

        Each block is consumed while it is still in cache, so the full
        ``[rows, vocab]`` logits matrix never exists. Each block's retained set
        is merged and repaired on the GPU before the buffer is reused, so no
        host synchronization or repeated head GEMM is needed for exact Top-K.
        """

        if hidden.ndim != 2 or hidden.shape[0] != rows or not hidden.is_contiguous():
            raise RuntimeError(
                "vocabulary slicing expects contiguous hidden states with "
                f"{rows} rows, got shape {tuple(hidden.shape)}"
            )
        layout = self._head_layout
        if layout is None:
            raise RuntimeError("vocabulary slicing requires the transposed head layout")
        transposed, ready = layout
        torch.cuda.current_stream(transposed.device).wait_event(ready)
        vocab_size = int(transposed.shape[1])
        cols = self._active_slice_cols
        buffer = self._head_buffers.slice_buffer(rows, cols)
        if buffer is None:
            raise RuntimeError("vocabulary slicing requires a slice logits buffer")
        temperatures = self.temperatures[:rows]

        def block(start: int) -> torch.Tensor:
            width = min(cols, vocab_size - start)
            # Pack the tail into the same storage with a contiguous row stride.
            # The previous slice has been consumed before this buffer is reused.
            target = buffer.view(-1)[: rows * width].view(rows, width)
            torch.mm(hidden, transposed[:, start : start + width], out=target)
            return target

        for start in range(0, vocab_size, cols):
            stream_slice(
                block(start),
                temperatures,
                top_k,
                scratch,
                rows,
                tile_offset=start // LOCAL_TILE_SIZE,
            )
        repair_emit(rows, top_k, scratch, *candidates, counts)

    def _try_transposed_head_logits(self, hidden: torch.Tensor) -> torch.Tensor | None:
        layout = self._head_layout
        if (
            not self._head_slice_request
            or layout is None
            or hidden.ndim != 2
            or not hidden.is_contiguous()
            or hidden.shape[1] != layout[0].shape[0]
        ):
            # The transposed GEMM belongs to the sliced path; the monolithic
            # head keeps the model's own ``compute_logits``.
            return None
        logits = self._head_buffers.full_buffer(hidden.shape[0])
        if logits is None or hidden.device != logits.device or hidden.dtype != logits.dtype:
            return None
        transposed, ready = layout
        torch.cuda.current_stream(hidden.device).wait_event(ready)
        torch.mm(hidden, transposed, out=logits)
        return logits

    def _forward_hidden(self, capture: bool) -> torch.Tensor:
        return self.hooks.forward_hidden(
            self.input_ids, self.positions, self.attention, self.slot_mappings, capture
        )

    def execute(
        self, scheduler_output: Any, original: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        previous_kind = self.previous.kind if self.previous is not None else None
        execution = self.admit(scheduler_output)
        worker = self.worker
        worker._current_scheduler_output = scheduler_output
        worker._current_beam_data = {}
        worker._current_beam_info = None
        assert self.pending is not None
        self.metrics.begin(self.pending[1], execution.entries[0].metadata, self)
        for entry in execution.entries:
            m = entry.metadata
            if m.kind is GRStageKind.PREFILL:
                self.temperatures.fill_(m.beam_params.temperature)
                self.prefixes.fill_(-1)
                self.hidden = None
                self.logits = self.hooks.prefill_logits(original, scheduler_output, *args, **kwargs)
            else:
                # Pure CPU work can run before the native Prefill staging fence.
                assert self.pending is not None
                self.prepared = self.prepare_cpu(m, self.pending[1])
                if previous_kind is GRStageKind.PREFILL:
                    # Native Prefill may asynchronously read reused CPU input
                    # buffers that the native state update mutates.  The
                    # runner-specific fence orders those Host writes.
                    self.hooks.prefill_stage_fence()
                # Decode does not upload these native input buffers again, so
                # subsequent Decode dispatches need no Host input-prep wait.
                self.hooks.consume_native_state(scheduler_output)
                self._bind_attention(m)
                self.prepare_decode(entry)
                self.logits = self.hooks.decode_logits(self)
        return None

    def sample(self, grammar_output: Any = None) -> AsyncGPUBeamOutput:
        if self.execution_error is not None:
            raise RuntimeError(
                "Asynchronous scheduling execute_model failed"
            ) from self.execution_error
        if grammar_output is not None or self.pending is None:
            raise RuntimeError("GR sample requires a pending dispatch and no grammar output")
        execution, slot = self.pending
        self.metrics.stop("forward")
        terminal_payload = False
        for entry in execution.entries:
            m = entry.metadata
            if m.produces_output and not self.cancelled:
                model_output = self.hidden if self.hidden is not None else self.logits
                if model_output is None:
                    raise RuntimeError("GR output stage has no native model logits")
                layout = BeamBatchExecutionView.from_stage_metadata(m)
                rows = layout.batch_size if m.kind is GRStageKind.PREFILL else entry.row_count
                candidate_width = layout.candidate_top_k
                device = model_output.device
                self.metrics.start("constraint_select")
                candidates = (
                    self.dynamic.candidate_buffers(rows, candidate_width)
                    if self.dynamic is not None
                    else (self.tokens[:rows], self.scores[:rows], self.valid[:rows])
                )
                # The backend's unchecked entry point validates nothing: it
                # addresses outputs as ``row * K + offset`` and skips shape
                # checks.  A candidate buffer whose column count disagrees with
                # the requested K would write past its allocation (K < rows of
                # the view) or leave candidate columns unwritten (K larger).
                # Check the geometry here, before the launch.
                expected = (rows, candidate_width)
                for name, tensor, dtype in (
                    ("candidate token ids", candidates[0], torch.int32),
                    ("candidate logprobs", candidates[1], torch.float32),
                    ("candidate valid mask", candidates[2], torch.bool),
                ):
                    if (
                        tuple(tensor.shape) != expected
                        or tensor.dtype != dtype
                        or tensor.device != device
                        or not tensor.is_contiguous()
                    ):
                        raise RuntimeError(
                            f"GR candidate {name} must be contiguous {dtype} "
                            f"{expected} on {device}, got "
                            f"{tuple(tensor.shape)} {tensor.dtype} {tensor.device}"
                        )
                if self.backend is not None:
                    self.backend.select_out_unchecked(
                        self.logits,
                        ConstraintMetadata(
                            self.prefixes[:rows],
                            self.temperatures[:rows],
                            0,
                            layout.candidate_top_k,
                            trie_ids=self.trie_ids_by_width.get(layout.input_width),
                        ),
                        out_token_ids=candidates[0],
                        out_logprobs=candidates[1],
                        out_valid_mask=candidates[2],
                        out_valid_counts=self.counts[:rows],
                        device_step=self.constraint_steps,
                        specialize_steps=self.hooks.specialize_constraint_steps,
                    )
                else:
                    sliced = self.hidden is not None
                    logits = None if sliced else model_output[:rows]
                    if sliced:
                        assert self._head_layout is not None
                        vocab_size = int(self._head_layout[0].shape[1])
                    else:
                        vocab_size = int(model_output.shape[-1])
                    if (
                        self.unconstrained_topk_scratch is None
                        or self.unconstrained_topk_scratch.top_k != layout.candidate_top_k
                        or self.unconstrained_topk_scratch.vocab_size != vocab_size
                    ):
                        self.unconstrained_topk_scratch = allocate_unconstrained_topk_scratch(
                            batch_size=len(self.bindings) * self.beam.state.beam_width,
                            vocab_size=vocab_size,
                            top_k=layout.candidate_top_k,
                            device=device,
                            keep_limit=self.gr_settings.topk_keep,
                            select=self.gr_settings.topk_select,
                        )
                    if sliced:
                        self._select_from_hidden_slices(
                            self.hidden,
                            rows,
                            layout.candidate_top_k,
                            self.unconstrained_topk_scratch,
                            candidates,
                            self.counts[:rows],
                        )
                    else:
                        unconstrained_topk_out(
                            logits,
                            self.temperatures[:rows],
                            layout.candidate_top_k,
                            self.unconstrained_topk_scratch,
                            candidates[0],
                            candidates[1],
                            candidates[2],
                            self.counts[:rows],
                        )
                self.metrics.stop("constraint_select")
                self.metrics.note_candidate_width(slot, int(candidates[0].shape[1]))
                timer = self.metrics if self.metrics.active_slot is not None else None
                for runner in getattr(self.beam, "runners", (self.beam,)):
                    runner._gr_metrics = timer
                candidate_batch = BeamCandidateBatch(layout, *candidates)
                if self.dynamic is not None:
                    self.dynamic.state.metrics = timer
                    self.metrics.start("beam_select")
                    self.dynamic.advance(layout, candidates, m.is_last_stage)
                    self.metrics.stop("beam_select")
                elif m.kind is GRStageKind.PREFILL:
                    self.beam.initialize_from_prefill(candidate_batch)
                else:
                    self.beam.advance(candidate_batch)
                if not m.is_last_stage:
                    self.metrics.snapshot(self, layout)
            state_rows = slice(0, len(self.bindings))
            control = self.device_control[slot, 0]
            control[:1].copy_(
                self.beam.state.num_generated_tokens[state_rows].amax().reshape(1) - control[:1]
            )
            control[1:2].copy_(
                _batch_finished(
                    self.beam.state.is_finished[state_rows],
                    self.beam.state.error_code[state_rows],
                ).reshape(1)
            )
            control[2:3].copy_(self.beam.state.error_code[state_rows].amax().reshape(1))
            # The fixed-length terminal stage returns one packed payload whose
            # header replaces the separate per-stage control D2H.
            if m.is_last_stage:
                self.metrics.start("final_pack")
                terminal_payload = self.final_output.enqueue(self, control)
                if not terminal_payload:
                    self.metrics.stop("final_pack")
        stream = torch.cuda.current_stream(self.beam.state.device)
        self.produced_ready[slot].record(stream)
        self.last_work.record(stream)
        if not terminal_payload:
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(self.produced_ready[slot])
                self.host_control[slot].copy_(self.device_control[slot], non_blocking=True)
                self.metrics.enqueue_copy(slot)
                self.copy_done[slot].record(self.copy_stream)
        self.hooks.finish_native_output()
        self.pending = None
        self.metrics.active_slot = None
        return AsyncGPUBeamOutput(self, execution, slot, terminal_payload)

    def completion_view(self, binding: BeamRequestBinding) -> dict[str, torch.Tensor]:
        if self.closed or binding not in self.bindings:
            raise RuntimeError("completion view requires a live matching binding")
        if self.pending is not None:
            raise RuntimeError("completion view requires the current dispatch to be sampled")
        torch.cuda.current_stream(self.beam.state.device).wait_event(self.last_work)
        return {
            name: value[binding.state_row : binding.state_row + 1]
            for name, value in vars(self.beam.state).items()
            if isinstance(value, torch.Tensor)
        }

    def request_release(self, metadata: GRRetireMetadata) -> AsyncGRRetireOutput:
        if (
            self.previous is None
            or metadata.session_id != self.binding.session_id
            or metadata.owner_generation != self.previous.owner_generation
            or self.pending is not None
        ):
            raise RuntimeError(
                "Asynchronous scheduling retirement identity does not match the drained execution"
            )
        self.final_output.abort(self)
        self.cancel(self.binding)
        return AsyncGRRetireOutput(self, metadata)

    def request_terminal_release(self) -> None:
        """Stop further work after a terminal receipt; release after all consumers drain."""
        with self.control_lock:
            if self.terminal_release_requested:
                return
            # A terminal control snapshot already proves that device execution
            # stopped. Avoid enqueueing the fallback cancel writes on the hot
            # path; any look-ahead receipt is covered by the latest last_work.
            self.final_output.abort(self)
            self.cancelled = True
            self.terminal_release_requested = True

    def release_terminal_if_drained(self) -> bool:
        """Release on the last terminal output consumer, without a control dispatch."""
        with self.control_lock:
            ready = (
                self.terminal_release_requested
                and not self.terminal_cleanup_pending
                and not self.closed
                and self.pending is None
                and not any(self.control_busy)
            )
            if ready:
                self.terminal_cleanup_pending = True
        if not ready:
            return False
        self.release(
            self.binding,
            consumer_done=self.final_output.consumer_done(self),
        )
        # Native Scheduler cleanup normally arrives in the next
        # SchedulerOutput.finished_req_ids. The fused terminal path has no
        # follow-up control dispatch, so retire the Worker's cached request
        # locally after all GPU and output consumers have drained.
        cleanup = SchedulerOutput.make_empty()
        cleanup.finished_req_ids.update(binding.native_request_id for binding in self.bindings)
        self.hooks.consume_native_state(cleanup)
        # release() closes the GPU owner before this native bookkeeping.
        # Publish the consumer's completion only after native cleanup succeeds.
        with self.control_lock:
            self.terminal_cleanup_pending = False
        return True

    @torch.inference_mode()  # type: ignore[misc]
    def cancel(self, binding: BeamRequestBinding) -> None:
        if binding not in self.bindings:
            raise ValueError("cancel binding mismatch")
        if self.closed or self.cancelled:
            return
        if self.pending is not None:
            raise RuntimeError("sample the dispatched work before cancellation")
        stream = torch.cuda.current_stream(self.beam.state.device)
        stream.wait_event(self.last_work)
        rows = slice(0, len(self.bindings))
        self.beam.state.is_finished[rows].fill_(True)
        self.beam.state.active_mask[rows].zero_()
        self.last_work.record(stream)
        self.cancelled = True

    def release(self, binding: BeamRequestBinding, *, consumer_done: torch.Event) -> None:
        if binding not in self.bindings:
            raise ValueError("release binding mismatch")
        if self.closed:
            return
        with self.control_lock:
            if self.pending is not None or any(self.control_busy):
                raise RuntimeError("GR release requires dispatch and output-consumer drain")
        torch.cuda.current_stream(self.beam.state.device).wait_event(self.last_work)
        self.beam.end_session(consumer_done=consumer_done)
        # Keep fixed-address graph storage for compatible, fully drained reuse.
        self.closed = True
        self.beam.session.context._gr_execution_owner = None
        self.beam.workspace.execution_owner = None
        for member in self.bindings:
            self.worker._beam_sessions.pop(member.native_request_id, None)


# The upstream base is intentionally outside the explicit mypy boundary.
class AsyncGRRetireOutput(AsyncModelRunnerOutput):  # type: ignore[misc]
    """Release proof is constructed only after GPU work and output consumers drain."""

    def __init__(self, owner: GPUBeamStageRunner, metadata: GRRetireMetadata) -> None:
        self.owner, self.metadata = owner, metadata
        self._lock = Lock()
        self._output = None

    def get_output(self) -> ModelRunnerOutput:
        with self._lock:
            if self._output is not None:
                return self._output
            owner = self.owner
            with owner.control_drained:
                owner.control_drained.wait_for(lambda: not any(owner.control_busy))
            owner.release(
                owner.binding,
                consumer_done=owner.final_output.consumer_done(owner),
            )
            if not owner.closed or owner.beam.workspace.execution_owner is owner:
                raise RuntimeError("Asynchronous scheduling Worker release did not complete")
            m = self.metadata
            out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
            setattr(
                out,
                GR_RETIRE_RESULT_ATTR,
                GRRetireResult(m.session_id, m.owner_generation, m.retire_id, True),
            )
            self._output = out
            # Startup graph capture releases its dummy owner directly and
            # never reaches this real-retirement path. Schedule only after a
            # serving session has fully drained and released its resources,
            # so the one-shot freeze covers lazily-created persistent state
            # without being consumed by the startup dummy session.
            from vllm_gr.v1.worker.gr_gc import (
                schedule_warmup_gc_freeze,
            )

            schedule_warmup_gc_freeze()
            return out


def execute_gr_retirement(worker: Any, scheduler_output: Any) -> AsyncGRRetireOutput:
    metadata = getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR)
    if (
        not isinstance(metadata, GRRetireMetadata)
        or scheduler_output.total_num_scheduled_tokens
        or scheduler_output.num_scheduled_tokens
        or getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
        or metadata.session_id not in scheduler_output.finished_req_ids
    ):
        raise ValueError("invalid Asynchronous scheduling retirement control frame")
    previous: AsyncGRRetireOutput | None = getattr(worker, "_gr_retirement_output", None)
    if previous is not None and previous.metadata == metadata:
        return previous
    owner: GPUBeamStageRunner | None = getattr(worker, "_gr_gpu_stage_runner", None)
    if owner is None or owner.closed:
        raise RuntimeError("Asynchronous scheduling retirement has no matching live Worker owner")
    result = owner.request_release(metadata)
    # Preserve the native finished_req_ids cleanup entry, without Forward or sampling.
    owner.hooks.prefill_stage_fence()
    owner.hooks.consume_retirement_state(scheduler_output, owner.bindings)
    worker._gr_retirement_output = result
    return result


def get_stage_runner(
    worker: Any,
    scheduler_output: Any,
    *,
    hooks_type: CudaStageHooksType,
    for_startup_capture: bool = False,
) -> GPUBeamStageRunner:
    """Validate and construct hooks before admitting sessions to Worker resources."""
    existing = getattr(worker, "_gr_gpu_stage_runner", None)
    if existing is not None:
        validate_stage_hooks_type(existing.hooks, hooks_type)
    if existing is not None and not existing.closed:
        if getattr(worker, "constraint_backend", None) is not existing.backend:
            raise ValueError("GR constraint backend binding changed before session release")
        return cast(GPUBeamStageRunner, existing)
    dispatch_session = validate_dispatch_session(scheduler_output)
    stages = validate_dispatch(scheduler_output, session=dispatch_session)
    (m,) = stages
    from vllm_gr.v1.worker.gr_context import (
        _create_beam_context,
        _get_gr_config,
    )

    cfg = _get_gr_config(worker)
    capacity = GRExecutionCapacity(cfg.beam_max_width)
    capacity.validate(m.beam_params)
    if m.execution_capacity is not None and m.execution_capacity != capacity:
        raise ValueError("Scheduler and Worker Asynchronous scheduling execution capacities differ")
    parallel = worker.parallel_config
    if (
        torch.device(worker.device).type != "cuda"
        or not worker.vllm_config.scheduler_config.async_scheduling
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or getattr(parallel, "decode_context_parallel_size", 1) != 1
        or getattr(parallel, "prefill_context_parallel_size", 1) != 1
        or worker.speculative_config is not None
        or worker.model_config.dtype not in (torch.float16, torch.bfloat16)
        or cfg.beam_max_width != m.beam_params.beam_width
        or cfg.beam_max_decode_steps < m.beam_params.max_tokens
    ):
        raise ValueError(
            "persistent GR requires CUDA async, TP/PP/DP=1, FP16/BF16 and matching capacity"
        )
    backend = getattr(worker, "constraint_backend", None)
    # A GRConfig without the switch (or a partial config object) keeps the
    # constrained default: only an explicit ``False`` selects full vocabulary.
    unconstrained = getattr(cfg, "constrained_generation", None) is False
    if not unconstrained and (
        backend is None or not isinstance(backend.get(), CUDAConstraintBackend)
    ):
        raise ValueError("persistent GR requires a loaded canonical CUDA Constraint Table")
    if (
        existing is not None
        and existing.graph is not None
        and existing.backend is not None
        and existing.backend is not backend
    ):
        raise ValueError(
            "Asynchronous scheduling captured constraint/model binding cannot change after startup"
        )
    if backend is not None and (
        not backend.is_ready
        or backend.info.table_metadata.vocab_size != worker.model_config.get_vocab_size()
        or any(table.device != worker.device for table in backend._tables())
    ):
        raise ValueError("constraint backend is not ready or its model/device binding differs")
    candidate_width = resolve_candidate_width(cfg)
    # A request-level per-stage K schedule overrides the configured scalar K.
    required_top_k = max(m.beam_params.per_parent_beam_widths or (candidate_width,))
    if backend is not None and (
        backend.info.max_rows
        < hooks_type.required_constraint_rows(len(m.request_ids), cfg.beam_max_width)
        or backend.info.max_top_k < required_top_k
    ):
        raise ValueError(
            "constraint backend capacity is smaller than requested Beam or candidate widths"
        )
    if (
        worker.model_config.is_encoder_decoder
        or worker.model_config.is_hybrid
        or worker.supports_mm_inputs
        or getattr(worker, "lora_config", None) is not None
        or getattr(worker.vllm_config, "kv_transfer_config", None) is not None
    ):
        raise ValueError("persistent GR requires a text decoder without LoRA or KV transfer")
    if len(worker.kv_cache_config.kv_cache_groups) != 1:
        raise ValueError("persistent GR requires one uniform KV group")
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

    layers = worker.compilation_config.static_forward_context
    for group in worker.kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            impl = getattr(layers.get(name), "impl", None)
            if not isinstance(impl, BeamAttentionImpl):
                raise ValueError(  # noqa: TRY004
                    "persistent GR requires Beam Attention on every KV layer"
                )
            # Packed beam queries do not preserve sliding-window positions
            # across the prefix/suffix split. No window is stored as (-1, -1).
            if impl.sliding_window != (-1, -1):
                raise ValueError(
                    "persistent GR split attention does not support sliding-window attention: "
                    f"{name} (window={impl.sliding_window})"
                )
    head_size = worker.model_config.get_head_size()
    if (
        head_size <= 0
        or head_size & (head_size - 1)
        or getattr(worker.model_config, "quantization", None) is not None
        or worker.cache_config.cache_dtype
        not in ("auto", str(worker.model_config.dtype).removeprefix("torch."))
    ):
        raise ValueError(
            "persistent GR requires unquantized FP16/BF16 KV and a power-of-two head size"
        )
    if getattr(worker, "_beam_postprocess_workspaces", None) is None:
        raise ValueError("persistent GR requires prewarmed Beam postprocess workspace")
    if getattr(worker, "_beam_sessions", {}):
        raise ValueError("another Beam session owns the Worker")
    workspace = worker._beam_postprocess_workspaces.require(
        BeamPostprocessWorkspaceKey(cfg.beam_max_width, candidate_width, worker.model_config.dtype)
    )
    view = BeamBatchExecutionView.from_stage_metadata(m)
    trie_count = backend.info.table_metadata.trie_count if backend is not None else 1
    trie_ids = None
    if m.session_init is not None:
        trie_ids = tuple(item.trie_id for item in m.session_init.items)
        if any(type(root) is not int or not 0 <= root < trie_count for root in trie_ids):
            raise ValueError("trie_id is outside the loaded constraint table")
    elif for_startup_capture:
        # Capture has no user request. Seed the same persistent selector buffers
        # with Trie 0; real sessions replace their contents before execution.
        trie_ids = (0,) * view.batch_size
    elif trie_count > 1:
        raise ValueError("multi-trie execution requires the initial batch request")
    context = getattr(worker, "_beam_context", None)
    if context is None:
        context = _create_beam_context(worker, cfg, max_batch=max(SUPPORTED_BATCH_SIZES))
        worker._beam_context = context
    if (
        context.registry.num_active_sessions
        or getattr(context, "_gr_execution_owner", None) is not None
        or workspace.execution_owner is not None
    ):
        raise ValueError("another Beam execution owns the registry or shared workspace")
    if (
        context.max_batch != workspace.max_batch
        or context.max_beams < m.beam_params.beam_width
        or context.max_decode_steps < m.beam_params.max_tokens
        or context.max_batch < view.batch_size
    ):
        raise ValueError("existing Beam context and workspace capacity are incompatible")
    # Check live storage before resetting state or allocating new sessions.
    # The key covers every pool and shared buffer; validating the final slot
    # also checks capacity for all contiguous request-major rows.
    # Do this before reuse_adapter so rejection cannot leave partially reset state.
    resource_key = BeamSearchRunner.validate_resources(
        context,
        workspace,
        beam_width=m.beam_params.beam_width,
        max_new_tokens=m.beam_params.max_tokens,
        kv_slot=view.batch_size - 1,
    )
    resources_match = existing is not None and all(
        runner._resource_key == resource_key for runner in existing.beam.runners
    )
    if existing is not None and existing.graph is not None and not resources_match:
        raise ValueError("captured Beam resource addresses or layouts changed")
    # The graph captures the adapter's fixed Forward and attention buffers,
    # not the request-local Beam state or its effective token budget.
    from vllm_gr.v1.worker.cuda.gpu_dynamic_beam import (
        GPUDynamicBeam,
        discard_dynamic_graphs,
        uses_dynamic_beam,
    )

    dynamic_requested = uses_dynamic_beam(m.beam_params)
    # Captures refer to an adapter's input/attention buffers. Keep one drained
    # adapter per B instead of replacing those addresses on every B transition.
    # A new schedule or resource binding invalidates the whole bounded cache.
    cache_enabled = (
        (
            getattr(cfg, "beam_adapter_cache_enabled", False)
            or (hooks_type.persistent_graph_install and len(cfg.deployment.beam_batch_sizes) > 1)
        )
        and not getattr(worker.model_config, "enforce_eager", False)
        and getattr(cfg, "beam_graph_enabled", False)
    )
    cache = getattr(worker, "_gr_gpu_stage_runner_cache", {})
    cache_key = (
        hooks_type,
        id(context),
        id(workspace),
        id(context.decision),
        id(backend),
        resource_key,
        m.beam_params.beam_width,
        view.dynamic_beam_widths,
        dynamic_requested,
    )
    cache_matches = cache_enabled and (
        getattr(worker, "_gr_gpu_stage_runner_cache_key", None) == cache_key
    )
    if cache_matches and view.batch_size in cache:
        existing = cache[view.batch_size]
        resources_match = all(
            runner._resource_key == resource_key for runner in existing.beam.runners
        )
    existing_dynamic = getattr(existing, "dynamic", None)
    reuse_adapter = (
        existing is not None
        and existing.closed
        and resources_match
        and (existing_dynamic is not None) == dynamic_requested
        and existing.beam.layout.dynamic_beam_widths == view.dynamic_beam_widths
        and existing.beam.state.beam_width == m.beam_params.beam_width
        and existing.beam.session.context is context
        and existing.beam.workspace is workspace
        and existing.beam.state.decision is context.decision
        and len(getattr(existing.beam, "runners", ())) == view.batch_size
        and all(
            runner._scratch_owner is workspace.state_scratch for runner in existing.beam.runners
        )
    )
    # Construction can reject an incomplete implementation or raise from its
    # initializer. Do it before invalidating graphs, resetting shared state or
    # reserving sessions, so the same Worker can admit a subsequent request.
    # A reused adapter retains its original hooks instance.
    hooks = None if reuse_adapter else hooks_type(worker)
    cache_cleared = False
    if (cache and not cache_matches) or (
        cache_matches and cache.get(view.batch_size) is existing and not reuse_adapter
    ):
        # A cached owner still has to pass every original compatibility check.
        discard_dynamic_graphs(worker, context)
        cache.clear()
        cache_cleared = True
    reuse_beam = (
        reuse_adapter
        and existing is not None
        and existing.beam.state.max_new_tokens == m.beam_params.max_tokens
        and existing.beam.state.stop_token_ids_config == m.beam_params.stop_token_ids
        and existing.beam.state.ignore_stop_tokens_config == m.beam_params.ignore_stop_tokens
        and existing.beam.state.candidate_width == candidate_width
    )
    if reuse_beam:
        assert existing is not None
        existing.beam.reset_for_session()
        state = existing.beam.state
    else:
        state = BeamSearchState(
            context.decision,
            beam_width=m.beam_params.beam_width,
            max_new_tokens=m.beam_params.max_tokens,
            stop_token_ids=m.beam_params.stop_token_ids,
            ignore_stop_tokens=m.beam_params.ignore_stop_tokens,
            # Completed-result storage is sized from the engine's K (W when K
            # is unset): the widest candidate row any stage can produce.  With
            # K unset the view's ``candidate_top_k`` is only this stage's output
            # width, which would undersize a schedule that widens later.
            candidate_width=candidate_width,
        )
    if (
        not reuse_adapter
        and not cache_matches
        and not cache_cleared
        and (existing_dynamic is not None or dynamic_requested)
    ):
        discard_dynamic_graphs(worker, context)
    sessions = []
    bindings = []
    for item_index, (request_id, prompt_tokens) in enumerate(
        zip(view.request_ids, view.num_prompt_tokens_by_request)
    ):
        slot = context.registry.allocate(request_id, m.beam_params.beam_width, prompt_tokens)
        if slot != item_index:
            raise RuntimeError("GR batch requires contiguous request-major KV slots")
        session = BeamSearchSession(request_id, slot, context)
        session.begin_session(m.beam_params.beam_width, prompt_tokens, decision_index=item_index)
        sessions.append(session)
        bindings.append(
            BeamRequestBinding(m.session_id, item_index, request_id, item_index, slot, item_index)
        )
    session_tuple = tuple(sessions)
    binding_tuple = tuple(bindings)
    if reuse_beam:
        assert existing is not None
        for runner, session in zip(existing.beam.runners, session_tuple):
            runner.rebind_session(session)
        beam = existing.beam
        beam.layout = view
    else:
        runners = tuple(
            BeamSearchRunner(
                session,
                state,
                workspace,
                state_index=index,
                workspace_index=index,
            )
            for index, session in enumerate(session_tuple)
        )
        beam = BeamBatchSearchRunner(runners, view)
    if reuse_adapter:
        assert existing is not None
        adapter = existing
        adapter.beam, adapter.bindings, adapter.backend = beam, binding_tuple, backend
        adapter.binding, adapter.batch_view = binding_tuple[0], view
        adapter.previous, adapter.pending = None, None
        adapter._dispatch_session, adapter._validated_dispatch = None, None
        adapter.closed, adapter.cancelled = False, False
        adapter.terminal_release_requested = False
        adapter.terminal_cleanup_pending = False
        adapter.execution_error = None
        adapter.prepared = None
        adapter.constraint_steps = beam.state.num_generated_tokens[: view.batch_size]
        adapter.configure_trie_ids(trie_ids)
        context._gr_execution_owner = adapter
        workspace.execution_owner = adapter
    else:
        assert hooks is not None
        hooks.prepare_new_adapter(view)
        adapter = GPUBeamStageRunner(
            worker, beam, binding_tuple, backend, hooks=hooks, trie_ids=trie_ids
        )
    if dynamic_requested:
        if adapter.dynamic is None:
            adapter.dynamic = GPUDynamicBeam(adapter)
        else:
            adapter.dynamic.reset()
        adapter.prefixes = workspace.prefix_token_ids[: view.batch_size].reshape(
            view.batch_size * view.beam_width, -1
        )
    worker._beam_sessions = {
        binding.native_request_id: session for binding, session in zip(binding_tuple, session_tuple)
    }
    worker._gr_gpu_stage_runner = adapter
    if cache_enabled:
        if (
            getattr(cfg, "beam_adapter_cache_enabled", False)
            or view.batch_size in cfg.deployment.beam_batch_sizes
        ):
            cache[view.batch_size] = adapter
        worker._gr_gpu_stage_runner_cache = cache
        worker._gr_gpu_stage_runner_cache_key = cache_key
    elif hasattr(worker, "_gr_gpu_stage_runner_cache_key"):
        del worker._gr_gpu_stage_runner_cache_key
    adapter._dispatch_session = dispatch_session
    adapter._validated_dispatch = (scheduler_output, stages, dispatch_session)
    return cast(GPUBeamStageRunner, adapter)
