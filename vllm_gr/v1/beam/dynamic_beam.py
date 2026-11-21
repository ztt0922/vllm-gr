# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Device-side state and contiguous KV banks for per-stage Beam widths.

The native KV operators derive request strides from the physical pool width.
Each active width therefore owns a contiguous, fixed-address pool. Dynamic
selection gathers history between these pools instead of passing a logical
width to a native in-place reorder with a different physical stride.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from typing import Any

import torch

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def initialize_packed_stage_pools(context: Any) -> None:
    """Pack a newly created, idle context before any graph captures its pools.

    The caller must never late-pack an existing context: graph capture can
    retain addresses outside this helper's ownership. The capture marker is
    an additional guard, rather than a way to discover arbitrary captures.
    Per-layer native attention still receives contiguous ``[B,W,H,T,D]``
    tensors, while dynamic reorder can process all K/V layers in one gather.
    """

    registry = getattr(context, "registry", None)
    if registry is None:
        raise ValueError("packed stage pools require a session registry")
    if registry.num_active_sessions:
        raise RuntimeError("cannot initialize packed pools while Beam sessions are active")
    if getattr(context, "_npu_beam_graph_captured", False):
        raise RuntimeError("cannot change KV pool addresses after Beam graph capture")
    if getattr(context, "_npu_packed_width_pools", None) is not None:
        return
    if getattr(context, "_npu_beam_width_pools", None):
        raise RuntimeError("packed initialization must precede stage width bank creation")
    pools: list[BeamAttentionPool] = context.pools
    if not pools:
        raise ValueError("packed stage pools require at least one KV layer")
    first = pools[0].unshared_key
    packed = torch.empty((2, len(pools), *first.shape), dtype=first.dtype, device=first.device)
    for layer, pool in enumerate(pools):
        packed[0, layer].copy_(pool.unshared_key)
        packed[1, layer].copy_(pool.unshared_value)
    for layer, pool in enumerate(pools):
        pool.unshared_key = packed[0, layer]
        pool.unshared_value = packed[1, layer]
    context._npu_packed_width_pools = {context.max_beams: packed}
    context._npu_packed_reorder_staging = {}


def get_stage_pools(context: Any, width: int) -> list[BeamAttentionPool]:
    """Return the persistent contiguous pool bank for one active Beam width.

    Startup graph capture and request execution use this same lookup. Banks
    are retained by the context, so a captured graph never sees a newly
    allocated replacement. The capacity-width bank is the original pool list.
    """

    width = _positive_int("width", width)
    if width > context.max_beams:
        raise ValueError("stage Beam width exceeds the context capacity")
    pools: list[BeamAttentionPool] = context.pools
    if width == context.max_beams:
        return pools
    banks: dict[int, list[BeamAttentionPool]] | None = getattr(
        context, "_npu_beam_width_pools", None
    )
    if banks is None:
        banks = {}
        context._npu_beam_width_pools = banks
    if width not in banks:
        packed_banks: dict[int, torch.Tensor] | None = getattr(
            context, "_npu_packed_width_pools", None
        )
        if packed_banks is None:
            banks[width] = [
                BeamAttentionPool(
                    max_batch=source.max_batch,
                    max_beams=width,
                    num_kv_heads=source.num_kv_heads,
                    max_decode_steps=source.max_decode_steps,
                    head_dim=source.head_dim,
                    device=source.unshared_key.device,
                    dtype=source.unshared_key.dtype,
                )
                for source in pools
            ]
        else:
            base = packed_banks[context.max_beams]
            shape = (*base.shape[:3], width, *base.shape[4:])
            packed = torch.zeros(shape, dtype=base.dtype, device=base.device)
            packed_banks[width] = packed
            bank = []
            for layer, source in enumerate(pools):
                pool = copy(source)
                pool.max_beams = width
                pool.unshared_key = packed[0, layer]
                pool.unshared_value = packed[1, layer]
                bank.append(pool)
            banks[width] = bank
    return banks[width]


def prune_stage_pools(context: Any, keep_widths: Iterable[int]) -> tuple[int, ...]:
    """Release unused width banks between sessions, retaining captured banks.

    The caller must include every width referenced by a captured graph over
    this context, in addition to the next session's Decode input widths.
    Releasing banks while a session owns this context is forbidden. The
    original capacity-width pools are never released by this function.

    Without pruning, successive schedules could retain one bank for every
    possible width. Keeping only captured widths and the upcoming schedule
    bounds live storage independently of schedules used by past sessions.
    """

    registry = getattr(context, "registry", None)
    if registry is None:
        raise ValueError("stage pool pruning requires a session registry")
    if registry.num_active_sessions:
        raise RuntimeError("cannot prune stage pools while Beam sessions are active")
    retained = {_positive_int("keep width", width) for width in keep_widths}
    if any(width > context.max_beams for width in retained):
        raise ValueError("retained Beam width exceeds the context capacity")
    banks: dict[int, list[BeamAttentionPool]] | None = getattr(
        context, "_npu_beam_width_pools", None
    )
    if not banks:
        return ()
    released = tuple(sorted(set(banks) - retained))
    packed_banks: dict[int, torch.Tensor] = getattr(context, "_npu_packed_width_pools", {})
    staging: dict[int, torch.Tensor] = getattr(context, "_npu_packed_reorder_staging", {})
    for width in released:
        del banks[width]
        packed_banks.pop(width, None)
        staging.pop(width, None)
    return released


def reorder_stage_kv(
    context: Any,
    source_width: int,
    target_width: int,
    parents: torch.Tensor,
    history_steps: int,
) -> None:
    """Gather request-local parent KV into the next stage's pool bank.

    ``parents`` has shape ``[B, target_width]`` and indexes ``source_width``
    parents per request. ``history_steps`` is the number of already-written
    suffix positions. ``gather`` produces separate staging storage before
    ``copy_``, including when both widths use the same pool and parents fan out.
    Packed banks copy the full time capacity, whose future positions are
    overwritten before use; legacy pools copy only valid history positions.
    Inactive requests are always left untouched.
    """

    source = get_stage_pools(context, source_width)
    target = get_stage_pools(context, target_width)
    history_steps = _positive_int("history_steps", history_steps)
    if history_steps > context.max_decode_steps:
        raise ValueError("KV history exceeds the context Decode capacity")
    if parents.ndim != 2 or parents.shape[1] != target_width:
        raise ValueError("KV parents must have shape [B, target_width]")
    batch = parents.shape[0]
    if not 1 <= batch <= context.max_batch:
        raise ValueError("KV parent batch exceeds the context capacity")
    if parents.dtype not in (torch.int32, torch.int64):
        raise ValueError("KV parents must use int32 or int64")
    if not source or len(source) != len(target):
        raise ValueError("KV pool banks must contain the same nonzero layer count")
    if parents.device != source[0].unshared_key.device:
        raise ValueError("KV parents and pools must share a device")
    packed_banks: dict[int, torch.Tensor] | None = getattr(context, "_npu_packed_width_pools", None)
    if packed_banks is not None:
        source_packed = packed_banks[source_width]
        target_packed = packed_banks[target_width]
        offsets = torch.arange(batch, dtype=torch.int64, device=parents.device)[:, None]
        global_parents = (parents.to(torch.int64) + offsets * source_width).reshape(-1)
        flattened = source_packed.flatten(2, 3)
        if batch < context.max_batch:
            # A leading active-B slice has gaps between layers; use one
            # compact gather followed by one strided active-slice copy.
            selected = flattened.index_select(2, global_parents)
            target_packed[:, :, :batch].copy_(
                selected.view(2, len(source), batch, target_width, *source_packed.shape[4:])
            )
        elif source_width != target_width:
            torch.index_select(flattened, 2, global_parents, out=target_packed.flatten(2, 3))
        else:
            staging: dict[int, torch.Tensor] = context._npu_packed_reorder_staging
            if target_width not in staging:
                staging[target_width] = torch.empty_like(target_packed)
            scratch = staging[target_width]
            torch.index_select(flattened, 2, global_parents, out=scratch.flatten(2, 3))
            target_packed.copy_(scratch)
        return
    indices = parents.to(torch.int64)[:, :, None, None, None]
    for old_pool, new_pool in zip(source, target):
        for name in ("unshared_key", "unshared_value"):
            old = getattr(old_pool, name)[:batch, :, :, :history_steps, :]
            new = getattr(new_pool, name)[:batch, :, :, :history_steps, :]
            gather_index = indices.expand(batch, target_width, *old.shape[2:])
            new.copy_(old.gather(1, gather_index))


class DynamicBeamState:
    """Own device-resident histories for one NPU stage-width schedule.

    The context owns KV bank storage and session lifecycle. This state owns
    compact candidate histories only; it never advances registry counters.
    Non-final winners are grouped by parent for the established Beam execution
    order, while final winners remain in descending cumulative score order.
    """

    def __init__(self, context: Any, batch_size: int, max_decode_steps: int) -> None:
        self.context = context
        self.batch_size = _positive_int("batch_size", batch_size)
        self.max_decode_steps = _positive_int("max_decode_steps", max_decode_steps)
        if self.batch_size > context.max_batch:
            raise ValueError("Beam state batch exceeds the context capacity")
        if self.max_decode_steps > context.max_decode_steps:
            raise ValueError("Beam history exceeds the context Decode capacity")
        self.device = context.pools[0].unshared_key.device
        self.current_width = 0
        self.num_generated_tokens = 0
        self.sequence: torch.Tensor | None = None
        self.scores: torch.Tensor | None = None
        self.parent_history: torch.Tensor | None = None

    def _validate_candidates(self, candidates: ConstraintOutput, rows: int, width: int) -> None:
        expected = (rows, width)
        tensors = (candidates.token_ids, candidates.logprobs, candidates.valid_mask)
        if any(tuple(tensor.shape) != expected for tensor in tensors):
            raise ValueError(f"Beam candidates must all have shape {expected}")
        if (
            candidates.token_ids.dtype is not torch.int32
            or candidates.logprobs.dtype is not torch.float32
            or candidates.valid_mask.dtype is not torch.bool
        ):
            raise ValueError(
                "Beam candidates must use int32 IDs, float32 scores, and bool validity"
            )
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError("Beam candidates and state must share a device")

    def initialize(self, prefill: ConstraintOutput, width: int) -> None:
        """Seed SA winners without recomputing their local probabilities.

        ``width`` is the survivor count the state keeps (the stage's output
        width, i.e. ``W``), while the Prefill candidates carry ``K`` columns.
        The two are independent, so the candidate geometry is validated against
        its own width and only ``min(W, K)`` beams are seeded: the candidates
        are score-ordered, so the first ``min(W, K)`` are the winners, and any
        beams beyond that stay inactive with ``-inf`` scores.
        """

        width = _positive_int("width", width)
        if self.num_generated_tokens:
            raise RuntimeError("dynamic Beam state has already been initialized")
        if width > self.context.max_beams:
            raise ValueError("initial Beam width exceeds the context capacity")
        candidate_width = prefill.token_ids.shape[1] if prefill.token_ids.ndim == 2 else 0
        if candidate_width < 1:
            raise ValueError("Beam candidates must be a nonempty 2-D tensor")
        self._validate_candidates(prefill, self.batch_size, candidate_width)
        live = min(width, candidate_width)
        shape = (self.batch_size, width, self.max_decode_steps)
        sequence = torch.zeros(shape, dtype=torch.int32, device=self.device)
        valid = prefill.valid_mask & torch.isfinite(prefill.logprobs)
        sequence[:, :live, 0].copy_(prefill.token_ids[:, :live].masked_fill(~valid[:, :live], 0))
        self.sequence = sequence
        scores = torch.full(
            (self.batch_size, width),
            float("-inf"),
            dtype=prefill.logprobs.dtype,
            device=self.device,
        )
        scores[:, :live].copy_(
            prefill.logprobs[:, :live].masked_fill(~valid[:, :live], float("-inf"))
        )
        self.scores = scores
        self.parent_history = torch.zeros_like(sequence)
        self.current_width = width
        self.num_generated_tokens = 1

    seed = initialize

    def select(
        self,
        candidates: ConstraintOutput,
        input_width: int,
        output_width: int,
        stage_index: int,
        *,
        is_last: bool = False,
        result_width: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Select the next stage width, independently of its input width.

        Each parent contributes ``candidate_width`` local candidates. The
        complete candidate pool must contain at least ``output_width`` items.
        Invalid candidates and invalid parent scores remain ``-inf``.
        The final return can be truncated to ``result_width`` without changing
        the full ``output_width`` survivor state used to define the schedule.
        """

        input_width = _positive_int("input_width", input_width)
        output_width = _positive_int("output_width", output_width)
        if self.sequence is None or self.scores is None or self.parent_history is None:
            raise RuntimeError("dynamic Beam state must be initialized from Prefill")
        if input_width != self.current_width:
            raise ValueError("candidate input width does not match the current Beam state")
        if output_width > self.context.max_beams:
            raise ValueError("output Beam width exceeds the context capacity")
        if (
            isinstance(stage_index, bool)
            or not isinstance(stage_index, int)
            or stage_index != self.num_generated_tokens
            or not 1 <= stage_index < self.max_decode_steps
        ):
            raise ValueError("Beam stage index must be the next unwritten sequence position")
        candidate_width = candidates.token_ids.shape[1] if candidates.token_ids.ndim == 2 else 0
        if input_width * candidate_width < output_width:
            raise ValueError("retained width exceeds the complete candidate pool")
        self._validate_candidates(candidates, self.batch_size * input_width, candidate_width)
        returned_width = (
            output_width if result_width is None else _positive_int("result_width", result_width)
        )
        if returned_width > output_width or (not is_last and returned_width != output_width):
            raise ValueError("only final Beam output may truncate the selected stage width")

        shape = (self.batch_size, input_width, candidate_width)
        candidate_valid = candidates.valid_mask & torch.isfinite(candidates.logprobs)
        local_scores = candidates.logprobs.masked_fill(~candidate_valid, float("-inf"))
        parent_scores = self.scores.masked_fill(~torch.isfinite(self.scores), float("-inf"))
        total_scores = parent_scores[:, :, None] + local_scores.reshape(shape)
        # Finite FP32 operands can still overflow when added. Reject these
        # paths before Top-K so nonfinite winners cannot displace valid ones.
        total_scores = total_scores.masked_fill(~torch.isfinite(total_scores), float("-inf"))
        selected_scores, flat_indices = total_scores.reshape(self.batch_size, -1).topk(
            output_width, dim=1, largest=True, sorted=True
        )
        parents = torch.div(flat_indices, candidate_width, rounding_mode="floor")
        if not is_last:
            # Parent IDs are at most 255, exactly representable in float32.
            # Ascend integer argsort falls back to AiCPU; use its AiCore path.
            order = parents.to(torch.float32).argsort(dim=1, stable=True)
            parents = parents.gather(1, order)
            flat_indices = flat_indices.gather(1, order)
            selected_scores = selected_scores.gather(1, order)
        token_ids = candidates.token_ids.reshape(self.batch_size, -1).gather(1, flat_indices)
        token_ids = token_ids.masked_fill(~torch.isfinite(selected_scores), 0)
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.stop("beam_select")
            metrics.start("lineage_kv")
        history_index = parents[:, :, None].expand(
            self.batch_size, output_width, self.max_decode_steps
        )
        sequence = self.sequence.gather(1, history_index)
        sequence[:, :, stage_index].copy_(token_ids)
        parent_history = self.parent_history.gather(1, history_index)
        parent_history[:, :, stage_index].copy_(parents.to(torch.int32))

        if not is_last:
            reorder_stage_kv(
                self.context, input_width, output_width, parents, history_steps=stage_index
            )
        if metrics is not None:
            metrics.stop("lineage_kv")
            metrics.start("beam_select")
        self.sequence = sequence
        self.scores = selected_scores
        self.parent_history = parent_history
        self.current_width = output_width
        self.num_generated_tokens = stage_index + 1

        local_parents = parents[:, :returned_width].to(torch.int32)
        request_offset = (
            torch.arange(self.batch_size, dtype=torch.int32, device=self.device)[:, None]
            * input_width
        )
        return {
            "out_token_ids": token_ids[:, :returned_width],
            "out_token_index": local_parents + request_offset,
            "out_log_probs": selected_scores[:, :returned_width],
            "out_sequence": sequence[:, :returned_width, : self.num_generated_tokens],
            "out_parent_history": parent_history[:, :returned_width, : self.num_generated_tokens],
            "local_parent_ids": local_parents,
        }


__all__ = [
    "DynamicBeamState",
    "get_stage_pools",
    "initialize_packed_stage_pools",
    "prune_stage_pools",
    "reorder_stage_kv",
]
