# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Compact final output for the persistent GPU Beam runner."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import triton

from vllm_gr.ops.cuda import beam_state as beam_state_ops
from vllm_gr.ops.cuda.beam_selection import (
    _hierarchical_topk_out,
    allocate_beam_selection_scratch,
)
from vllm_gr.ops.cuda.beam_state import FINISH_STOP
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchParams,
    BeamBatchResult,
    BeamOutputOptions,
    BeamOutputSequence,
    BeamRequestOutput,
    BeamSequenceColumns,
)

_CONTROL_WORDS = 3

if TYPE_CHECKING:
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
        BeamRequestBinding,
        GPUBeamStageRunner,
    )


@dataclass(frozen=True, slots=True)
class _PayloadLayout:
    """Fixed compact wire layout: count followed by one row per returned Beam."""

    width: int
    steps: int
    row_stride: int
    batch_size: int
    member_stride: int
    size: int

    @classmethod
    def create(cls, width: int, steps: int, batch_size: int = 1) -> _PayloadLayout:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("terminal payload batch_size must be positive")
        row_stride = 2 * steps + 4
        member_stride = 1 + width * row_stride
        return cls(
            width,
            steps,
            row_stride,
            batch_size,
            member_stride,
            _CONTROL_WORDS + batch_size * member_stride,
        )


def _build_batch_result(
    payload: Any,
    layout: _PayloadLayout,
    *,
    session_id: str,
    bindings: tuple[BeamRequestBinding, ...],
    options: BeamOutputOptions,
    columnar: bool = False,
) -> BeamBatchResult:
    values = payload.numpy() if isinstance(payload, torch.Tensor) else payload
    outputs = []
    steps = layout.steps
    stride = layout.row_stride
    for request_index, binding in enumerate(bindings):
        start = _CONTROL_WORDS + request_index * layout.member_stride
        count = int(values[start])
        rows = values[start + 1 : start + layout.member_stride].reshape(layout.width, stride)
        valid_rows = rows[:count]
        if (
            columnar
            and count >= 64
            and np.all(valid_rows[:, -3] == steps)
            and not np.any(valid_rows[:, -2] == FINISH_STOP)
        ):
            outputs.append(
                BeamRequestOutput(
                    binding.item_index,
                    binding.native_request_id,
                    BeamSequenceColumns(
                        count,
                        steps,
                        valid_rows[:, :steps].astype("<i4", copy=False).tobytes(),
                        (
                            valid_rows.view(np.float32)[:, -4].astype("<f4", copy=False).tobytes()
                            if options.include_cumulative_logprob
                            else None
                        ),
                        (
                            valid_rows[:, steps : 2 * steps].astype("<i4", copy=False).tobytes()
                            if options.include_parent_beam_indices
                            else None
                        ),
                    ),
                )
            )
            continue
        sequences = []
        for row in struct.iter_unpack(f"={2 * steps}ifiii", rows[:count].tobytes()):
            length, reason, stop_token = row[-3:]
            stopped = reason == FINISH_STOP
            token_count = length - (stopped and not options.include_stop_token)
            sequences.append(
                BeamOutputSequence(
                    row[:token_count],
                    row[-4] if options.include_cumulative_logprob else None,
                    (row[steps : steps + length] if options.include_parent_beam_indices else ()),
                    "stop" if stopped else "length",
                    stop_token if stopped else None,
                )
            )
        if not sequences:
            return BeamBatchResult(
                protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
                batch_id=session_id,
                output=None,
                failure=BeamBatchFailure(
                    "no_valid_beams",
                    "No valid final Beam candidates",
                    binding.item_index,
                ),
            )
        outputs.append(
            BeamRequestOutput(
                binding.item_index,
                binding.native_request_id,
                tuple(sequences),
            )
        )
    return BeamBatchResult(
        BEAM_BATCH_PROTOCOL_VERSION,
        session_id,
        BeamBatchOutput(tuple(outputs)),
        None,
    )


class GPUBeamFinalOutput:
    """Rank and materialize final paths on GPU, then transfer one compact payload."""

    def __init__(
        self,
        owner: GPUBeamStageRunner,
        binding: BeamRequestBinding | tuple[BeamRequestBinding, ...],
        params: BeamBatchParams,
        options: BeamOutputOptions,
    ) -> None:
        state = owner.beam.state
        bindings = binding if isinstance(binding, tuple) else (binding,)
        if not bindings:
            raise ValueError("terminal output requires at least one binding")
        self._storage_key = (
            state.beam_width,
            state.max_new_tokens,
            state.completed_capacity,
            state.device,
            len(bindings),
        )
        self.layout = _PayloadLayout.create(state.beam_width, state.max_new_tokens, len(bindings))
        with torch.inference_mode(False):
            self.device = torch.empty(self.layout.size, dtype=torch.int32, device=state.device)
            self.host = torch.empty(self.layout.size, dtype=torch.int32, pin_memory=True)
            self._zero_score = torch.zeros((1, 1), dtype=torch.float32, device=state.device)
            self._ranking_scores = torch.empty(
                (1, state.completed_capacity), dtype=torch.float32, device=state.device
            )
            self._selection = allocate_beam_selection_scratch(
                1,
                1,
                state.completed_capacity,
                state.beam_width,
                state.device,
            )
        self.host_numpy = self.host.numpy()
        self.pack_ready = torch.Event()
        self.ready = torch.Event()
        self.done = torch.Event()
        self._lock = Lock()
        self.reset(owner, binding, params, options)
        self.final_d2h_count = 0
        self.final_d2h_bytes = 0

    def reset(
        self,
        owner: GPUBeamStageRunner,
        binding: BeamRequestBinding | tuple[BeamRequestBinding, ...],
        params: BeamBatchParams,
        options: BeamOutputOptions,
    ) -> None:
        with self._lock:
            self.bindings = binding if isinstance(binding, tuple) else (binding,)
            self.binding = self.bindings[0]
            self.options = options
            self.length_penalty = params.length_penalty
            self.prompt_lengths = tuple(
                runner.session._state.prefix_len
                for runner in getattr(owner.beam, "runners", (owner.beam,))
            )
            self._result: BeamBatchResult | None = None
            self._enqueued = False
            self._delivered = False
            self._aborted = False
            self._done_recorded = False
            self._control_embedded = False

    def is_compatible(self, owner: GPUBeamStageRunner) -> bool:
        state = owner.beam.state
        return self._storage_key == (
            state.beam_width,
            state.max_new_tokens,
            state.completed_capacity,
            state.device,
            len(getattr(owner, "bindings", (owner.binding,))),
        )

    def _record_done(self, owner: GPUBeamStageRunner) -> None:
        if self._done_recorded:
            return
        stream = torch.cuda.current_stream(owner.beam.state.device)
        stream.wait_event(owner.last_work)
        if self._enqueued:
            stream.wait_event(self.ready)
        self.done.record(stream)
        self._done_recorded = True

    def _enqueue(
        self,
        owner: GPUBeamStageRunner,
        control: torch.Tensor | None = None,
    ) -> None:
        if self._enqueued:
            return
        state = owner.beam.state
        result_width = self.options.num_return_sequences
        produced = torch.cuda.current_stream(state.device)
        produced.wait_event(owner.last_work)
        if control is not None:
            self.device[:_CONTROL_WORDS].copy_(control)
            self._control_embedded = True
        if getattr(owner, "dynamic", None) is not None:
            owner.dynamic.pack_final(self)
        else:
            for request_index, binding in enumerate(self.bindings):
                row = binding.state_row
                ranking_scores = state.completed_scores[row : row + 1]
                if self.length_penalty != 0:
                    beam_state_ops._rank_final_beams_kernel[
                        (triton.cdiv(state.completed_capacity, 256),)
                    ](
                        ranking_scores,
                        state.completed_lengths[row],
                        state.completed_finish_reasons[row],
                        self._ranking_scores,
                        CAPACITY=state.completed_capacity,
                        PROMPT_LENGTH=self.prompt_lengths[request_index],
                        MAX_TOKENS=state.max_new_tokens,
                        LENGTH_PENALTY=self.length_penalty,
                        BLOCK=256,
                    )
                    ranking_scores = self._ranking_scores
                _, selected_indices, _ = _hierarchical_topk_out(
                    self._zero_score,
                    ranking_scores,
                    None,
                    1,
                    1,
                    state.completed_capacity,
                    result_width,
                    self._selection,
                )
                member_offset = _CONTROL_WORDS + request_index * self.layout.member_stride
                beam_state_ops._pack_final_beams_kernel[(result_width,)](
                    selected_indices,
                    state.num_completed[row : row + 1],
                    state.completed_token_ids[row],
                    state.completed_scores[row],
                    state.completed_parent_ids[row],
                    state.completed_lengths[row],
                    state.completed_finish_reasons[row],
                    state.history_token_ids[row],
                    state.history_parent_ids[row],
                    self.device[member_offset:],
                    WIDTH=state.beam_width,
                    STEPS=state.max_new_tokens,
                    RESULT_WIDTH=result_width,
                    ROW_STRIDE=self.layout.row_stride,
                    ROW_BLOCK=triton.next_power_of_2(2 * state.max_new_tokens),
                )
        # Fallback finalization after pending was cleared must not touch a successor.
        metrics = getattr(owner, "metrics", None) if control is not None else None
        if metrics is not None:
            metrics.stop("final_pack")
            metrics.snapshot(owner, None, final=True)
        self.pack_ready.record(produced)
        copy_words = (
            _CONTROL_WORDS
            + (len(self.bindings) - 1) * self.layout.member_stride
            + 1
            + result_width * self.layout.row_stride
        )
        with torch.cuda.stream(owner.copy_stream):
            owner.copy_stream.wait_event(self.pack_ready)
            self.host[:copy_words].copy_(self.device[:copy_words], non_blocking=True)
            if metrics is not None and owner.pending is not None:
                metrics.enqueue_copy(owner.pending[1])
            self.ready.record(owner.copy_stream)
        self.final_d2h_count += 1
        self.final_d2h_bytes += copy_words * self.device.element_size()
        self._enqueued = True

    def enqueue(self, owner: GPUBeamStageRunner, control: torch.Tensor) -> bool:
        """Submit final selection and D2H while the producing sample is still active."""

        with self._lock:
            if self._delivered or self._aborted:
                return False
            self._enqueue(owner, control)
            return self._enqueued and self._control_embedded

    def wait_embedded_control(self) -> list[int]:
        """Wait once for the terminal payload and return its control header."""

        with self._lock:
            if not self._enqueued or not self._control_embedded:
                raise RuntimeError("terminal Beam payload has no embedded control header")
            self.ready.synchronize()
            return self.host_numpy[:_CONTROL_WORDS].tolist()

    def abort(self, owner: GPUBeamStageRunner) -> None:
        with self._lock:
            if not self._delivered:
                self._aborted = True
            self._record_done(owner)

    def failure(self, owner: GPUBeamStageRunner, result: BeamBatchResult) -> BeamBatchResult | None:
        with self._lock:
            if self._delivered or self._aborted:
                return None
            self._result = result
            self._delivered = True
            self._record_done(owner)
            return result

    def consume(
        self,
        owner: GPUBeamStageRunner,
        *,
        ready: bool = False,
    ) -> BeamBatchResult | None:
        with self._lock:
            if self._delivered or self._aborted:
                return None
            # Early-stop termination is not known to the Host at sample time;
            # it uses this fallback. The fixed-length path is already enqueued.
            self._enqueue(owner)
            if not ready:
                self.ready.synchronize()
            self._result = _build_batch_result(
                self.host_numpy,
                self.layout,
                session_id=self.binding.session_id,
                bindings=self.bindings,
                options=self.options,
                columnar=True,
            )
            self._delivered = True
            self._record_done(owner)
            return self._result

    def consumer_done(self, owner: GPUBeamStageRunner) -> torch.Event:
        with self._lock:
            self._record_done(owner)
            return self.done


__all__ = ["GPUBeamFinalOutput"]
