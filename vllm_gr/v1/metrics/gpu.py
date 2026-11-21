# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Sampled CUDA events and compact survivors, piggybacked on existing D2H fences."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from vllm_gr.v1.metrics.telemetry import metrics_config

_COMPONENTS = (
    "forward",
    "constraint_select",
    "beam_select",
    "lineage_kv",
    "final_pack",
)


def pack_survivors(output, tokens, cumulative, parents, parent_scores, mask, *, first: bool):
    """Int32 wire words preserve token IDs and FP32 score bits exactly."""
    local = cumulative if first else cumulative - parent_scores.gather(1, parents.long())
    output[..., 0].copy_(tokens)
    output[..., 1].copy_(cumulative.contiguous().view(torch.int32))
    output[..., 2].copy_(local.contiguous().view(torch.int32))
    output[..., 3].copy_(mask)


def unpack_survivors(host, width: int) -> list:
    """Must be called after the owner's existing copy completion fence."""
    rows = host[:, :width]
    tokens = rows[..., 0].tolist()
    cumulative = rows[..., 1].contiguous().view(torch.float32).tolist()
    local = rows[..., 2].contiguous().view(torch.float32).tolist()
    valid = rows[..., 3].tolist()
    return [
        list(zip(t, c, local_scores, v, strict=True))
        for t, c, local_scores, v in zip(tokens, cumulative, local, valid, strict=True)
    ]


class GPUStageMetrics:
    """Two slots mirror the owner's dispatch controls; no extra synchronize()."""

    def __init__(self, owner: Any, capacity: int) -> None:
        self.config = metrics_config()
        self.enabled = self.config.enabled
        self.active_slot: int | None = None
        self.slots: list[dict] = []
        if not self.enabled:
            return
        state = owner.beam.state
        batch, width = len(owner.bindings), state.beam_width
        for _ in range(capacity):
            self.slots.append(
                {
                    "events": {
                        c: [
                            (torch.Event(enable_timing=True), torch.Event(enable_timing=True))
                            for _ in range(2 * batch + 1)
                        ]
                        for c in _COMPONENTS
                    },
                    "used": {},
                    "running": set(),
                    "sampled": False,
                    "device": torch.empty(
                        (batch, width, 4), dtype=torch.int32, device=state.device
                    ),
                    "host": torch.empty((batch, width, 4), dtype=torch.int32, pin_memory=True),
                    "parents": torch.empty(
                        (batch, width), dtype=torch.float32, device=state.device
                    ),
                }
            )

    def begin(self, slot: int, metadata: Any, owner: Any) -> None:
        if not self.enabled:
            return
        record = self.slots[slot]
        record["sampled"] = metadata.owner_generation % self.config.sample_every == 0
        record["used"] = {}
        record["running"] = set()
        record["first"] = metadata.stage_index == 0
        record["last"] = metadata.is_last_stage
        record["width"] = 0
        record["candidate_width"] = None
        self.active_slot = slot if record["sampled"] else None
        if self.active_slot is not None:
            record["parents"].copy_(owner.beam.state.beam_scores[: len(owner.bindings)])
            self.start("forward")

    def start(self, name: str) -> None:
        if self.active_slot is None:
            return
        record = self.slots[self.active_slot]
        index = record["used"].get(name, 0)
        record["events"][name][index][0].record()
        record["running"].add(name)

    def stop(self, name: str) -> None:
        if self.active_slot is None:
            return
        record = self.slots[self.active_slot]
        if name not in record["running"]:
            return
        index = record["used"].get(name, 0)
        record["events"][name][index][1].record()
        record["used"][name] = index + 1
        record["running"].remove(name)

    @contextmanager
    def measure(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def snapshot(self, owner: Any, view: Any, *, final: bool = False) -> None:
        if self.active_slot is None:
            return
        record = self.slots[self.active_slot]
        batch = len(owner.bindings)
        if final:
            # Fixed Beam accumulates completed candidates separately. Only the
            # final packed, sorted return set defines the terminal distribution.
            output = owner.final_output
            width, steps = output.options.num_return_sequences, output.layout.steps
            for i in range(batch):
                start = 3 + i * output.layout.member_stride
                rows = output.device[start + 1 : start + output.layout.member_stride].view(
                    output.layout.width, output.layout.row_stride
                )[:width]
                lengths = rows[:, 2 * steps + 1].long().clamp(1, steps)
                tokens = rows[:, :steps].gather(1, (lengths - 1)[:, None]).view(1, width)
                parents = rows[:, steps : 2 * steps].gather(1, (lengths - 1)[:, None])
                parents = parents.view(1, width).clamp(0, record["parents"].shape[1] - 1)
                scores = rows[:, 2 * steps].contiguous().view(torch.float32).view(1, width)
                mask = torch.arange(width, device=rows.device)[None, :] < output.device[start]
                pack_survivors(
                    record["device"][i : i + 1, :width],
                    tokens,
                    scores,
                    parents,
                    record["parents"][i : i + 1],
                    mask,
                    first=record["first"],
                )
            record["population"] = "returned"
        else:
            state, width = owner.beam.state, view.output_width
            parents = (
                owner.dynamic.state.parent_history[:, :width, view.stage_index]
                if owner.dynamic is not None
                else state.parent_ids[:batch, :width]
            )
            pack_survivors(
                record["device"][:, :width],
                state.input_ids[:batch, :width],
                state.beam_scores[:batch, :width],
                parents,
                record["parents"],
                state.active_mask[:batch, :width],
                first=record["first"],
            )
            record["population"] = "retained"
        record["width"] = width

    def enqueue_copy(self, slot: int) -> None:
        if not self.enabled:
            return
        record = self.slots[slot]
        if record["sampled"] and record["width"]:
            # Caller is already on copy_stream after its producer wait_event.
            record["host"].copy_(record["device"], non_blocking=True)

    def note_candidate_width(self, slot: int, width: int) -> None:
        if self.enabled:
            self.slots[slot]["candidate_width"] = width

    def consume(self, slot: int) -> dict | None:
        if not self.enabled:
            return None
        record = self.slots[slot]
        if not record["sampled"]:
            return {"sampled": False, "candidate_width": record["candidate_width"]}
        timings = {
            name: sum(start.elapsed_time(end) for start, end in record["events"][name][:n]) / 1000
            for name, n in record["used"].items()
        }
        return {
            "sampled": True,
            "candidate_width": record["candidate_width"],
            "timings": timings,
            "rows": unpack_survivors(record["host"], record["width"]),
            "population": record.get("population", "retained"),
        }
