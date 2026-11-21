# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Real CUDA stream/fence tests for metrics, also runnable through unittest."""

import unittest
from typing import Any
import os
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from vllm_gr.v1.metrics.gpu import GPUStageMetrics
from vllm_gr.v1.metrics.telemetry import metrics_config


pytestmark = [pytest.mark.gpu_test, pytest.mark.l2]


@unittest.skipUnless(
    torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "cuda",
    "CUDA unavailable",
)
class CUDAStageMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(
            os.environ,
            {
                "VLLM_GR_METRICS_ENABLED": "1",
                "VLLM_GR_METRICS_SAMPLE_EVERY": "1",
            },
        )
        self.env.start()
        metrics_config.cache_clear()

    def tearDown(self) -> None:
        self.env.stop()
        metrics_config.cache_clear()

    def owner(self, batch: Any = 2, width: Any = 4) -> Any:
        state = NS(
            beam_width=width,
            device="cuda",
            beam_scores=torch.full((batch, width), -2.0, device="cuda"),
            input_ids=torch.arange(batch * width, device="cuda", dtype=torch.int32).view(
                batch, width
            ),
            parent_ids=torch.zeros((batch, width), device="cuda", dtype=torch.int32),
            active_mask=torch.ones((batch, width), device="cuda", dtype=torch.bool),
        )
        return NS(beam=NS(state=state), bindings=tuple(range(batch)), dynamic=None)

    def test_copy_fence_covers_events_and_survivor_snapshot(self) -> None:
        owner = self.owner()
        metrics = GPUStageMetrics(owner, 2)
        copy_stream = torch.cuda.Stream()
        produced = torch.Event()
        copy_done = torch.Event()
        metadata = NS(owner_generation=2, stage_index=1, is_last_stage=False)
        metrics.begin(0, metadata, owner)
        owner.beam.state.beam_scores.fill_(-3.0)
        metrics.stop("forward")
        metrics.snapshot(owner, NS(output_width=4))
        produced.record()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(produced)
            metrics.enqueue_copy(0)
            copy_done.record()
        # This is the pre-existing owner fence; consume adds no wait.
        copy_done.synchronize()
        result = metrics.consume(0)
        self.assertGreaterEqual(result["timings"]["forward"], 0)
        self.assertEqual(result["rows"][1][0], (4, -3.0, -1.0, 1))

    def test_forward_timer_matches_independent_cuda_events(self) -> None:
        owner = self.owner(batch=1)
        metrics = GPUStageMetrics(owner, 2)
        a = torch.randn((256, 256), device="cuda")
        b = torch.randn_like(a)
        output = torch.empty_like(a)
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        metrics.begin(0, NS(owner_generation=2, stage_index=1, is_last_stage=False), owner)
        start.record()
        for _ in range(32):
            torch.mm(a, b, out=output)
        end.record()
        metrics.stop("forward")
        metrics.snapshot(owner, NS(output_width=4))
        produced = torch.Event()
        copied = torch.Event()
        copy_stream = torch.cuda.Stream()
        produced.record()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(produced)
            metrics.enqueue_copy(0)
            copied.record()
        copied.synchronize()
        actual_ms = metrics.consume(0)["timings"]["forward"] * 1000
        reference_ms = start.elapsed_time(end)
        self.assertGreater(reference_ms, 0)
        self.assertLessEqual(abs(actual_ms - reference_ms), max(0.15, 0.1 * reference_ms))

    def test_terminal_uses_packed_count_and_ranked_rows(self) -> None:
        owner = self.owner(batch=1)
        metrics = GPUStageMetrics(owner, 2)
        # Same layout as GPUBeamFinalOutput: 3 controls, count, W*(2*steps+4).
        payload = torch.zeros(4 + 4 * 10, dtype=torch.int32, device="cuda")
        payload[3] = 2
        rows = payload[4:].view(4, 10)
        rows[:2, :3] = torch.tensor([[1, 2, 8], [1, 2, 9]], device="cuda")
        rows[:2, 5] = 0
        # A business-ranked payload can put a lower model-score row first.
        rows[:2, 6] = torch.tensor([-4.0, -3.0], device="cuda").view(torch.int32)
        rows[:2, 7] = 3
        owner.final_output = NS(
            device=payload,
            options=NS(num_return_sequences=3),
            layout=NS(steps=3, width=4, member_stride=41, row_stride=10),
        )
        metadata = NS(owner_generation=2, stage_index=2, is_last_stage=True)
        metrics.begin(0, metadata, owner)
        metrics.stop("forward")
        metrics.snapshot(owner, None, final=True)
        metrics.enqueue_copy(0)
        done = torch.Event()
        done.record()
        done.synchronize()
        result = metrics.consume(0)
        self.assertEqual(result["population"], "returned")
        self.assertEqual(sum(v[3] for v in result["rows"][0]), 2)
        self.assertEqual(result["rows"][0][0][:3], (8, -4.0, -2.0))


if __name__ == "__main__":
    unittest.main()
