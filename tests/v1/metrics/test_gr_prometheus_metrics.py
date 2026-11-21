# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU contracts for Prometheus reporting, queue handoff and real Prometheus exposition."""

import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event
from types import SimpleNamespace as NS
import unittest
from typing import Any
from unittest.mock import patch

import msgspec
import pytest
import torch

from vllm_gr.v1.metrics import prometheus as prom
from vllm_gr.v1.metrics.gpu import GPUStageMetrics, pack_survivors, unpack_survivors
from vllm_gr.v1.metrics.telemetry import metrics_config
from vllm_gr.v1.metrics.telemetry import (
    TELEMETRY_KEY,
    invalid_sequence_reason,
    stage_finished,
    stage_started,
    terminal_metrics,
)
from vllm_gr.v1.metrics.timer import WallTimer


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class Client:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def observe_timer(self, name: Any, value: Any, tags: Any = None) -> None:
        self.events.append(("timer", name, value, tags))

    def increment_counter(self, name: Any, value: Any, tags: Any = None) -> None:
        self.events.append(("rate_counter", name, value, tags))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def session(batch: Any = 2, steps: Any = 3) -> Any:
    return NS(
        owner_generation=9,
        batch_size=batch,
        total_stages=steps,
        session_id="batch",
        request_ids=tuple(f"r{i}" for i in range(batch)),
        request=NS(beam_params=NS(beam_width=4, max_tokens=steps, dynamic_beam_widths=None)),
        _metrics_dispatch_timers={},
        _metrics_started=False,
        _metrics_decode_timer=None,
        _metrics_decode_seconds=None,
        _metrics_prompt_work=(),
    )


def metadata(s: Any, index: Any) -> Any:
    return NS(
        session_id=s.session_id,
        dispatch_id=index,
        stage_index=index,
        owner_generation=s.owner_generation,
        kind="prefill" if index == 0 else "decode",
        decode_step=None if index == 0 else index - 1,
        produces_output=True,
        beam_params=s.request.beam_params,
        request_ids=s.request_ids,
        num_prompt_tokens_by_request=(10,) * s.batch_size,
        num_prompt_tokens=10 * s.batch_size,
    )


class CollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_config.cache_clear()
        self.client = Client()
        self.collector = prom.PrometheusMetricsCollector(self.client)

    def values(self, name: Any, **tags: Any) -> Any:
        return [
            v
            for _, n, v, t in self.client.events
            if n == name and all(t.get(k) == value for k, value in tags.items())
        ]

    def test_prompt_partition_and_duplicate_frame(self) -> None:
        value = {
            "version": 1,
            "sid_level": 1,
            "batch_size": 2,
            "stage_seconds": 0.0125,
            "prefill_seconds": 0.0125,
            "prompt_work": [(10, 3), (20, 20)],
            "widths": [1, 4, 4],
        }
        self.collector.observe("m", (1,), value)
        self.collector.observe("m", (1,), value)
        self.assertEqual(sum(self.values("prompt.tokens")), 30)
        self.assertEqual(self.values("prompt.length"), [10, 20])
        self.assertEqual(sum(self.values("prompt.tokens_by_source", source="local_compute")), 23)
        self.assertEqual(sum(self.values("cache.prefix.tokens", status="hit")), 7)
        self.assertEqual(self.values("prompt.samples"), [2])
        self.assertEqual(self.values("stage.elapsed.ms"), [12.5])
        self.assertEqual(len(self.values("beam.width", kind="target")), 2)
        self.assertTrue(
            all(k == "rate_counter" for k, n, _, _ in self.client.events if n == "prompt.tokens")
        )

    def test_device_timer_nll_and_padding_population(self) -> None:
        self.collector.observe(
            "m",
            (1,),
            {
                "version": 1,
                "sid_level": 2,
                "batch_size": 1,
                "stage_seconds": 0.01,
                "device": {
                    "sampled": True,
                    "timings": {"forward": 0.00125, "lineage_kv": 0.000125},
                    "population": "retained",
                    "rows": [[(7, -3.0, -1.0, 1), (8, -math.inf, -1.0, 0), (9, math.nan, -1.0, 1)]],
                },
            },
        )
        self.assertEqual(self.values("stage.forward.ms"), [1.25])
        self.assertEqual(self.values("gr.processing.ms", component="lineage_kv"), [0.125])
        self.assertEqual(self.values("sid.score.nll", score_kind="cumulative"), [3.0])
        self.assertEqual(self.values("sid.score.count", score_kind="local"), [1])
        self.assertEqual(self.values("beam.width", kind="effective"), [1])
        self.assertEqual(self.values("token_code", token_code="7"), [1])
        self.assertEqual(self.values("token_code", token_code="8"), [])

    def test_token_counts_aggregate_and_bound_labels(self) -> None:
        with patch.dict(os.environ, {"VLLM_GR_METRICS_TOKEN_CODE_LIMIT": "2"}):
            metrics_config.cache_clear()
            self.collector.observe(
                "m",
                (1,),
                {
                    "version": 1,
                    "sid_level": 1,
                    "batch_size": 1,
                    "stage_seconds": 0,
                    "device": {"rows": [[(i, -1.0, -1.0, 1) for i in (10, 11, 12, 13, 10)]]},
                },
            )
        self.assertEqual(self.values("token_code", token_code="10"), [2])
        self.assertEqual(self.values("token_code", token_code="other"), [2])
        self.assertEqual(self.values("token_code", token_code="12"), [])

    def test_unsampled_geometry_has_no_device_sample(self) -> None:
        self.collector.observe(
            "m",
            (1,),
            {
                "version": 1,
                "sid_level": 3,
                "batch_size": 1,
                "stage_seconds": 0.1,
                "widths": [4, 4, 17],
                "device": {"sampled": False, "candidate_width": 17},
            },
        )
        self.assertEqual(self.values("beam.width", kind="candidate_per_parent"), [17])
        self.assertEqual(self.values("stage.device_samples"), [])

    def test_failure_is_batch_count_not_missing_beam_slots(self) -> None:
        self.collector.observe(
            "m",
            (1,),
            {
                "version": 1,
                "terminal": True,
                "result_counts": [0, 0],
                "invalid_counts": {},
                "failure": "no_valid_beams",
            },
        )
        self.assertEqual(self.values("beam.result_count"), [0, 0])
        self.assertEqual(self.values("batch.failures", reason="no_valid_beams"), [1])
        self.assertEqual(self.values("beam.invalid.result_count"), [])

    def test_unknown_failure_reason_is_bounded(self) -> None:
        self.collector.observe(
            "m",
            (1,),
            {"version": 1, "terminal": True, "failure": "some detailed request-specific error"},
        )
        self.assertEqual(self.values("batch.failures", reason="other"), [1])

    def test_exporter_observation_exception_does_not_abort_inference(self) -> None:
        def fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("observation failed")

        self.client.observe_timer = fail  # type: ignore[method-assign]
        with self.assertLogs(prom.logger, "ERROR"):
            self.collector.observe(
                "m",
                (1,),
                {
                    "version": 1,
                    "sid_level": 1,
                    "batch_size": 1,
                    "stage_seconds": 0,
                    "prompt_work": [(5, 5)],
                },
            )
        self.assertGreater(self.collector.emission_errors, 0)
        self.assertEqual(self.values("prompt.tokens"), [5])

    def test_plain_outputs_do_not_initialize_exporter(self) -> None:
        with patch.object(prom, "get_metrics") as get:
            prom.consume_outputs(NS(), [NS(kv_transfer_params=None)])
            get.assert_not_called()

    def test_transport_consumption_with_native_logs_disabled(self) -> None:
        from vllm_gr.v1.engine.output_processor import make_patched_process_outputs

        value = {
            "version": 1,
            "generation": 9,
            "dispatch_id": 0,
            "sid_level": 1,
            "batch_size": 1,
            "stage_seconds": 0.01,
            "prompt_work": [[10, 4]],
        }
        params = msgspec.msgpack.decode(
            msgspec.msgpack.encode(
                {TELEMETRY_KEY: [value], "gr_metrics_model_name": "m", "business": "kept"}
            )
        )
        outputs = [NS(request_id="r", kv_transfer_params=params, finish_reason=None)]
        wrapped = make_patched_process_outputs(lambda self, out: 42)
        with (
            patch.object(prom, "get_metrics", return_value=self.collector),
            patch("vllm_gr.v1.engine.output_processor.note_admission_rejections"),
        ):
            self.assertEqual(wrapped(NS(log_stats=False), outputs), 42)
        self.assertEqual(self.values("prompt.tokens"), [10])
        self.assertEqual(outputs[0].kv_transfer_params, {"business": "kept"})

    def test_new_generation_reused_request_id_is_counted(self) -> None:
        processor = NS()
        with patch.object(prom, "get_metrics", return_value=self.collector):
            for generation in (1, 2):
                value = {
                    "version": 1,
                    "generation": generation,
                    "dispatch_id": 0,
                    "sid_level": 1,
                    "batch_size": 1,
                    "stage_seconds": 0,
                    "prompt_work": [[5, 5]],
                }
                prom.consume_outputs(
                    processor,
                    [
                        NS(
                            request_id="r",
                            kv_transfer_params={
                                TELEMETRY_KEY: [value],
                                "gr_metrics_model_name": "m",
                            },
                        )
                    ],
                )
        self.assertEqual(sum(self.values("prompt.tokens")), 10)

    def test_disabled_metrics(self) -> None:
        with patch.dict(os.environ, {"VLLM_GR_METRICS_ENABLED": "0"}):
            metrics_config.cache_clear()
            with patch.object(prom, "get_metrics") as get:
                prom.consume_outputs(NS(), [NS(kv_transfer_params={TELEMETRY_KEY: [{}]})])
                get.assert_not_called()


class TimerTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        metrics_config.cache_clear()

    def test_wall_timer_context_and_idempotent_stop(self) -> None:
        with patch("vllm_gr.v1.metrics.timer.perf_counter", side_effect=[1.0, 1.125]):
            with WallTimer() as timer:
                pass
            self.assertEqual(timer.stop(), 0.125)
        with self.assertRaises(RuntimeError):
            timer.start()

    def test_wall_timer_closes_on_exception(self) -> None:
        timer = WallTimer()
        with patch("vllm_gr.v1.metrics.timer.perf_counter", side_effect=[1.0, 1.3]):
            with self.assertRaises(ValueError), timer:
                raise ValueError("conversion failed")
        self.assertAlmostEqual(timer.seconds, 0.3)

    def test_three_stages_and_actual_candidate_extent(self) -> None:
        s = session()
        with patch(
            "vllm_gr.v1.metrics.timer.perf_counter",
            side_effect=[1.0, 1.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5],
        ):
            values = []
            for index in range(3):
                m = metadata(s, index)
                stage_started(s, m, NS(num_scheduled_tokens={"r0": 3, "r1": 7}))
                values.append(
                    stage_finished(
                        s, m, NS(metrics={"candidate_width": 17, "sampled": False}), error=False
                    )
                )
        self.assertEqual(values[0]["prompt_work"], ((10, 3), (10, 7)))
        self.assertAlmostEqual(values[0]["prefill_seconds"], 0.1)
        self.assertEqual(values[2]["widths"], (4, 4, 17))
        self.assertAlmostEqual(s._metrics_decode_seconds, 0.4)
        self.assertEqual(stage_finished(s, metadata(s, 2), NS(), error=False), {})

    def test_prefill_only_decode_zero_and_structural_invalidity(self) -> None:
        s = session(batch=1, steps=1)
        with patch("vllm_gr.v1.metrics.timer.perf_counter", side_effect=[1.0, 1.2, 1.2, 1.3]):
            stage_started(s, metadata(s, 0), NS(num_scheduled_tokens={"r0": 10}))
            stage_finished(s, metadata(s, 0), NS(metrics=None), error=False)
        good = NS(token_ids=(1,), cumulative_logprob=-1.0, parent_beam_indices=(0,))
        bad = NS(token_ids=(1,), cumulative_logprob=math.nan, parent_beam_indices=())
        value = terminal_metrics(
            s,
            NS(
                failure=None,
                output=NS(items=[NS(item_index=0, sequences=(good, bad), business="kept")]),
            ),
        )
        self.assertEqual(value["decode_seconds"], 0)
        self.assertEqual(value["result_counts"], [1])
        self.assertEqual(value["invalid_counts"], {"non_finite_score": 1})

    def test_dynamic_parent_bounds(self) -> None:
        s = session(batch=1)
        s.request.beam_params.dynamic_beam_widths = (2, 4, 3)
        seq = NS(token_ids=(1, 2, 3), cumulative_logprob=-1.0, parent_beam_indices=(0, 1, 3))
        self.assertIsNone(invalid_sequence_reason(seq, s.request.beam_params))
        seq.parent_beam_indices = (0, 2, 3)
        self.assertEqual(invalid_sequence_reason(seq, s.request.beam_params), "invalid_parent")

    def test_malformed_output_does_not_crash_core_metrics(self) -> None:
        s = session()
        s._metrics_started = True
        value = terminal_metrics(s, NS(failure=None, output=object()))
        self.assertEqual(value["result_counts"], [0, 0])
        self.assertEqual(value["invalid_counts"], {"malformed_result": 1})

    def test_bad_optional_worker_metrics_are_ignored(self) -> None:
        s = session()
        stage_started(s, metadata(s, 0), NS(num_scheduled_tokens={"r0": 10, "r1": 10}))
        value = stage_finished(s, metadata(s, 0), NS(metrics="invalid"), error=False)
        self.assertEqual(value["widths"], (1, 4, 4))
        self.assertNotIn("device", value)

    def test_failed_first_stage_has_no_successful_prefill_timer(self) -> None:
        s = session()
        stage_started(s, metadata(s, 0), NS(num_scheduled_tokens={"r0": 10, "r1": 10}))
        value = stage_finished(s, metadata(s, 0), NS(metrics={}), error=True)
        self.assertIn("stage_seconds", value)
        self.assertNotIn("prefill_seconds", value)
        self.assertIsNone(s._metrics_decode_timer)


class GPUWireTests(unittest.TestCase):
    def test_token_bits_parent_order_and_padding(self) -> None:
        output = torch.empty((1, 3, 4), dtype=torch.int32)
        pack_survivors(
            output,
            torch.tensor([[2**25 + 1, 7, 0]], dtype=torch.int32),
            torch.tensor([[-4.0, -3.0, -math.inf]]),
            torch.tensor([[1, 0, 0]]),
            torch.tensor([[-2.0, -3.0, -5.0]]),
            torch.tensor([[True, True, False]]),
            first=False,
        )
        rows = unpack_survivors(output, 3)
        self.assertEqual(rows[0][0], (2**25 + 1, -4.0, -1.0, 1))
        self.assertEqual(rows[0][1], (7, -3.0, -1.0, 1))
        self.assertEqual(rows[0][2][3], 0)

    def test_slot_sampling_and_auxiliary_timing(self) -> None:
        class EventStub:
            def record(self) -> None:
                pass

            def elapsed_time(self, end: Any) -> Any:
                return 2.0

            def synchronize(self) -> None:
                raise AssertionError("Metrics must reuse the owner's existing fence")

        state = NS(
            beam_width=4,
            device="cpu",
            beam_scores=torch.tensor([[-1.0, -2.0, -3.0, -4.0]]),
            input_ids=torch.tensor([[5, 6, 7, 0]]),
            parent_ids=torch.tensor([[0, 1, 2, 0]]),
            active_mask=torch.tensor([[True, True, True, False]]),
        )
        owner = NS(beam=NS(state=state), bindings=(0,), dynamic=None)
        original_empty = torch.empty

        def empty(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("pin_memory", None)
            return original_empty(*args, **kwargs)

        with patch.dict(os.environ, {"VLLM_GR_METRICS_SAMPLE_EVERY": "10"}):
            metrics_config.cache_clear()
            with (
                patch("torch.Event", side_effect=lambda **kwargs: EventStub()),
                patch("torch.empty", empty),
            ):
                timer = GPUStageMetrics(owner, 2)
        m = NS(stage_index=1, owner_generation=10, is_last_stage=False)
        timer.begin(0, m, owner)
        timer.stop("forward")
        timer.stop("forward")  # Prefill stops before auxiliary branches; sample stops again safely.
        with timer.measure("constraint_select"):
            pass
        timer.note_candidate_width(0, 4)
        timer.snapshot(owner, NS(output_width=4))
        timer.enqueue_copy(0)
        timer.begin(1, NS(stage_index=1, owner_generation=11, is_last_stage=False), owner)
        timer.note_candidate_width(1, 17)
        sampled = timer.consume(0)
        self.assertEqual(sampled["timings"]["forward"], 0.002)
        self.assertEqual(sampled["timings"]["constraint_select"], 0.002)
        self.assertEqual(sampled["rows"][0][0][:3], (5, -1.0, 0.0))
        self.assertEqual(timer.consume(1), {"sampled": False, "candidate_width": 17})
        metrics_config.cache_clear()

    def test_disabled_allocates_no_device_metrics(self) -> None:
        with patch.dict(os.environ, {"VLLM_GR_METRICS_ENABLED": "0"}):
            metrics_config.cache_clear()
            timer = GPUStageMetrics(NS(), 2)
            self.assertEqual(timer.slots, [])
            self.assertIsNone(timer.consume(0))
        metrics_config.cache_clear()


class ReporterTests(unittest.TestCase):
    def test_idle_reporter_runs_flush_lifecycle(self) -> None:
        flushed = Event()
        client = Client()
        client.flush = flushed.set  # type: ignore[method-assign]
        reporter = prom.PrometheusMetricsReporter(lambda: client)
        try:
            self.assertTrue(flushed.wait(3))
        finally:
            reporter.close()

    def test_background_report_and_flush(self) -> None:
        client = Client()
        reporter = prom.PrometheusMetricsReporter(lambda: client)
        try:
            self.assertTrue(
                reporter.observe(
                    "m",
                    (1,),
                    {"version": 1, "sid_level": 1, "batch_size": 1, "stage_seconds": 0.001},
                )
            )
            self.assertTrue(reporter.flush())
            self.assertEqual(client.events[0][:3], ("timer", "stage.elapsed.ms", 1.0))
        finally:
            reporter.close()

    def test_full_queue_drops_telemetry_without_blocking_inference(self) -> None:
        entered, release = Event(), Event()
        client = Client()
        original = client.observe_timer

        def emit(*args: Any, **kwargs: Any) -> None:
            entered.set()
            release.wait(5)
            original(*args, **kwargs)

        client.observe_timer = emit  # type: ignore[method-assign]
        reporter = prom.PrometheusMetricsReporter(lambda: client, queue_size=1)
        value = {"version": 1, "sid_level": 1, "batch_size": 1, "stage_seconds": 0.001}
        try:
            reporter.observe("m", (1,), value)
            self.assertTrue(entered.wait(5))
            self.assertTrue(reporter.observe("m", (2,), value))
            with self.assertLogs(prom.logger, "WARNING"):
                self.assertFalse(reporter.observe("m", (3,), value))
            self.assertEqual(reporter.dropped, 1)
            release.set()
            self.assertFalse(reporter.flush())
            self.assertTrue(any(name == "metrics.dropped" for _, name, _, _ in client.events))
        finally:
            release.set()
            reporter.close()

    def test_exporter_initialization_failure_is_visible_and_fail_open(self) -> None:
        def factory() -> None:
            raise ImportError("exporter missing")

        with self.assertLogs(prom.logger, "ERROR"):
            reporter = prom.PrometheusMetricsReporter(factory)
            try:
                self.assertTrue(reporter.observe("m", (1,), {"version": 1}))
                self.assertFalse(reporter.flush())
                self.assertIn("exporter missing", reporter.client_error)
            finally:
                reporter.close()

    def test_flush_error_is_reported_before_barrier_completes(self) -> None:
        client = Client()

        def fail() -> None:
            raise OSError("flush failed")

        client.flush = fail  # type: ignore[method-assign]
        reporter = prom.PrometheusMetricsReporter(lambda: client)
        with self.assertLogs(prom.logger, "ERROR"):
            self.assertFalse(reporter.flush())
            self.assertGreater(reporter.report_errors, 0)
            reporter.close()


class PrometheusExportTests(unittest.TestCase):
    def setUp(self) -> None:
        from prometheus_client import CollectorRegistry

        self.env = patch.dict(os.environ)
        self.env.start()
        self.registry = CollectorRegistry()
        self.client = prom.PrometheusClient("vllm_gr", registry=self.registry)
        self.collector = prom.PrometheusMetricsCollector(self.client)

    def tearDown(self) -> None:
        self.env.stop()
        metrics_config.cache_clear()

    def value(self, name: Any, **labels: Any) -> Any:
        return self.registry.get_sample_value("vllm_gr_" + name, labels)

    def test_real_histogram_and_counter_exposition(self) -> None:
        self.collector.observe(
            "test-model",
            (1,),
            {
                "version": 1,
                "sid_level": 1,
                "batch_size": 1,
                "stage_seconds": 0.00125,
                "prompt_work": [(10, 3)],
                "device": {
                    "sampled": True,
                    "timings": {"forward": 0.0005},
                    "rows": [[(7, -3.0, -1.0, 1)]],
                },
            },
        )
        stage = {"model": "test-model", "sid_level": "1"}
        self.assertEqual(self.value("stage_elapsed_ms_sum", **stage), 1.25)
        self.assertEqual(self.value("stage_elapsed_ms_count", **stage), 1)
        self.assertEqual(self.value("stage_elapsed_ms_bucket", **stage, le="1.0"), 0)
        self.assertEqual(self.value("stage_elapsed_ms_bucket", **stage, le="2.5"), 1)
        self.assertEqual(self.value("prompt_tokens_total", model="test-model"), 10)
        self.assertEqual(
            self.value("cache_prefix_tokens_total", model="test-model", status="hit"), 7
        )
        self.assertEqual(
            self.value(
                "sid_score_nll_sum", **stage, population="retained", score_kind="cumulative"
            ),
            3,
        )

    def test_shared_registry_reuses_families_without_reset(self) -> None:
        second = prom.PrometheusClient("vllm_gr", registry=self.registry)
        self.client.increment_counter("prompt.tokens", 3, {"model": "m"})
        second.increment_counter("prompt.tokens", 7, {"model": "m"})
        self.client.observe_timer("stage.elapsed.ms", 1.25, {"model": "m", "sid_level": "1"})
        second.observe_timer("stage.elapsed.ms", 2.5, {"model": "m", "sid_level": "1"})
        self.client.close()
        self.assertEqual(self.value("prompt_tokens_total", model="m"), 10)
        self.assertEqual(self.value("stage_elapsed_ms_sum", model="m", sid_level="1"), 3.75)
        self.assertEqual(self.value("stage_elapsed_ms_count", model="m", sid_level="1"), 2)

    def test_http_scrape_contains_real_observations(self) -> None:
        from urllib.request import urlopen

        self.client.observe_timer("stage.elapsed.ms", 1.25, {"model": "m", "sid_level": "1"})
        server, thread = prom.start_metrics_server(0, registry=self.registry)
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/metrics", timeout=3) as response:
                text = response.read().decode()
            self.assertIn("# TYPE vllm_gr_stage_elapsed_ms histogram", text)
            self.assertIn('vllm_gr_stage_elapsed_ms_sum{model="m",sid_level="1"} 1.25', text)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

    def test_native_registry_receives_frontend_observations(self) -> None:
        from prometheus_client import REGISTRY, generate_latest

        with patch.dict(os.environ), patch.object(prom, "_families", prom.WeakKeyDictionary()):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            client = prom.PrometheusClient("gr_registry_contract")
            try:
                client.increment_counter("prompt.samples", 2, {"model": "registry-model"})
                self.assertIs(prom.get_metrics_registry(), REGISTRY)
                expected = b'gr_registry_contract_prompt_samples_total{model="registry-model"} 2.0'
                self.assertIn(expected, generate_latest(prom.get_metrics_registry()))
            finally:
                for metric in client._metrics.values():
                    REGISTRY.unregister(metric)

    def test_multiprocess_scrape_aggregates_frontend_processes(self) -> None:
        writer = """
import sys
from vllm_gr.v1.metrics.prometheus import PrometheusClient
client=PrometheusClient("gr_mp_contract")
client.increment_counter("prompt.tokens",float(sys.argv[1]),{"model":"m"})
client.observe_timer("stage.elapsed.ms",float(sys.argv[2]),{"model":"m","sid_level":"1"})
"""
        reader = """
import json
from vllm_gr.v1.metrics.prometheus import get_metrics_registry
registry=get_metrics_registry()
print(json.dumps([registry.get_sample_value("gr_mp_contract_prompt_tokens_total",{"model":"m"}),
 registry.get_sample_value("gr_mp_contract_stage_elapsed_ms_sum",{"model":"m","sid_level":"1"}),
 registry.get_sample_value("gr_mp_contract_stage_elapsed_ms_count",{"model":"m","sid_level":"1"})]))
"""
        with tempfile.TemporaryDirectory() as directory:
            env = {
                **os.environ,
                "PROMETHEUS_MULTIPROC_DIR": directory,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            for tokens, elapsed in ((3, 1.25), (7, 2.5)):
                result = subprocess.run(
                    [sys.executable, "-c", writer, str(tokens), str(elapsed)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [sys.executable, "-c", reader],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [10, 3.75, 2])

    def test_no_private_sdk_is_needed_by_frontend(self) -> None:
        script = """
import builtins
original=builtins.__import__
def checked(name,*args,**kwargs):
 if name=="bytedance" or name.startswith("bytedance."):raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=checked
from prometheus_client import CollectorRegistry,generate_latest
from vllm_gr.v1.metrics.prometheus import PrometheusClient,PrometheusMetricsCollector
registry=CollectorRegistry()
client=PrometheusClient("public_gr",registry=registry)
PrometheusMetricsCollector(client).observe("m",(1,),dict(version=1,sid_level=1,
 batch_size=1,stage_seconds=.00125,prompt_work=[(10,3)]))
assert b"public_gr_stage_elapsed_ms_sum" in generate_latest(registry)
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_optional_server_is_started_once_per_process(self) -> None:
        with (
            patch.dict(os.environ, {"VLLM_GR_METRICS_PORT": "8000"}),
            patch.object(prom, "_servers", {}),
            patch.object(prom, "start_metrics_server") as start,
        ):
            metrics_config.cache_clear()
            prom._create_client()
            prom._create_client()
            start.assert_called_once_with(8000, "127.0.0.1")

    def test_invalid_scrape_port_is_rejected(self) -> None:
        for value in ("-1", "65536", "invalid"):
            with patch.dict(os.environ, {"VLLM_GR_METRICS_PORT": value}):
                metrics_config.cache_clear()
                with self.assertRaises(ValueError):
                    metrics_config()

    def test_worker_clock_modules_have_no_prometheus_import(self) -> None:
        root = Path(__file__).parents[3] / "vllm_gr/v1/metrics"
        for name in ("timer.py", "telemetry.py", "gpu.py"):
            path = root / name
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all("prometheus" not in item.name for item in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn("prometheus", node.module or "")


if __name__ == "__main__":
    unittest.main()
