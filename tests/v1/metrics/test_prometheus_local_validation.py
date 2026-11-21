# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Independent local oracles for native transport, Prometheus scrape data and real clocks.

Synthetic stage completions exercise real telemetry and native output codecs.
These tests do not claim to exercise a checkpoint or a running vLLM engine.
"""

import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
from typing import Any
from unittest.mock import patch

import msgspec
import pytest

from vllm_gr.v1.metrics import prometheus as prom
from vllm_gr.v1.metrics.telemetry import metrics_config, stage_finished, stage_started
from vllm_gr.v1.metrics.telemetry import terminal_metrics
from vllm_gr.v1.metrics.timer import WallTimer

PREFIX = "vllm_gr.local_validation"
MODEL = "local-validation-model"


def text(value: Any) -> Any:
    return value.decode() if isinstance(value, bytes) else value


def record(metric_type: Any, name: Any, value: Any, **tags: Any) -> Any:
    return (
        metric_type,
        PREFIX + "." + name,
        round(float(value), 8),
        tuple(sorted({"model": MODEL, **tags}.items())),
    )


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class LocalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        from prometheus_client import CollectorRegistry

        self.env = patch.dict(
            os.environ,
            {
                "VLLM_GR_METRICS_ENABLED": "1",
                "VLLM_GR_METRICS_SAMPLE_EVERY": "1",
                "VLLM_GR_METRICS_TOKEN_CODE_LIMIT": "2",
                "VLLM_GR_METRICS_PREFIX": PREFIX,
                "VLLM_GR_METRICS_QUEUE_SIZE": "256",
            },
        )
        self.env.start()
        metrics_config.cache_clear()
        self.registry = CollectorRegistry()
        self.client = prom.PrometheusClient(registry=self.registry)
        reporter = prom.PrometheusMetricsReporter(lambda: self.client)
        self.reporter_patch = patch.object(prom, "_reporter", reporter)
        self.reporter_patch.start()
        self.messages: list[Any] = []
        self.scrape = ""

    def finish(self) -> Any:
        from prometheus_client import generate_latest
        from prometheus_client.parser import text_string_to_metric_families

        reporter = prom._reporter
        if reporter is not None and not reporter._closed:
            self.assertTrue(
                reporter.flush(5),
                {
                    "client_error": reporter.client_error,
                    "report_errors": reporter.report_errors,
                    "emission_errors": (
                        reporter.collector.emission_errors if reporter.collector else None
                    ),
                    "dropped": reporter.dropped,
                },
            )
            reporter.close()
        self.scrape = generate_latest(self.registry).decode()
        self.messages = [
            {"kind": family.type, "name": sample.name, "value": sample.value, "tags": sample.labels}
            for family in text_string_to_metric_families(self.scrape)
            for sample in family.samples
            if not sample.name.endswith("_created")
        ]
        return self.messages

    def tearDown(self) -> None:
        if prom._reporter is not None:
            prom._reporter.close()
        artifact_dir = os.environ.get("PROMETHEUS_METRICS_VALIDATION_ARTIFACT_DIR")
        if artifact_dir:
            self.finish()
            path = Path(artifact_dir) / (self._testMethodName + ".prom")
            path.write_text(self.scrape)
        self.reporter_patch.stop()
        self.env.stop()
        metrics_config.cache_clear()

    def evidence(self, name: Any, data: Any) -> None:
        artifact_dir = os.environ.get("PROMETHEUS_METRICS_VALIDATION_ARTIFACT_DIR")
        if artifact_dir:
            (Path(artifact_dir) / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")

    def assert_scrape(self, expected: Any) -> None:
        from collections import defaultdict

        values = defaultdict(list)
        expected_series: dict[Any, float] = {}
        for kind, name, value, tags in expected:
            suffix = name[len(PREFIX) + 1 :]
            wire = prom.metric_name(PREFIX, suffix)
            values[kind, suffix, tags].append(value)
            if kind == "timer":
                count_key, sum_key = (wire + "_count", tags), (wire + "_sum", tags)
                expected_series[count_key] = expected_series.get(count_key, 0) + 1
                expected_series[sum_key] = expected_series.get(sum_key, 0) + value
            else:
                key = wire + "_total", tags
                expected_series[key] = expected_series.get(key, 0) + value
        actual = {
            (m["name"], tuple(sorted(m["tags"].items()))): m["value"]
            for m in self.finish()
            if not m["name"].endswith("_bucket")
        }
        self.evidence(
            "prometheus_" + self._testMethodName,
            {
                "expected_observations": len(expected),
                "expected_series": len(expected_series),
                "received_series": len(actual),
                "missing": [repr(key) for key in expected_series.keys() - actual.keys()],
                "unexpected": [repr(key) for key in actual.keys() - expected_series.keys()],
                "expected": expected,
            },
        )
        self.assertEqual(
            expected_series.keys(), actual.keys(), "Missing or unexpected metric series"
        )
        for key, value in expected_series.items():
            self.assertAlmostEqual(actual[key], value, places=7, msg=str(key))
        for (kind, suffix, tags), observations in values.items():
            if kind != "timer":
                continue
            buckets = prom.HISTOGRAMS[suffix][2]
            for upper in (*buckets, float("inf")):
                labels = dict(tags, le=str(float(upper)) if math.isfinite(upper) else "+Inf")
                name = prom.metric_name(PREFIX, suffix) + "_bucket"
                count = self.registry.get_sample_value(name, labels)
                self.assertEqual(count, sum(value <= upper for value in observations))

    def test_three_stage_native_transport_and_exact_prometheus_records(self) -> None:
        from vllm.v1.engine import EngineCoreOutputs
        from vllm_gr.v1.engine.output_processor import make_patched_process_outputs
        from vllm_gr.v1.core.sched.scheduler_patch import _attach_gr_metrics

        params = NS(beam_width=3, max_tokens=3, dynamic_beam_widths=None)
        s = NS(
            owner_generation=9,
            batch_size=2,
            total_stages=3,
            session_id="local-batch",
            request_ids=("r0", "r1"),
            request=NS(beam_params=params, items=(NS(request=NS(client_index=0)),)),
            _metrics_dispatch_timers={},
            _metrics_started=False,
            _metrics_decode_timer=None,
            _metrics_decode_seconds=None,
            _metrics_prompt_work=(),
        )
        scheduler = NS(vllm_config=NS(model_config=NS(served_model_name=[MODEL])))
        devices = [
            {
                "sampled": True,
                "timings": {
                    "forward": 0.0005,
                    "lineage_kv": 0.000125,
                    "output_materialize": 0.00025,
                },
                "population": "retained",
                "rows": [
                    [(7, -3.0, -1.0, 1), (8, -math.inf, -1.0, 0), (9, math.nan, -1.0, 1)],
                    [(7, -4.0, -2.0, 1), (10, -5.0, -3.0, 1), (12, -6.0, -4.0, 1)],
                ],
            },
            {"sampled": False, "candidate_width": 17},
            {
                "sampled": True,
                "timings": {"final_pack": 0.00075},
                "population": "returned",
                "rows": [[(11, -6.0, -2.0, 1)], [(7, -7.0, -1.0, 1)]],
            },
        ]
        result: dict[Any, Any] = {}
        # Independent fixture: 12.5/5/20 ms stages, 37.5 ms decode, 7 hit + 23 miss tokens.
        with patch(
            "vllm_gr.v1.metrics.timer.perf_counter",
            side_effect=[1.0, 1.0125, 1.0125, 1.02, 1.025, 1.03, 1.05, 1.05],
        ):
            for index in range(3):
                metadata = NS(
                    session_id=s.session_id,
                    dispatch_id=index,
                    stage_index=index,
                    owner_generation=9,
                    kind="prefill" if index == 0 else "decode",
                    decode_step=None if index == 0 else index - 1,
                    produces_output=True,
                    beam_params=params,
                    request_ids=s.request_ids,
                    num_prompt_tokens_by_request=(10, 20),
                    num_prompt_tokens=30,
                )
                stage_started(s, metadata, NS(num_scheduled_tokens={"r0": 3, "r1": 20}))
                value = stage_finished(s, metadata, NS(metrics=devices[index]), error=False)
                _attach_gr_metrics(scheduler, result, s, value)
        good0 = NS(token_ids=(7, 8, 11), cumulative_logprob=-6.0, parent_beam_indices=(0, 0, 0))
        good1 = NS(token_ids=(7, 8, 7), cumulative_logprob=-7.0, parent_beam_indices=(0, 0, 0))
        invalid = NS(token_ids=(7, 8, 7), cumulative_logprob=math.nan, parent_beam_indices=())
        terminal = terminal_metrics(
            s,
            NS(
                failure=None,
                output=NS(
                    items=[
                        NS(item_index=0, sequences=(good0,)),
                        NS(item_index=1, sequences=(good1, invalid)),
                    ]
                ),
            ),
        )
        _attach_gr_metrics(scheduler, result, s, terminal)
        result[0].outputs[0].kv_transfer_params["business"] = "preserved"
        encoded = msgspec.msgpack.encode(result[0])
        processor = NS(log_stats=False)

        def original(self: Any, outputs: Any) -> Any:
            self.assertion = [output.kv_transfer_params for output in outputs]
            return "original-return-value"

        wrapped = make_patched_process_outputs(original)
        with patch("vllm_gr.v1.engine.output_processor.note_admission_rejections"):
            for _ in range(2):  # A freshly decoded replay must not count the same dispatch again.
                decoded = msgspec.msgpack.decode(encoded, type=EngineCoreOutputs)
                self.assertEqual(wrapped(processor, decoded.outputs), "original-return-value")
                self.assertEqual(processor.assertion, [{"business": "preserved"}])
                self.assertEqual(decoded.outputs[0].new_token_ids, [])

        expected: list[Any] = []

        def add(metric_type: Any, name: Any, value: Any, **tags: Any) -> None:
            expected.append(record(metric_type, name, value, **tags))

        for level, elapsed, widths in (
            (1, 12.5, (1, 3, 3)),
            (2, 5.0, (3, 3, 17)),
            (3, 20.0, (3, 3, 3)),
        ):
            add("timer", "stage.elapsed.ms", elapsed, sid_level=str(level))
            for kind, width in zip(("input", "target", "candidate_per_parent"), widths):
                for _ in range(2):
                    add("timer", "beam.width", width, sid_level=str(level), kind=kind)
        add("timer", "request.prefill.ms", 12.5)
        add("rate_counter", "prompt.samples", 2)
        for prompt, work, hit in ((10, 3, 7), (20, 20, 0)):
            add("rate_counter", "prompt.tokens", prompt)
            add("timer", "prompt.length", prompt)
            add("rate_counter", "cache.prefix.tokens", hit, status="hit")
            add("rate_counter", "cache.prefix.tokens", work, status="miss")
            add("rate_counter", "prompt.tokens_by_source", work, source="local_compute")
            add("rate_counter", "prompt.tokens_by_source", hit, source="local_cache_hit")
        for level, population, effective, scores, tokens in (
            (
                1,
                "retained",
                (1, 3),
                ((3, 1), (4, 2), (5, 3), (6, 4)),
                {"7": 2, "10": 1, "other": 1},
            ),
            (3, "returned", (1, 1), ((6, 2), (7, 1)), {"11": 1, "7": 1}),
        ):
            tags = {"sid_level": str(level)}
            add("rate_counter", "stage.device_samples", 1, **tags)
            for width in effective:
                add("timer", "beam.width", width, **tags, kind="effective")
            for cumulative, local in scores:
                add(
                    "timer",
                    "sid.score.nll",
                    cumulative,
                    **tags,
                    population=population,
                    score_kind="cumulative",
                )
                add(
                    "timer",
                    "sid.score.nll",
                    local,
                    **tags,
                    population=population,
                    score_kind="local",
                )
            for kind in ("cumulative", "local"):
                add(
                    "rate_counter",
                    "sid.score.count",
                    len(scores),
                    **tags,
                    population=population,
                    score_kind=kind,
                )
            for token, count in tokens.items():
                add(
                    "rate_counter",
                    "token_code",
                    count,
                    **tags,
                    population=population,
                    token_code=token,
                )
        add("timer", "stage.forward.ms", 0.5, sid_level="1")
        add(
            "timer",
            "gr.processing.ms",
            0.125,
            sid_level="1",
            component="lineage_kv",
            clock="device",
        )
        add(
            "timer",
            "gr.processing.ms",
            0.25,
            sid_level="1",
            component="output_materialize",
            clock="host",
        )
        add(
            "timer", "gr.processing.ms", 0.75, sid_level="3", component="final_pack", clock="device"
        )
        for _ in range(2):
            add("timer", "beam.result_count", 1)
        add("rate_counter", "beam.invalid.result_count", 1, reason="non_finite_score")
        add("timer", "request.decode.ms", 37.5)
        self.assert_scrape(expected)

    def test_wall_clock_calibration_through_real_prometheus(self) -> None:
        before = time.perf_counter()
        with WallTimer() as timer:
            time.sleep(0.02)
        reference_ms = (time.perf_counter() - before) * 1000
        prom.get_metrics().observe(
            MODEL,
            (1,),
            {"version": 1, "sid_level": 1, "batch_size": 1, "stage_seconds": timer.seconds},
        )
        self.finish()
        labels = {"model": MODEL, "sid_level": "1"}
        name = prom.metric_name(PREFIX, "stage.elapsed.ms")
        actual = self.registry.get_sample_value(name + "_sum", labels)
        self.assertEqual(self.registry.get_sample_value(name + "_count", labels), 1)
        self.evidence(
            "prometheus_host_calibration", {"reference_ms": reference_ms, "reported_ms": actual}
        )
        self.assertGreaterEqual(actual, 20)
        self.assertLessEqual(abs(actual - reference_ms), 1)


if __name__ == "__main__":
    unittest.main()
