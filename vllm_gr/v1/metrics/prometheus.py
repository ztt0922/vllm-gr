# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Frontend Prometheus Histogram/Counter reporting through a bounded queue.

Core and Worker clocks are backend independent. Prometheus is imported only
when the frontend creates a client or explicitly starts a scrape server.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import regex as re
from collections import Counter, OrderedDict
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any
from weakref import WeakKeyDictionary

from vllm_gr.v1.metrics.telemetry import TELEMETRY_KEY, metrics_config

logger = logging.getLogger(__name__)


_TIME_BUCKETS_MS = (
    0,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    30000,
    60000,
)
_COUNT_BUCKETS = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
)
_SCORE_BUCKETS = (0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)

# Keys remain identical to the telemetry's existing metric names. The wire
# names replace punctuation with underscores; Counter adds its standard _total.
HISTOGRAMS = {
    "request.prefill.ms": (
        "First SID dispatch to its result processing, in milliseconds.",
        ("model",),
        _TIME_BUCKETS_MS,
    ),
    "request.decode.ms": (
        "First SID result to last SID result, in milliseconds.",
        ("model",),
        _TIME_BUCKETS_MS,
    ),
    "stage.elapsed.ms": (
        "SID dispatch to result processing, in milliseconds.",
        ("model", "sid_level"),
        _TIME_BUCKETS_MS,
    ),
    "stage.forward.ms": (
        "SID preparation and forward/logits device interval, in milliseconds.",
        ("model", "sid_level"),
        _TIME_BUCKETS_MS,
    ),
    "gr.processing.ms": (
        "GR component execution interval, in milliseconds.",
        ("model", "sid_level", "component", "clock"),
        _TIME_BUCKETS_MS,
    ),
    "prompt.length": ("Effective input tokens per prompt row.", ("model",), _COUNT_BUCKETS),
    "beam.width": (
        "Configured or effective beam width per input row.",
        ("model", "sid_level", "kind"),
        _COUNT_BUCKETS,
    ),
    "beam.result_count": (
        "Structurally valid final SID candidates per input row.",
        ("model",),
        _COUNT_BUCKETS,
    ),
    "sid.score.nll": (
        "Negative raw model log probability of valid selected SID candidates.",
        ("model", "sid_level", "population", "score_kind"),
        _SCORE_BUCKETS,
    ),
}
COUNTERS = {
    "prompt.samples": ("Input rows entering the first GR stage.", ("model",)),
    "prompt.tokens": ("Effective input prompt tokens, including prefix reuse.", ("model",)),
    "cache.prefix.tokens": ("Reused or locally computed prompt tokens.", ("model", "status")),
    "prompt.tokens_by_source": (
        "Effective prompt tokens by computation source.",
        ("model", "source"),
    ),
    "stage.device_samples": ("Received sampled SID device batches.", ("model", "sid_level")),
    "sid.score.count": (
        "Valid candidates observed in each SID score distribution.",
        ("model", "sid_level", "population", "score_kind"),
    ),
    "token_code": (
        "Selected valid SID token occurrences, with a bounded token label budget.",
        ("model", "sid_level", "population", "token_code"),
    ),
    "beam.invalid.result_count": ("Invalid final SID records by reason.", ("model", "reason")),
    "batch.failures": ("Published GR batch failures by bounded reason.", ("model", "reason")),
    "metrics.dropped": ("Telemetry frames dropped by the bounded frontend queue.", ("reason",)),
}

_families: WeakKeyDictionary = WeakKeyDictionary()
_families_lock = Lock()
_servers: dict[tuple[int, str, int], Any] = {}
_servers_lock = Lock()


def metric_name(prefix: str, name: str) -> str:
    """Convert existing dotted names into valid Prometheus metric names."""
    result = re.sub(r"[^a-zA-Z0-9_:]", "_", f"{prefix}.{name}")
    return result if re.match(r"[a-zA-Z_:]", result) else "_" + result


def get_metrics_registry():
    """Use the same scrape registry convention as the native vLLM /metrics route."""
    from prometheus_client import REGISTRY, CollectorRegistry, multiprocess

    if os.environ.get("PROMETHEUS_MULTIPROC_DIR") is not None:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return REGISTRY


def start_metrics_server(port: int = 8000, addr: str = "127.0.0.1", *, registry=None):
    """Expose metrics for offline/RPC serving; native HTTP serving already has /metrics."""
    from prometheus_client import start_http_server

    return start_http_server(
        port, addr=addr, registry=get_metrics_registry() if registry is None else registry
    )


class PrometheusClient:
    """Lazy metric families shared by clients in a process; values retain their old units."""

    def __init__(self, prefix: str | None = None, *, registry=None):
        from prometheus_client import REGISTRY

        self.registry = REGISTRY if registry is None else registry
        self.prefix = metrics_config().prefix if prefix is None else prefix
        self._metrics: dict[str, Any] = {}
        self._labels: dict[str, tuple[str, ...]] = {}

    def _metric(self, kind: str, name: str):
        from prometheus_client import Counter as PrometheusCounter, Histogram

        if name in self._metrics:
            return self._metrics[name]
        specs = HISTOGRAMS if kind == "timer" else COUNTERS
        description, labels, *buckets = specs[name]
        full_name = metric_name(self.prefix, name)
        with _families_lock:
            families = _families.setdefault(self.registry, {})
            cached = families.get(full_name)
            if cached is not None:
                previous_kind, previous_labels, metric = cached
                if (previous_kind, previous_labels) != (kind, labels):
                    raise ValueError(f"Metric {full_name} already uses different type/labels")
            else:
                metric = (
                    Histogram(
                        full_name, description, labels, buckets=buckets[0], registry=self.registry
                    )
                    if kind == "timer"
                    else PrometheusCounter(full_name, description, labels, registry=self.registry)
                )
                families[full_name] = kind, labels, metric
        self._metrics[name] = metric
        self._labels[name] = labels
        return metric

    def _sample(self, kind: str, name: str, tags: dict | None):
        metric = self._metric(kind, name)
        labels = self._labels[name]
        tags = tags or {}
        unknown = set(tags).difference(labels)
        if unknown:
            raise ValueError(f"Unexpected labels for {name}: {sorted(unknown)}")
        return metric.labels(**{label: str(tags.get(label, "")) for label in labels})

    def observe_timer(self, name: str, value: float, tags: dict | None = None) -> None:
        self._sample("timer", name, tags).observe(value)

    def increment_counter(self, name: str, value: float, tags: dict | None = None) -> None:
        self._sample("counter", name, tags).inc(value)

    def flush(self) -> None:
        # Observations are immediately visible in the registry. Reporter.flush
        # supplies the barrier that waits for all queued telemetry to be observed.
        pass

    def close(self) -> None:
        # Registry families must outlive the reporter so their final values can
        # still be scraped. Closing a reporter must not reset shared counters.
        pass


def _create_client():
    config = metrics_config()
    client = PrometheusClient()
    if config.prometheus_port is not None:
        key = os.getpid(), config.prometheus_addr, config.prometheus_port
        with _servers_lock:
            if key not in _servers:
                _servers[key] = start_metrics_server(config.prometheus_port, config.prometheus_addr)
    return client


class PrometheusMetricsCollector:
    """One reporter-thread collector; no metric object is created in the Worker."""

    def __init__(self, client: Any):
        self.client = client
        self.emission_errors = 0
        self._seen: OrderedDict[tuple, None] = OrderedDict()
        self._codes: dict[tuple[str, str], set[int]] = {}

    def _emit(self, kind: str, name: str, value: float, tags: dict) -> None:
        if not math.isfinite(value):
            return
        try:
            method = self.client.observe_timer if kind == "timer" else self.client.increment_counter
            method(name, value, tags=tags)
        except Exception:
            self.emission_errors += 1
            if self.emission_errors == 1:
                logger.exception("GR Prometheus observation failed; inference continues")

    def _timer(self, name: str, value: float, tags: dict) -> None:
        self._emit("timer", name, value, tags)

    def _counter(self, name: str, value: int, tags: dict) -> None:
        self._emit("rate_counter", name, value, tags)

    def observe(self, model: str, key: tuple, value: dict) -> None:
        if value.get("version") != 1 or key in self._seen:
            return
        self._seen[key] = None
        if len(self._seen) > 4096:
            self._seen.popitem(last=False)
        tags = {"model": model}
        if value.get("terminal"):
            for n in value.get("result_counts", ()):
                self._timer("beam.result_count", n, tags)
            for reason, n in value.get("invalid_counts", {}).items():
                self._counter("beam.invalid.result_count", n, {**tags, "reason": reason})
            if value.get("failure"):
                reason = value["failure"]
                known = {
                    "no_valid_beams",
                    "beam_device_error",
                    "worker_output_error",
                    "beam_final_output_error",
                    "worker_protocol_error",
                }
                self._counter(
                    "batch.failures", 1, {**tags, "reason": reason if reason in known else "other"}
                )
            if "decode_seconds" in value:
                self._timer("request.decode.ms", value["decode_seconds"] * 1000, tags)
            return

        level = str(value["sid_level"])
        stage_tags = {**tags, "sid_level": level}
        self._timer("stage.elapsed.ms", value["stage_seconds"] * 1000, stage_tags)
        if "prefill_seconds" in value:
            self._timer("request.prefill.ms", value["prefill_seconds"] * 1000, tags)
        prompts = value.get("prompt_work", ())
        if prompts:
            self._counter("prompt.samples", len(prompts), tags)
        for prompt, work in prompts:
            self._counter("prompt.tokens", prompt, tags)
            self._timer("prompt.length", prompt, tags)
            hit = prompt - work
            self._counter("cache.prefix.tokens", hit, {**tags, "status": "hit"})
            self._counter("cache.prefix.tokens", work, {**tags, "status": "miss"})
            self._counter("prompt.tokens_by_source", work, {**tags, "source": "local_compute"})
            self._counter("prompt.tokens_by_source", hit, {**tags, "source": "local_cache_hit"})
        for kind, n in zip(("input", "target", "candidate_per_parent"), value.get("widths", ())):
            for _ in range(value["batch_size"]):
                self._timer("beam.width", n, {**stage_tags, "kind": kind})

        device = value.get("device")
        if not device or not device.get("sampled", True):
            return
        self._counter("stage.device_samples", 1, stage_tags)
        for component, seconds in device.get("timings", {}).items():
            if component == "forward":
                self._timer("stage.forward.ms", seconds * 1000, stage_tags)
            else:
                clock = "host" if component == "output_materialize" else "device"
                self._timer(
                    "gr.processing.ms",
                    seconds * 1000,
                    {**stage_tags, "component": component, "clock": clock},
                )
        population = device.get("population", "retained")
        population_tags = {**stage_tags, "population": population}
        codes = self._codes.setdefault((model, level), set())
        limit = metrics_config().token_code_limit
        token_counts: Counter[str] = Counter()
        score_counts: Counter[str] = Counter()
        for row in device.get("rows", ()):
            valid = [v for v in row if v[3] and v[0] >= 0 and math.isfinite(v[1])]
            self._timer("beam.width", len(valid), {**stage_tags, "kind": "effective"})
            for token, cumulative, local, _ in valid:
                for kind, score in (("cumulative", cumulative), ("local", local)):
                    if math.isfinite(score) and score <= 0:
                        self._timer(
                            "sid.score.nll", -score, {**population_tags, "score_kind": kind}
                        )
                        score_counts[kind] += 1
                if limit:
                    token = int(token)
                    if token in codes or len(codes) < limit:
                        codes.add(token)
                        label = str(token)
                    else:
                        label = "other"
                    token_counts[label] += 1
        for kind, n in score_counts.items():
            self._counter("sid.score.count", n, {**population_tags, "score_kind": kind})
        for token, n in token_counts.items():
            self._counter("token_code", n, {**population_tags, "token_code": token})


class PrometheusMetricsReporter:
    """Nonblocking inference handoff; metric updates run on one background thread."""

    def __init__(self, client_factory=_create_client, queue_size: int | None = None):
        self.pid = os.getpid()
        self._queue: Queue = Queue(maxsize=queue_size or metrics_config().queue_size)
        self._closed = False
        self.dropped = 0
        self.client_error: str | None = None
        self.report_errors = 0
        self.collector: PrometheusMetricsCollector | None = None
        self._thread = Thread(
            target=self._run, args=(client_factory,), daemon=True, name="gr-prometheus-metrics"
        )
        self._thread.start()
        atexit.register(self.close)

    def observe(self, model: str, key: tuple, value: dict) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait((model, key, value))
            return True
        except Full:
            self.dropped += 1
            if self.dropped == 1:
                logger.warning("GR Prometheus queue full; telemetry dropped, inference continues")
            return False

    def _run(self, client_factory) -> None:
        try:
            self.collector = PrometheusMetricsCollector(client_factory())
        except Exception as error:
            self.client_error = str(error)
            logger.exception("GR Prometheus initialization failed; GR reporting disabled")
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                # Preserve the reporter's idle/barrier lifecycle. Prometheus
                # updates are immediate and its client flush is a no-op.
                self._flush_client()
                continue
            try:
                if item is None:
                    try:
                        if self.collector is not None:
                            self.collector.client.close()
                    except Exception:
                        self.report_errors += 1
                        logger.exception("GR Prometheus client close failed")
                    return
                if isinstance(item, Event):
                    try:
                        self._flush_client()
                    finally:
                        item.set()
                elif self.collector is not None:
                    if self.dropped:
                        reported = getattr(self, "_reported_drops", 0)
                        if self.dropped > reported:
                            self.collector._counter(
                                "metrics.dropped", self.dropped - reported, {"reason": "queue_full"}
                            )
                            self._reported_drops = self.dropped
                    self.collector.observe(*item)
            except Exception:
                self.report_errors += 1
                logger.exception("GR Prometheus report failed; inference continues")
            finally:
                self._queue.task_done()

    def _flush_client(self) -> None:
        if self.collector is None:
            return
        try:
            self.collector.client.flush()
        except Exception:
            self.report_errors += 1
            if self.report_errors == 1:
                logger.exception("GR Prometheus client flush failed")

    def flush(self, timeout: float = 5.0) -> bool:
        if self._closed or self.pid != os.getpid():
            return False
        deadline = monotonic() + max(0.0, timeout)
        barrier = Event()
        try:
            self._queue.put(barrier, timeout=max(0.0, deadline - monotonic()))
        except Full:
            return False
        return (
            barrier.wait(max(0.0, deadline - monotonic()))
            and self.collector is not None
            and not self.client_error
            and self.collector.emission_errors == 0
            and self.report_errors == 0
            and self.dropped == 0
        )

    def close(self, timeout: float = 2.0) -> None:
        if self._closed or self.pid != os.getpid():
            return
        self.flush(timeout)
        self._closed = True
        try:
            self._queue.put(None, timeout=timeout)
        except Full:
            return
        self._thread.join(timeout=timeout)


_reporter: PrometheusMetricsReporter | None = None
_reporter_lock = Lock()


def get_metrics() -> PrometheusMetricsReporter:
    global _reporter
    with _reporter_lock:
        if _reporter is None or _reporter.pid != os.getpid():
            _reporter = PrometheusMetricsReporter()
        return _reporter


def consume_outputs(processor: Any, outputs: Any) -> None:
    if not metrics_config().enabled:
        return
    for output in outputs:
        params = getattr(output, "kv_transfer_params", None) or {}
        values = params.pop(TELEMETRY_KEY, ())
        if not values:
            continue
        model = params.pop("gr_metrics_model_name", "gr")
        for value in values:
            key = (
                id(processor),
                output.request_id,
                value["generation"],
                "terminal" if value.get("terminal") else value["dispatch_id"],
            )
            get_metrics().observe(model, key, value)


def flush_metrics(timeout: float = 5.0) -> bool:
    """Drain queued observations to the registry; this does not acknowledge a scrape."""
    return _reporter.flush(timeout) if _reporter is not None else True
