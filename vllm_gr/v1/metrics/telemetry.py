# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Small CPU-only telemetry values transported through the native output channel.

No Prometheus registry lives in EngineCore or the Worker. Their CPU values
travel to the frontend reporting thread through the native output channel.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from vllm_gr.v1.metrics.timer import WallTimer

TELEMETRY_KEY = "gr_metrics"


@dataclass(frozen=True)
class MetricsConfig:
    enabled: bool = True
    sample_every: int = 1
    token_code_limit: int = 512
    prefix: str = "vllm_gr"
    queue_size: int = 256
    prometheus_port: int | None = None
    prometheus_addr: str = "127.0.0.1"


@lru_cache(maxsize=1)
def metrics_config() -> MetricsConfig:
    """Read once per process; set these variables before starting the engine."""
    enabled = os.environ.get("VLLM_GR_METRICS_ENABLED", "1")
    if enabled not in ("0", "1"):
        raise ValueError("VLLM_GR_METRICS_ENABLED must be 0 or 1")
    every = int(os.environ.get("VLLM_GR_METRICS_SAMPLE_EVERY", "1"))
    limit = int(os.environ.get("VLLM_GR_METRICS_TOKEN_CODE_LIMIT", "512"))
    if every < 1 or not 0 <= limit <= 4096:
        raise ValueError("metrics sample interval must be >=1 and token code limit in [0,4096]")
    prefix = os.environ.get("VLLM_GR_METRICS_PREFIX", "vllm_gr").rstrip(".")
    queue_size = int(os.environ.get("VLLM_GR_METRICS_QUEUE_SIZE", "256"))
    if not prefix or not 1 <= queue_size <= 4096:
        raise ValueError("metrics prefix must be nonempty and queue size in [1,4096]")
    port_value = os.environ.get("VLLM_GR_METRICS_PORT")
    port = int(port_value) if port_value is not None else None
    addr = os.environ.get("VLLM_GR_METRICS_ADDR", "127.0.0.1")
    if (port is not None and not 0 <= port <= 65535) or not addr:
        raise ValueError("metrics port must be in [0,65535] and address must be nonempty")
    return MetricsConfig(enabled == "1", every, limit, prefix, queue_size, port, addr)


def stage_started(session: Any, metadata: Any, scheduler_output: Any) -> None:
    if not metrics_config().enabled:
        return
    session._metrics_dispatch_timers[metadata.dispatch_id] = WallTimer().start()
    if metadata.stage_index == 0:
        session._metrics_started = True
        # V1 forbids external KV transfer, partial prefill and recomputation.
        # num_scheduled_tokens is therefore the actual local prompt work.
        prompts = metadata.num_prompt_tokens_by_request
        work = tuple(int(scheduler_output.num_scheduled_tokens[r]) for r in session.request_ids)
        if all(0 <= n <= p for p, n in zip(prompts, work, strict=True)):
            session._metrics_prompt_work = tuple(zip(prompts, work, strict=True))


def stage_finished(session: Any, metadata: Any, worker_result: Any, *, error: bool) -> dict:
    """Called only for a live, identity-checked dispatch, once per physical batch."""
    timer = session._metrics_dispatch_timers.pop(metadata.dispatch_id, None)
    if timer is None:
        return {}
    elapsed = timer.stop()
    terminal = getattr(worker_result, "terminal_result", None)
    successful = (
        not error
        and worker_result is not None
        and getattr(worker_result, "produced_token_count", 1) == 1
        and (terminal is None or terminal.failure is None)
    )
    if successful:
        if metadata.stage_index == 0:
            session._metrics_decode_timer = WallTimer().start()
        if metadata.stage_index == session.total_stages - 1:
            decode_timer = session._metrics_decode_timer
            if decode_timer is not None:
                decode_seconds = decode_timer.stop()
                session._metrics_decode_seconds = (
                    0.0 if session.total_stages == 1 else decode_seconds
                )
    value: dict[str, Any] = {
        "version": 1,
        "generation": session.owner_generation,
        "dispatch_id": metadata.dispatch_id,
        "sid_level": metadata.stage_index + 1,
        "batch_size": session.batch_size,
        "stage_seconds": elapsed,
    }
    if metadata.stage_index == 0:
        value["prompt_work"] = session._metrics_prompt_work
        if successful:
            value["prefill_seconds"] = elapsed
    if not error and worker_result is not None:
        from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView

        view = BeamBatchExecutionView.from_stage_metadata(metadata)
        sample = getattr(worker_result, "metrics", None)
        sample = sample if isinstance(sample, dict) else None
        candidate_width = (sample or {}).get("candidate_width")
        if type(candidate_width) is not int or candidate_width < 1:
            candidate_width = view.candidate_top_k
        value["widths"] = (view.input_width, view.output_width, candidate_width)
        if sample:
            value["device"] = sample
    return value


def terminal_metrics(session: Any, result: Any) -> dict:
    """Keep failures distinct from invalid returned sequences and padding."""
    if not session._metrics_started:
        return {}
    from vllm_gr.v1.beam.batch_contracts import BeamSequenceColumns

    value: dict[str, Any] = {
        "version": 1,
        "generation": session.owner_generation,
        "terminal": True,
        "result_counts": [0] * session.batch_size,
        "invalid_counts": {},
    }
    if result.failure is None and result.output is not None:
        items = getattr(result.output, "items", None)
        if not isinstance(items, (tuple, list)):
            value["invalid_counts"]["malformed_result"] = 1
            items = ()
        for item in items:
            index = getattr(item, "item_index", None)
            sequences = getattr(item, "sequences", None)
            if (
                type(index) is int
                and 0 <= index < session.batch_size
                and isinstance(sequences, BeamSequenceColumns)
            ):
                from vllm_gr.v1.beam.batch_contracts import validate_complete_sequences

                try:
                    validate_complete_sequences(sequences, session.request)
                except (TypeError, ValueError):
                    counts = value["invalid_counts"]
                    counts["malformed_result"] = counts.get("malformed_result", 0) + 1
                else:
                    value["result_counts"][index] += sequences.count
                continue
            if (
                type(index) is not int
                or not 0 <= index < session.batch_size
                or not isinstance(sequences, (tuple, list))
            ):
                counts = value["invalid_counts"]
                counts["malformed_result"] = counts.get("malformed_result", 0) + 1
                continue
            for sequence in sequences:
                try:
                    reason = invalid_sequence_reason(sequence, session.request.beam_params)
                except (TypeError, ValueError, AttributeError):
                    reason = "malformed_sequence"
                if reason is None:
                    value["result_counts"][index] += 1
                else:
                    counts = value["invalid_counts"]
                    counts[reason] = counts.get(reason, 0) + 1
        if session._metrics_decode_seconds is not None:
            value["decode_seconds"] = session._metrics_decode_seconds
    else:
        value["failure"] = getattr(result.failure, "code", "unknown")
    return value


def invalid_sequence_reason(sequence: Any, params: Any) -> str | None:
    """Structural invalidity of returned records, never unused capacity slots."""
    tokens = sequence.token_ids
    reason = getattr(sequence, "finish_reason", "length")
    if len(tokens) > params.max_tokens or (reason == "length" and len(tokens) != params.max_tokens):
        return "invalid_length"
    if any(type(token) is not int or token < 0 for token in tokens):
        return "invalid_token"
    score = sequence.cumulative_logprob
    if score is not None and not math.isfinite(score):
        return "non_finite_score"
    parents = sequence.parent_beam_indices
    if parents:
        widths = params.dynamic_beam_widths or (params.beam_width,) * params.max_tokens
        bounds = (1, *widths[:-1])
        if len(parents) != len(tokens) or any(
            type(p) is not int or not 0 <= p < bound for p, bound in zip(parents, bounds)
        ):
            return "invalid_parent"
    return None
