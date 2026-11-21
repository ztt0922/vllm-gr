# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam Search V1 state layered on vLLM's native asynchronous scheduler."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchRequest,
    BeamBatchResult,
    GRAdmissionError,
)
from vllm_gr.v1.beam.dispatch import (
    GR_BATCH_RESULT_ATTR,
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRDeviceStateRef,
    GRExecutionCapacity,
    GRRetireMetadata,
    GRRetireResult,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
)
from vllm_gr.v1.core.sched.session import (
    GRResourceStatus,
    GRSessionState,
    GRSessionStatus,
    GRStageCompletion,
)
from vllm_gr.v1.engine.beam_search import (
    validate_gr_device_config,
    validate_gr_execution_config,
    validate_gr_request,
)

GR_LOOKAHEAD_DEPTH = 2
_DRAINING_STATUSES = frozenset(
    {
        GRSessionStatus.FINISHING,
        GRSessionStatus.ABORTED,
        GRSessionStatus.FAILED,
    }
)


class _GRWorkerProtocolError(ValueError):
    """A malformed per-request Worker result; never escapes EngineCore."""


def _sessions(scheduler: Any) -> dict[str, GRSessionState]:
    sessions = getattr(scheduler, "_gr_v1_sessions", None)
    if sessions is None:
        sessions = {}
        scheduler._gr_v1_sessions = sessions
    return sessions


def _request_sessions(scheduler: Any) -> dict[str, str]:
    mapping = getattr(scheduler, "_gr_v1_request_sessions", None)
    if mapping is None:
        mapping = {}
        scheduler._gr_v1_request_sessions = mapping
    return mapping


def has_gr_cleanup_work(scheduler: Any) -> bool:
    """Keep terminal delivery and Worker retirement live under native pause."""
    # Retained sessions are not necessarily runnable: native pause policy
    # decides that, and EngineCore's batch queue drains ordinary in-flight
    # dispatches. Only GR cleanup adds work beyond those native checks.
    return any(
        session.terminal_result_pending
        or session.resource_status in (GRResourceStatus.DRAINING, GRResourceStatus.RETIRING)
        for session in getattr(scheduler, "_gr_v1_sessions", {}).values()
    )


def _progress(scheduler: Any) -> None:
    scheduler._gr_progress = getattr(scheduler, "_gr_progress", 0) + 1


def _admit_owner(scheduler: Any) -> GRSessionState | None:
    sessions = _sessions(scheduler)
    owner_id = getattr(scheduler, "_gr_owner", None)
    if owner_id is not None:
        return sessions[owner_id]
    # Dict insertion order is the session FIFO, independent of native priority.
    for session in sessions.values():
        if session.resource_status is GRResourceStatus.WAITING:
            generation = getattr(scheduler, "_gr_owner_generation", 0) + 1
            scheduler._gr_owner_generation = generation
            scheduler._gr_owner = session.session_id
            session.owner_generation = generation
            session.resource_status = GRResourceStatus.RUNNING
            _progress(scheduler)
            return session
    return None


class _AdmissionQueue:
    """Filtered native queue: hidden requests retain their objects and FIFO positions."""

    def __init__(self, queue: Any, owner_ids: tuple[str, ...]):
        self.queue = queue
        self.owner_ids = frozenset(owner_ids)

    def __iter__(self):
        return (r for r in self.queue if r.request_id in self.owner_ids)

    def __len__(self):
        return sum(1 for _ in self)

    def __bool__(self):
        return any(True for _ in self)

    def peek_request(self):
        return next(iter(self))

    def pop_request(self):
        request = self.peek_request()
        self.queue.remove_request(request)
        return request

    def __getattr__(self, name):
        return getattr(self.queue, name)


@contextmanager
def gr_admission(scheduler: Any):
    """Gate native selection and reserve the full logical KV extent on first allocation."""
    owner = _admit_owner(scheduler)
    if owner is None:
        yield
        return
    waiting, skipped = scheduler.waiting, scheduler.skipped_waiting
    budget = scheduler.max_num_scheduled_tokens
    manager = scheduler.kv_cache_manager
    allocate = manager.allocate_slots

    owner_ids = owner.request_ids
    owner_id_set = frozenset(owner_ids)

    def reserve(request, num_new_tokens, *args, **kwargs):
        if request.request_id not in owner_id_set:
            raise RuntimeError("Asynchronous scheduling allocation escaped the execution owner")
        computed = request.num_computed_tokens + kwargs.get("num_new_computed_tokens", 0)
        extent = request.num_prompt_tokens + owner.max_decode_steps
        kwargs["num_lookahead_tokens"] = max(0, extent - computed - num_new_tokens)
        result = allocate(request, num_new_tokens, *args, **kwargs)
        if result is None and owner.issued_stages:
            raise RuntimeError(
                "Asynchronous scheduling lost reserved KV capacity after Beam generation"
            )
        return result

    scheduler.waiting = _AdmissionQueue(waiting, owner_ids)
    scheduler.skipped_waiting = _AdmissionQueue(skipped, owner_ids)
    manager.allocate_slots = reserve
    if not can_issue_gr_stage(scheduler, owner):
        scheduler.max_num_scheduled_tokens = 0
    try:
        yield
    finally:
        scheduler.waiting, scheduler.skipped_waiting = waiting, skipped
        scheduler.max_num_scheduled_tokens = budget
        manager.allocate_slots = allocate


def _drain_resources(scheduler: Any, session: GRSessionState) -> None:
    session.resource_status = GRResourceStatus.DRAINING
    if (
        not session.worker_resources_allocated
        and not session._in_flight_dispatches
        and not session.terminal_result_pending
    ):
        _release_owner(scheduler, session)


def _release_owner(scheduler: Any, session: GRSessionState) -> None:
    session.resource_status = GRResourceStatus.CLOSED
    _sessions(scheduler).pop(session.session_id, None)
    request_sessions = _request_sessions(scheduler)
    for request_id in session.request_ids:
        if request_sessions.get(request_id) == session.session_id:
            request_sessions.pop(request_id, None)
    if getattr(scheduler, "_gr_owner", None) == session.session_id:
        scheduler._gr_owner = None
    _progress(scheduler)


def attach_gr_retirement(scheduler: Any, output: Any) -> None:
    owner = _sessions(scheduler).get(getattr(scheduler, "_gr_owner", None))
    if (
        owner is None
        or owner.resource_status is not GRResourceStatus.DRAINING
        or owner._in_flight_dispatches
        or not owner.worker_resources_allocated
    ):
        return
    if output.total_num_scheduled_tokens:
        raise RuntimeError("Asynchronous scheduling retirement must be a zero-token control frame")
    retire_id = getattr(scheduler, "_gr_retire_id", 0) + 1
    scheduler._gr_retire_id = retire_id
    owner.retirement = GRRetireMetadata(owner.session_id, owner.owner_generation, retire_id)
    owner.resource_status = GRResourceStatus.RETIRING
    setattr(output, GR_RETIRE_OUTPUT_ATTR, owner.retirement)
    output.finished_req_ids.add(owner.session_id)
    _progress(scheduler)


def consume_gr_retirement(scheduler: Any, output: Any, model_output: Any) -> GRSessionState | None:
    metadata = getattr(output, GR_RETIRE_OUTPUT_ATTR, None)
    if metadata is None:
        return None
    session = _sessions(scheduler).get(metadata.session_id)
    if session is None or session.retirement != metadata:
        return None  # A duplicate/stale receipt cannot release the current owner.
    result = getattr(model_output, GR_RETIRE_RESULT_ATTR, None)
    expected = GRRetireResult(
        metadata.session_id, metadata.owner_generation, metadata.retire_id, True
    )
    if (
        not isinstance(result, GRRetireResult)
        or type(result.owner_generation) is not int
        or type(result.retire_id) is not int
        or result.released is not True
        or result != expected
    ):
        # A missing/mismatched release proof is a Worker protocol failure.
        # Never continue service by guessing that the shared buffers are free.
        raise RuntimeError(
            "Asynchronous scheduling Worker did not prove release of the matching execution owner"
        )
    session.worker_resources_allocated = False
    session.resource_status = GRResourceStatus.CLOSED
    if not session.terminal_result_pending:
        _release_owner(scheduler, session)
    return session


def _install_gr_prompt_cache_guard(scheduler: Any) -> None:
    """Cap native async output caching as well as schedule-time caching."""
    manager = getattr(scheduler, "kv_cache_manager", None)
    original = getattr(manager, "cache_blocks", None)
    if (
        manager is None
        or not callable(original)
        or getattr(manager, "_gr_prompt_cache_guard", False)
    ):
        return

    def cache_prompt(request: Any, num_computed_tokens: int) -> Any:
        if getattr(request, "gr_batch_request", None) is not None:
            num_computed_tokens = min(num_computed_tokens, request.num_prompt_tokens)
        return original(request, num_computed_tokens)

    manager.cache_blocks = cache_prompt
    manager._gr_prompt_cache_guard = True


def _prepare_gr_session(scheduler: Any, request: BeamBatchRequest) -> GRSessionState:
    """Validate admission without mutating session, owner or capacity state."""
    if not scheduler.scheduler_config.async_scheduling:
        raise RuntimeError("beam_search requires vLLM async_scheduling")
    try:
        config = getattr(scheduler, "vllm_config", None)
        validate_gr_request(
            request, config=config, trie_count=getattr(scheduler, "_gr_trie_count", None)
        )
        if config is not None:
            validate_gr_execution_config(config)
            validate_gr_device_config(config, request)
    except ValueError as error:
        raise GRAdmissionError(str(error)) from error
    session_id = request.batch_id
    engine_requests = tuple(item.request for item in request.items)
    request_ids = tuple(engine_request.request_id for engine_request in engine_requests)
    sessions = getattr(scheduler, "_gr_v1_sessions", {})
    if session_id in sessions:
        raise GRAdmissionError(f"GR session already exists: {session_id}")
    request_sessions = _request_sessions(scheduler)
    duplicate_ids = tuple(
        request_id for request_id in request_ids if request_id in request_sessions
    )
    if duplicate_ids:
        raise GRAdmissionError(f"GR request already belongs to a session: {duplicate_ids[0]}")
    capacity = getattr(scheduler, "_gr_execution_capacity", None)
    if capacity is None:
        from vllm_gr.config import GRConfig

        width = GRConfig.from_vllm_config(config).beam_max_width if config is not None else None
        if config is not None and width is None:
            raise ValueError(
                "Asynchronous scheduling requires beam_max_width to be fixed at startup"
            )
        capacity = GRExecutionCapacity(width or request.beam_params.beam_width)
    try:
        capacity.validate(request.beam_params)
    except ValueError as error:
        raise GRAdmissionError(str(error)) from error
    manager = getattr(scheduler, "kv_cache_manager", None)
    if manager is not None:
        pool = manager.block_pool
        block_size = scheduler.cache_config.block_size
        required_blocks = 0
        for engine_request in engine_requests:
            extent = len(engine_request.prompt_token_ids) + request.beam_params.max_tokens - 1
            required_blocks += (extent + block_size - 1) // block_size
            if extent >= manager.max_model_len:
                raise GRAdmissionError(
                    "Asynchronous scheduling request exceeds the total native KV capacity"
                )
        if required_blocks > pool.num_gpu_blocks - 1:
            raise GRAdmissionError(
                "Asynchronous scheduling batch exceeds the total native KV capacity"
            )
    scheduler_config = scheduler.scheduler_config
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", len(engine_requests))
    if len(engine_requests) > max_num_seqs:
        raise GRAdmissionError("Beam batch size exceeds the native Scheduler max_num_seqs")
    total_prompt_tokens = sum(
        len(engine_request.prompt_token_ids) for engine_request in engine_requests
    )
    scheduling_token_budget = getattr(scheduler, "max_num_scheduled_tokens", None)
    if scheduling_token_budget is None:
        scheduling_token_budget = getattr(
            scheduler_config, "max_num_batched_tokens", total_prompt_tokens
        )
    if total_prompt_tokens > scheduling_token_budget:
        raise GRAdmissionError(
            "Beam batch Prefill exceeds the effective native scheduling token "
            "budget; chunked Prefill is not supported by this scheduling layer"
        )
    return GRSessionState(
        session_id=session_id,
        request=request,
        # The final Prefill invocation produces output token 0; the remaining
        # max_tokens - 1 invocations are Decode stages.
        max_decode_steps=max(0, request.beam_params.max_tokens - 1),
    )


def _register_gr_session(scheduler: Any, session: GRSessionState) -> None:
    """Publish validated state after the native ADD succeeds on the core thread."""
    _install_gr_prompt_cache_guard(scheduler)
    if getattr(scheduler, "_gr_execution_capacity", None) is None:
        scheduler._gr_execution_capacity = GRExecutionCapacity(
            session.request.beam_params.beam_width
        )
    _sessions(scheduler)[session.session_id] = session
    request_sessions = _request_sessions(scheduler)
    request_sessions.update((request_id, session.session_id) for request_id in session.request_ids)
    _admit_owner(scheduler)


def add_gr_session(scheduler: Any, request: BeamBatchRequest) -> GRSessionState:
    """Register one validated fixed-membership batch session."""
    session = _prepare_gr_session(scheduler, request)
    _register_gr_session(scheduler, session)
    return session


def can_issue_gr_stage(scheduler: Any, session: GRSessionState) -> bool:
    return (
        getattr(scheduler, "_gr_owner", None) == session.session_id
        and session.resource_status is GRResourceStatus.RUNNING
        and session.status in (GRSessionStatus.WAITING, GRSessionStatus.RUNNING)
        and all(request_id in scheduler.requests for request_id in session.request_ids)
        and len(session._in_flight_dispatches) < GR_LOOKAHEAD_DEPTH
    )


def attach_gr_stage_metadata(
    scheduler: Any, scheduler_output: Any, session: GRSessionState
) -> GRStageMetadata | None:
    request_ids = session.request_ids
    scheduled_ids = tuple(
        request_id
        for request_id in request_ids
        if request_id in scheduler_output.num_scheduled_tokens
    )
    if not scheduled_ids:
        return None
    if scheduled_ids != request_ids:
        raise RuntimeError(
            "native Scheduler split an atomic GR batch stage; all fixed members "
            "must be scheduled together: "
            f"session_id={session.session_id!r}, expected_ids={request_ids!r}, "
            f"scheduled_session_ids={scheduled_ids!r}, "
            f"scheduled_output_ids={tuple(scheduler_output.num_scheduled_tokens)!r}, "
            f"scheduled_tokens={dict(scheduler_output.num_scheduled_tokens)!r}"
        )
    native_requests = tuple(scheduler.requests.get(request_id) for request_id in request_ids)
    if any(native_request is None for native_request in native_requests):
        raise RuntimeError("native Scheduler lost a GR batch member")
    if not can_issue_gr_stage(scheduler, session):
        raise RuntimeError("native Scheduler selected a blocked Asynchronous scheduling dispatch")

    progress = tuple(
        {
            "request_id": request_id,
            "scheduled_tokens": scheduler_output.num_scheduled_tokens[request_id],
            "num_computed_tokens": native_request.num_computed_tokens,
            "num_prompt_tokens": native_request.num_prompt_tokens,
            "num_output_placeholders": getattr(native_request, "num_output_placeholders", None),
            "is_prefill_chunk": bool(native_request.is_prefill_chunk),
        }
        for request_id, native_request in zip(request_ids, native_requests, strict=True)
    )
    prefill_states = tuple(item["is_prefill_chunk"] for item in progress)
    if any(prefill_states):
        raise RuntimeError(
            "beam_search cannot schedule partial Prefill or recompute: "
            f"session_id={session.session_id!r}, progress={progress!r}"
        )
    dispatch_id = session._next_dispatch_id
    # Native _update_after_schedule already advanced this extent, independently
    # of host consumption of preceding dispatches.
    stage_indices = tuple(
        int(native_request.num_computed_tokens - native_request.num_prompt_tokens)
        for native_request in native_requests
    )
    if len(set(stage_indices)) != 1:
        raise RuntimeError(
            "GR batch members crossed the Prefill/Decode stage boundary: "
            f"session_id={session.session_id!r}, dispatch_id={dispatch_id}, "
            f"scheduled_output_ids={tuple(scheduler_output.num_scheduled_tokens)!r}, "
            f"stage_indices={stage_indices!r}, progress={progress!r}"
        )
    stage_index = stage_indices[0]
    if not 0 <= stage_index < session.total_stages:
        raise RuntimeError(
            "native GR stage exceeds the effective generation budget: "
            f"session_id={session.session_id!r}, dispatch_id={dispatch_id}, "
            f"stage_index={stage_index}, total_stages={session.total_stages}, "
            f"progress={progress!r}"
        )
    previous = session.latest_device_state
    expected_previous = (
        None if stage_index == 0 else GRDeviceStateRef(session.session_id, stage_index - 1)
    )
    if previous != expected_previous:
        raise RuntimeError(
            "GR rejects recompute or nonconsecutive native progress: "
            f"session_id={session.session_id!r}, dispatch_id={dispatch_id}, "
            f"stage_index={stage_index}, previous={previous!r}, "
            f"expected_previous={expected_previous!r}, "
            f"scheduled_output_ids={tuple(scheduler_output.num_scheduled_tokens)!r}, "
            f"session_request_ids={request_ids!r}, "
            f"in_flight_dispatches={tuple(sorted(session._in_flight_dispatches))!r}, "
            f"issued_stages={session.issued_stages}, "
            f"completed_stages={session.completed_stages}, progress={progress!r}"
        )
    kind = GRStageKind.PREFILL if stage_index == 0 else GRStageKind.DECODE
    output_state = GRDeviceStateRef(session.session_id, stage_index)
    prompt_tokens = tuple(int(request.num_prompt_tokens) for request in native_requests)
    manager = getattr(scheduler, "kv_cache_manager", None)
    kv_blocks: tuple[tuple[int, ...], ...] | None = None
    if manager is not None:
        # Scheduler owns the native block refcounts. Python references in a
        # Worker cannot prevent block reuse after abort or preemption.
        touched: list[Any] = []
        lease_groups: list[tuple[Any, ...]] = []
        block_rows: list[tuple[int, ...]] = []
        for request_id in request_ids:
            kv = manager.get_blocks(request_id)
            for group in kv.blocks:
                touched.extend(group)
                lease_groups.append(tuple(reversed(group)))
            # Only one uniform KV group can be represented as one row per
            # request. Keep the lease for every group even when publication is
            # unavailable; the Worker rejects unsupported multi-group GR.
            if len(kv.blocks) == 1:
                block_ids = getattr(kv, "get_block_ids", None)
                if callable(block_ids):
                    groups = block_ids()
                    if len(groups) == 1:
                        block_rows.append(tuple(groups[0]))
                else:
                    block_rows.append(
                        tuple(
                            block if isinstance(block, int) else getattr(block, "block_id", block)
                            for block in kv.blocks[0]
                        )
                    )
        if len(block_rows) == len(request_ids):
            kv_blocks = tuple(block_rows)
        if touched:
            manager.block_pool.touch(touched)
            session._kv_dispatch_leases[dispatch_id] = tuple(lease_groups)
    metadata = GRStageMetadata(
        session_id=session.session_id,
        dispatch_id=dispatch_id,
        stage_index=stage_index,
        kind=kind,
        decode_step=None if kind == GRStageKind.PREFILL else stage_index - 1,
        is_last_stage=stage_index == session.total_stages - 1,
        input_device_state=session.latest_device_state,
        output_device_state=output_state,
        beam_params=session.request.beam_params,
        output_options=session.request.output_options,
        num_prompt_tokens=sum(prompt_tokens),
        produces_output=True,
        session_init=session.request if stage_index == 0 else None,
        owner_generation=session.owner_generation,
        execution_capacity=scheduler._gr_execution_capacity,
        kv_blocks=kv_blocks,
        request_ids=request_ids,
        num_prompt_tokens_by_request=prompt_tokens,
    )
    session._dispatch_records[dispatch_id] = metadata
    from vllm_gr.v1.metrics.telemetry import stage_started

    stage_started(session, metadata, scheduler_output)
    session._next_dispatch_id += 1
    session._in_flight_dispatches.add(dispatch_id)
    session.worker_resources_allocated = True
    _progress(scheduler)
    session.status = GRSessionStatus.RUNNING
    return metadata


def _stage_completion(
    session: GRSessionState,
    stage_index: int,
    model_output: Any,
    *,
    finished: bool = False,
    device_state: GRDeviceStateRef | None = None,
    error: str | None = None,
) -> GRStageCompletion:
    return GRStageCompletion(
        session_id=session.session_id,
        stage_index=stage_index,
        finished=finished,
        device_state=device_state,
        model_output=model_output,
        error=error,
    )


def _result_mapping(model_output: Any, attribute: str) -> Mapping[Any, Any] | None:
    value = getattr(model_output, attribute, None)
    if value is not None and not isinstance(value, Mapping):
        raise _GRWorkerProtocolError(f"Worker returned an invalid {attribute} mapping")
    return value


def _read_worker_payload(
    session: GRSessionState,
    metadata: GRStageMetadata,
    model_output: Any,
) -> tuple[GRWorkerResult, BeamBatchResult | None, int, bool]:
    """Require a Worker receipt with matching identity for every active dispatch."""
    worker_results = _result_mapping(model_output, GR_WORKER_RESULT_ATTR)
    worker_result = worker_results.get(session.session_id) if worker_results is not None else None
    if worker_result is None:
        raise _GRWorkerProtocolError("Worker omitted GRWorkerResult for a dispatched GR stage")
    if not isinstance(worker_result, GRWorkerResult):
        raise _GRWorkerProtocolError("Worker returned an invalid GRWorkerResult")
    if (
        worker_result.session_id != session.session_id
        or type(worker_result.dispatch_id) is not int
        or worker_result.dispatch_id != metadata.dispatch_id
        or type(worker_result.stage_index) is not int
        or worker_result.stage_index != metadata.stage_index
        or type(worker_result.owner_generation) is not int
        or worker_result.owner_generation != metadata.owner_generation
    ):
        raise _GRWorkerProtocolError("Worker returned a GR result for a different dispatch")
    if type(worker_result.finished) is not bool:
        raise _GRWorkerProtocolError("Worker returned an invalid finished flag")
    if type(worker_result.resources_released) is not bool:
        raise _GRWorkerProtocolError("Worker returned an invalid resource-release proof")

    terminal_result = worker_result.terminal_result
    if terminal_result is None:
        terminal_results = _result_mapping(model_output, GR_BATCH_RESULT_ATTR)
        if terminal_results is not None:
            terminal_result = terminal_results.get(session.session_id)
    if terminal_result is not None:
        if not isinstance(terminal_result, BeamBatchResult):
            raise _GRWorkerProtocolError("Worker returned an invalid BeamBatchResult")
        if terminal_result.batch_id != session.session_id:
            raise _GRWorkerProtocolError("Worker returned a terminal result for another session")

    produced = worker_result.produced_token_count
    worker_finished = worker_result.finished
    if worker_result.resources_released and not worker_finished:
        raise _GRWorkerProtocolError("Worker released GR resources before a terminal receipt")
    if not isinstance(produced, int) or isinstance(produced, bool) or produced not in (0, 1):
        raise _GRWorkerProtocolError("Worker returned an invalid produced_token_count")
    if (
        metadata.produces_output
        and produced == 0
        and not worker_finished
        and terminal_result is None
    ):
        raise _GRWorkerProtocolError("Worker returned no token for a non-terminal logical stage")
    if not metadata.produces_output and produced != 0:
        raise _GRWorkerProtocolError("Worker produced a token for a non-output Prefill chunk")
    if produced == 1 and worker_result.device_state != metadata.output_device_state:
        raise _GRWorkerProtocolError("Worker returned a device state for a different logical stage")
    return worker_result, terminal_result, produced, worker_finished


def update_gr_from_output(
    scheduler: Any,
    session: GRSessionState,
    scheduler_output: Any,
    model_output: Any,
) -> GRStageCompletion:
    metadata = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {}).get(session.session_id)
    if metadata is None:
        return _stage_completion(session, session.completed_stages, model_output)

    if (
        metadata.owner_generation != session.owner_generation
        or metadata.dispatch_id not in session._in_flight_dispatches
    ):
        return _stage_completion(session, metadata.stage_index, model_output)
    # The Future has completed, so this submitted work no longer holds native
    # KV blocks, even when its receipt is malformed. This is not proof that
    # the Worker's persistent resources have been released.
    session._in_flight_dispatches.remove(metadata.dispatch_id)
    _progress(scheduler)
    lease_groups = session._kv_dispatch_leases.pop(metadata.dispatch_id, ())
    for blocks in lease_groups:
        # Match native per-group free calls: uncached blocks are prepended,
        # cached blocks appended. Flattening groups changes eviction order.
        scheduler.kv_cache_manager.block_pool.free_blocks(blocks)
    try:
        if session._dispatch_records.get(metadata.dispatch_id) != metadata:
            raise _GRWorkerProtocolError("Scheduler dispatch identity changed")
        worker_result, terminal_result, _, worker_finished = _read_worker_payload(
            session, metadata, model_output
        )
    except _GRWorkerProtocolError as error:
        return _fail_gr_dispatch(scheduler, session, metadata, model_output, str(error))

    if worker_result.resources_released:
        session.worker_resources_allocated = False
    if session.status in _DRAINING_STATUSES:
        # Cancellation and the first terminal result win over queued successors.
        _drain_resources(scheduler, session)
        return _stage_completion(session, metadata.stage_index, model_output)

    finished = worker_finished or terminal_result is not None
    if finished or metadata.is_last_stage:
        _begin_terminal_drain(session, terminal_result)
    return _stage_completion(
        session,
        metadata.stage_index,
        model_output,
        finished=finished,
        device_state=worker_result.device_state,
    )


def _begin_terminal_drain(session: GRSessionState, result: BeamBatchResult | None) -> None:
    session.status = GRSessionStatus.FINISHING
    session.terminal_result_pending = True
    session._pending_terminal_result = result
    session.resource_status = GRResourceStatus.DRAINING


def finish_after_native_termination(session: GRSessionState, member_id: str) -> None:
    """Drain outstanding work, preserving any terminal result already buffered."""
    result = session._pending_terminal_result
    if result is None:
        result = _make_failure_result(
            session,
            "native_termination_without_gr_result",
            f"Native batch member {member_id} terminated before GR produced a terminal result",
        )
    _begin_terminal_drain(session, result)


def _make_failure_result(session: GRSessionState, code: str, message: str) -> BeamBatchResult:
    return BeamBatchResult(
        protocol_version=session.request.protocol_version,
        batch_id=session.session_id,
        output=None,
        failure=BeamBatchFailure(code=code, message=message, item_index=None),
    )


def _fail_gr_dispatch(
    scheduler: Any,
    session: GRSessionState,
    metadata: GRStageMetadata,
    model_output: Any,
    message: str,
) -> GRStageCompletion:
    """Turn a malformed Worker response into one request-scoped failure."""
    session.resource_status = GRResourceStatus.DRAINING
    if session.status not in (GRSessionStatus.ABORTED, GRSessionStatus.FAILED):
        session.status = GRSessionStatus.FINISHING
        session.terminal_result_pending = True
        # A protocol error must not leave an earlier success pending.
        if (
            session._pending_terminal_result is None
            or session._pending_terminal_result.failure is None
        ):
            session._pending_terminal_result = _make_failure_result(
                session, "worker_protocol_error", message
            )
    _drain_resources(scheduler, session)
    return _stage_completion(
        session,
        metadata.stage_index,
        model_output,
        finished=True,
        error=message,
    )


def _gr_session_ready_to_finalize(session: GRSessionState) -> bool:
    return (
        session.status == GRSessionStatus.FINISHING
        and session.terminal_result_pending
        and not session._terminal_emitted
        and not session._in_flight_dispatches
        and not session.worker_resources_allocated
    )


def _gr_terminal_result(session: GRSessionState) -> BeamBatchResult:
    result = session._pending_terminal_result
    if result is not None:
        return result
    return _make_failure_result(
        session,
        "worker_result_unavailable",
        "Worker did not return a terminal BeamBatchResult",
    )


def _close_gr_session(scheduler: Any, session: GRSessionState) -> None:
    if session._terminal_emitted:
        return
    session._terminal_emitted = True
    session.terminal_result_pending = False
    session.status = GRSessionStatus.FINISHED
    _release_owner(scheduler, session)


def finish_gr_session(
    scheduler: Any, session: GRSessionState, status: GRSessionStatus
) -> BeamBatchResult | None:
    session.status = status
    session.terminal_result_pending = False
    _drain_resources(scheduler, session)
    if status == GRSessionStatus.FAILED:
        return _make_failure_result(session, "engine_failure", "GR session failed in EngineCore")
    return None


def abort_gr_session(scheduler: Any, session_id: str) -> None:
    from vllm.v1.request import RequestStatus

    session = _sessions(scheduler).get(session_id)
    if session is not None:
        session.status = GRSessionStatus.ABORTED
        session.terminal_result_pending = False
    request_ids = session.request_ids if session is not None else (session_id,)
    scheduler.finish_requests(list(request_ids), RequestStatus.FINISHED_ABORTED)
    if session is not None and not session._in_flight_dispatches:
        _drain_resources(scheduler, session)


def release_gr_session(scheduler: Any, session_id: str) -> None:
    session = _sessions(scheduler).get(session_id)
    if session is None:
        return
    if session._in_flight_dispatches:
        raise RuntimeError("cannot release a GR session with in-flight stages")
    if session.worker_resources_allocated:
        raise RuntimeError(
            "Asynchronous scheduling release requires a matching Worker retirement proof"
        )
    _release_owner(scheduler, session)


def install_gr_scheduler_methods(scheduler_cls: type[Any]) -> None:
    scheduler_cls.add_gr_session = add_gr_session
    scheduler_cls.can_issue_gr_stage = can_issue_gr_stage
    scheduler_cls.attach_gr_stage_metadata = attach_gr_stage_metadata
    scheduler_cls.update_gr_from_output = update_gr_from_output
    scheduler_cls.finish_gr_session = finish_gr_session
    scheduler_cls.abort_gr_session = abort_gr_session
    scheduler_cls.release_gr_session = release_gr_session
