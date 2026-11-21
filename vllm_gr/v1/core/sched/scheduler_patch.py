# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Install the GR session lifecycle on the native vLLM scheduler."""

from __future__ import annotations

from functools import wraps

from vllm.logger import init_logger

logger = init_logger(__name__)
_SCHEDULER_PATCH_MARKER = "_vllm_gr_beam_scheduler_patch"
_SCHEDULER_UPDATE_PATCH_MARKER = "_vllm_gr_update_from_output_patch"
_GR_FINALIZING_IDS_ATTR = "_gr_v1_finalizing_request_ids"
_GR_NATIVE_UPDATE_IDS_ATTR = "_gr_v1_native_update_request_ids"
_GR_RESET_PREFIX_CACHE_PATCH_MARKER = "_vllm_gr_v1_reset_prefix_cache"
_GR_SCHEDULE_PATCH_DEPTH_ATTR = "_vllm_gr_schedule_patch_depth"


def _expand_gr_abort_request_ids(scheduler, request_ids, finished_status):
    """Expand cancellation of any GR member to its complete fixed batch."""
    if request_ids is None or getattr(finished_status, "name", "") != "FINISHED_ABORTED":
        return request_ids
    if request_ids is None:
        return None
    requested = (request_ids,) if isinstance(request_ids, str) else tuple(request_ids)
    sessions = getattr(scheduler, "_gr_v1_sessions", {})
    request_sessions = getattr(scheduler, "_gr_v1_request_sessions", {})
    expanded: list[str] = []
    seen: set[str] = set()
    for request_id in requested:
        session_id = request_sessions.get(request_id, request_id)
        session = sessions.get(session_id)
        members = session.request_ids if session is not None else (request_id,)
        for member_id in members:
            if member_id in seen:
                continue
            seen.add(member_id)
            expanded.append(member_id)
    return expanded


def _sync_gr_sessions_after_native_finish(scheduler, finished_requests, finished_status) -> None:
    """Mirror the native finished Request objects into GR session state."""
    from vllm_gr.v1.core.sched.session import GRSessionStatus

    status_name = getattr(finished_status, "name", "")
    sessions = getattr(scheduler, "_gr_v1_sessions", {})
    request_sessions = getattr(scheduler, "_gr_v1_request_sessions", {})
    gr_finalizing_ids = getattr(scheduler, _GR_FINALIZING_IDS_ATTR, ())
    gr_native_update_ids = getattr(scheduler, _GR_NATIVE_UPDATE_IDS_ATTR, ())
    for request in finished_requests:
        req_id = request.request_id
        session = sessions.get(request_sessions.get(req_id, req_id))
        if session is None:
            continue
        # Native EOS/stop detection may finish a Request while processing a GR
        # dispatch. The update wrapper reconciles that transition after GR has
        # consumed the same output; it is not an external abort/error.
        if req_id in gr_native_update_ids:
            continue
        gr_terminal_delivery = (
            req_id in gr_finalizing_ids
            and session.status == GRSessionStatus.FINISHING
            and session.terminal_result_pending
        )
        if gr_terminal_delivery:
            continue
        if status_name == "FINISHED_ABORTED":
            session.status = GRSessionStatus.ABORTED
        elif status_name == "FINISHED_ERROR":
            session.status = GRSessionStatus.FAILED
        else:
            session.status = GRSessionStatus.FINISHING
        session.terminal_result_pending = False

        from vllm_gr.v1.core.sched.gr_async_scheduler import _drain_resources

        _drain_resources(scheduler, session)


def _patch_scheduler_reset_prefix_cache(scheduler_cls) -> None:
    original_reset = scheduler_cls.reset_prefix_cache
    if getattr(original_reset, _GR_RESET_PREFIX_CACHE_PATCH_MARKER, False):
        return

    @wraps(original_reset)
    def patched_reset_prefix_cache(self, *args, **kwargs):
        reset_running_requests = kwargs.get("reset_running_requests", args[0] if args else False)
        from vllm_gr.v1.core.sched.session import GRResourceStatus, GRSessionStatus

        # Native forced reset cannot safely preempt active GR resources. Weight
        # updates may leave unstarted sessions queued after pause/drain, so allow
        # cache clearing for those while still rejecting executing/retiring work.
        sessions = getattr(self, "_gr_v1_sessions", {}).values()
        if reset_running_requests and any(
            session.status is not GRSessionStatus.WAITING
            or session.resource_status is not GRResourceStatus.WAITING
            or session.in_flight_stages
            for session in sessions
        ):
            logger.warning(
                "Rejecting reset_prefix_cache(reset_running_requests=True) "
                "while GR sessions are active"
            )
            return False
        return original_reset(self, *args, **kwargs)

    setattr(patched_reset_prefix_cache, _GR_RESET_PREFIX_CACHE_PATCH_MARKER, True)
    scheduler_cls.reset_prefix_cache = patched_reset_prefix_cache


def apply_scheduler_patch():
    """Install the native scheduler hooks for Beam dispatch and completion."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_gr.v1.beam.batch_contracts import GRAdmissionError
    from vllm_gr.v1.beam.dispatch import GR_STAGE_OUTPUT_ATTR
    from vllm_gr.v1.core.sched.gr_async_scheduler import (
        _prepare_gr_session,
        _register_gr_session,
        has_gr_cleanup_work,
        install_gr_scheduler_methods,
    )

    install_gr_scheduler_methods(Scheduler)

    if not getattr(Scheduler.has_requests, "_gr_retirement_work", False):
        original_has_requests = Scheduler.has_requests

        @wraps(original_has_requests)
        def has_requests(self):
            return original_has_requests(self) or has_gr_cleanup_work(self)

        has_requests._gr_retirement_work = True
        Scheduler.has_requests = has_requests

    if not getattr(Scheduler.add_request, "_vllm_gr_v1_add_request", False):
        original_add_request = Scheduler.add_request

        @wraps(original_add_request)
        def patched_add_request(self, request):
            batch_request = getattr(request, "gr_batch_request", None)
            sessions = getattr(self, "_gr_v1_sessions", {})
            if (batch_request is None and sessions) or (
                batch_request is not None
                and any(
                    getattr(r, "gr_batch_request", None) is None for r in self.requests.values()
                )
            ):
                raise GRAdmissionError(
                    "Asynchronous scheduling requests cannot mix with ordinary or legacy Beam requests"
                )
            if batch_request is not None:
                member_index = getattr(request, "gr_batch_member_index", None)
                matching_indices = tuple(
                    index
                    for index, item in enumerate(batch_request.items)
                    if item.request.request_id == request.request_id
                )
                if len(matching_indices) != 1 or (
                    member_index is not None
                    and (type(member_index) is not int or member_index != matching_indices[0])
                ):
                    raise GRAdmissionError("native request does not match its GR batch member")
                if request.request_id in self.requests:
                    raise GRAdmissionError(f"GR request already exists: {request.request_id}")
                session = sessions.get(batch_request.batch_id)
                if session is not None:
                    if session.request is not batch_request:
                        raise GRAdmissionError(
                            f"GR session already exists: {batch_request.batch_id}"
                        )
                    return original_add_request(self, request)
                # The caller supplied the complete immutable batch. Validate
                # it once before publishing the first native member; the core
                # handles every member synchronously before it can schedule.
                duplicate_id = next(
                    (
                        item.request.request_id
                        for item in batch_request.items
                        if item.request.request_id in self.requests
                    ),
                    None,
                )
                if duplicate_id is not None:
                    raise GRAdmissionError(f"GR request already exists: {duplicate_id}")
                session = _prepare_gr_session(self, batch_request)
                result = original_add_request(self, request)
                _register_gr_session(self, session)
                return result
            return original_add_request(self, request)

        setattr(patched_add_request, "_vllm_gr_v1_add_request", True)
        Scheduler.add_request = patched_add_request

    if not getattr(Scheduler.finish_requests, "_vllm_gr_v1_finish_requests", False):
        original_finish_requests = Scheduler.finish_requests

        @wraps(original_finish_requests)
        def patched_finish_requests(self, request_ids, finished_status):
            request_ids = _expand_gr_abort_request_ids(self, request_ids, finished_status)
            result = original_finish_requests(self, request_ids, finished_status)
            _sync_gr_sessions_after_native_finish(self, result, finished_status)
            return result

        setattr(patched_finish_requests, "_vllm_gr_v1_finish_requests", True)
        Scheduler.finish_requests = patched_finish_requests

    _patch_scheduler_reset_prefix_cache(Scheduler)
    _patch_scheduler_update_from_output(Scheduler)

    if getattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self, throttle_prefills: bool = False):
        # Some platform plugins replace Scheduler with a subclass whose
        # schedule() delegates to super().schedule(). If both implementations
        # were patched, that produces nested GR wrappers around one native
        # scheduling pass. Only the outermost wrapper may apply admission and
        # annotate the SchedulerOutput; the inner wrapper must preserve the
        # subclass' normal super() chain without processing the output twice.
        schedule_patch_depth = getattr(self, _GR_SCHEDULE_PATCH_DEPTH_ATTR, 0)
        if schedule_patch_depth:
            return _original_schedule(self, throttle_prefills=throttle_prefills)

        # Cache only real prompt KV for V1: generated tokens are device-owned
        # placeholders. Ordinary requests retain native cache semantics.
        original_cache_blocks = self.kv_cache_manager.coordinator.cache_blocks
        cached_patch = getattr(self, "_gr_cache_blocks_patch", None)
        if cached_patch is None or cached_patch[0] is not original_cache_blocks:

            def cache_prompt_blocks(request, num_computed_tokens):
                if getattr(request, "gr_batch_request", None) is not None:
                    num_computed_tokens = min(num_computed_tokens, request.num_prompt_tokens)
                return original_cache_blocks(request, num_computed_tokens)

            cached_patch = (original_cache_blocks, cache_prompt_blocks)
            self._gr_cache_blocks_patch = cached_patch
        self.kv_cache_manager.coordinator.cache_blocks = cached_patch[1]
        from vllm_gr.v1.core.sched.gr_async_scheduler import (
            attach_gr_retirement,
            gr_admission,
        )

        setattr(self, _GR_SCHEDULE_PATCH_DEPTH_ATTR, schedule_patch_depth + 1)
        try:
            with gr_admission(self):
                output = _original_schedule(self, throttle_prefills=throttle_prefills)

            gr_stages = {}
            sessions = getattr(self, "_gr_v1_sessions", {})
            request_sessions = getattr(self, "_gr_v1_request_sessions", {})
            scheduled_session_ids = dict.fromkeys(
                request_sessions[req_id]
                for req_id in output.num_scheduled_tokens
                if req_id in request_sessions
            )
            for session_id in scheduled_session_ids:
                session = sessions.get(session_id)
                if session is None:
                    continue
                metadata = self.attach_gr_stage_metadata(output, session)
                if metadata is not None:
                    gr_stages[session_id] = metadata
            setattr(output, GR_STAGE_OUTPUT_ATTR, gr_stages)
            attach_gr_retirement(self, output)
            return output
        finally:
            try:
                self.kv_cache_manager.coordinator.cache_blocks = original_cache_blocks
            finally:
                if schedule_patch_depth:
                    setattr(self, _GR_SCHEDULE_PATCH_DEPTH_ATTR, schedule_patch_depth)
                else:
                    delattr(self, _GR_SCHEDULE_PATCH_DEPTH_ATTR)

    Scheduler.schedule = patched_schedule
    setattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, True)
    Scheduler._patched_for_beam_requests = True
    logger.debug("Scheduler.schedule patched for beam-requests propagation.")


def _find_gr_output(result, request_id):
    for engine_outputs in result.values():
        for engine_output in engine_outputs.outputs:
            if engine_output.request_id == request_id:
                return engine_outputs, engine_output
    return None, None


def _native_request_terminated(scheduler, result, request_id) -> bool:
    _, target_output = _find_gr_output(result, request_id)
    if target_output is not None and target_output.finish_reason is not None:
        return True
    for engine_outputs in result.values():
        if request_id in (engine_outputs.finished_requests or ()):
            return True
    return request_id not in scheduler.requests


def _reconcile_native_gr_termination(scheduler, result, session) -> None:
    from vllm_gr.v1.core.sched.gr_async_scheduler import finish_after_native_termination
    from vllm_gr.v1.core.sched.session import GRSessionStatus

    if session.status not in (GRSessionStatus.WAITING, GRSessionStatus.RUNNING):
        return
    terminated_member = next(
        (
            request_id
            for request_id in session.request_ids
            if _native_request_terminated(scheduler, result, request_id)
        ),
        None,
    )
    if terminated_member is not None:
        finish_after_native_termination(session, terminated_member)


def _suppress_native_terminal(result, request_id, target_output) -> None:
    if target_output is not None:
        target_output.finish_reason = None
    for engine_outputs in result.values():
        if engine_outputs.finished_requests is not None:
            engine_outputs.finished_requests.discard(request_id)


def _suppress_batch_member_outputs(result, session) -> None:
    """Keep internal batch members off the single batch-result channel."""
    internal_ids = frozenset(session.request_ids[1:])
    if not internal_ids:
        return
    for engine_outputs in result.values():
        engine_outputs.outputs[:] = [
            output for output in engine_outputs.outputs if output.request_id not in internal_ids
        ]
        if engine_outputs.finished_requests is not None:
            engine_outputs.finished_requests.difference_update(internal_ids)


def _finish_native_gr_request(scheduler, request, request_id, request_status) -> None:
    if request is None or request.is_finished():
        return

    # Every physical dispatch has drained. Any remaining placeholders are
    # speculative reservations and no longer represent real outputs.
    request.num_output_placeholders = 0
    finalizing_ids = getattr(scheduler, _GR_FINALIZING_IDS_ATTR, None)
    if finalizing_ids is None:
        finalizing_ids = set()
        setattr(scheduler, _GR_FINALIZING_IDS_ATTR, finalizing_ids)
    finalizing_ids.add(request_id)
    try:
        scheduler.finish_requests(request_id, request_status)
    finally:
        finalizing_ids.discard(request_id)
        if not finalizing_ids:
            delattr(scheduler, _GR_FINALIZING_IDS_ATTR)


def _discard_pending_native_finish(
    scheduler, client_index, request_id, *, worker_released: bool = False
) -> None:
    finished_by_client = getattr(scheduler, "finished_req_ids_dict", None)
    if finished_by_client is not None:
        pending_finished = finished_by_client.get(client_index)
        if pending_finished is not None:
            pending_finished.discard(request_id)
            if not pending_finished:
                finished_by_client.pop(client_index, None)
    if not worker_released:
        return
    # A fused release retires the Worker's cached request in the consuming
    # frame, so no retirement control frame will carry this ID to the Worker.
    # Leaving it pending would keep has_requests() true forever: while
    # PAUSED_ALL, schedule() early-returns before it can consume the set, so
    # the engine would never look idle and the pause Future would never
    # complete.
    pending_finished_ids = getattr(scheduler, "finished_req_ids", None)
    if pending_finished_ids is not None:
        pending_finished_ids.discard(request_id)


def _finalize_ready_gr_session(scheduler, result, session) -> None:
    """Publish one terminal native output after speculative work drains."""
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
    from vllm.v1.request import RequestStatus

    from vllm_gr.v1.core.sched.gr_async_scheduler import (
        _close_gr_session,
        _gr_session_ready_to_finalize,
        _gr_terminal_result,
    )
    from vllm_gr.v1.core.sched.session import GRSessionStatus

    request_id = session.session_id
    target_outputs, target_output = _find_gr_output(result, request_id)

    if not _gr_session_ready_to_finalize(session):
        # Native stop detection can race ahead of a queued GR successor. Keep
        # the request open at the frontend until the GR session has drained;
        # the finalizer will synthesize the one authoritative terminal frame.
        if session.status == GRSessionStatus.FINISHING and session.terminal_result_pending:
            for member_id in session.request_ids:
                _, member_output = _find_gr_output(result, member_id)
                _suppress_native_terminal(result, member_id, member_output)
        return

    terminal_result = _gr_terminal_result(session)
    native_requests = {
        member_id: scheduler.requests.get(member_id) for member_id in session.request_ids
    }
    native_request = native_requests[request_id]
    client_index = (
        native_request.client_index
        if native_request is not None
        else session.request.items[0].request.client_index
    )

    request_status = (
        RequestStatus.FINISHED_ERROR
        if terminal_result.failure is not None
        else RequestStatus.FINISHED_LENGTH_CAPPED
    )
    finish_reason = (
        FinishReason.ERROR if terminal_result.failure is not None else FinishReason.LENGTH
    )

    for member_id, member_request in native_requests.items():
        _finish_native_gr_request(scheduler, member_request, member_id, request_status)

    if target_outputs is None:
        target_outputs = result.get(client_index)
        if target_outputs is None:
            target_outputs = EngineCoreOutputs(outputs=[])
            result[client_index] = target_outputs
    if target_output is None:
        target_output = EngineCoreOutput(
            request_id=request_id,
            new_token_ids=[],
            finish_reason=finish_reason,
        )
        target_outputs.outputs.append(target_output)
    else:
        # The GR result, rather than the native placeholder token, owns the
        # externally visible terminal status.
        target_output.finish_reason = finish_reason

    kv_params = dict(target_output.kv_transfer_params or {})
    kv_params["gr_batch_result"] = terminal_result
    target_output.kv_transfer_params = kv_params
    from vllm_gr.v1.metrics.telemetry import terminal_metrics

    _attach_gr_metrics(scheduler, result, session, terminal_metrics(session, terminal_result))
    finished_requests = set(target_outputs.finished_requests or ())
    finished_requests.add(request_id)
    target_outputs.finished_requests = finished_requests

    # finish_requests() ran after upstream assembled this step's outputs. Avoid
    # publishing the same finished ID again on the next update. A Worker that
    # already proved the release has no cached request left, so the pending
    # native finish must not survive either.
    for member_id in session.request_ids:
        _discard_pending_native_finish(
            scheduler,
            client_index,
            member_id,
            worker_released=not session.worker_resources_allocated,
        )

    _close_gr_session(scheduler, session)


def _attach_gr_metrics(scheduler, result, session, value) -> None:
    """Reuse native IPC; telemetry-only frames contain no generated token."""
    if not value:
        return
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
    from vllm_gr.v1.metrics.telemetry import TELEMETRY_KEY

    _, target = _find_gr_output(result, session.session_id)
    if target is None:
        client = session.request.items[0].request.client_index
        outputs = result.setdefault(client, EngineCoreOutputs(outputs=[]))
        target = EngineCoreOutput(request_id=session.session_id, new_token_ids=[])
        outputs.outputs.append(target)
    params = dict(target.kv_transfer_params or {})
    params[TELEMETRY_KEY] = [*params.get(TELEMETRY_KEY, ()), value]
    model_config = getattr(getattr(scheduler, "vllm_config", None), "model_config", None)
    name = getattr(model_config, "served_model_name", None) or "gr"
    params["gr_metrics_model_name"] = name[0] if isinstance(name, (list, tuple)) else str(name)
    target.kv_transfer_params = params


def _patch_scheduler_update_from_output(scheduler_cls) -> None:
    """Consume V1 stage/retirement receipts and publish the terminal batch result."""
    if getattr(scheduler_cls.update_from_output, _SCHEDULER_UPDATE_PATCH_MARKER, False):
        return

    _original_update = scheduler_cls.update_from_output

    @wraps(_original_update)
    def patched_update_from_output(self, scheduler_output, model_runner_output, *args, **kwargs):
        from vllm_gr.v1.beam.dispatch import GR_STAGE_OUTPUT_ATTR
        from vllm_gr.v1.core.sched.gr_async_scheduler import consume_gr_retirement

        retired_session = consume_gr_retirement(self, scheduler_output, model_runner_output)
        gr_stages = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {})
        sessions = getattr(self, "_gr_v1_sessions", {})
        dispatched_sessions = {
            sid: sessions[sid]
            for sid, metadata in gr_stages.items()
            if sid in sessions
            and metadata.owner_generation == sessions[sid].owner_generation
            and metadata.dispatch_id in sessions[sid]._in_flight_dispatches
        }
        # GR dispatches never mix with ordinary requests. Stale results must
        # not reach native token accounting, even if a new owner reuses the ID.
        if gr_stages and not dispatched_sessions:
            return {}
        completions = [
            self.update_gr_from_output(session, scheduler_output, model_runner_output)
            for session in dispatched_sessions.values()
        ]
        native_update_ids = getattr(self, _GR_NATIVE_UPDATE_IDS_ATTR, None)
        if native_update_ids is None:
            native_update_ids = set()
            setattr(self, _GR_NATIVE_UPDATE_IDS_ATTR, native_update_ids)
        dispatched_request_ids = {
            request_id
            for session in dispatched_sessions.values()
            for request_id in session.request_ids
        }
        added_update_ids = dispatched_request_ids.difference(native_update_ids)
        native_update_ids.update(added_update_ids)
        try:
            # A malformed active receipt is a request failure, not a native
            # sampled token. The normal finalizer clears remaining placeholders
            # only after drain and matching Worker release confirmation.
            result = (
                {}
                if any(completion.error for completion in completions)
                else _original_update(self, scheduler_output, model_runner_output, *args, **kwargs)
            )
        finally:
            native_update_ids.difference_update(added_update_ids)
            if not native_update_ids:
                delattr(self, _GR_NATIVE_UPDATE_IDS_ATTR)

        for session in dispatched_sessions.values():
            _suppress_batch_member_outputs(result, session)

        updated_sessions = []
        for session in dispatched_sessions.values():
            _reconcile_native_gr_termination(self, result, session)
            updated_sessions.append(session)

        if retired_session is not None and retired_session.terminal_result_pending:
            updated_sessions.append(retired_session)
        for outputs in result.values():
            for output in outputs.outputs:
                if output.request_id in dispatched_sessions:
                    output.new_token_ids = []

        from vllm_gr.v1.beam.dispatch import GR_WORKER_RESULT_ATTR
        from vllm_gr.v1.metrics.telemetry import stage_finished

        worker_results = getattr(model_runner_output, GR_WORKER_RESULT_ATTR, {}) or {}
        if not isinstance(worker_results, dict):
            worker_results = {}
        for session, completion in zip(dispatched_sessions.values(), completions, strict=True):
            metadata = gr_stages[session.session_id]
            value = stage_finished(
                session,
                metadata,
                worker_results.get(session.session_id),
                error=bool(completion.error),
            )
            _attach_gr_metrics(self, result, session, value)

        # Account for the stage before finalization, including Prefill-only
        # sessions. Publish terminal metrics only after physical dispatches drain.
        for session in updated_sessions:
            _finalize_ready_gr_session(self, result, session)

        return result

    setattr(patched_update_from_output, _SCHEDULER_UPDATE_PATCH_MARKER, True)
    scheduler_cls.update_from_output = patched_update_from_output
