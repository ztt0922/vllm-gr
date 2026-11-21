# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Beam stage dispatch and retirement contracts shared by Scheduler and Worker."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from operator import attrgetter
from typing import Any

from vllm_gr.config import MAX_BEAM_WIDTH
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    DEFAULT_CONSTRAINT_RESOURCE_ID,
    SUPPORTED_BEAM_BATCH_SIZES,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
    validate_beam_batch_request,
)
from vllm_gr.v1.beam.execution_batch import (
    BeamBatchExecutionView,
    BeamExecutionContractError,
)

GR_STAGE_OUTPUT_ATTR = "gr_stage_metadata"
GR_WORKER_RESULT_ATTR = "gr_worker_results"
GR_BATCH_RESULT_ATTR = "gr_batch_results"
GR_RETIRE_OUTPUT_ATTR = "gr_retire_metadata"
GR_RETIRE_RESULT_ATTR = "gr_retire_result"


def validate_dispatch_request(request: BeamBatchRequest, *, trie_count: int | None = None) -> None:
    """Validate shared V1 semantics before native ADD or Worker allocation."""
    validate_beam_batch_request(request)
    if trie_count is not None and any(item.trie_id >= trie_count for item in request.items):
        raise ValueError("trie_id is outside the loaded constraint table")
    if len(request.items) not in SUPPORTED_BEAM_BATCH_SIZES:
        raise ValueError(
            "beam_search supports batch sizes "
            f"{SUPPORTED_BEAM_BATCH_SIZES}, got {len(request.items)}"
        )
    if request.items[0].request.request_id != request.batch_id:
        raise ValueError("batch_id and the first native request_id must match")
    for item in request.items:
        native = item.request
        params = native.sampling_params
        if (
            native.prompt_embeds is not None
            or native.prompt_token_ids is None
            or len(native.prompt_token_ids) == 0
        ):
            raise ValueError("beam_search requires a nonempty token prompt")
        if native.mm_features or native.lora_request is not None:
            raise ValueError("beam_search does not support multimodal inputs or LoRA")
        if params is None or params.ignore_eos is not True:
            raise ValueError("beam_search requires ignore_eos=True")
        if params.max_tokens != request.beam_params.max_tokens or params.n != 1:
            raise ValueError("native and Beam generation budgets must match and n must be 1")
    _validate_gr_generation_params(request.beam_params)


class GRStageKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


def _validate_gr_generation_params(params: BeamBatchParams) -> None:
    """Validate V1 generation semantics without constructing physical capacity."""
    if params.ignore_stop_tokens is not True:
        raise ValueError("beam_search requires ignore_eos=True")
    if params.catalog_strategy is not None:
        raise ValueError("Constraint Table does not support catalog_strategy")
    if (
        type(params.beam_width) is not int
        or not 1 <= params.beam_width <= MAX_BEAM_WIDTH
        or type(params.max_tokens) is not int
        or params.max_tokens < 1
    ):
        raise ValueError(f"Asynchronous scheduling requires W in [1, {MAX_BEAM_WIDTH}] and N >= 1")


@dataclass(frozen=True, slots=True)
class GRExecutionCapacity:
    """Shared Asynchronous scheduling admission contract; storage capacity does not enable concurrency."""

    beam_width: int
    active_sessions: int = 1
    dispatches_per_session: int = 2

    def __post_init__(self) -> None:
        if type(self.beam_width) is not int or not 1 <= self.beam_width <= MAX_BEAM_WIDTH:
            raise ValueError(
                "Asynchronous scheduling requires a fixed physical beam width in "
                f"[1, {MAX_BEAM_WIDTH}]"
            )
        if (
            type(self.active_sessions) is not int
            or self.active_sessions != 1
            or type(self.dispatches_per_session) is not int
            or self.dispatches_per_session != 2
        ):
            raise ValueError(
                "Asynchronous scheduling requires one execution owner and two dispatch slots"
            )

    def validate(self, params: BeamBatchParams) -> None:
        _validate_gr_generation_params(params)
        if params.beam_width != self.beam_width:
            raise ValueError("Asynchronous scheduling requires the startup beam width")


@dataclass(frozen=True, slots=True)
class GRRetireMetadata:
    session_id: str
    owner_generation: int
    retire_id: int


@dataclass(frozen=True, slots=True)
class GRRetireResult:
    session_id: str
    owner_generation: int
    retire_id: int
    released: bool


@dataclass(frozen=True, slots=True)
class GRDeviceStateRef:
    """Reference to Beam state that remains owned by the Worker device."""

    session_id: str
    producer_stage_index: int


@dataclass(frozen=True, slots=True)
class GRStageMetadata:
    """GR metadata attached to one native vLLM ``SchedulerOutput``.

    Stage 0 is Prefill. For effective budget N, stages 1..N-1 are Decode
    steps 0..N-2. ``dispatch_id`` identifies the physical invocation separately
    from native logical progress. Worker echoes both identities and generation.

    SchedulerOutput.gr_stage_metadata is keyed by session_id. Asynchronous scheduling admits one
    execution owner; owner_generation distinguishes reuse of a session ID.
    session_init carries the caller-formed batch request (Beam parameters, output
    options and beam_params.constraint_resource_id) once, with Prefill only.
    Native SchedulerOutput supplies prompt/KV data. Stage k > 0 references
    (session_id, k - 1) even before host completion; A4 orders producer writes
    and successor reads through its execution stream or device events.
    """

    session_id: str
    dispatch_id: int
    stage_index: int
    kind: GRStageKind
    decode_step: int | None
    is_last_stage: bool
    input_device_state: GRDeviceStateRef | None
    output_device_state: GRDeviceStateRef
    beam_params: BeamBatchParams
    output_options: BeamOutputOptions
    num_prompt_tokens: int
    produces_output: bool = True
    session_init: BeamBatchRequest | None = None
    owner_generation: int = 0
    execution_capacity: GRExecutionCapacity | None = None
    # One row of prompt block IDs per request_id, in the same order. GR
    # execution supports one uniform KV cache group; for B=1 this retains
    # the existing single-row representation used by V1 and V2 runners.
    # None means block IDs were not published.
    kv_blocks: tuple[tuple[int, ...], ...] | None = None
    # The batch is formed by the caller and remains ordered and immutable for
    # the complete session. B=1 keeps the original representation.
    request_ids: tuple[str, ...] = ()
    num_prompt_tokens_by_request: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GRWorkerResult:
    """A4/A5 attach ModelRunnerOutput.gr_v1_results: dict[str, GRWorkerResult].

    Keys match session_id, including multiple sessions in one native output.
    Both GR attachments must survive executor IPC; async output wrappers must
    preserve gr_v1_results on the inner ModelRunnerOutput returned by get_output.
    Intermediate token IDs, parents and scores remain device-resident.
    terminal_result is supplied once for the terminating stage, with batch_id
    matching session_init.batch_id. A3 retains it until successors are drained,
    then delivers it through the frontend's wait_final channel.
    resources_released is supplied once by the last terminal output consumer;
    it replaces the normal-path zero-token retirement round trip.
    """

    session_id: str
    dispatch_id: int
    stage_index: int
    produced_token_count: int
    device_state: GRDeviceStateRef
    finished: bool
    terminal_result: BeamBatchResult | None = None
    resources_released: bool = False
    owner_generation: int = 0
    # Compact CPU telemetry; device tensors and events never cross executor IPC.
    metrics: dict[str, Any] | None = None


# Every GRStageMetadata field is either proven once per session and then compared
# by equality on each stage (SESSION_FIELDS), or re-checked on each stage
# (STAGE_FIELDS).  A test asserts the two lists partition the dataclass, so a
# field added later cannot silently fall between the two halves.
SESSION_FIELDS = (
    "session_id",
    "owner_generation",
    "beam_params",
    "output_options",
    # Only checked to be nonzero here; equality with the per-request total is
    # proven by the view in ``validate_dispatch_session``, then compared.
    "num_prompt_tokens",
    "request_ids",
    "num_prompt_tokens_by_request",
)
STAGE_FIELDS = (
    "dispatch_id",
    "stage_index",
    "kind",
    "decode_step",
    "is_last_stage",
    "input_device_state",
    "output_device_state",
    "produces_output",
    "session_init",
    "execution_capacity",
    "kv_blocks",
)
_session_fields = attrgetter(*SESSION_FIELDS)


@dataclass(frozen=True, slots=True)
class GRDispatchSession:
    """Session-invariant dispatch facts, proven once per Beam session.

    ``fields`` is a *value snapshot* of ``SESSION_FIELDS``, not a reference to
    the stage-0 metadata: every later stage must carry equal values, so
    :func:`validate_dispatch_step` compares values and never object identity.
    ``capacity`` is the execution capacity proven for this session.

    The snapshot stays trustworthy because its members cannot be changed in
    place: ``GRStageMetadata`` is a frozen dataclass, ``BeamBatchParams`` and
    ``BeamOutputOptions`` are frozen ``msgspec`` structs, and the rest are
    ``str``, ``int`` and ``tuple``.  A write after validation therefore raises
    instead of silently invalidating the record.  The one escape hatch,
    ``msgspec.structs.force_setattr``, is used only for
    ``BeamBatchItem.request`` while translating a Prefill payload; it never
    touches a session field, and ``session_init`` is a stage field that must
    be ``None`` on Decode anyway.  Fields whose types ``msgspec`` equality
    would conflate (``3.0`` vs ``3``, ``1`` vs ``True``) are re-derived per
    stage by :func:`validate_dispatch_step`, so the record is not their only
    guard.

    A record is valid for exactly one logical Beam session: build it from the
    Prefill dispatch, reuse it for that session's Decode stages, rebuild it
    whenever no record is available (a caller that was not handed one), and
    drop it when the session ends or the adapter is recycled.  ``session_id``
    and ``owner_generation`` are session fields, so a record from another
    logical batch is rejected rather than reused.
    """

    session_id: str
    fields: tuple[Any, ...]
    capacity: GRExecutionCapacity


def validate_dispatch_session(scheduler_output: Any) -> GRDispatchSession:
    """Validate the stage-invariant half of a GR dispatch.

    These facts cannot change between the Prefill and Decode stages of one
    session: the single-entry metadata map, the logical batch identity, the
    generation parameters (including the execution capacity), the
    output-producing nonempty prompt, and ``session_init``.
    """
    mapping = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("GR dispatch requires a nonempty metadata mapping")
    if len(mapping) != 1 or getattr(scheduler_output, "beam_data", {}):
        raise ValueError("GR dispatch cannot mix ordinary or legacy Beam requests")
    key, m = next(iter(mapping.items()))
    if not isinstance(m, GRStageMetadata) or key != m.session_id:
        raise ValueError("invalid GR metadata identity")
    view = BeamBatchExecutionView.from_stage_metadata(m)
    if view.batch_size not in SUPPORTED_BEAM_BATCH_SIZES:
        raise ValueError(
            f"GR requires one immutable logical batch with B in {SUPPORTED_BEAM_BATCH_SIZES}"
        )
    if not m.num_prompt_tokens:
        raise ValueError("GR requires a nonempty prompt and complete, output-producing Prefill")
    params = m.beam_params
    if (
        params.dynamic_beam_widths is not None
        and len(params.dynamic_beam_widths) != params.max_tokens
    ):
        raise ValueError("dynamic_beam_widths must contain one width per effective step")
    capacity = m.execution_capacity or GRExecutionCapacity(params.beam_width)
    capacity.validate(params)
    if (
        type(params.beam_width) is not int
        or not 1 <= params.beam_width <= MAX_BEAM_WIDTH
        or type(params.max_tokens) is not int
        or params.max_tokens < 1
        or not math.isfinite(params.temperature)
        or params.temperature < 0
        or params.constraint_resource_id != DEFAULT_CONSTRAINT_RESOURCE_ID
    ):
        raise ValueError(f"GR requires W in [1, {MAX_BEAM_WIDTH}] and valid generation params")
    if m.session_init is not None:
        request = m.session_init
        validate_dispatch_request(request)
        if (
            m.kind is not GRStageKind.PREFILL
            or request.protocol_version != BEAM_BATCH_PROTOCOL_VERSION
            or len(request.items) != view.batch_size
            or request.batch_id != key
            or tuple(item.item_index for item in request.items) != tuple(range(view.batch_size))
            or tuple(item.request.request_id for item in request.items) != view.request_ids
            or tuple(
                len(item.request.prompt_token_ids)
                if item.request.prompt_token_ids is not None
                else 0
                for item in request.items
            )
            != view.num_prompt_tokens_by_request
            or request.beam_params != params
            or request.output_options != m.output_options
        ):
            raise ValueError("GR session_init disagrees with dispatch metadata")
    return GRDispatchSession(session_id=key, fields=_session_fields(m), capacity=capacity)


def validate_dispatch_step(
    scheduler_output: Any,
    session: GRDispatchSession,
    *,
    previous: GRStageMetadata | None = None,
    binding: Any = None,
    bindings: tuple[Any, ...] | None = None,
) -> tuple[GRStageMetadata, ...]:
    """Validate everything one stage can still change, plus invariant equality."""
    key = session.session_id
    mapping = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("GR dispatch requires a nonempty metadata mapping")
    if len(mapping) != 1 or getattr(scheduler_output, "beam_data", {}):
        raise ValueError("GR dispatch cannot mix ordinary or legacy Beam requests")
    ((map_key, m),) = mapping.items()
    if not isinstance(m, GRStageMetadata) or map_key != key or m.session_id != key:
        raise ValueError("invalid GR metadata identity")
    params = m.beam_params
    # ``msgspec`` struct equality compares values, so it cannot tell ``3.0`` from
    # ``3`` or ``1`` from ``True``, while the uncached path rejects both.  Repeat
    # the cheap structural predicates instead of trusting the record; none of them
    # allocates the per-row execution view.
    session.capacity.validate(params)
    for name, values in (
        ("dynamic_beam_widths", params.dynamic_beam_widths),
        ("num_prompt_tokens_by_request", m.num_prompt_tokens_by_request),
    ):
        if values is not None and any(type(value) is not int or value <= 0 for value in values):
            raise ValueError(f"{name} must contain positive integers")
    if params.candidate_width is not None and (
        type(params.candidate_width) is not int or params.candidate_width <= 0
    ):
        raise ValueError("candidate_width must be a positive integer")
    # Equality with the record covers every session-invariant field.  ``previous``
    # is compared as well, so a caller that passes a record from one stage and a
    # ``previous`` from another cannot slip through.
    if _session_fields(m) != session.fields or (
        previous is not None and _session_fields(previous) != session.fields
    ):
        raise BeamExecutionContractError("logical Beam batch identity changed across stages")
    request_ids, prompt_counts = m.request_ids, m.num_prompt_tokens_by_request
    scheduled = scheduler_output.num_scheduled_tokens
    if len(scheduled) != len(request_ids) or any(r not in scheduled for r in request_ids):
        raise ValueError(
            f"GR requires one immutable logical batch with B in {SUPPORTED_BEAM_BATCH_SIZES}"
        )
    if any(type(count) is not int or count <= 0 for count in scheduled.values()):
        raise ValueError("GR scheduled counts must be positive integers")
    total = sum(scheduled.values())
    if getattr(scheduler_output, "total_num_scheduled_tokens", total) != total:
        raise ValueError("GR dispatch total disagrees with native scheduled counts")
    for number in (m.dispatch_id, m.stage_index, m.num_prompt_tokens, m.owner_generation):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError("GR dispatch indices and prompt length must be nonnegative integers")
    if m.produces_output is not True:
        raise ValueError("GR requires a nonempty prompt and complete, output-producing Prefill")
    if not isinstance(m.is_last_stage, bool) or (
        m.decode_step is not None and type(m.decode_step) is not int
    ):
        raise ValueError("GR dispatch flags and step have invalid types")
    if m.execution_capacity is not None and m.execution_capacity != session.capacity:
        raise BeamExecutionContractError("execution_capacity changed across stages")
    if m.session_init is not None and m.kind is not GRStageKind.PREFILL:
        raise ValueError("GR session_init disagrees with dispatch metadata")
    if params.per_parent_beam_widths is not None:
        # Only the per-stage schedule feasibility rule lives in the view; every
        # other view check is session-level and already proven.
        BeamBatchExecutionView.from_stage_metadata(m)
    new_requests = {r.req_id: r for r in getattr(scheduler_output, "scheduled_new_reqs", ())}
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    for request_id, prompt_tokens in zip(request_ids, prompt_counts):
        computed = None
        if request_id in new_requests:
            native = new_requests[request_id]
            if (
                m.kind is GRStageKind.DECODE
                or (len(native.prompt_token_ids) if native.prompt_token_ids is not None else 0)
                != prompt_tokens
            ):
                raise ValueError("GR native request identity or prompt geometry changed")
            computed = native.num_computed_tokens
        elif cached is not None and request_id in cached.req_ids:
            if request_id in cached.resumed_req_ids and m.stage_index > 0:
                raise ValueError("GR rejects recompute after generation")
            computed = cached.num_computed_tokens[cached.req_ids.index(request_id)]
        if computed is None and (new_requests or cached is not None):
            raise ValueError("GR metadata has no matching native request")
        if computed is not None:
            if type(computed) is not int or computed < 0:
                raise ValueError("invalid native computed token count")
            if computed + scheduled[request_id] != prompt_tokens + m.stage_index:
                raise ValueError(
                    "GR stage disagrees with native token extent; partial Prefill/recompute"
                )
            if m.kind is GRStageKind.PREFILL and computed >= prompt_tokens:
                raise ValueError("GR Prefill requires remaining prompt tokens")
    if not 0 <= m.stage_index < params.max_tokens:
        raise ValueError("GR logical stage exceeds generation budget")
    if m.output_device_state != GRDeviceStateRef(key, m.stage_index):
        raise ValueError("GR output device-state identity mismatch")
    if m.kind is GRStageKind.PREFILL:
        if m.stage_index != 0 or m.decode_step is not None or m.input_device_state is not None:
            raise ValueError("GR rejects recompute after generation")
    elif m.kind is GRStageKind.DECODE:
        if (
            not m.produces_output
            or m.decode_step != m.stage_index - 1
            or m.input_device_state != GRDeviceStateRef(key, m.stage_index - 1)
            or any(scheduled[request_id] != 1 for request_id in request_ids)
        ):
            raise ValueError("invalid GR Decode dependency or logical token count")
    else:
        raise ValueError("unknown GR stage kind")
    if m.is_last_stage != (m.produces_output and m.stage_index == params.max_tokens - 1):
        raise ValueError("GR last-stage flag disagrees with effective budget")
    if binding is not None and key != binding.session_id:
        raise ValueError("a second active GR session cannot use shared execution scratch")
    if bindings is not None and (
        tuple(bound.native_request_id for bound in bindings) != request_ids
        or any(bound.session_id != key for bound in bindings)
    ):
        raise ValueError("GR batch changed the shared execution binding")
    if previous is None:
        if m.dispatch_id != 0 or m.kind is not GRStageKind.PREFILL:
            raise ValueError("GR session must start with dispatch 0 Prefill")
    else:
        if (
            m.dispatch_id != previous.dispatch_id + 1
            or m.stage_index != previous.stage_index + 1
            or m.owner_generation != previous.owner_generation
            or params != previous.beam_params
            or m.output_options != previous.output_options
            or m.num_prompt_tokens != previous.num_prompt_tokens
        ):
            raise ValueError("GR dispatch order, configuration or prompt binding changed")
    return (m,)


def validate_dispatch(
    scheduler_output: Any,
    *,
    previous: GRStageMetadata | None = None,
    binding: Any = None,
    bindings: tuple[Any, ...] | None = None,
    session: GRDispatchSession | None = None,
) -> tuple[GRStageMetadata, ...]:
    """Validate the whole metadata map before any Worker resource mutation.

    ``session`` carries facts already proven by :func:`validate_dispatch_session`
    for an earlier stage of the same Beam session.  With it, the session-level
    fields are guaranteed by equality with the recorded values rather than by
    re-deriving them.  Without it the record is rebuilt from this dispatch, so a
    caller holding only one stage still gets the complete check.
    """
    if session is None:
        session = validate_dispatch_session(scheduler_output)
    return validate_dispatch_step(
        scheduler_output, session, previous=previous, binding=binding, bindings=bindings
    )
