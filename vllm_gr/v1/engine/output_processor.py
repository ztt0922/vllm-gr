# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from contextvars import ContextVar
from functools import wraps

from vllm.logger import init_logger

logger = init_logger(__name__)


_PROCESS_OUTPUTS_PATCH_MARKER = "_vllm_gr_admission_rejection_outputs_patch"

# Terminal admission rejections, keyed by the request id the *engine* used,
# parked between the output that carries one and the serving layer that reports
# it. A request the engine refuses ends on a bare error output whose reason
# lives in ``kv_transfer_params``; the serving loop only ever sees the finish
# reason, so the message needs somewhere to wait in between.
#
# The engine's name for a request is not the name its caller is answered under.
# The frontend resolves the caller's own name into the id it reports and then
# extends it before submitting it, and the engine extends it again, so an
# ordinary request's refusal is filed under a name no serving loop ever sees.
# ``_CLIENT_REQUEST_IDS_ATTR`` records which engine request each caller-owned
# name became, and a claim is an exact lookup under one name or the other --
# never a prefix, since the caller supplies its own name and one may extend
# another's. Pooled rather than assumed empty because a refused V1 session is
# reported by the beam client off the output itself and claims nothing.
_ADMISSION_REJECTIONS_ATTR = "_gr_admission_rejections"
_CLIENT_REQUEST_IDS_ATTR = "_gr_client_request_ids"
_ADMISSION_REJECTION_MAX = 256

# The name the caller of the request being handled in this task will be told.
# Both of the OpenAI frontends resolve a caller-supplied id through
# ``BaseServing._base_request_id`` and extend it before submitting it to the
# engine, so the two names differ for every ordinary request and only the
# frontend holds both. It publishes the caller's name here as it resolves it
# (see ``patch_client_request_id``), and both the submit path and the report
# path read it back; outside an OpenAI request it is unset, which keeps
# in-process callers and beam sessions on the engine's own names.
_CLIENT_REQUEST_ID: ContextVar[str | None] = ContextVar("vllm_gr_client_request_id", default=None)


def note_admission_rejections(processor, engine_core_outputs) -> None:
    """Park the reason of every refused output in ``engine_core_outputs``."""
    from vllm_gr.v1.beam.batch_contracts import admission_rejection

    for ec_output in engine_core_outputs:
        message = admission_rejection(ec_output)
        if message is None:
            # A finished output that is not itself a refusal says its id now
            # belongs to a request this pool was never told about. Whatever is
            # parked under that id described an earlier one and must not answer
            # for this one: the caller picks its own id and may reuse it.
            if getattr(ec_output, "finish_reason", None) is not None:
                forget_admission_rejection(processor, getattr(ec_output, "request_id", ""))
            continue
        rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
        if rejections is None:
            rejections = {}
            setattr(processor, _ADMISSION_REJECTIONS_ATTR, rejections)
        while len(rejections) >= _ADMISSION_REJECTION_MAX:
            rejections.pop(next(iter(rejections)))
        rejections[getattr(ec_output, "request_id", "")] = message


def forget_admission_rejection(processor, request_id: str) -> None:
    """Drop whatever is parked for ``request_id``, if anything is."""
    rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
    if rejections:
        rejections.pop(request_id, None)


def remember_client_request_id(processor, client_request_id: str, engine_request_id: str) -> None:
    """Record which engine request a caller-owned name was submitted as.

    The frontend resolves the caller's name into the id it reports and extends
    it once more before the engine is handed the request, so this pairing of
    the two exists nowhere else. Only the latest engine request per name is
    kept: a name the caller reuses has to find the request it was last
    submitted for, and not the one before it.
    """
    ids = getattr(processor, _CLIENT_REQUEST_IDS_ATTR, None)
    if ids is None:
        ids = {}
        setattr(processor, _CLIENT_REQUEST_IDS_ATTR, ids)
    while len(ids) >= _ADMISSION_REJECTION_MAX:
        ids.pop(next(iter(ids)))
    ids[client_request_id] = engine_request_id


def engine_request_id_for(processor, client_request_id: str) -> str | None:
    """The engine request a caller-owned name was last submitted as, if any."""
    ids = getattr(processor, _CLIENT_REQUEST_IDS_ATTR, None)
    return (ids or {}).get(client_request_id)


def take_admission_rejection(
    engine_client, request_id: str, client_request_id: str | None = None
) -> str | None:
    """Consume the parked reason for the request this caller is asking about.

    Two names can stand for the request that was refused: ``request_id``, the
    one the serving layer reports, and ``client_request_id``, the caller's own,
    which the frontend resolved into the engine's when it submitted the request
    (see ``remember_client_request_id``). Which of the two the reason was filed
    under depends on how the refusal reached the serving layer, so both are
    tried -- and both are looked up exactly, and nothing looser. That is what
    keeps the answer honest: the caller supplies its own name -- an
    ``X-Request-Id`` header, or the body's ``request_id`` -- so a match by
    shared prefix would let one caller's name decide another request's refusal
    and report a real engine fault as a client mistake.
    """
    processor = getattr(engine_client, "output_processor", None)
    rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
    if not rejections:
        return None
    message = rejections.pop(request_id, None)
    if message is not None:
        return message
    if client_request_id is None:
        return None
    engine_request_id = engine_request_id_for(processor, client_request_id)
    if engine_request_id is None:
        return None
    return rejections.pop(engine_request_id, None)


def make_patched_process_outputs(original_process_outputs):
    """Preserve admission reasons while forwarding every output to vLLM."""
    if getattr(original_process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, False):
        return original_process_outputs

    @wraps(original_process_outputs)
    def patched_process_outputs(self, engine_core_outputs, *args, **kwargs):
        from vllm_gr.v1.metrics.prometheus import consume_outputs

        consume_outputs(self, engine_core_outputs)
        note_admission_rejections(self, engine_core_outputs)
        return original_process_outputs(self, engine_core_outputs, *args, **kwargs)

    setattr(patched_process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, True)
    return patched_process_outputs
