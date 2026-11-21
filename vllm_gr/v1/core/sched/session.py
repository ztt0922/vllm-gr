# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Scheduler-owned session state and stage completion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, cast

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchRequest,
    BeamBatchResult,
)
from vllm_gr.v1.beam.dispatch import (
    GRDeviceStateRef,
    GRRetireMetadata,
    GRStageMetadata,
)


class GRSessionStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHING = "finishing"
    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"


class GRResourceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    DRAINING = "draining"
    RETIRING = "retiring"
    CLOSED = "closed"


@dataclass(slots=True)
class GRSessionState:
    """EngineCore scheduling state for one fixed-membership GR batch session.

    Native Requests own token progress and the effective budget. Frozen dispatch
    records retain dependencies and receipt identity after native termination.
    """

    session_id: str
    request: BeamBatchRequest
    max_decode_steps: int
    status: GRSessionStatus = GRSessionStatus.WAITING
    terminal_result_pending: bool = False
    resource_status: GRResourceStatus = GRResourceStatus.WAITING
    owner_generation: int = 0
    retirement: GRRetireMetadata | None = None
    worker_resources_allocated: bool = False
    # EngineCore-private lifecycle state. These fields are excluded from the
    # constructor and never cross the Scheduler/Worker boundary.
    _next_dispatch_id: int = field(default=0, init=False, repr=False)
    _dispatch_records: dict[int, GRStageMetadata] = field(
        default_factory=dict, init=False, repr=False
    )
    _in_flight_dispatches: set[int] = field(default_factory=set, init=False, repr=False)
    _pending_terminal_result: BeamBatchResult | None = field(default=None, init=False, repr=False)
    _terminal_emitted: bool = field(default=False, init=False, repr=False)
    _kv_dispatch_leases: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _metrics_dispatch_timers: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _metrics_started: bool = field(default=False, init=False, repr=False)
    _metrics_decode_timer: Any = field(default=None, init=False, repr=False)
    _metrics_decode_seconds: float | None = field(default=None, init=False, repr=False)
    _metrics_prompt_work: tuple[tuple[int, int], ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.session_id != self.request.batch_id:
            raise ValueError("session_id must match the Beam batch_id")
        if not self.request.items:
            raise ValueError("Beam Search V1 batch must not be empty")
        request_ids = self.request_ids
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Beam Search V1 batch request IDs must be unique")
        if self.max_decode_steps < 0:
            raise ValueError("max_decode_steps must be non-negative")
        if self.max_decode_steps != self.request.beam_params.max_tokens - 1:
            raise ValueError("max_decode_steps must equal effective max_tokens minus 1")

    @property
    def total_stages(self) -> int:
        return cast(int, self.request.beam_params.max_tokens)

    @property
    def issued_stages(self) -> int:
        """Diagnostic count derived from physical dispatch records."""
        return len(self._dispatch_records)

    @property
    def completed_stages(self) -> int:
        return len(self._dispatch_records) - len(self._in_flight_dispatches)

    @property
    def request_ids(self) -> tuple[str, ...]:
        """Stable caller-defined member order for every batch stage."""
        return tuple(item.request.request_id for item in self.request.items)

    @property
    def batch_size(self) -> int:
        return len(self.request.items)

    @property
    def in_flight_stages(self) -> int:
        return len(self._in_flight_dispatches)

    @property
    def latest_device_state(self) -> GRDeviceStateRef | None:
        if not self._dispatch_records:
            return None
        return next(reversed(self._dispatch_records.values())).output_device_state


@dataclass(frozen=True, slots=True)
class GRStageCompletion:
    """GR state derived from one native vLLM ``ModelRunnerOutput``."""

    session_id: str
    stage_index: int
    finished: bool
    device_state: GRDeviceStateRef | None = None
    model_output: ModelRunnerOutput | None = None
    error: str | None = None


class GRAsyncSchedulerInterface(Protocol):
    """Hooks added by patching vLLM Scheduler; this is not a second scheduler."""

    def add_gr_session(self, request: BeamBatchRequest) -> GRSessionState:
        """Create one persistent scheduler state and add it once."""
        ...

    def can_issue_gr_stage(self, session: GRSessionState) -> bool:
        """Whether another stage may be scheduled without consuming a Future.

        GR bounds outstanding physical dispatches; native placeholders own progress.
        This is what permits Prefill-to-Decode and Decode-to-Decode overlap.
        """
        ...

    def attach_gr_stage_metadata(
        self,
        scheduler_output: SchedulerOutput,
        session: GRSessionState,
    ) -> GRStageMetadata | None:
        """Attach one GR stage to the output of native ``Scheduler.schedule``."""
        ...

    def update_gr_from_output(
        self,
        session: GRSessionState,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> GRStageCompletion:
        """Match a receipt and retire its dispatch and KV lease exactly once."""
        ...

    def finish_gr_session(
        self,
        session: GRSessionState,
        status: GRSessionStatus,
    ) -> BeamBatchResult | None:
        """On completion/cancel/error, stop new stage issuance.

        A3/A4 must drain or invalidate issued successors without overwriting
        terminal Beam state, retiring their placeholders and completions.
        Return None until all device work and terminal copies are complete;
        only then release KV/workspace and return the retained result once.
        Host-observed finished=True alone never authorizes resource release.
        """
        ...
