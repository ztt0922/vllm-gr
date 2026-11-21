# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native result lifetime at the CUDA stage's asynchronous output boundary."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.v1.metrics.gpu import GPUStageMetrics
from vllm_gr.v1.worker.cuda import gpu_beam_stage_runner as stages
from vllm_gr.v1.worker.cuda.gpu_stage_hooks import CudaStageHooksType
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1
from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize("hooks_type", [CudaStageHooksV1, CudaStageHooksV2])  # type: ignore[misc]
@pytest.mark.parametrize("has_native_output", [False, True])  # type: ignore[misc]
def test_finish_native_output_preserves_resources_and_bookkeeping(
    hooks_type: CudaStageHooksType, has_native_output: bool
) -> None:
    resources = {
        "_gr_v2_runtime": object(),
        "_beam_decode_graph": object(),
        "_gr_gpu_stage_runner_cache": {1: object()},
        "_beam_sessions": {"request": object()},
        "postprocess_num_computed_tokens": Mock(side_effect=AssertionError("counter update")),
        "_update_states": Mock(side_effect=AssertionError("native bookkeeping")),
    }
    worker = SimpleNamespace(**resources)
    if has_native_output:
        worker.execute_model_state = object()
        worker.kv_connector_output = object()
    hooks = hooks_type(worker)
    for _ in range(2):
        hooks.finish_native_output()
        assert vars(worker) == {
            **resources,
            "execute_model_state": None,
            "kv_connector_output": None,
        }


def test_v1_prefill_retains_native_state_until_output_completion() -> None:
    logits = torch.zeros((1, 4))
    state = SimpleNamespace(logits=logits)
    connector = object()
    worker = SimpleNamespace(execute_model_state=state, kv_connector_output=connector)
    hooks = CudaStageHooksV1(worker)
    assert hooks.prefill_logits(Mock(return_value=None), object()) is logits
    assert worker.execute_model_state is state
    assert worker.kv_connector_output is connector
    hooks.finish_native_output()
    assert worker.execute_model_state is None
    assert worker.kv_connector_output is None


@pytest.fixture(params=[CudaStageHooksV1, CudaStageHooksV2], ids=["v1", "v2"])  # type: ignore[misc]
def sample_owner(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    # A cancelled successor dispatch still publishes a control receipt.
    # CPU tensors and fake events isolate ordering; device tests cover real DMA.
    worker = SimpleNamespace(execute_model_state=object(), kv_connector_output=object())
    execution = SimpleNamespace(
        entries=(
            SimpleNamespace(metadata=SimpleNamespace(produces_output=True, is_last_stage=False)),
        )
    )
    owner = SimpleNamespace(
        worker=worker,
        hooks=request.param(worker),
        execution_error=None,
        pending=(execution, 0),
        cancelled=True,
        hidden=None,
        logits=None,
        bindings=(object(),),
        beam=SimpleNamespace(
            state=SimpleNamespace(
                device=torch.device("cpu"),
                num_generated_tokens=torch.ones(1, dtype=torch.int32),
                is_finished=torch.ones(1, dtype=torch.bool),
                error_code=torch.zeros(1, dtype=torch.int32),
            )
        ),
        device_control=torch.tensor([[[1, 0, 0]]], dtype=torch.int32),
        host_control=torch.empty((1, 1, 3), dtype=torch.int32),
        produced_ready=[Mock()],
        last_work=Mock(),
        copy_stream=Mock(),
        copy_done=[Mock()],
    )
    monkeypatch.setattr(
        "vllm_gr.v1.metrics.gpu.metrics_config", lambda: SimpleNamespace(enabled=False)
    )
    owner.metrics = GPUStageMetrics(owner, stages.CONTROL_CAPACITY)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=object()))
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(
        owner.hooks, "finish_native_output", Mock(wraps=owner.hooks.finish_native_output)
    )
    return owner


def test_sample_finishes_native_output_after_enqueue_before_clearing_pending(
    sample_owner: Any,
) -> None:
    owner = sample_owner
    pending = owner.pending
    native_state, connector = owner.worker.execute_model_state, owner.worker.kv_connector_output
    finish = owner.hooks.finish_native_output

    def observe_finish() -> None:
        owner.produced_ready[0].record.assert_called_once()
        owner.last_work.record.assert_called_once()
        owner.copy_done[0].record.assert_called_once_with(owner.copy_stream)
        assert owner.pending is pending
        assert owner.worker.execute_model_state is native_state
        assert owner.worker.kv_connector_output is connector

    finish.side_effect = observe_finish
    receipt = stages.GPUBeamStageRunner.sample(owner)
    finish.assert_called_once_with()
    assert receipt.execution is pending[0] and receipt.slot == pending[1]
    assert owner.pending is None
    assert owner.host_control.tolist() == [[[0, 1, 0]]]
    # A replacement hook owns the native field policy; the core must not also
    # write the fields after the hook returns.
    assert owner.worker.execute_model_state is native_state
    assert owner.worker.kv_connector_output is connector


@pytest.mark.parametrize(  # type: ignore[misc]
    "failure_at", ["execute", "grammar", "pending", "logits", "enqueue", "terminal_enqueue"]
)
def test_failed_sample_preserves_native_output(sample_owner: Any, failure_at: str) -> None:
    owner = sample_owner
    error = RuntimeError("injected failure")
    grammar = None
    if failure_at == "execute":
        owner.execution_error = error
    elif failure_at == "grammar":
        grammar = object()
    elif failure_at == "pending":
        owner.pending = None
    elif failure_at == "logits":
        owner.cancelled = False
    elif failure_at == "terminal_enqueue":
        owner.pending[0].entries[0].metadata.is_last_stage = True
        owner.final_output = SimpleNamespace(enqueue=Mock(side_effect=error))
    else:
        owner.copy_done[0].record.side_effect = error
    pending = owner.pending
    native_state, connector = owner.worker.execute_model_state, owner.worker.kv_connector_output
    with pytest.raises(RuntimeError) as failure:
        stages.GPUBeamStageRunner.sample(owner, grammar)
    if failure_at == "execute":
        assert failure.value.__cause__ is error
    elif failure_at in ("enqueue", "terminal_enqueue"):
        assert failure.value is error
    owner.hooks.finish_native_output.assert_not_called()
    assert owner.pending is pending
    assert owner.worker.execute_model_state is native_state
    assert owner.worker.kv_connector_output is connector


def test_execute_failure_is_propagated_to_sampling(
    sample_owner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_gr.v1.worker.gr_execution import execute_gr_stage

    owner = sample_owner
    owner.worker.device = "cuda"
    error = RuntimeError("native execution failed")
    owner.execute = Mock(side_effect=error)
    monkeypatch.setattr(stages, "get_stage_runner", Mock(return_value=owner))
    pending = owner.pending
    native_state = owner.worker.execute_model_state
    with pytest.raises(RuntimeError) as execute_failure:
        execute_gr_stage(
            owner.worker, SchedulerOutput.make_empty(), Mock(), hooks_type=type(owner.hooks)
        )
    assert execute_failure.value is error
    with pytest.raises(RuntimeError, match="execute_model failed") as sample_failure:
        stages.GPUBeamStageRunner.sample(owner)
    assert sample_failure.value.__cause__ is error
    owner.hooks.finish_native_output.assert_not_called()
    assert owner.pending is pending
    assert owner.worker.execute_model_state is native_state


def test_native_output_finish_failure_propagates_without_publishing_receipt(
    sample_owner: Any,
) -> None:
    owner = sample_owner
    pending = owner.pending
    error = RuntimeError("native output cleanup failed")
    owner.hooks.finish_native_output.side_effect = error
    native_state, connector = owner.worker.execute_model_state, owner.worker.kv_connector_output
    with pytest.raises(RuntimeError) as failure:
        stages.GPUBeamStageRunner.sample(owner)
    assert failure.value is error
    owner.copy_done[0].record.assert_called_once_with(owner.copy_stream)
    owner.hooks.finish_native_output.assert_called_once_with()
    assert owner.pending is pending
    assert owner.worker.execute_model_state is native_state
    assert owner.worker.kv_connector_output is connector
