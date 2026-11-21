# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Metrics for columnar results and sampled asynchronous output."""

from dataclasses import replace
from struct import pack
from threading import Condition
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock

import msgspec
import pytest
from vllm.v1.outputs import ModelRunnerOutput

from vllm_gr.v1.beam.batch_contracts import BeamSequenceColumns
from vllm_gr.v1.beam.dispatch import GR_WORKER_RESULT_ATTR, GRWorkerResult
from vllm_gr.v1.metrics.telemetry import invalid_sequence_reason, terminal_metrics
from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import AsyncGPUBeamOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def column_session() -> Any:
    return NS(
        _metrics_started=True,
        _metrics_decode_seconds=0.02,
        owner_generation=7,
        batch_size=1,
        request=NS(
            beam_params=NS(max_tokens=3, beam_width=4, dynamic_beam_widths=None),
            output_options=NS(
                num_return_sequences=2,
                include_cumulative_logprob=True,
                include_parent_beam_indices=False,
            ),
        ),
    )


def test_columnar_results_are_counted_without_materializing_sequences() -> None:
    columns = BeamSequenceColumns(2, 3, pack("<6i", 1, 2, 3, 4, 5, 6), pack("<2f", -1, -2), None)
    result = NS(failure=None, output=NS(items=(NS(item_index=0, sequences=columns),)))
    value = terminal_metrics(column_session(), result)
    assert value["result_counts"] == [2]
    assert value["invalid_counts"] == {}
    assert value["decode_seconds"] == 0.02
    assert result.output.items[0].sequences is columns


def test_malformed_columns_do_not_abort_metric_collection() -> None:
    columns = BeamSequenceColumns(2, 3, b"invalid", pack("<2f", -1, -2), None)
    result = NS(failure=None, output=NS(items=(NS(item_index=0, sequences=columns),)))
    value = terminal_metrics(column_session(), result)
    assert value["result_counts"] == [0]
    assert value["invalid_counts"] == {"malformed_result": 1}


def test_valid_early_stop_is_not_counted_as_an_invalid_length() -> None:
    sequence = NS(
        token_ids=(1,), cumulative_logprob=-1.0, parent_beam_indices=(0,), finish_reason="stop"
    )
    assert invalid_sequence_reason(sequence, column_session().request.beam_params) is None


@pytest.mark.parametrize("sampled", [False, True])  # type: ignore[misc]
def test_terminal_output_transports_metrics_once_after_existing_fence(sampled: Any) -> None:
    sample: dict[str, Any] = {"sampled": sampled, "candidate_width": 4}
    if sampled:
        sample["timings"] = {"forward": 0.001}
    owner = NS(
        final_output=NS(wait_embedded_control=Mock(return_value=[1, 1, 0])),
        metrics=NS(consume=Mock(return_value=sample)),
        control_drained=Condition(),
        control_busy=[True],
        request_terminal_release=Mock(),
        release_terminal_if_drained=Mock(return_value=False),
    )
    execution = NS(entries=(NS(metadata=NS(is_last_stage=True)),))
    receipt = AsyncGPUBeamOutput(owner, execution, 0, True)
    result = GRWorkerResult("batch", 0, 0, 1, None, True, owner_generation=7)
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
    setattr(output, GR_WORKER_RESULT_ATTR, {"batch": result})

    def convert(*args: Any, **kwargs: Any) -> Any:
        owner.final_output.wait_embedded_control.assert_called_once()
        owner.metrics.consume.assert_called_once_with(0)
        return output

    receipt._convert = Mock(side_effect=convert)
    assert receipt.get_output() is output
    assert receipt.get_output() is output
    receipt._convert.assert_called_once()
    assert owner.control_busy == [False]
    carried = getattr(output, GR_WORKER_RESULT_ATTR)["batch"].metrics
    assert carried is sample
    if sampled:
        assert carried["timings"]["output_materialize"] >= 0
    else:
        assert "timings" not in carried
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(carried))["sampled"] is sampled
    assert replace(result, metrics=carried).owner_generation == 7
