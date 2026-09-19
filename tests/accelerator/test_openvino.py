from __future__ import annotations

import importlib.util

import pytest
import torch
from freetoken.accelerator.openvino import OpenVINODenseIsland


def test_openvino_island_rejects_cpu_device():
    with pytest.raises(ValueError, match="explicit GPU"):
        OpenVINODenseIsland(torch.ones(4, 8), max_batch_tokens=2, device="CPU")


def test_openvino_island_rejects_out_of_bucket_input():
    if importlib.util.find_spec("openvino") is None:
        pytest.skip("OpenVINO unavailable")
    island = OpenVINODenseIsland(torch.ones(4, 8), max_batch_tokens=2)

    with pytest.raises(ValueError, match="between 1 and 2"):
        island(torch.ones(3, 8))


@pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None, reason="OpenVINO unavailable"
)
def test_openvino_executes_dense_island_on_intel_gpu():
    weight = torch.arange(32, dtype=torch.float16).reshape(4, 8) / 32
    bias = torch.arange(4, dtype=torch.float16) / 16
    hidden_states = torch.arange(16, dtype=torch.float16).reshape(2, 8) / 8
    expected = hidden_states @ weight.T + bias

    island = OpenVINODenseIsland(
        weight, bias, max_batch_tokens=4, activation="none", device="GPU"
    )
    result = island(hidden_states)

    assert island.execution_info.execution_devices == ("GPU.0",)
    assert "Intel" in island.execution_info.full_device_name
    torch.testing.assert_close(result.output, expected, rtol=2e-3, atol=2e-3)
    assert result.inference_seconds > 0
