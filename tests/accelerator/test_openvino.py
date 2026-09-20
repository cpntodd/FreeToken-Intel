from __future__ import annotations

import builtins
import importlib.util

import pytest
import torch
import torch.nn.functional as F
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
    assert result.backend == "openvino"
    assert result.fallback_reason is None


class _CompileFailureCore:
    def compile_model(self, *_args, **_kwargs):
        raise RuntimeError("forced compile failure")


class _InferenceFailureCompiledModel:
    def get_property(self, _name):
        return ["GPU.0"]

    def __call__(self, _inputs):
        raise RuntimeError("forced inference failure")


class _InferenceFailureCore:
    def compile_model(self, *_args, **_kwargs):
        return _InferenceFailureCompiledModel()

    def get_property(self, _device, _name):
        return "Intel Arc test device"


class _UnexpectedExecutionCompiledModel:
    def get_property(self, _name):
        return ["CPU"]


class _UnexpectedExecutionCore:
    def compile_model(self, *_args, **_kwargs):
        return _UnexpectedExecutionCompiledModel()


def _xpu_device():
    if not torch.xpu.is_available():
        pytest.skip("Intel XPU unavailable")
    return torch.device("xpu")


def _xpu_reference(weight, bias, hidden_states, activation):
    device = hidden_states.device
    with torch.inference_mode():
        output = F.linear(
            hidden_states.to(torch.float16),
            weight.to(device=device, dtype=torch.float16),
            None if bias is None else bias.to(device=device, dtype=torch.float16),
        )
        if activation == "gelu":
            output = F.gelu(output, approximate="none")
        elif activation == "silu":
            output = F.silu(output)
    torch.xpu.synchronize(device)
    return output


@pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None, reason="OpenVINO unavailable"
)
@pytest.mark.parametrize("activation", ["none", "gelu", "silu"])
def test_openvino_compile_failure_uses_explicit_xpu_fallback(activation):
    device = _xpu_device()
    weight = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32
    bias = torch.arange(4, dtype=torch.float32) / 16
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 8
    expected = _xpu_reference(weight, bias, hidden_states.to(device), activation)

    island = OpenVINODenseIsland(
        weight,
        bias,
        max_batch_tokens=4,
        activation=activation,
        fallback="xpu",
        core=_CompileFailureCore(),
    )
    result = island(hidden_states.to(device))

    assert island.execution_info is None
    assert result.backend == "xpu"
    assert "forced compile failure" in result.fallback_reason
    torch.testing.assert_close(result.output, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None, reason="OpenVINO unavailable"
)
@pytest.mark.parametrize("activation", ["none", "gelu", "silu"])
def test_openvino_inference_failure_uses_explicit_xpu_fallback(activation):
    device = _xpu_device()
    weight = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32
    bias = torch.arange(4, dtype=torch.float32) / 16
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 8
    xpu_hidden_states = hidden_states.to(device)
    expected = _xpu_reference(weight, bias, xpu_hidden_states, activation)

    island = OpenVINODenseIsland(
        weight,
        bias,
        max_batch_tokens=4,
        activation=activation,
        fallback="xpu",
        core=_InferenceFailureCore(),
    )
    result = island(xpu_hidden_states)

    assert island.execution_info.execution_devices == ("GPU.0",)
    assert result.backend == "xpu"
    assert "forced inference failure" in result.fallback_reason
    torch.testing.assert_close(result.output, expected, rtol=1e-3, atol=1e-3)


def test_openvino_non_gpu_execution_report_uses_explicit_xpu_fallback():
    device = _xpu_device()
    weight = torch.eye(4, 8)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 8).to(device)
    island = OpenVINODenseIsland(
        weight,
        max_batch_tokens=2,
        fallback="xpu",
        core=_UnexpectedExecutionCore(),
    )

    result = island(hidden_states)

    assert island.execution_info is None
    assert result.backend == "xpu"
    assert "unexpected execution devices" in result.fallback_reason
    torch.testing.assert_close(
        result.output, hidden_states[:, :4].to(torch.float16), rtol=0, atol=0
    )


@pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None, reason="OpenVINO unavailable"
)
def test_openvino_compile_failure_does_not_fallback_by_default():
    with pytest.raises(RuntimeError, match="forced compile failure"):
        OpenVINODenseIsland(
            torch.ones(4, 8),
            max_batch_tokens=2,
            core=_CompileFailureCore(),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None, reason="OpenVINO unavailable"
)
def test_openvino_inference_failure_does_not_fallback_by_default():
    island = OpenVINODenseIsland(
        torch.ones(4, 8),
        max_batch_tokens=2,
        core=_InferenceFailureCore(),
    )

    with pytest.raises(RuntimeError, match="forced inference failure"):
        island(torch.ones(1, 8))


@pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None, reason="OpenVINO unavailable"
)
def test_openvino_non_gpu_execution_report_does_not_fallback_by_default():
    with pytest.raises(RuntimeError, match="unexpected execution devices"):
        OpenVINODenseIsland(
            torch.ones(4, 8),
            max_batch_tokens=2,
            core=_UnexpectedExecutionCore(),
        )


def test_openvino_xpu_fallback_rejects_cpu_input():
    island = OpenVINODenseIsland(
        torch.ones(4, 8),
        max_batch_tokens=2,
        fallback="xpu",
        core=_CompileFailureCore(),
    )

    with pytest.raises(RuntimeError, match="requires an XPU input"):
        island(torch.ones(1, 8))


def _hide_openvino_import(monkeypatch):
    original_import = builtins.__import__

    def import_without_openvino(name, *args, **kwargs):
        if name == "openvino" or name.startswith("openvino."):
            raise ImportError("forced OpenVINO import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_openvino)


def test_missing_openvino_uses_explicit_xpu_fallback(monkeypatch):
    device = _xpu_device()
    _hide_openvino_import(monkeypatch)
    weight = torch.eye(4, 8)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    island = OpenVINODenseIsland(weight, max_batch_tokens=2, fallback="xpu")
    result = island(hidden_states.to(device))

    assert result.backend == "xpu"
    assert "forced OpenVINO import failure" in result.fallback_reason
    torch.testing.assert_close(
        result.output, hidden_states[:, :4].to(device, torch.float16)
    )


def test_missing_openvino_does_not_fallback_by_default(monkeypatch):
    _hide_openvino_import(monkeypatch)

    with pytest.raises(RuntimeError, match="optional openvino package"):
        OpenVINODenseIsland(torch.ones(4, 8), max_batch_tokens=2)
