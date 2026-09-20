from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.accelerator import (
    CudaRuntime,
    XpuRuntime,
    discover_accelerators,
    probe_accelerators,
    resolve_runtime,
    validate_runtime_request,
)
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.server.args import parse_args


class FakeApi:
    def __init__(self, *, available: bool, names: tuple[str, ...], graph: bool = False):
        self._available = available
        self._names = names
        self.Stream = lambda: "stream"
        self.Event = lambda **kwargs: ("event", kwargs)
        if graph:
            self.CUDAGraph = object
            self.graph = lambda *_args, **_kwargs: None

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self._names)

    def get_device_properties(self, index):
        return SimpleNamespace(name=self._names[index], total_memory=(index + 1) * 1024)

    def set_device(self, device):
        self.selected = device

    def set_stream(self, stream):
        self.selected_stream = stream

    def synchronize(self, device=None):
        self.synchronized = device

    def empty_cache(self):
        self.emptied = True

    def mem_get_info(self, device=None):
        return 512, 1024


class FakeTorch:
    def __init__(self, *, cuda: FakeApi, xpu: FakeApi | None):
        self.cuda = cuda
        if xpu is not None:
            self.xpu = xpu

    @staticmethod
    def device(kind, index):
        return f"{kind}:{index}"


def test_auto_preserves_cuda_priority():
    fake = FakeTorch(
        cuda=FakeApi(available=True, names=("NVIDIA",), graph=True),
        xpu=FakeApi(available=True, names=("Intel Arc",)),
    )

    runtime = resolve_runtime("auto", fake)

    assert isinstance(runtime, CudaRuntime)
    assert runtime.capabilities().graph_capture is True


def test_xpu_is_selected_explicitly_and_reports_capabilities():
    fake = FakeTorch(
        cuda=FakeApi(available=False, names=()),
        xpu=FakeApi(available=True, names=("Intel Arc B580",)),
    )

    runtime = resolve_runtime("xpu", fake)
    capabilities = runtime.capabilities()

    assert isinstance(runtime, XpuRuntime)
    assert capabilities.device == "xpu:0"
    assert capabilities.name == "Intel Arc B580"
    assert capabilities.total_memory == 1024
    assert capabilities.graph_capture is False


def test_explicit_unavailable_backend_never_falls_back():
    fake = FakeTorch(
        cuda=FakeApi(available=True, names=("NVIDIA",)),
        xpu=FakeApi(available=False, names=()),
    )

    with pytest.raises(RuntimeError, match="xpu: unavailable"):
        resolve_runtime("xpu", fake)


def test_discovery_reports_both_backend_namespaces():
    fake = FakeTorch(
        cuda=FakeApi(available=True, names=("NVIDIA",)),
        xpu=FakeApi(available=True, names=("Intel Arc B580", "Intel Arc A770")),
    )

    devices = discover_accelerators(fake)

    assert [(device.device, device.name) for device in devices] == [
        ("cuda:0", "NVIDIA"),
        ("xpu:0", "Intel Arc B580"),
        ("xpu:1", "Intel Arc A770"),
    ]


def test_probe_reports_unavailable_and_available_backends():
    fake = FakeTorch(
        cuda=FakeApi(available=False, names=()),
        xpu=FakeApi(available=True, names=("Intel Arc B580",)),
    )

    discovery = probe_accelerators(fake)

    assert [backend.status for backend in discovery.backends] == [
        "unavailable",
        "available",
    ]
    assert (
        discovery.backends[0].message == "PyTorch reports this backend as unavailable"
    )
    assert discovery.backends[1].device_count == 1
    assert [device.device for device in discovery.devices] == ["xpu:0"]


def test_probe_surfaces_backend_initialization_errors():
    fake = FakeTorch(cuda=FakeApi(available=False, names=()), xpu=None)

    discovery = probe_accelerators(fake)

    xpu = discovery.backends[1]
    assert xpu.status == "error"
    assert "does not include the XPU backend" in xpu.message
    assert discovery.devices == ()


def test_probe_preserves_devices_when_another_device_probe_fails():
    xpu = FakeApi(available=True, names=("Intel Arc B580", "Intel Arc A770"))
    original_get_device_properties = xpu.get_device_properties

    def fail_second_device(index):
        if index == 1:
            raise RuntimeError("properties query failed")
        return original_get_device_properties(index)

    xpu.get_device_properties = fail_second_device
    fake = FakeTorch(cuda=FakeApi(available=False, names=()), xpu=xpu)

    discovery = probe_accelerators(fake)

    assert [device.device for device in discovery.devices] == ["xpu:0"]
    assert discovery.backends[1].status == "partial"
    assert discovery.backends[1].device_count == 1
    assert "xpu:1: RuntimeError: properties query failed" in (
        discovery.backends[1].message or ""
    )


def test_missing_xpu_api_has_a_clear_error():
    fake = FakeTorch(cuda=FakeApi(available=False, names=()), xpu=None)

    with pytest.raises(RuntimeError, match="does not include the XPU backend"):
        resolve_runtime("xpu", fake)


def test_invalid_accelerator_name_is_rejected():
    fake = FakeTorch(cuda=FakeApi(available=False, names=()), xpu=None)

    with pytest.raises(ValueError, match="auto, cuda, xpu"):
        resolve_runtime("vulkan", fake)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "has_cuda_device_ids", "message"),
    [
        (2, False, "single-GPU"),
        (1, True, "CUDA identifiers"),
    ],
)
def test_xpu_rejects_unsupported_worker_topologies(
    tensor_parallel_size, has_cuda_device_ids, message
):
    runtime = XpuRuntime(
        FakeTorch(
            cuda=FakeApi(available=False, names=()),
            xpu=FakeApi(available=True, names=("Intel Arc",)),
        )
    )

    with pytest.raises(ValueError, match=message):
        validate_runtime_request(
            runtime,
            tensor_parallel_size=tensor_parallel_size,
            has_cuda_device_ids=has_cuda_device_ids,
        )


def test_engine_config_accepts_explicit_single_gpu_xpu():
    config = EngineConfig(
        model_path="/models/test",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
        accelerator="xpu",
    )

    assert config.accelerator == "xpu"


def test_engine_config_rejects_xpu_tensor_parallelism():
    with pytest.raises(ValueError, match="single-GPU"):
        EngineConfig(
            model_path="/models/test",
            tp_info=DistributedInfo(rank=0, size=2),
            dtype=torch.float16,
            accelerator="xpu",
        )


def test_cli_exposes_accelerator_selection():
    args, run_shell = parse_args(
        ["--model", "/models/test", "--dtype", "float16", "--accelerator", "xpu"]
    )

    assert args.accelerator == "xpu"
    assert run_shell is False
