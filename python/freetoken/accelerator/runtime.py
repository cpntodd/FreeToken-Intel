from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch

AcceleratorKind = Literal["cuda", "xpu"]
AcceleratorProbeState = Literal["available", "unavailable", "error", "partial"]


def supports_pinned_host_memory(torch_module: Any = torch) -> bool:
    """Return whether an available CUDA or XPU runtime can use pinned host buffers."""
    for name in ("cuda", "xpu"):
        backend = getattr(torch_module, name, None)
        is_available = getattr(backend, "is_available", None)
        if callable(is_available) and is_available():
            return True
    return False


@dataclass(frozen=True)
class AcceleratorCapabilities:
    kind: AcceleratorKind
    index: int
    name: str
    total_memory: int
    device_id: str | None
    uuid: str | None
    driver_version: str | None
    platform_name: str | None
    graph_capture: bool
    streams: bool
    events: bool

    @property
    def device(self) -> str:
        return f"{self.kind}:{self.index}"


@dataclass(frozen=True)
class AcceleratorBackendStatus:
    kind: AcceleratorKind
    status: AcceleratorProbeState
    device_count: int
    message: str | None


@dataclass(frozen=True)
class AcceleratorDiscovery:
    devices: tuple[AcceleratorCapabilities, ...]
    backends: tuple[AcceleratorBackendStatus, ...]


class AcceleratorRuntime(ABC):
    kind: AcceleratorKind

    def __init__(self, torch_module: Any = torch):
        self.torch = torch_module

    @property
    @abstractmethod
    def api(self) -> Any: ...

    def is_available(self) -> bool:
        is_available = getattr(self.api, "is_available", None)
        return bool(is_available is not None and is_available())

    def device_count(self) -> int:
        return int(self.api.device_count()) if self.is_available() else 0

    def device(self, index: int = 0):
        self._validate_index(index)
        return self.torch.device(self.kind, index)

    def set_device(self, device) -> None:
        self.api.set_device(device)

    def stream(self):
        return self.api.Stream()

    def stream_context(self, stream):
        return self.api.stream(stream)

    def set_stream(self, stream) -> None:
        self.api.set_stream(stream)

    def current_stream(self):
        return self.api.current_stream()

    def event(self, *, enable_timing: bool = False):
        return self.api.Event(enable_timing=enable_timing)

    def synchronize(self, device=None) -> None:
        self.api.synchronize(device)

    def empty_cache(self) -> None:
        self.api.empty_cache()

    def reset_peak_memory_stats(self, device=None) -> None:
        reset = getattr(self.api, "reset_peak_memory_stats", None)
        if reset is not None:
            reset(device)

    def is_initialized(self) -> bool:
        is_initialized = getattr(self.api, "is_initialized", None)
        return bool(is_initialized is not None and is_initialized())

    def memory_info(self, device=None) -> tuple[int, int]:
        free, total = self.api.mem_get_info(device)
        return int(free), int(total)

    def memory_reserved(self, device=None) -> int:
        return int(self.api.memory_reserved(device))

    def capabilities(self, index: int = 0) -> AcceleratorCapabilities:
        self._validate_index(index)
        props = self.api.get_device_properties(index)
        return AcceleratorCapabilities(
            kind=self.kind,
            index=index,
            name=str(props.name),
            total_memory=int(props.total_memory),
            device_id=_format_device_id(getattr(props, "device_id", None)),
            uuid=_optional_string(getattr(props, "uuid", None)),
            driver_version=_optional_string(getattr(props, "driver_version", None)),
            platform_name=_optional_string(getattr(props, "platform_name", None)),
            graph_capture=self._supports_graph_capture(),
            streams=hasattr(self.api, "Stream") and hasattr(self.api, "set_stream"),
            events=hasattr(self.api, "Event"),
        )

    def _validate_index(self, index: int) -> None:
        count = self.device_count()
        if not 0 <= index < count:
            raise RuntimeError(
                f"cannot use {self.kind}:{index}: only {count} device(s) available"
            )

    def _supports_graph_capture(self) -> bool:
        return hasattr(self.api, "CUDAGraph") and hasattr(self.api, "graph")


class CudaRuntime(AcceleratorRuntime):
    kind: AcceleratorKind = "cuda"

    @property
    def api(self) -> Any:
        return self.torch.cuda


class XpuRuntime(AcceleratorRuntime):
    kind: AcceleratorKind = "xpu"

    @property
    def api(self) -> Any:
        api = getattr(self.torch, "xpu", None)
        if api is None:
            raise RuntimeError("this PyTorch build does not include the XPU backend")
        return api

    def _supports_graph_capture(self) -> bool:
        return False


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _format_device_id(value: Any) -> str | None:
    if value is None:
        return None
    return f"0x{value:04X}" if isinstance(value, int) else str(value)


def _runtime(kind: AcceleratorKind, torch_module: Any = torch) -> AcceleratorRuntime:
    if kind == "cuda":
        return CudaRuntime(torch_module)
    if kind == "xpu":
        return XpuRuntime(torch_module)
    raise ValueError(f"unsupported accelerator {kind!r}")


def resolve_runtime(
    preferred: str = "auto", torch_module: Any = torch
) -> AcceleratorRuntime:
    if preferred not in {"auto", "cuda", "xpu"}:
        raise ValueError("accelerator must be one of: auto, cuda, xpu")

    kinds: tuple[AcceleratorKind, ...] = (
        ("cuda", "xpu") if preferred == "auto" else (preferred,)
    )  # type: ignore[assignment]
    failures: list[str] = []
    for kind in kinds:
        try:
            runtime = _runtime(kind, torch_module)
            if runtime.is_available() and runtime.device_count() > 0:
                return runtime
            failures.append(f"{kind}: unavailable")
        except (AttributeError, RuntimeError) as exc:
            failures.append(f"{kind}: {exc}")
    raise RuntimeError(
        "no requested accelerator is available (" + "; ".join(failures) + ")"
    )


def probe_accelerators(
    torch_module: Any = torch,
) -> AcceleratorDiscovery:
    discovered: list[AcceleratorCapabilities] = []
    backends: list[AcceleratorBackendStatus] = []
    for kind in ("cuda", "xpu"):
        try:
            runtime = _runtime(kind, torch_module)
            if not runtime.is_available():
                backends.append(
                    AcceleratorBackendStatus(
                        kind=kind,
                        status="unavailable",
                        device_count=0,
                        message="PyTorch reports this backend as unavailable",
                    )
                )
                continue

            device_count = runtime.device_count()
            if device_count <= 0:
                backends.append(
                    AcceleratorBackendStatus(
                        kind=kind,
                        status="unavailable",
                        device_count=0,
                        message="The runtime reports no devices",
                    )
                )
                continue

            backend_devices: list[AcceleratorCapabilities] = []
            failures: list[str] = []
            for index in range(device_count):
                try:
                    backend_devices.append(runtime.capabilities(index))
                # A failed device query must not hide other devices or backends.
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{kind}:{index}: {type(exc).__name__}: {exc}")

            discovered.extend(backend_devices)
            if failures:
                state: AcceleratorProbeState = "partial" if backend_devices else "error"
                message = (
                    f"{len(failures)} of {device_count} device probe(s) failed: "
                    + "; ".join(failures)
                )
            else:
                state = "available"
                message = None
            backends.append(
                AcceleratorBackendStatus(
                    kind=kind,
                    status=state,
                    device_count=len(backend_devices),
                    message=message,
                )
            )
        # Keep discovery best-effort so one broken driver cannot hide the other.
        except Exception as exc:  # noqa: BLE001
            backends.append(
                AcceleratorBackendStatus(
                    kind=kind,
                    status="error",
                    device_count=0,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
    return AcceleratorDiscovery(tuple(discovered), tuple(backends))


def discover_accelerators(
    torch_module: Any = torch,
) -> tuple[AcceleratorCapabilities, ...]:
    return probe_accelerators(torch_module).devices


def validate_runtime_request(
    runtime: AcceleratorRuntime,
    *,
    tensor_parallel_size: int,
    has_cuda_device_ids: bool,
) -> None:
    if runtime.kind == "xpu" and tensor_parallel_size != 1:
        raise ValueError("the XPU backend currently supports single-GPU inference only")
    if runtime.kind == "xpu" and has_cuda_device_ids:
        raise ValueError(
            "--gpu contains CUDA identifiers and cannot be used with --accelerator xpu"
        )
