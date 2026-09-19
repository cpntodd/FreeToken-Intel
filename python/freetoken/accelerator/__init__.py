from .runtime import (
    AcceleratorCapabilities,
    AcceleratorRuntime,
    CudaRuntime,
    XpuRuntime,
    discover_accelerators,
    resolve_runtime,
    validate_runtime_request,
)


def __getattr__(name: str):
    if name in {
        "OpenVINODenseIsland",
        "OpenVINOExecutionInfo",
        "OpenVINOIslandResult",
    }:
        from . import openvino

        return getattr(openvino, name)
    raise AttributeError(name)

__all__ = [
    "AcceleratorCapabilities",
    "AcceleratorRuntime",
    "CudaRuntime",
    "OpenVINODenseIsland",
    "OpenVINOExecutionInfo",
    "OpenVINOIslandResult",
    "XpuRuntime",
    "discover_accelerators",
    "resolve_runtime",
    "validate_runtime_request",
]
