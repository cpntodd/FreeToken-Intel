from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

Activation = Literal["none", "gelu", "silu"]


@dataclass(frozen=True)
class OpenVINOExecutionInfo:
    requested_device: str
    execution_devices: tuple[str, ...]
    full_device_name: str


@dataclass(frozen=True)
class OpenVINOIslandResult:
    output: torch.Tensor
    input_copy_seconds: float
    inference_seconds: float
    output_copy_seconds: float


class OpenVINODenseIsland:
    """A bounded dense OpenVINO GPU compute island.

    The island deliberately uses host staging while PyTorch XPU/OpenVINO USM
    interoperability remains unproven. Copy timings are returned to make that
    cost observable. Compilation and execution reject CPU participation.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        max_batch_tokens: int,
        activation: Activation = "none",
        device: str = "GPU",
        core: Any | None = None,
    ) -> None:
        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        if activation not in {"none", "gelu", "silu"}:
            raise ValueError("activation must be one of: none, gelu, silu")
        if not device.upper().startswith("GPU"):
            raise ValueError("OpenVINO compute islands require an explicit GPU device")
        if weight.ndim != 2:
            raise ValueError("weight must have shape [out_features, in_features]")
        if bias is not None and tuple(bias.shape) != (weight.shape[0],):
            raise ValueError("bias must have shape [out_features]")

        try:
            import openvino as ov
            from openvino import opset13 as ops
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO compute islands require the optional openvino package"
            ) from exc

        self.max_batch_tokens = max_batch_tokens
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.device = device
        self._core = core if core is not None else ov.Core()

        weight_np = np.ascontiguousarray(_as_fp16_numpy(weight).T)
        parameter = ops.parameter(
            [-1, self.in_features], ov.Type.f16, name="hidden_states"
        )
        output = ops.matmul(parameter, ops.constant(weight_np), False, False)
        if bias is not None:
            output = ops.add(output, ops.constant(_as_fp16_numpy(bias)))
        if activation == "gelu":
            output = ops.gelu(output, "erf")
        elif activation == "silu":
            output = ops.multiply(output, ops.sigmoid(output))

        model = ov.Model([output], [parameter], "freetoken_dense_island")
        self._compiled = self._core.compile_model(
            model, device, {"PERFORMANCE_HINT": "LATENCY"}
        )
        execution_devices = tuple(self._compiled.get_property("EXECUTION_DEVICES"))
        if not execution_devices or any(
            not execution_device.upper().startswith("GPU")
            for execution_device in execution_devices
        ):
            raise RuntimeError(
                "OpenVINO compiled the island for unexpected execution devices: "
                f"{execution_devices!r}"
            )
        self.execution_info = OpenVINOExecutionInfo(
            requested_device=device,
            execution_devices=execution_devices,
            full_device_name=str(self._core.get_property(device, "FULL_DEVICE_NAME")),
        )

    def __call__(self, hidden_states: torch.Tensor) -> OpenVINOIslandResult:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.in_features:
            raise ValueError(
                "hidden_states must have shape "
                f"[tokens, {self.in_features}], got {tuple(hidden_states.shape)}"
            )
        if not 0 < hidden_states.shape[0] <= self.max_batch_tokens:
            raise ValueError(
                f"token count must be between 1 and {self.max_batch_tokens}"
            )

        source_device = hidden_states.device
        started = perf_counter()
        host_input = _as_fp16_numpy(hidden_states)
        input_copy_seconds = perf_counter() - started

        started = perf_counter()
        result = self._compiled({"hidden_states": host_input})[0]
        inference_seconds = perf_counter() - started

        started = perf_counter()
        output = torch.from_numpy(np.array(result, copy=True)).to(source_device)
        if source_device.type == "xpu":
            torch.xpu.synchronize(source_device)
        output_copy_seconds = perf_counter() - started
        return OpenVINOIslandResult(
            output=output,
            input_copy_seconds=input_copy_seconds,
            inference_seconds=inference_seconds,
            output_copy_seconds=output_copy_seconds,
        )


def _as_fp16_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
