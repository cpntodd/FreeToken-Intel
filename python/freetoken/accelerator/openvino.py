from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

Activation = Literal["none", "gelu", "silu"]
FallbackPolicy = Literal["none", "xpu"]
ExecutionBackend = Literal["openvino", "xpu"]


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
    backend: ExecutionBackend = "openvino"
    fallback_reason: str | None = None


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
        fallback: FallbackPolicy = "none",
        core: Any | None = None,
    ) -> None:
        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        if activation not in {"none", "gelu", "silu"}:
            raise ValueError("activation must be one of: none, gelu, silu")
        if fallback not in {"none", "xpu"}:
            raise ValueError("fallback must be one of: none, xpu")
        if not device.upper().startswith("GPU"):
            raise ValueError("OpenVINO compute islands require an explicit GPU device")
        if weight.ndim != 2:
            raise ValueError("weight must have shape [out_features, in_features]")
        if bias is not None and tuple(bias.shape) != (weight.shape[0],):
            raise ValueError("bias must have shape [out_features]")

        self.max_batch_tokens = max_batch_tokens
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.device = device
        self.activation = activation
        self.fallback = fallback
        self._fallback_weight_source = weight.detach() if fallback == "xpu" else None
        self._fallback_bias_source = (
            bias.detach() if fallback == "xpu" and bias is not None else None
        )
        self._fallback_weight: torch.Tensor | None = None
        self._fallback_bias: torch.Tensor | None = None
        self._compiled = None
        self._fallback_reason: str | None = None
        self.execution_info: OpenVINOExecutionInfo | None = None

        try:
            import openvino as ov
            from openvino import opset13 as ops
        except ImportError as exc:
            if fallback != "xpu":
                raise RuntimeError(
                    "OpenVINO compute islands require the optional openvino package"
                ) from exc
            self._fallback_reason = f"OpenVINO import failed: {exc}"
            return

        try:
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
            compiled = self._core.compile_model(
                model, device, {"PERFORMANCE_HINT": "LATENCY"}
            )
            execution_devices = tuple(compiled.get_property("EXECUTION_DEVICES"))
            if not execution_devices or any(
                not execution_device.upper().startswith("GPU")
                for execution_device in execution_devices
            ):
                raise RuntimeError(
                    "OpenVINO compiled the island for unexpected execution devices: "
                    f"{execution_devices!r}"
                )
            self._compiled = compiled
            self.execution_info = OpenVINOExecutionInfo(
                requested_device=device,
                execution_devices=execution_devices,
                full_device_name=str(
                    self._core.get_property(device, "FULL_DEVICE_NAME")
                ),
            )
        except Exception as exc:
            if fallback != "xpu":
                raise
            self._fallback_reason = (
                f"OpenVINO compilation failed: {type(exc).__name__}: {exc}"
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

        if self._compiled is None:
            return self._run_xpu_fallback(
                hidden_states,
                self._fallback_reason or "OpenVINO did not produce a compiled model",
            )

        source_device = hidden_states.device
        try:
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
        except Exception as exc:
            if self.fallback != "xpu":
                raise
            return self._run_xpu_fallback(
                hidden_states,
                f"OpenVINO inference failed: {type(exc).__name__}: {exc}",
            )

    def _run_xpu_fallback(
        self, hidden_states: torch.Tensor, reason: str
    ) -> OpenVINOIslandResult:
        if self.fallback != "xpu":
            raise RuntimeError(reason)
        if hidden_states.device.type != "xpu" or not torch.xpu.is_available():
            raise RuntimeError(
                "the explicit XPU fallback requires an XPU input and available XPU"
            )
        if self._fallback_weight_source is None:
            raise RuntimeError("XPU fallback weights were not retained")

        source_device = hidden_states.device
        started = perf_counter()
        x = hidden_states.detach().to(dtype=torch.float16)
        if (
            self._fallback_weight is None
            or self._fallback_weight.device != source_device
        ):
            self._fallback_weight = self._fallback_weight_source.to(
                device=source_device, dtype=torch.float16
            )
        if self._fallback_bias_source is not None and (
            self._fallback_bias is None or self._fallback_bias.device != source_device
        ):
            self._fallback_bias = self._fallback_bias_source.to(
                device=source_device, dtype=torch.float16
            )
        torch.xpu.synchronize(source_device)
        input_copy_seconds = perf_counter() - started

        started = perf_counter()
        with torch.inference_mode():
            output = torch.nn.functional.linear(
                x, self._fallback_weight, self._fallback_bias
            )
            if self.activation == "gelu":
                output = torch.nn.functional.gelu(output, approximate="none")
            elif self.activation == "silu":
                output = torch.nn.functional.silu(output)
        torch.xpu.synchronize(source_device)
        inference_seconds = perf_counter() - started
        return OpenVINOIslandResult(
            output=output,
            input_copy_seconds=input_copy_seconds,
            inference_seconds=inference_seconds,
            output_copy_seconds=0.0,
            backend="xpu",
            fallback_reason=reason,
        )


def _as_fp16_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
