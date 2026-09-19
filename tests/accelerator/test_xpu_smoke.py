from __future__ import annotations

import pytest
import torch
from freetoken.accelerator import resolve_runtime


@pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU unavailable"
)
def test_xpu_executes_matmul_on_selected_device():
    runtime = resolve_runtime("xpu")
    device = runtime.device(0)
    values = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)

    result = values @ values.T
    runtime.api.current_stream().synchronize()

    assert result.device.type == "xpu"
    assert result.sum().item() == pytest.approx(3680.0)


@pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU unavailable"
)
def test_xpu_executes_eager_q4_0_matmul():
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q4_0, dequantize

    rows, in_features = 8, 64
    packed = torch.arange(rows * (in_features // 32) * 18, dtype=torch.uint8).reshape(
        rows, -1
    )
    # Finite fp16 block scales; arbitrary bytes may encode NaN/Inf scales.
    blocks = packed.view(-1, 18)
    blocks[:, :2] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    x = torch.randn(2, in_features, dtype=torch.bfloat16)
    weight = dequantize(packed, GGML_Q4_0, x.dtype).reshape(rows, in_features)
    expected = x @ weight.T

    actual = fused_mul_mat_gguf(x.to("xpu"), packed.to("xpu"), GGML_Q4_0).cpu()
    torch.xpu.synchronize()

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)
