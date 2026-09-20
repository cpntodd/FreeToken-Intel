from __future__ import annotations

import pytest
import torch


def _reference(x, state, weight, indices):
    selected = state.index_select(0, indices.to(torch.int64))
    window = torch.cat((selected, x.unsqueeze(-1)), dim=-1)
    output = torch.nn.functional.silu((window * weight.unsqueeze(0)).sum(dim=-1))
    expected_state = state.clone()
    expected_state.index_copy_(0, indices.to(torch.int64), window[..., 1:])
    return output, expected_state


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sycl_causal_conv1d_decode_matches_reference_and_updates_state(dtype):
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode

    torch.manual_seed(7)
    x_cpu = torch.randn(2, 257, dtype=dtype)
    state_cpu = torch.randn(5, 257, 3, dtype=dtype)
    weight_cpu = torch.randn(257, 4, dtype=dtype)
    indices_cpu = torch.tensor([1, 4], dtype=torch.int32)
    expected_output, expected_state = _reference(
        x_cpu, state_cpu, weight_cpu, indices_cpu
    )

    x = x_cpu.to("xpu")
    state = state_cpu.to("xpu")
    output = causal_conv1d_decode(
        x,
        state,
        weight_cpu.to("xpu"),
        indices_cpu.to("xpu"),
    )
    torch.xpu.synchronize()

    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(output.cpu(), expected_output, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sycl_q8_0_matvec_matches_dequantized_reference(dtype):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequantize

    generator = torch.Generator().manual_seed(29)
    qweight = torch.randint(0, 256, (11, 68), dtype=torch.uint8, generator=generator)
    qweight[:, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(3, 64, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_Q8_0, dtype).reshape(11, 64).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_Q8_0)
    torch.xpu.synchronize()

    tolerance = 1e-5 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sycl_q4_k_matvec_matches_dequantized_reference(dtype):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q4_K, dequantize

    generator = torch.Generator().manual_seed(31)
    qweight = torch.randint(0, 256, (11, 288), dtype=torch.uint8, generator=generator)
    qweight[:, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    qweight[:, 2:4] = torch.tensor([0, 48], dtype=torch.uint8)
    x_cpu = torch.randn(3, 512, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_Q4_K, dtype).reshape(11, 512).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_Q4_K)
    torch.xpu.synchronize()

    tolerance = 1e-5 if dtype == torch.float32 else 8e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
