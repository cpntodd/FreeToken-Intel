from __future__ import annotations

import builtins

import pytest
import torch


def _reference(x, state, weight, indices):
    selected = state.index_select(0, indices.to(torch.int64))
    window = torch.cat((selected, x.unsqueeze(-1)), dim=-1)
    output = torch.nn.functional.silu((window * weight.unsqueeze(0)).sum(dim=-1))
    expected_state = state.clone()
    expected_state.index_copy_(0, indices.to(torch.int64), window[..., 1:])
    return output, expected_state


def test_sycl_wrapper_preserves_extension_abi_import_error(monkeypatch):
    from freetoken.kernel.sycl.causal_conv1d import pq2_0_matvec_sycl

    real_import = builtins.__import__

    def fail_extension_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "freetoken.kernel" and "_sycl_kernels" in fromlist:
            raise ImportError("undefined SYCL runtime symbol")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_extension_import)
    with pytest.raises(ImportError, match="undefined SYCL runtime symbol"):
        pq2_0_matvec_sycl(torch.empty((1, 128)), torch.empty((1, 34)))


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
    torch.testing.assert_close(
        output.cpu(), expected_output, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_q8_0_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequantize

    generator = torch.Generator().manual_seed(29)
    qweight = torch.randint(0, 256, (11, 68), dtype=torch.uint8, generator=generator)
    qweight[:, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 64, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_Q8_0, dtype).reshape(11, 64).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_Q8_0)
    torch.xpu.synchronize()

    tolerance = 2e-5 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_q4_k_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q4_K, dequantize

    generator = torch.Generator().manual_seed(31)
    qweight = torch.randint(0, 256, (11, 288), dtype=torch.uint8, generator=generator)
    qweight[:, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    qweight[:, 2:4] = torch.tensor([0, 48], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_Q4_K, dtype).reshape(11, 512).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_Q4_K)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 8e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_q2_k_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q2_K, dequantize

    generator = torch.Generator().manual_seed(37)
    qweight = torch.randint(0, 256, (11, 168), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 84)
    blocks[:, :, 80:82] = torch.tensor([0, 52], dtype=torch.uint8)
    blocks[:, :, 82:84] = torch.tensor([0, 48], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_Q2_K, dtype).reshape(11, 512).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_Q2_K)
    torch.xpu.synchronize()

    tolerance = 1e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_pq2_0_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_PQ2_0, dequantize

    generator = torch.Generator().manual_seed(43)
    qweight = torch.randint(0, 256, (11, 136), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 4, 34)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_PQ2_0, dtype).reshape(11, 512).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_PQ2_0)
    torch.xpu.synchronize()

    tolerance = 2e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_q3_k_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q3_K, dequantize

    generator = torch.Generator().manual_seed(41)
    qweight = torch.randint(0, 256, (11, 220), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 110)
    blocks[:, :, 108:110] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_Q3_K, dtype).reshape(11, 512).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_Q3_K)
    torch.xpu.synchronize()

    tolerance = 1e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_q5_k_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q5_K, dequantize

    generator = torch.Generator().manual_seed(43)
    qweight = torch.randint(0, 256, (11, 352), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 176)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    blocks[:, :, 2:4] = torch.tensor([0, 48], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_Q5_K, dtype).reshape(11, 512).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_Q5_K)
    torch.xpu.synchronize()

    tolerance = 1e-5 if dtype == torch.float32 else 8e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 3, 4, 5])
def test_sycl_q6_k_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q6_K, dequantize

    generator = torch.Generator().manual_seed(83)
    qweight = torch.randint(0, 256, (11, 420), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 210)
    blocks[:, :, 208:210] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    weight = dequantize(qweight, GGML_Q6_K, torch.float32).reshape(11, 512)
    expected = (x_cpu.float() @ weight.T).to(dtype)

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_Q6_K)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
def test_sycl_q6_k_matvec_rejects_invalid_geometry():
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q6_K

    x = torch.randn(1, 300, device="xpu")
    qweight = torch.empty((2, 210), dtype=torch.uint8, device="xpu")
    with pytest.raises(RuntimeError, match="divisible by 256"):
        fused_mul_mat_gguf(x, qweight, GGML_Q6_K)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
def test_sycl_q6_k_matvec_rejects_unsupported_input_dtype():
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q6_K

    x = torch.randn(1, 256, dtype=torch.float16, device="xpu")
    qweight = torch.empty((2, 210), dtype=torch.uint8, device="xpu")
    with pytest.raises(RuntimeError, match="supports float32 and bfloat16"):
        fused_mul_mat_gguf(x, qweight, GGML_Q6_K)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
def test_fused_mul_mat_gguf_dispatches_q6_k_to_sycl(monkeypatch):
    from freetoken.kernel.sycl import causal_conv1d
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q6_K

    x = torch.empty((1, 256), device="xpu")
    qweight = torch.empty((2, 210), dtype=torch.uint8, device="xpu")
    expected = torch.empty((1, 2), device="xpu")
    calls = []

    def dispatch(actual_x, actual_qweight):
        calls.append((actual_x, actual_qweight))
        return expected

    monkeypatch.setattr(causal_conv1d, "q6_k_matvec_sycl", dispatch)
    actual = fused_mul_mat_gguf(x, qweight, GGML_Q6_K)

    assert actual is expected
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is qweight


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_iq3_xxs_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_IQ3_XXS, dequantize

    generator = torch.Generator().manual_seed(47)
    qweight = torch.randint(0, 256, (11, 196), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 98)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_IQ3_XXS, dtype).reshape(11, 512).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_IQ3_XXS)
    torch.xpu.synchronize()

    tolerance = 1e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_iq2_s_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_IQ2_S, dequantize

    generator = torch.Generator().manual_seed(53)
    qweight = torch.randint(0, 256, (11, 164), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 82)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    expected = x_cpu @ dequantize(qweight, GGML_IQ2_S, dtype).reshape(11, 512).T

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_IQ2_S)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_iq3_s_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_IQ3_S, dequantize

    generator = torch.Generator().manual_seed(59)
    qweight = torch.randint(0, 256, (11, 220), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 110)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    weight = dequantize(qweight, GGML_IQ3_S, torch.float32).reshape(11, 512)
    expected = (x_cpu.float() @ weight.T).to(dtype)

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_IQ3_S)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_iq2_xxs_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_IQ2_XXS, dequantize

    generator = torch.Generator().manual_seed(61)
    qweight = torch.randint(0, 256, (11, 132), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 66)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    weight = dequantize(qweight, GGML_IQ2_XXS, torch.float32).reshape(11, 512)
    expected = (x_cpu.float() @ weight.T).to(dtype)

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_IQ2_XXS)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_iq2_xs_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_IQ2_XS, dequantize

    generator = torch.Generator().manual_seed(67)
    qweight = torch.randint(0, 256, (11, 148), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 74)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    weight = dequantize(qweight, GGML_IQ2_XS, torch.float32).reshape(11, 512)
    expected = (x_cpu.float() @ weight.T).to(dtype)

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_IQ2_XS)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_iq4_xs_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_IQ4_XS, dequantize

    generator = torch.Generator().manual_seed(71)
    qweight = torch.randint(0, 256, (11, 272), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 136)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    weight = dequantize(qweight, GGML_IQ4_XS, torch.float32).reshape(11, 512)
    expected = (x_cpu.float() @ weight.T).to(dtype)

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_IQ4_XS)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_iq1_s_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_IQ1_S, dequantize

    generator = torch.Generator().manual_seed(73)
    qweight = torch.randint(0, 256, (11, 100), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 50)
    blocks[:, :, :2] = torch.tensor([0, 52], dtype=torch.uint8)
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    weight = dequantize(qweight, GGML_IQ1_S, torch.float32).reshape(11, 512)
    expected = (x_cpu.float() @ weight.T).to(dtype)

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_IQ1_S)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [3, 5])
def test_sycl_iq1_m_matvec_matches_dequantized_reference(dtype, batch):
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_IQ1_M, dequantize

    generator = torch.Generator().manual_seed(79)
    qweight = torch.randint(0, 256, (11, 112), dtype=torch.uint8, generator=generator)
    blocks = qweight.view(11, 2, 56)
    blocks[:, :, 48:56] = 0
    blocks[:, :, 53] = 0x40
    blocks[:, :, 55] = 0x30
    x_cpu = torch.randn(batch, 512, dtype=dtype, generator=generator)
    weight = dequantize(qweight, GGML_IQ1_M, torch.float32).reshape(11, 512)
    expected = (x_cpu.float() @ weight.T).to(dtype)

    actual = fused_mul_mat_gguf(x_cpu.to("xpu"), qweight.to("xpu"), GGML_IQ1_M)
    torch.xpu.synchronize()

    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
