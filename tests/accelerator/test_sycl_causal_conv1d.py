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
