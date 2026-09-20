from __future__ import annotations

import torch


def causal_conv1d_decode_sycl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    try:
        from freetoken.kernel import _sycl_kernels
    except ImportError as error:
        raise RuntimeError(
            "the native SYCL extension is not installed; rebuild with "
            "FREETOKEN_ACCELERATOR=xpu and the oneAPI icpx compiler"
        ) from error
    return _sycl_kernels.causal_conv1d_decode(
        x,
        conv_state,
        weight,
        conv_state_indices.to(dtype=torch.int32),
    )


def q8_0_matvec_sycl(x: torch.Tensor, qweight: torch.Tensor) -> torch.Tensor:
    try:
        from freetoken.kernel import _sycl_kernels
    except ImportError as error:
        raise RuntimeError(
            "the native SYCL extension is not installed; rebuild with "
            "FREETOKEN_ACCELERATOR=xpu and the oneAPI icpx compiler"
        ) from error
    return _sycl_kernels.q8_0_matvec(x, qweight)


def q4_k_matvec_sycl(x: torch.Tensor, qweight: torch.Tensor) -> torch.Tensor:
    try:
        from freetoken.kernel import _sycl_kernels
    except ImportError as error:
        raise RuntimeError(
            "the native SYCL extension is not installed; rebuild with "
            "FREETOKEN_ACCELERATOR=xpu and the oneAPI icpx compiler"
        ) from error
    return _sycl_kernels.q4_k_matvec(x, qweight)


__all__ = ["causal_conv1d_decode_sycl", "q4_k_matvec_sycl", "q8_0_matvec_sycl"]
