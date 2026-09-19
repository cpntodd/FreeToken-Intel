from __future__ import annotations

import functools
import importlib.util
from typing import TYPE_CHECKING

import torch

from .utils import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    if importlib.util.find_spec("tvm_ffi") is None:
        size = min(x.numel(), y.numel())
        mismatch = torch.nonzero(x[:size] != y[:size], as_tuple=False)
        return int(mismatch[0].item()) if mismatch.numel() else size
    return _load_radix_module().fast_compare_key(x, y)
