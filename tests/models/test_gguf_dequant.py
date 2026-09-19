from __future__ import annotations

import gguf
import numpy as np
import torch
from freetoken.models.gguf.dequant import GGML_Q4_K, dequantize


def test_q4_k_dequant_matches_gguf_reference():
    generator = np.random.default_rng(1234)
    raw = generator.integers(0, 256, size=(7, 144), dtype=np.uint8)
    raw[:, 0:2] = np.array([0.25], dtype=np.float16).view(np.uint8)
    raw[:, 2:4] = np.array([0.125], dtype=np.float16).view(np.uint8)

    expected = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q4_K)
    actual = dequantize(torch.from_numpy(raw), GGML_Q4_K, torch.float32)

    torch.testing.assert_close(actual, torch.from_numpy(expected).reshape(-1))
