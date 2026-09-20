from __future__ import annotations

import gguf
import numpy as np
import pytest
import torch
from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_IQ1_M,
    GGML_IQ1_S,
    GGML_IQ2_S,
    GGML_IQ2_XS,
    GGML_IQ2_XXS,
    GGML_IQ3_S,
    GGML_IQ3_XXS,
    GGML_IQ4_XS,
    GGML_PQ2_0,
    GGML_Q2_K,
    GGML_Q3_K,
    GGML_Q4_1,
    GGML_Q4_K,
    GGML_Q5_0,
    GGML_Q5_1,
    GGML_Q5_K,
    GGML_Q8_0,
    dequantize,
    row_bytes,
)


def test_q4_k_dequant_matches_gguf_reference():
    generator = np.random.default_rng(1234)
    raw = generator.integers(0, 256, size=(7, 144), dtype=np.uint8)
    raw[:, 0:2] = np.array([0.25], dtype=np.float16).view(np.uint8)
    raw[:, 2:4] = np.array([0.125], dtype=np.float16).view(np.uint8)

    expected = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q4_K)
    actual = dequantize(torch.from_numpy(raw), GGML_Q4_K, torch.float32)

    torch.testing.assert_close(actual, torch.from_numpy(expected).reshape(-1))


def test_pq2_0_dequant_uses_prism_128_value_blocks():
    raw = torch.zeros((1, 34), dtype=torch.uint8)
    raw[0, :2] = torch.tensor([0, 56], dtype=torch.uint8)  # fp16 0.5
    raw[0, 2:] = 0b11_10_01_00

    actual = dequantize(raw, GGML_PQ2_0, torch.float32)
    expected = torch.tensor([-0.5, 0.0, 0.5, 1.0] * 32)

    torch.testing.assert_close(actual, expected)
    assert BLOCK_SHAPE[GGML_PQ2_0] == (128, 34)
    assert row_bytes(5120, GGML_PQ2_0) == 1360


@pytest.mark.parametrize(
    ("quant_type", "block_size"),
    [
        (GGML_Q2_K, 84),
        (GGML_Q3_K, 110),
        (GGML_Q4_1, 20),
        (GGML_Q5_0, 22),
        (GGML_Q5_1, 24),
        (GGML_Q5_K, 176),
        (GGML_IQ1_S, 50),
        (GGML_IQ1_M, 56),
        (GGML_IQ2_XXS, 66),
        (GGML_IQ2_XS, 74),
        (GGML_IQ2_S, 82),
        (GGML_IQ3_XXS, 98),
        (GGML_IQ3_S, 110),
        (GGML_IQ4_XS, 136),
        (GGML_Q8_0, 34),
    ],
)
def test_additional_quant_dequant_matches_gguf_reference(quant_type, block_size):
    generator = np.random.default_rng(5678 + quant_type)
    raw = generator.integers(0, 256, size=(7, block_size), dtype=np.uint8)
    if quant_type == GGML_Q2_K:
        raw[:, 80:82] = np.array([0.25], dtype=np.float16).view(np.uint8)
        raw[:, 82:84] = np.array([0.125], dtype=np.float16).view(np.uint8)
    elif quant_type == GGML_Q3_K:
        raw[:, 108:110] = np.array([0.25], dtype=np.float16).view(np.uint8)
    elif quant_type == GGML_Q5_0:
        raw[:, 0:2] = np.array([0.25], dtype=np.float16).view(np.uint8)
    elif quant_type == GGML_Q4_1:
        raw[:, 0:4] = np.array([0.25, 0.125], dtype=np.float16).view(np.uint8)
    elif quant_type == GGML_Q5_1:
        raw[:, 0:4] = np.array([0.25, 0.125], dtype=np.float16).view(np.uint8)
    elif (
        quant_type
        in {
            GGML_IQ2_XXS,
            GGML_IQ2_XS,
            GGML_IQ2_S,
            GGML_IQ3_XXS,
            GGML_IQ3_S,
            GGML_IQ4_XS,
        }
        or quant_type == GGML_IQ1_S
    ):
        raw[:, 0:2] = np.array([0.25], dtype=np.float16).view(np.uint8)
    elif quant_type == GGML_IQ1_M:
        raw[:, 48:56] = 0
        raw[:, 53] = 0x40
        raw[:, 55] = 0x30
    else:
        raw[:, 0:2] = np.array([0.25], dtype=np.float16).view(np.uint8)
        raw[:, 2:4] = np.array([0.125], dtype=np.float16).view(np.uint8)

    gguf_type = gguf.GGMLQuantizationType(quant_type)
    expected = gguf.dequantize(raw, gguf_type)
    actual = dequantize(torch.from_numpy(raw), quant_type, torch.float32)

    torch.testing.assert_close(actual, torch.from_numpy(expected).reshape(-1))


def test_qwen35_quant_types_have_exact_ggml_block_sizes():
    expected = {
        GGML_Q2_K: 84,
        GGML_Q3_K: 110,
        GGML_Q5_K: 176,
        GGML_IQ2_XXS: 66,
        GGML_IQ2_XS: 74,
        GGML_IQ3_XXS: 98,
        GGML_IQ1_S: 50,
        GGML_IQ3_S: 110,
        GGML_IQ2_S: 82,
        GGML_IQ4_XS: 136,
        GGML_IQ1_M: 56,
    }

    for quant_type, type_size in expected.items():
        assert BLOCK_SHAPE[quant_type] == (256, type_size)
        assert row_bytes(5120, quant_type) == 20 * type_size


def test_q4_1_has_standard_ggml_block_geometry():
    assert BLOCK_SHAPE[GGML_Q4_1] == (32, 20)
    assert row_bytes(4096, GGML_Q4_1) == 2560


def test_q5_0_has_standard_ggml_block_geometry():
    assert BLOCK_SHAPE[GGML_Q5_0] == (32, 22)
    assert row_bytes(4096, GGML_Q5_0) == 2816


def test_q5_1_has_standard_ggml_block_geometry():
    assert BLOCK_SHAPE[GGML_Q5_1] == (32, 24)
    assert row_bytes(4096, GGML_Q5_1) == 3072
