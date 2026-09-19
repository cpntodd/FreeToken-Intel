"""GGML block-quant dequantization in pure torch (the formats this repo's GGUF
checkpoints use: Q4_0, Q6_K, plus trivial F32/F16/BF16).

This is the *reference / CPU* path, NOT the engine's hot path: GGUF weights stay
packed and are dequantized inside the borrowed ggml CUDA kernels (see
``freetoken.kernel.gguf``). These routines are used only to (a) materialize the few
dense F32/F16 tensors at load (norms, scales, router) via :func:`dequantize`, and
(b) cross-check the CUDA kernels in tests. The ``BLOCK_SHAPE`` table and
:func:`row_bytes` are the type metadata the packed (kernel) path also relies on.

Each ``dequant_*`` takes the raw little-endian bytes as a ``uint8`` tensor whose
final axis spans whole blocks, and returns the values in *storage order* (ggml's
fastest axis first); the caller reshapes to the torch shape (``dims[::-1]``). The
math mirrors ``ggml-quants.c``.
"""

from __future__ import annotations

import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_IQ2_XXS = 16
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
GGML_IQ1_S = 19
GGML_IQ3_S = 21
GGML_IQ2_S = 22
GGML_IQ4_XS = 23
GGML_IQ1_M = 29
GGML_BF16 = 30

# (block numel, bytes per block) per ggml type.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q8_0: (32, 34),
    GGML_Q2_K: (256, 84),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    GGML_IQ2_XXS: (256, 66),
    GGML_IQ2_XS: (256, 74),
    GGML_IQ3_XXS: (256, 98),
    GGML_IQ1_S: (256, 50),
    GGML_IQ3_S: (256, 110),
    GGML_IQ2_S: (256, 82),
    GGML_IQ4_XS: (256, 136),
    GGML_IQ1_M: (256, 56),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ2_XXS: "IQ2_XXS",
    GGML_IQ2_XS: "IQ2_XS",
    GGML_IQ3_XXS: "IQ3_XXS",
    GGML_IQ1_S: "IQ1_S",
    GGML_IQ3_S: "IQ3_S",
    GGML_IQ2_S: "IQ2_S",
    GGML_IQ4_XS: "IQ4_XS",
    GGML_IQ1_M: "IQ1_M",
}


def row_bytes(numel: int, ggml_type: int) -> int:
    """Packed byte length of one row of ``numel`` elements in ``ggml_type`` blocks.

    Single source of truth for the ``numel // block * type_size`` math shared by the
    packed-weight ops (``GGUFLinear``/``GGUFEmbedding``) and the expert bank loaders.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert numel % block == 0, (
        f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
    )
    return numel // block * type_size


def _f16_scales(raw: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reinterpret bytes ``[lo:hi]`` (2 per block) of each block row as fp16 -> fp32 [N,1]."""
    return raw[:, lo:hi].contiguous().view(torch.float16).to(torch.float32)


def dequant_q4_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0: per 32-elem block = fp16 scale ``d`` + 16 packed nibbles; ``w = d*(q-8)``.

    Byte ``j`` of the 16 holds element ``j`` in its low nibble and ``j+16`` in its high
    nibble, so storage order within the block is ``[lo0..lo15, hi0..hi15]``.
    """
    raw = raw.reshape(-1, 18)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    qs = raw[:, 2:18]  # [N,16] uint8
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    q = torch.cat([lo, hi], dim=1)  # [N,32]
    return ((q - 8.0) * d).reshape(-1).to(out_dtype)


def dequant_q2_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q2_K: 16 groups of 16 values with four-bit scales and minima."""
    raw = raw.reshape(-1, 84)
    scales = raw[:, 0:16].to(torch.int32)
    qs = raw[:, 16:80].to(torch.int32)
    d = _f16_scales(raw, 80, 82)
    dmin = _f16_scales(raw, 82, 84)
    output = torch.empty((raw.shape[0], 256), dtype=torch.float32, device=raw.device)
    group = 0
    for half in range(2):
        packed = qs[:, half * 32 : (half + 1) * 32]
        for shift in range(0, 8, 2):
            for offset in (0, 16):
                scale = scales[:, group : group + 1]
                values = (packed[:, offset : offset + 16] >> shift) & 0x03
                output[:, group * 16 : (group + 1) * 16] = (
                    d * (scale & 0x0F).to(torch.float32) * values.to(torch.float32)
                    - dmin * (scale >> 4).to(torch.float32)
                )
                group += 1
    return output.reshape(-1).to(out_dtype)


def dequant_q3_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q3_K: packed low two bits, a high-bit mask, and signed six-bit scales."""
    raw = raw.reshape(-1, 110)
    hmask = raw[:, 0:32].to(torch.int32)
    qs = raw[:, 32:96].to(torch.int32)
    packed_scales = raw[:, 96:108].to(torch.int64)
    d = _f16_scales(raw, 108, 110)

    words = []
    for offset in (0, 4, 8):
        word = torch.zeros(raw.shape[0], dtype=torch.int64, device=raw.device)
        for byte in range(4):
            word |= packed_scales[:, offset + byte] << (8 * byte)
        words.append(word)
    mask2 = 0x03030303
    mask4 = 0x0F0F0F0F
    scale_words = (
        (words[0] & mask4) | (((words[2] >> 0) & mask2) << 4),
        (words[1] & mask4) | (((words[2] >> 2) & mask2) << 4),
        ((words[0] >> 4) & mask4) | (((words[2] >> 4) & mask2) << 4),
        ((words[1] >> 4) & mask4) | (((words[2] >> 6) & mask2) << 4),
    )
    scales = torch.stack(
        [(word >> (8 * byte)) & 0xFF for word in scale_words for byte in range(4)],
        dim=1,
    ).to(torch.float32) - 32.0

    output = torch.empty((raw.shape[0], 256), dtype=torch.float32, device=raw.device)
    group = 0
    for half in range(2):
        packed = qs[:, half * 32 : (half + 1) * 32]
        for shift in range(0, 8, 2):
            for offset in (0, 16):
                low = (packed[:, offset : offset + 16] >> shift) & 0x03
                high = hmask[:, offset : offset + 16] & (1 << (group // 2))
                values = low - torch.where(high != 0, 0, 4)
                output[:, group * 16 : (group + 1) * 16] = (
                    d * scales[:, group : group + 1] * values.to(torch.float32)
                )
                group += 1
    return output.reshape(-1).to(out_dtype)


def dequant_q6_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q6_K: 256-elem super-block = 128B low nibbles + 64B high 2-bits + 16 int8
    sub-scales + fp16 ``d``. Direct vectorization of ggml's two-half loop."""
    raw = raw.reshape(-1, 210)
    n = raw.shape[0]
    ql = raw[:, 0:128]  # [n,128]
    qh = raw[:, 128:192]  # [n,64]
    sc = raw[:, 192:208].view(torch.int8).to(torch.float32)  # [n,16]
    d = _f16_scales(raw, 208, 210)  # [n,1]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # l in 0..15 -> is=0; l in 16..31 -> is=1 (per ggml: is = l/16).
    is_idx = (torch.arange(32, device=raw.device) // 16)  # [32] in {0,1}
    for h in range(2):  # two 128-elem halves of the super-block
        qlh = ql[:, h * 64:(h + 1) * 64]  # [n,64]
        qhh = qh[:, h * 32:(h + 1) * 32]  # [n,32]
        sch = sc[:, h * 8:(h + 1) * 8]  # [n,8]
        a = qlh[:, 0:32].to(torch.int32)  # ql[l]
        b = qlh[:, 32:64].to(torch.int32)  # ql[l+32]
        hb = qhh.to(torch.int32)  # qh[l]
        q1 = ((a & 0x0F) | (((hb >> 0) & 3) << 4)) - 32
        q2 = ((b & 0x0F) | (((hb >> 2) & 3) << 4)) - 32
        q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
        q4 = ((b >> 4) | (((hb >> 6) & 3) << 4)) - 32
        s1 = sch.index_select(1, is_idx + 0).to(torch.float32)
        s2 = sch.index_select(1, is_idx + 2).to(torch.float32)
        s3 = sch.index_select(1, is_idx + 4).to(torch.float32)
        s4 = sch.index_select(1, is_idx + 6).to(torch.float32)
        base = h * 128
        y[:, base + 0:base + 32] = d * s1 * q1.to(torch.float32)
        y[:, base + 32:base + 64] = d * s2 * q2.to(torch.float32)
        y[:, base + 64:base + 96] = d * s3 * q3.to(torch.float32)
        y[:, base + 96:base + 128] = d * s4 * q4.to(torch.float32)
    return y.reshape(-1).to(out_dtype)


def dequant_q4_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_K: 256 values with eight 6-bit scales/mins and packed 4-bit quants."""
    raw = raw.reshape(-1, 144)
    d = _f16_scales(raw, 0, 2)
    dmin = _f16_scales(raw, 2, 4)
    packed_scales = raw[:, 4:16].to(torch.int32)
    qs = raw[:, 16:144]

    scales = torch.empty((raw.shape[0], 8), dtype=torch.float32, device=raw.device)
    mins = torch.empty_like(scales)
    for index in range(8):
        if index < 4:
            scales[:, index] = (packed_scales[:, index] & 0x3F).to(torch.float32)
            mins[:, index] = (packed_scales[:, index + 4] & 0x3F).to(torch.float32)
        else:
            scales[:, index] = (
                (packed_scales[:, index + 4] & 0x0F)
                | ((packed_scales[:, index - 4] >> 6) << 4)
            ).to(torch.float32)
            mins[:, index] = (
                (packed_scales[:, index + 4] >> 4)
                | ((packed_scales[:, index] >> 6) << 4)
            ).to(torch.float32)

    output = torch.empty((raw.shape[0], 256), dtype=torch.float32, device=raw.device)
    for group in range(4):
        packed = qs[:, group * 32 : (group + 1) * 32]
        low = (packed & 0x0F).to(torch.float32)
        high = (packed >> 4).to(torch.float32)
        first = group * 2
        output[:, group * 64 : group * 64 + 32] = (
            d * scales[:, first : first + 1] * low
            - dmin * mins[:, first : first + 1]
        )
        output[:, group * 64 + 32 : (group + 1) * 64] = (
            d * scales[:, first + 1 : first + 2] * high
            - dmin * mins[:, first + 1 : first + 2]
        )
    return output.reshape(-1).to(out_dtype)


def dequant_q5_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q5_K: Q4_K scales/minima with one high quant bit per value."""
    raw = raw.reshape(-1, 176)
    d = _f16_scales(raw, 0, 2)
    dmin = _f16_scales(raw, 2, 4)
    packed_scales = raw[:, 4:16].to(torch.int32)
    qh = raw[:, 16:48].to(torch.int32)
    qs = raw[:, 48:176].to(torch.int32)

    scales = torch.empty((raw.shape[0], 8), dtype=torch.float32, device=raw.device)
    mins = torch.empty_like(scales)
    for index in range(8):
        if index < 4:
            scales[:, index] = (packed_scales[:, index] & 0x3F).to(torch.float32)
            mins[:, index] = (packed_scales[:, index + 4] & 0x3F).to(torch.float32)
        else:
            scales[:, index] = (
                (packed_scales[:, index + 4] & 0x0F)
                | ((packed_scales[:, index - 4] >> 6) << 4)
            ).to(torch.float32)
            mins[:, index] = (
                (packed_scales[:, index + 4] >> 4)
                | ((packed_scales[:, index] >> 6) << 4)
            ).to(torch.float32)

    output = torch.empty((raw.shape[0], 256), dtype=torch.float32, device=raw.device)
    for group in range(4):
        packed = qs[:, group * 32 : (group + 1) * 32]
        high_low = ((qh & (1 << (2 * group))) != 0).to(torch.int32) << 4
        high_high = ((qh & (2 << (2 * group))) != 0).to(torch.int32) << 4
        first = group * 2
        output[:, group * 64 : group * 64 + 32] = (
            d * scales[:, first : first + 1]
            * ((packed & 0x0F) + high_low).to(torch.float32)
            - dmin * mins[:, first : first + 1]
        )
        output[:, group * 64 + 32 : (group + 1) * 64] = (
            d * scales[:, first + 1 : first + 2]
            * ((packed >> 4) + high_high).to(torch.float32)
            - dmin * mins[:, first + 1 : first + 2]
        )
    return output.reshape(-1).to(out_dtype)


_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q2_K: dequant_q2_k,
    GGML_Q3_K: dequant_q3_k,
    GGML_Q4_K: dequant_q4_k,
    GGML_Q5_K: dequant_q5_k,
    GGML_Q6_K: dequant_q6_k,
}


def dequantize(raw: torch.Tensor, ggml_type: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``raw`` (uint8) of any supported ggml type to flat ``out_dtype``."""
    if ggml_type == GGML_F32:
        return raw.view(torch.float32).to(out_dtype)
    if ggml_type == GGML_F16:
        return raw.view(torch.float16).to(out_dtype)
    if ggml_type == GGML_BF16:
        return raw.view(torch.bfloat16).to(out_dtype)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"dequant for ggml type {GGML_NAME.get(ggml_type, ggml_type)} not implemented"
        )
    return fn(raw, out_dtype)


__all__ = [
    "BLOCK_SHAPE",
    "GGML_BF16",
    "GGML_F16",
    "GGML_F32",
    "GGML_IQ1_M",
    "GGML_IQ1_S",
    "GGML_IQ2_S",
    "GGML_IQ2_XS",
    "GGML_IQ2_XXS",
    "GGML_IQ3_S",
    "GGML_IQ3_XXS",
    "GGML_IQ4_XS",
    "GGML_NAME",
    "GGML_Q2_K",
    "GGML_Q3_K",
    "GGML_Q4_0",
    "GGML_Q4_K",
    "GGML_Q5_K",
    "GGML_Q6_K",
    "GGML_Q8_0",
    "dequant_q2_k",
    "dequant_q3_k",
    "dequant_q4_0",
    "dequant_q4_k",
    "dequant_q5_k",
    "dequant_q6_k",
    "dequantize",
    "row_bytes",
]
