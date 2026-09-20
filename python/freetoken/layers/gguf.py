"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp kernels (no bf16 copy ever materialized).

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_IQ1_M,
    GGML_IQ1_S,
    GGML_IQ2_S,
    GGML_IQ2_XS,
    GGML_IQ2_XXS,
    GGML_IQ3_S,
    GGML_IQ3_XXS,
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_NAME,
    GGML_PQ2_0,
    GGML_Q2_K,
    GGML_Q3_K,
    GGML_Q4_0,
    GGML_Q4_1,
    GGML_Q4_K,
    GGML_Q5_0,
    GGML_Q5_1,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    row_bytes,
)

from .base import BaseOP, OPList

# ggml type groups for kernel dispatch (subset we build kernels for).
_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}
# standard + k-quants: both an MMVQ (small-batch GEMV) and MMQ (large-batch) kernel exist.
_MMVQ = {GGML_Q4_0, GGML_Q4_K, GGML_Q8_0, GGML_Q6_K}
_MMQ = {GGML_Q4_0, GGML_Q4_K, GGML_Q8_0, GGML_Q6_K}
_DEQUANT = {GGML_Q4_0, GGML_Q4_K, GGML_Q8_0, GGML_Q6_K}

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6


def fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type."""
    if x.device.type == "xpu":
        if qweight_type in _UNQUANTIZED:
            from freetoken.models.gguf.dequant import dequantize

            weight = dequantize(qweight, qweight_type, x.dtype).reshape(
                qweight.shape[0], -1
            )
            return x @ weight.T

        from freetoken.kernel.sycl.causal_conv1d import (
            iq1_m_matvec_sycl,
            iq1_s_matvec_sycl,
            iq2_s_matvec_sycl,
            iq2_xs_matvec_sycl,
            iq2_xxs_matvec_sycl,
            iq3_s_matvec_sycl,
            iq3_xxs_matvec_sycl,
            iq4_nl_matvec_sycl,
            iq4_xs_matvec_sycl,
            pq2_0_matvec_sycl,
            q2_k_matvec_sycl,
            q3_k_matvec_sycl,
            q4_0_matvec_sycl,
            q4_1_matvec_sycl,
            q4_k_matvec_sycl,
            q5_0_matvec_sycl,
            q5_1_matvec_sycl,
            q5_k_matvec_sycl,
            q6_k_matvec_sycl,
            q8_0_matvec_sycl,
        )

        # Keep prompt batches on the measured XPU matmul path; SYCL is for decode.
        if qweight_type == GGML_Q4_0 and x.shape[0] == 1:
            return q4_0_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q4_1 and x.shape[0] == 1:
            return q4_1_matvec_sycl(x, qweight)
        if qweight_type == GGML_IQ4_NL and x.shape[0] == 1:
            return iq4_nl_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q5_0 and x.shape[0] == 1:
            return q5_0_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q5_1 and x.shape[0] == 1:
            return q5_1_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q8_0:
            return q8_0_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q2_K:
            return q2_k_matvec_sycl(x, qweight)
        if qweight_type == GGML_PQ2_0:
            return pq2_0_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q3_K:
            return q3_k_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q4_K:
            return q4_k_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q5_K:
            return q5_k_matvec_sycl(x, qweight)
        if qweight_type == GGML_Q6_K:
            return q6_k_matvec_sycl(x, qweight)
        if qweight_type == GGML_IQ3_XXS:
            from freetoken.models.gguf.dequant import _iq_signs, _iq_table

            return iq3_xxs_matvec_sycl(
                x,
                qweight,
                _iq_table("IQ3_XXS", x.device),
                _iq_signs(x.device),
            )
        if qweight_type == GGML_IQ2_XXS:
            from freetoken.models.gguf.dequant import _iq_signs, _iq_table

            return iq2_xxs_matvec_sycl(
                x,
                qweight,
                _iq_table("IQ2_XXS", x.device),
                _iq_signs(x.device),
            )
        if qweight_type == GGML_IQ2_XS:
            from freetoken.models.gguf.dequant import _iq_signs, _iq_table

            return iq2_xs_matvec_sycl(
                x,
                qweight,
                _iq_table("IQ2_XS", x.device),
                _iq_signs(x.device),
            )
        if qweight_type == GGML_IQ4_XS:
            return iq4_xs_matvec_sycl(x, qweight)
        if qweight_type == GGML_IQ1_S:
            from freetoken.models.gguf.dequant import _iq_table

            return iq1_s_matvec_sycl(x, qweight, _iq_table("IQ1_S", x.device))
        if qweight_type == GGML_IQ1_M:
            from freetoken.models.gguf.dequant import _iq_table

            return iq1_m_matvec_sycl(x, qweight, _iq_table("IQ1_M", x.device))
        if qweight_type == GGML_IQ2_S:
            from freetoken.models.gguf.dequant import _iq_table

            return iq2_s_matvec_sycl(x, qweight, _iq_table("IQ2_S", x.device))
        if qweight_type == GGML_IQ3_S:
            from freetoken.models.gguf.dequant import _iq_table

            return iq3_s_matvec_sycl(x, qweight, _iq_table("IQ3_S", x.device))
    if x.device.type != "cuda":
        from freetoken.models.gguf.dequant import dequantize

        # Bound temporary memory, especially for tied vocabulary heads: a
        # 262k x 3840 Q4 table expands to nearly 2 GiB before Q4 unpacking
        # intermediates. Dequantize output rows in small compute islands.
        outputs = []
        for start in range(0, qweight.shape[0], 4096):
            packed = qweight[start : start + 4096]
            weight = dequantize(packed, qweight_type, x.dtype).reshape(
                packed.shape[0], -1
            )
            outputs.append(x @ weight.T)
        return torch.cat(outputs, dim=-1)

    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in _UNQUANTIZED:
        return x @ qweight.T
    if x.shape[0] <= _MMVQ_SAFE and qweight_type in _MMVQ:
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _MMQ:
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _DEQUANT:
        block, type_size = BLOCK_SHAPE[qweight_type]
        in_features = qweight.shape[1] // type_size * block
        weight = ggml_dequantize(
            qweight, qweight_type, out_features, in_features, x.dtype
        )
        return x @ weight.T
    raise NotImplementedError(
        f"unsupported GGUF type {GGML_NAME.get(qweight_type, qweight_type)}"
    )


def _hadamard_transform_torch(
    x: torch.Tensor, signs: torch.Tensor, block_size: int, *, inverse: bool
) -> torch.Tensor:
    shape = x.shape
    width = shape[-1]
    values = x.float().reshape(-1, width).clone()
    signs = signs.float().reshape(1, width)
    if not inverse:
        values = values * signs
    values = values.reshape(-1, block_size)
    stride = 1
    while stride < block_size:
        groups = values.view(-1, block_size // (2 * stride), 2, stride)
        left = groups[:, :, 0, :].clone()
        right = groups[:, :, 1, :].clone()
        groups[:, :, 0, :] = left + right
        groups[:, :, 1, :] = left - right
        stride *= 2
    values = values.reshape(-1, width) / block_size**0.5
    if inverse:
        values = values * signs
    return values.reshape(shape).to(x.dtype)


def _apply_hadamard_transform(
    x: torch.Tensor,
    hadamard_config,
    weight_name: str,
    *,
    inverse: bool = False,
    gdn_permutation: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    names = (
        hadamard_config.inverse_weight_names
        if inverse
        else hadamard_config.weight_names
    )
    if weight_name not in names:
        return x

    shape = x.shape
    width = shape[-1]
    if gdn_permutation is not None:
        key_heads, repeat, head_dim = gdn_permutation
        x = (
            x.reshape(-1, repeat, key_heads, head_dim)
            .transpose(1, 2)
            .contiguous()
            .reshape(-1, width)
        )
    else:
        x = x.reshape(-1, width).contiguous()

    signs = hadamard_config.signs_on(width, x.device)
    if x.device.type == "xpu":
        from freetoken.kernel.sycl.causal_conv1d import hadamard_transform_sycl

        result = hadamard_transform_sycl(
            x, signs, hadamard_config.block_size, inverse=inverse
        )
    else:
        result = _hadamard_transform_torch(
            x, signs, hadamard_config.block_size, inverse=inverse
        )
    return result.reshape(shape)


class GGUFLinear(BaseOP):
    """Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        has_bias: bool = False,
        *,
        hadamard_config=None,
        hadamard_weight_name: str | None = None,
        hadamard_permutation: tuple[int, int, int] | None = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self._hadamard_config = hadamard_config
        self._hadamard_weight_name = hadamard_weight_name
        self._hadamard_permutation = hadamard_permutation
        self.qweight = torch.empty(
            out_features, row_bytes(in_features, quant_type), dtype=torch.uint8
        )
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._hadamard_config is not None and self._hadamard_weight_name is not None:
            x = _apply_hadamard_transform(
                x,
                self._hadamard_config,
                self._hadamard_weight_name,
                gdn_permutation=self._hadamard_permutation,
            )
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out


class GGUFMergedLinear(BaseOP):
    """Logical fused projection backed by independently quantized GGUF sources."""

    def __init__(
        self,
        in_features: int,
        parts: list[tuple[int, int]],
        *,
        hadamard_config=None,
        hadamard_weight_names: list[str] | None = None,
    ):
        if hadamard_weight_names is not None and len(hadamard_weight_names) != len(
            parts
        ):
            raise ValueError("Hadamard weight names must match merged projection parts")
        self.parts = OPList(
            [
                GGUFLinear(
                    in_features,
                    out_features,
                    quant_type,
                    hadamard_config=hadamard_config,
                    hadamard_weight_name=(
                        hadamard_weight_names[index]
                        if hadamard_weight_names is not None
                        else None
                    ),
                )
                for index, (out_features, quant_type) in enumerate(parts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([part.forward(x) for part in self.parts.op_list], dim=-1)


class GGUFEmbedding(BaseOP):
    """Vocab embedding stored as a native GGUF block-quantized table.

    The full table is never dequantized: only the looked-up rows are gathered (in
    packed form) and dequantized per lookup, matching vLLM's ``_apply_gguf_embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        embed_scale: float | None = None,
        *,
        hadamard_config=None,
        hadamard_weight_name: str | None = None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self._hadamard_config = hadamard_config
        self._hadamard_weight_name = hadamard_weight_name
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten()
        rows = self.qweight.index_select(0, flat)  # [n, row_bytes] packed
        if rows.device.type == "cuda":
            from freetoken.kernel.gguf import ggml_dequantize

            y = ggml_dequantize(
                rows,
                self._quant_type,
                flat.shape[0],
                self.embedding_dim,
                torch.bfloat16,
            )
        else:
            from freetoken.models.gguf.dequant import dequantize

            y = dequantize(rows, self._quant_type, torch.bfloat16).reshape(
                flat.shape[0], self.embedding_dim
            )
        y = y.view(*x.shape, self.embedding_dim)
        if self._hadamard_config is not None and self._hadamard_weight_name is not None:
            y = _apply_hadamard_transform(
                y,
                self._hadamard_config,
                self._hadamard_weight_name,
                inverse=True,
            )
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


__all__ = ["GGUFEmbedding", "GGUFLinear", "GGUFMergedLinear", "fused_mul_mat_gguf"]
