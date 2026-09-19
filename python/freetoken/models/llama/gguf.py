"""Correctness-first GGUF adapter for dense Llama checkpoints.

Unlike Gemma 4's native packed path, common Q4_K_M Llama files mix Q4_K and Q6_K
within fused FreeToken projections. The loader therefore dequantizes each source
tensor independently to BF16 before fusing it. This costs more VRAM but preserves
the existing Llama architecture and provides a reliable XPU baseline.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from freetoken.models.config import ModelConfig, RotaryConfig
from freetoken.models.gguf.dequant import dequantize

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _rope_scaling(shim: GgufConfigShim, head_dim: int, base: float):
    """Recover llama3 scaling encoded by llama.cpp's rope frequency divisors."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    model_path = getattr(shim, "model_path", None)
    if model_path is None:
        return None
    for tensor in iter_gguf_tensors(model_path):
        if tensor.name != "rope_freqs.weight":
            continue
        divisors = tensor.packed().reshape(-1).view(torch.float32)
        if torch.all(divisors == 1):
            return None
        factor = float(divisors.max().item())
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        for original_context in (4096, 8192, 16384, 32768):
            wave_len = 2 * math.pi / inv_freq
            smooth = torch.clamp(
                (original_context / wave_len - 1.0) / (4.0 - 1.0), 0, 1
            )
            expected = 1.0 / ((1 - smooth) / factor + smooth)
            if torch.allclose(divisors, expected, rtol=1e-5, atol=1e-5):
                return {
                    "rope_type": "llama3",
                    "factor": factor,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": original_context,
                }
        raise ValueError(
            "unsupported Llama GGUF rope_freqs.weight; refusing to ignore "
            "checkpoint frequency scaling"
        )
    return None


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    metadata = shim.metadata

    def get(key: str):
        value = metadata.get(f"llama.{key}")
        if value is None:
            raise KeyError(f"missing GGUF metadata key llama.{key}")
        return value

    hidden_size = int(get("embedding_length"))
    num_heads = int(get("attention.head_count"))
    head_dim = int(metadata.get("llama.attention.key_length") or hidden_size // num_heads)
    rope_base = float(get("rope.freq_base"))
    return ModelConfig(
        num_layers=int(get("block_count")),
        num_qo_heads=num_heads,
        num_kv_heads=int(get("attention.head_count_kv")),
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(get("feed_forward_length")),
        rms_norm_eps=float(get("attention.layer_norm_rms_epsilon")),
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=int(get("rope.dimension_count")),
            max_position=int(get("context_length")),
            base=rope_base,
            scaling=_rope_scaling(shim, head_dim, rope_base),
        ),
        hidden_act="silu",
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="llama",
        architectures=list(shim.architectures),
    )


def _to_bf16(tensor, *, rows_per_chunk: int = 512) -> torch.Tensor:
    packed = tensor.packed()
    if len(tensor.shape) == 1:
        return dequantize(packed.reshape(-1), tensor.ggml_type, torch.bfloat16).reshape(
            tensor.shape
        )

    rows = []
    for start in range(0, packed.shape[0], rows_per_chunk):
        chunk = packed[start : start + rows_per_chunk]
        rows.append(
            dequantize(chunk, tensor.ggml_type, torch.bfloat16).reshape(
                chunk.shape[0], tensor.shape[1]
            )
        )
    return torch.cat(rows, dim=0)


def _reverse_rope_permute(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Restore HF/FreeToken q/k row order from llama.cpp's GGUF RoPE layout."""
    head_dim = weight.shape[0] // num_heads
    return (
        weight.reshape(num_heads, head_dim // 2, 2, weight.shape[1])
        .transpose(1, 2)
        .reshape_as(weight)
    )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    from freetoken.distributed import get_tp_info
    from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata

    del device, include_moe_experts
    if not include_non_moe:
        return
    if get_tp_info().size != 1:
        raise NotImplementedError("Llama GGUF loading currently supports TP=1 only")

    metadata = load_gguf_metadata(model_path)
    num_q_heads = int(metadata["llama.attention.head_count"])
    num_kv_heads = int(metadata["llama.attention.head_count_kv"])

    qkv: dict[int, dict[str, torch.Tensor]] = {}
    gate_up: dict[int, dict[str, torch.Tensor]] = {}
    for tensor in iter_gguf_tensors(model_path):
        name = tensor.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.weight", _to_bf16(tensor)
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(tensor)
            continue
        if name == "output.weight":
            yield "lm_head.weight", _to_bf16(tensor)
            continue
        if name == "rope_freqs.weight":
            continue
        if not name.startswith("blk."):
            raise ValueError(f"unmapped Llama GGUF tensor: {name}")

        layer = int(name.split(".")[1])
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"
        if suffix == "attn_norm.weight":
            yield f"{base}.input_layernorm.weight", _to_bf16(tensor)
        elif suffix == "ffn_norm.weight":
            yield f"{base}.post_attention_layernorm.weight", _to_bf16(tensor)
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.weight", _to_bf16(tensor)
        elif suffix == "ffn_down.weight":
            yield f"{base}.mlp.down_proj.weight", _to_bf16(tensor)
        elif suffix in {"attn_q.weight", "attn_k.weight", "attn_v.weight"}:
            slot = suffix.removeprefix("attn_").removesuffix(".weight")
            weight = _to_bf16(tensor)
            if slot == "q":
                weight = _reverse_rope_permute(weight, num_q_heads)
            elif slot == "k":
                weight = _reverse_rope_permute(weight, num_kv_heads)
            qkv.setdefault(layer, {})[slot] = weight
        elif suffix in {"ffn_gate.weight", "ffn_up.weight"}:
            slot = suffix.removeprefix("ffn_").removesuffix(".weight")
            gate_up.setdefault(layer, {})[slot] = _to_bf16(tensor)
        else:
            raise ValueError(f"unmapped Llama GGUF tensor: {name}")

        parts = qkv.get(layer)
        if parts is not None and parts.keys() >= {"q", "k", "v"}:
            yield f"{base}.self_attn.qkv_proj.weight", torch.cat(
                [parts["q"], parts["k"], parts["v"]], dim=0
            )
            del qkv[layer]
        parts = gate_up.get(layer)
        if parts is not None and parts.keys() >= {"gate", "up"}:
            yield f"{base}.mlp.gate_up_proj.weight", torch.cat(
                [parts["gate"], parts["up"]], dim=0
            )
            del gate_up[layer]

    if qkv or gate_up:
        raise ValueError(
            f"incomplete Llama GGUF fused groups: qkv={sorted(qkv)}, "
            f"gate_up={sorted(gate_up)}"
        )


__all__ = ["iter_gguf_weights", "parse_gguf_config"]
