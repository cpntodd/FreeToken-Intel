"""GGUF metadata adapter for dense Qwen3.5-family hybrid models."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from freetoken.layers import BaseOP
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    metadata = shim.metadata
    if any(key.startswith("prism.hadamard.") for key in metadata):
        raise NotImplementedError(
            "Prism Hadamard-transformed GGUF weights are not supported; "
            "use a non-transformed checkpoint or a Prism-compatible runtime"
        )

    def get(key: str):
        value = metadata.get(f"qwen35.{key}")
        if value is None:
            raise KeyError(f"missing GGUF metadata key qwen35.{key}")
        return value

    stored_blocks = int(get("block_count"))
    mtp_layers = int(metadata.get("qwen35.nextn_predict_layers") or 0)
    num_layers = stored_blocks - mtp_layers
    if num_layers <= 0:
        raise ValueError(
            f"invalid qwen35 layer counts: {stored_blocks} blocks, {mtp_layers} MTP layers"
        )

    hidden_size = int(get("embedding_length"))
    num_heads = int(get("attention.head_count"))
    num_kv_heads = int(get("attention.head_count_kv"))
    head_dim = int(get("attention.key_length"))
    value_dim = int(get("attention.value_length"))
    if value_dim != head_dim:
        raise ValueError(
            f"qwen35 full-attention key/value widths differ: {head_dim} != {value_dim}"
        )

    interval = int(get("full_attention_interval"))
    if interval <= 0:
        full_ids: tuple[int, ...] = ()
        linear_ids = tuple(range(num_layers))
    else:
        full_ids = tuple(i for i in range(num_layers) if (i + 1) % interval == 0)
        linear_ids = tuple(i for i in range(num_layers) if (i + 1) % interval != 0)

    linear_head_dim = int(get("ssm.state_size"))
    linear_key_heads = int(get("ssm.group_count"))
    linear_inner_size = int(get("ssm.inner_size"))
    if linear_inner_size % linear_head_dim:
        raise ValueError(
            "qwen35 ssm.inner_size must be divisible by ssm.state_size: "
            f"{linear_inner_size} % {linear_head_dim}"
        )
    linear_value_heads = linear_inner_size // linear_head_dim
    time_step_rank = int(get("ssm.time_step_rank"))
    if time_step_rank != linear_value_heads:
        raise ValueError(
            "qwen35 ssm.time_step_rank must match the value-head count: "
            f"{time_step_rank} != {linear_value_heads}"
        )

    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(get("rope.dimension_count")),
        max_position=int(get("context_length")),
        base=float(get("rope.freq_base")),
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=linear_key_heads,
        num_value_heads=linear_value_heads,
        key_head_dim=linear_head_dim,
        value_head_dim=linear_head_dim,
        conv_kernel_dim=int(get("ssm.conv_kernel")),
        output_gate="silu",
    )

    tensor_types: dict[str, int] | None = None
    if Path(shim.model_path).is_file():
        from freetoken.models.gguf.reader import iter_gguf_tensors

        tensor_types = {}
        for tensor in iter_gguf_tensors(shim.model_path):
            if tensor.name.startswith("blk."):
                layer = int(tensor.name.split(".", 2)[1])
                if layer >= num_layers:
                    continue
            tensor_types[tensor.name] = tensor.ggml_type

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(get("feed_forward_length")),
        hidden_act="silu",
        rms_norm_eps=float(get("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=True,
        model_type="qwen3_5_text",
        architectures=list(shim.architectures),
        moe_enabled=False,
        use_qk_norm=True,
        gguf_tensor_types=tensor_types,
        attention_groups=tuple(
            sorted(
                (full_group, linear_group),
                key=lambda group: group.layer_ids[0] if group.layer_ids else 1 << 30,
            )
        ),
    )


def is_gguf_model(config: ModelConfig) -> bool:
    return config.gguf_tensor_types is not None


class GGUFUntiedLMHead(BaseOP):
    def __init__(self, in_features: int, out_features: int, quant_type: int):
        from freetoken.layers.gguf import GGUFLinear

        self.proj = GGUFLinear(in_features, out_features, quant_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return self.proj.forward(x)


def _type(config: ModelConfig, name: str) -> int:
    assert config.gguf_tensor_types is not None
    try:
        return config.gguf_tensor_types[name]
    except KeyError as error:
        raise ValueError(f"missing GGUF tensor required by Qwen3.8: {name}") from error


def convert_qwen35_to_gguf(model, config: ModelConfig) -> None:
    """Replace every large dense projection with its native per-tensor GGUF form."""
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, GGUFMergedLinear

    assert config.gguf_tensor_types is not None
    inner = model.model
    inner.embed_tokens = GGUFEmbedding(
        config.vocab_size, config.hidden_size, _type(config, "token_embd.weight")
    )

    for layer_id, layer in enumerate(inner.layers.op_list):
        prefix = f"blk.{layer_id}"
        if layer._is_linear:
            op = layer.linear_attn
            op.in_proj = GGUFMergedLinear(
                config.hidden_size,
                [
                    (op.conv_dim, _type(config, f"{prefix}.attn_qkv.weight")),
                    (op.value_dim, _type(config, f"{prefix}.attn_gate.weight")),
                    (op.num_v_heads, _type(config, f"{prefix}.ssm_beta.weight")),
                    (op.num_v_heads, _type(config, f"{prefix}.ssm_alpha.weight")),
                ],
            )
            op.out_proj = GGUFLinear(
                op.value_dim,
                config.hidden_size,
                _type(config, f"{prefix}.ssm_out.weight"),
            )
            op.gguf_tiled_v = True
        else:
            op = layer.self_attn
            op.qkv_proj = GGUFMergedLinear(
                config.hidden_size,
                [
                    (op._qkv_split[0], _type(config, f"{prefix}.attn_q.weight")),
                    (op._qkv_split[1], _type(config, f"{prefix}.attn_k.weight")),
                    (op._qkv_split[2], _type(config, f"{prefix}.attn_v.weight")),
                ],
            )
            op.o_proj = GGUFLinear(
                op.qo_attn_dim,
                config.hidden_size,
                _type(config, f"{prefix}.attn_output.weight"),
            )

        layer.mlp.gate_up_proj = GGUFMergedLinear(
            config.hidden_size,
            [
                (config.intermediate_size, _type(config, f"{prefix}.ffn_gate.weight")),
                (config.intermediate_size, _type(config, f"{prefix}.ffn_up.weight")),
            ],
        )
        layer.mlp.down_proj = GGUFLinear(
            config.intermediate_size,
            config.hidden_size,
            _type(config, f"{prefix}.ffn_down.weight"),
        )

    if config.tie_word_embeddings:
        raise NotImplementedError(
            "tied Qwen3.8 GGUF output heads are not yet supported"
        )
    model.lm_head = GGUFUntiedLMHead(
        config.hidden_size, config.vocab_size, _type(config, "output.weight")
    )


_DENSE_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
}


def _to_bf16(tensor) -> torch.Tensor:
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(
        tensor.packed().reshape(-1), tensor.ggml_type, torch.bfloat16
    ).reshape(tensor.shape)


def _v_tiled_to_grouped(value: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    group = config.linear_attention_group()
    assert group is not None
    ratio = group.num_value_heads // group.num_key_heads
    shape = value.shape
    return (
        value.reshape(ratio, group.num_key_heads, *shape[1:])
        .transpose(0, 1)
        .contiguous()
        .reshape(shape)
    )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Map Qwen3.8 GGUF tensors into the independently packed model modules."""
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    del device, include_moe_experts, include_vision
    assert include_non_moe
    config = parse_gguf_config(cached_load_hf_config(model_path))
    for tensor in iter_gguf_tensors(model_path):
        name = tensor.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", tensor.packed()
            continue
        if name == "output.weight":
            yield "lm_head.proj.qweight", tensor.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(tensor)
            continue
        if not name.startswith("blk."):
            continue
        layer_id = int(name.split(".", 2)[1])
        if layer_id >= config.num_layers:
            continue
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer_id}"
        if suffix in _DENSE_MAP:
            yield f"{base}.{_DENSE_MAP[suffix]}", _to_bf16(tensor)
            continue
        if suffix == "ssm_a":
            value = _v_tiled_to_grouped(_to_bf16(tensor), config)
            yield f"{base}.linear_attn.A_log", torch.log(-value.float())
            continue
        if suffix == "ssm_dt.bias":
            value = _v_tiled_to_grouped(_to_bf16(tensor), config).float()
            yield f"{base}.linear_attn.dt_bias", value
            continue
        if suffix == "ssm_conv1d.weight":
            value = _to_bf16(tensor).unsqueeze(1)
            yield f"{base}.linear_attn.conv1d.weight", value
            continue

        packed_slots = {
            "attn_q.weight": ("self_attn.qkv_proj.parts.0.qweight"),
            "attn_k.weight": ("self_attn.qkv_proj.parts.1.qweight"),
            "attn_v.weight": ("self_attn.qkv_proj.parts.2.qweight"),
            "attn_output.weight": ("self_attn.o_proj.qweight"),
            "attn_qkv.weight": ("linear_attn.in_proj.parts.0.qweight"),
            "attn_gate.weight": ("linear_attn.in_proj.parts.1.qweight"),
            "ssm_beta.weight": ("linear_attn.in_proj.parts.2.qweight"),
            "ssm_alpha.weight": ("linear_attn.in_proj.parts.3.qweight"),
            "ssm_out.weight": ("linear_attn.out_proj.qweight"),
            "ffn_gate.weight": ("mlp.gate_up_proj.parts.0.qweight"),
            "ffn_up.weight": ("mlp.gate_up_proj.parts.1.qweight"),
            "ffn_down.weight": ("mlp.down_proj.qweight"),
        }
        target = packed_slots.get(suffix)
        if target is None:
            raise ValueError(f"unmapped Qwen3.8 GGUF tensor: {name}")
        yield f"{base}.{target}", tensor.packed()


__all__ = [
    "convert_qwen35_to_gguf",
    "is_gguf_model",
    "iter_gguf_weights",
    "parse_gguf_config",
]
