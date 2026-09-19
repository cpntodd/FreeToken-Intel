"""GGUF metadata adapter for dense Qwen3.5-family hybrid models."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        attention_groups=tuple(
            sorted(
                (full_group, linear_group),
                key=lambda group: group.layer_ids[0] if group.layer_ids else 1 << 30,
            )
        ),
    )


__all__ = ["parse_gguf_config"]
