"""GGUF metadata adapter for dense Qwen3.5-family hybrid models."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from freetoken.layers import BaseOP
from freetoken.models.config import (
    FullAttentionGroupConfig,
    GGUFHadamardConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


_PRISM_HADAMARD_PREFIX = "prism.hadamard."
_PRISM_HADAMARD_KEYS = {
    "version",
    "block_size",
    "transform",
    "axis",
    "sign_mode",
    "weight_names",
    "sign_widths",
    "sign_values",
    "gdn_v_grouped",
    "inverse_weight_names",
}
_PRISM_HADAMARD_WEIGHT_KINDS = {
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_out.weight",
}


def _is_supported_hadamard_weight(name: str, num_layers: int) -> bool:
    if name == "output.weight":
        return True
    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "blk" or parts[2] not in _PRISM_HADAMARD_WEIGHT_KINDS:
        return False
    try:
        layer_id = int(parts[1])
    except ValueError:
        return False
    return 0 <= layer_id < num_layers


def _parse_prism_hadamard(
    metadata: dict,
    tensor_types: dict[str, int] | None,
    num_layers: int,
    tie_word_embeddings: bool,
) -> GGUFHadamardConfig | None:
    keys = {key for key in metadata if key.startswith(_PRISM_HADAMARD_PREFIX)}
    if not keys:
        return None
    suffixes = {key.removeprefix(_PRISM_HADAMARD_PREFIX) for key in keys}
    unknown = suffixes - _PRISM_HADAMARD_KEYS
    if unknown:
        raise NotImplementedError(
            f"unsupported Prism Hadamard metadata keys: {sorted(unknown)}"
        )

    def required(name: str):
        key = f"{_PRISM_HADAMARD_PREFIX}{name}"
        if key not in metadata:
            raise ValueError(f"incomplete Prism Hadamard metadata: missing {key}")
        return metadata[key]

    version = int(required("version"))
    if version != 1:
        raise NotImplementedError(f"unsupported Prism Hadamard version {version}")
    block_size = int(required("block_size"))
    if block_size < 2 or block_size > 1024 or block_size & (block_size - 1):
        raise ValueError(
            "Prism Hadamard block_size must be a power of two from 2 through 1024"
        )
    transform = required("transform")
    if transform != "normalized-sylvester-walsh-hadamard":
        raise NotImplementedError(f"unsupported Prism Hadamard transform {transform!r}")
    axis = required("axis")
    if axis != "input-last-dimension":
        raise NotImplementedError(f"unsupported Prism Hadamard axis {axis!r}")
    sign_mode = required("sign_mode")
    if sign_mode != "explicit":
        raise NotImplementedError(
            f"unsupported Prism Hadamard sign mode {sign_mode!r}; explicit signs are required"
        )

    raw_names = required("weight_names")
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        raise ValueError("Prism Hadamard weight_names must be a non-empty array")
    weight_names = tuple(raw_names)
    if any(not isinstance(name, str) for name in weight_names):
        raise ValueError("Prism Hadamard weight_names must contain strings")
    if len(set(weight_names)) != len(weight_names):
        raise ValueError("Prism Hadamard weight_names contains duplicates")
    unsupported = [
        name for name in weight_names if not _is_supported_hadamard_weight(name, num_layers)
    ]
    if unsupported:
        raise NotImplementedError(
            "Prism Hadamard weights are not wired to verified Qwen3.5 GGUF paths: "
            f"{unsupported[:4]}"
        )
    if tensor_types is not None:
        missing = set(weight_names) - tensor_types.keys()
        if missing:
            raise ValueError(
                "Prism Hadamard weight_names reference missing GGUF tensors: "
                f"{sorted(missing)[:4]}"
            )

    raw_inverse_names = metadata.get(f"{_PRISM_HADAMARD_PREFIX}inverse_weight_names", [])
    if not isinstance(raw_inverse_names, (list, tuple)):
        raise ValueError("Prism Hadamard inverse_weight_names must be an array")
    inverse_names = tuple(raw_inverse_names)
    if any(name != "token_embd.weight" for name in inverse_names):
        raise NotImplementedError(
            "Prism Hadamard inverse transforms are supported only for token_embd.weight"
        )
    if len(set(inverse_names)) != len(inverse_names):
        raise ValueError("Prism Hadamard inverse_weight_names contains duplicates")
    if set(weight_names) & set(inverse_names):
        raise ValueError("a GGUF tensor cannot use both forward and inverse Hadamard paths")
    if tie_word_embeddings and "token_embd.weight" in inverse_names:
        raise NotImplementedError(
            "Prism Hadamard inverse token embeddings with a tied LM head are not supported"
        )
    if tensor_types is not None and set(inverse_names) - tensor_types.keys():
        raise ValueError("Prism Hadamard inverse_weight_names references a missing GGUF tensor")

    raw_widths = required("sign_widths")
    raw_signs = required("sign_values")
    if not isinstance(raw_widths, (list, tuple)) or not raw_widths:
        raise ValueError("explicit Prism Hadamard sign_widths must be a non-empty array")
    if not isinstance(raw_signs, (list, tuple)):
        raise ValueError("explicit Prism Hadamard sign_values must be an array")
    widths = tuple(int(width) for width in raw_widths)
    if len(set(widths)) != len(widths) or any(
        width <= 0 or width % block_size for width in widths
    ):
        raise ValueError("Prism Hadamard sign widths must be unique positive block multiples")
    if len(raw_signs) != sum(widths):
        raise ValueError("Prism Hadamard sign_values length does not match sign_widths")

    signs_by_width: dict[int, tuple[int, ...]] = {}
    offset = 0
    for width in widths:
        values = raw_signs[offset : offset + width]
        if any(value not in (-1, 1) for value in values):
            raise ValueError("Prism Hadamard sign values must be +/-1")
        signs_by_width[width] = tuple(int(value) for value in values)
        offset += width

    gdn_v_grouped = metadata.get(f"{_PRISM_HADAMARD_PREFIX}gdn_v_grouped", False)
    if not isinstance(gdn_v_grouped, bool):
        raise ValueError("Prism Hadamard gdn_v_grouped must be boolean")
    return GGUFHadamardConfig(
        block_size=block_size,
        weight_names=frozenset(weight_names),
        inverse_weight_names=frozenset(inverse_names),
        signs_by_width=signs_by_width,
        gdn_v_grouped=gdn_v_grouped,
    )


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

    hadamard = _parse_prism_hadamard(
        metadata, tensor_types, num_layers, shim.tie_word_embeddings
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
        gguf_tensor_types=tensor_types,
        gguf_hadamard=hadamard,
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
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        *,
        hadamard_config: GGUFHadamardConfig | None = None,
        hadamard_weight_name: str = "output.weight",
    ):
        from freetoken.layers.gguf import GGUFLinear

        self.proj = GGUFLinear(
            in_features,
            out_features,
            quant_type,
            hadamard_config=hadamard_config,
            hadamard_weight_name=hadamard_weight_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return self.proj.forward(x)


class GGUFTiedLMHead:
    """Tied LM head backed by the native packed GGUF embedding table."""

    def __init__(self, embedding, quant_type: int):
        self._embedding = embedding
        self._quant_type = quant_type

    def state_dict(self, *, prefix: str = "", result=None):
        return result if result is not None else {}

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        state_dict.pop(f"{prefix}.weight", None)
        state_dict.pop(f"{prefix}.bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self._embedding.qweight, self._quant_type)


def _type(config: ModelConfig, name: str) -> int:
    assert config.gguf_tensor_types is not None
    try:
        return config.gguf_tensor_types[name]
    except KeyError as error:
        raise ValueError(f"missing GGUF tensor required by Qwen3.8: {name}") from error


def convert_qwen35_to_gguf(model, config: ModelConfig) -> None:
    """Replace every large dense projection with its native per-tensor GGUF form."""
    from freetoken.layers.gguf import GGUFEmbedding

    assert config.gguf_tensor_types is not None
    inner = model.model

    hadamard = config.gguf_hadamard
    transformed_weights: set[str] = set()

    def merged(in_features: int, parts: list[tuple[str, int]]):
        from freetoken.layers.gguf import GGUFMergedLinear

        names = [name for name, _ in parts]
        if hadamard is not None:
            for name, _ in parts:
                if name in hadamard.weight_names:
                    if in_features not in hadamard.signs_by_width:
                        raise ValueError(
                            f"Prism Hadamard signs do not cover {name} input width "
                            f"{in_features}"
                        )
                    transformed_weights.add(name)
        return GGUFMergedLinear(
            in_features,
            [(out_features, _type(config, name)) for name, out_features in parts],
            hadamard_config=hadamard,
            hadamard_weight_names=names,
        )

    def linear(
        name: str,
        in_features: int,
        out_features: int,
        *,
        permutation: tuple[int, int, int] | None = None,
    ):
        from freetoken.layers.gguf import GGUFLinear

        if hadamard is not None and name in hadamard.weight_names:
            if in_features not in hadamard.signs_by_width:
                raise ValueError(
                    f"Prism Hadamard signs do not cover {name} input width {in_features}"
                )
            transformed_weights.add(name)
        return GGUFLinear(
            in_features,
            out_features,
            _type(config, name),
            hadamard_config=hadamard,
            hadamard_weight_name=name,
            hadamard_permutation=permutation,
        )

    if (
        hadamard is not None
        and "token_embd.weight" in hadamard.inverse_weight_names
        and config.hidden_size not in hadamard.signs_by_width
    ):
        raise ValueError(
            "Prism Hadamard signs do not cover token_embd.weight feature width "
            f"{config.hidden_size}"
        )
    inner.embed_tokens = GGUFEmbedding(
        config.vocab_size,
        config.hidden_size,
        _type(config, "token_embd.weight"),
        hadamard_config=hadamard,
        hadamard_weight_name="token_embd.weight",
    )

    for layer_id, layer in enumerate(inner.layers.op_list):
        prefix = f"blk.{layer_id}"
        if layer._is_linear:
            op = layer.linear_attn
            op.in_proj = merged(
                config.hidden_size,
                [
                    (f"{prefix}.attn_qkv.weight", op.conv_dim),
                    (f"{prefix}.attn_gate.weight", op.value_dim),
                    (f"{prefix}.ssm_beta.weight", op.num_v_heads),
                    (f"{prefix}.ssm_alpha.weight", op.num_v_heads),
                ],
            )
            permutation = None
            if hadamard is not None and hadamard.gdn_v_grouped:
                group = config.linear_attention_group()
                assert group is not None
                permutation = (
                    group.num_key_heads,
                    group.num_value_heads // group.num_key_heads,
                    group.value_head_dim,
                )
            op.out_proj = linear(
                f"{prefix}.ssm_out.weight",
                op.value_dim,
                config.hidden_size,
                permutation=permutation,
            )
            op.gguf_tiled_v = True
        else:
            op = layer.self_attn
            op.qkv_proj = merged(
                config.hidden_size,
                [
                    (f"{prefix}.attn_q.weight", op._qkv_split[0]),
                    (f"{prefix}.attn_k.weight", op._qkv_split[1]),
                    (f"{prefix}.attn_v.weight", op._qkv_split[2]),
                ],
            )
            op.o_proj = linear(
                f"{prefix}.attn_output.weight",
                op.qo_attn_dim,
                config.hidden_size,
            )

        layer.mlp.gate_up_proj = merged(
            config.hidden_size,
            [
                (f"{prefix}.ffn_gate.weight", config.intermediate_size),
                (f"{prefix}.ffn_up.weight", config.intermediate_size),
            ],
        )
        layer.mlp.down_proj = linear(
            f"{prefix}.ffn_down.weight",
            config.intermediate_size,
            config.hidden_size,
        )

    if config.tie_word_embeddings:
        model.lm_head = GGUFTiedLMHead(
            inner.embed_tokens, _type(config, "token_embd.weight")
        )
    else:
        model.lm_head = GGUFUntiedLMHead(
            config.hidden_size,
            config.vocab_size,
            _type(config, "output.weight"),
            hadamard_config=hadamard,
            hadamard_weight_name="output.weight",
        )
        if hadamard is not None and "output.weight" in hadamard.weight_names:
            if config.hidden_size not in hadamard.signs_by_width:
                raise ValueError("Prism Hadamard signs do not cover output.weight")
            transformed_weights.add("output.weight")

    if hadamard is not None:
        missing = hadamard.weight_names - transformed_weights
        if missing:
            raise ValueError(
                "Prism Hadamard weights were not attached to a Qwen3.5 projection: "
                f"{sorted(missing)[:4]}"
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
            if config.tie_word_embeddings:
                continue
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
