import pytest
import torch
from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config


def _shim(**overrides):
    metadata = {
        "qwen35.block_count": 65,
        "qwen35.nextn_predict_layers": 1,
        "qwen35.context_length": 262144,
        "qwen35.embedding_length": 5120,
        "qwen35.feed_forward_length": 17408,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35.rope.freq_base": 10_000_000.0,
        "qwen35.rope.dimension_count": 64,
        "qwen35.full_attention_interval": 4,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.time_step_rank": 48,
        "qwen35.ssm.inner_size": 6144,
    }
    metadata.update(overrides)
    return GgufConfigShim(
        architectures=["Qwen3_5GGUFForCausalLM"],
        model_path="unused.gguf",
        model_type="qwen35",
        metadata=metadata,
        vocab_size=248320,
        tie_word_embeddings=False,
    )


def test_qwen38_metadata_excludes_mtp_and_builds_hybrid_groups():
    config = parse_gguf_config(_shim())

    assert config.num_layers == 64
    assert config.hidden_size == 5120
    assert config.intermediate_size == 17408
    assert config.head_dim == 256
    assert config.rotary_config.rotary_dim == 64
    full = config.attention_group_for_layer(3)
    assert full.layer_ids == tuple(range(3, 64, 4))
    linear = config.linear_attention_group()
    assert linear.layer_ids[:4] == (0, 1, 2, 4)
    assert linear.num_key_heads == 16
    assert linear.num_value_heads == 48
    assert linear.key_head_dim == linear.value_head_dim == 128
    assert linear.conv_kernel_dim == 4


def test_qwen35_metadata_rejects_inconsistent_delta_head_geometry():
    try:
        parse_gguf_config(_shim(**{"qwen35.ssm.time_step_rank": 47}))
    except ValueError as error:
        assert "value-head count" in str(error)
    else:
        raise AssertionError("expected inconsistent GDN geometry to be rejected")


def test_qwen35_metadata_rejects_prism_hadamard_weights():
    shim = _shim(**{"prism.hadamard.version": 1})

    with pytest.raises(NotImplementedError, match="Hadamard-transformed"):
        parse_gguf_config(shim)


def test_mixed_gguf_linear_keeps_each_projection_in_its_own_quant_format():
    from freetoken.layers.gguf import GGUFMergedLinear
    from freetoken.models.gguf.dequant import GGML_Q2_K, GGML_Q5_K, dequantize

    generator = torch.Generator().manual_seed(19)
    q2 = torch.randint(0, 256, (3, 84), dtype=torch.uint8, generator=generator)
    q2[:, 80:82] = torch.tensor([0, 52], dtype=torch.uint8)
    q2[:, 82:84] = torch.tensor([0, 48], dtype=torch.uint8)
    q5 = torch.randint(0, 256, (2, 176), dtype=torch.uint8, generator=generator)
    q5[:, 0:2] = torch.tensor([0, 52], dtype=torch.uint8)
    q5[:, 2:4] = torch.tensor([0, 48], dtype=torch.uint8)
    layer = GGUFMergedLinear(256, [(3, GGML_Q2_K), (2, GGML_Q5_K)])
    layer.parts.op_list[0].qweight = q2
    layer.parts.op_list[1].qweight = q5
    x = torch.randn(4, 256, generator=generator)

    actual = layer.forward(x)
    expected = torch.cat(
        (
            x @ dequantize(q2, GGML_Q2_K, x.dtype).reshape(3, 256).T,
            x @ dequantize(q5, GGML_Q5_K, x.dtype).reshape(2, 256).T,
        ),
        dim=-1,
    )

    torch.testing.assert_close(actual, expected)
    assert set(layer.state_dict()) == {"parts.0.qweight", "parts.1.qweight"}


def test_qwen35_gguf_v_head_layout_round_trip():
    op = object.__new__(Qwen3_5GatedDeltaNet)
    op.num_k_heads = 2
    op.num_v_heads = 6
    grouped = torch.arange(2 * 6 * 4).reshape(2, 6, 4)

    tiled = op._v_grouped_to_tiled(grouped)

    torch.testing.assert_close(tiled[:, :, 0], grouped[:, [0, 3, 1, 4, 2, 5], 0])
    torch.testing.assert_close(op._v_tiled_to_grouped(tiled), grouped)
