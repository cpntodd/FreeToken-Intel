from freetoken.models.gguf.config import GgufConfigShim
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
