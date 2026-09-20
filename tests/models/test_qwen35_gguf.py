from dataclasses import replace
from types import SimpleNamespace

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


def _hadamard_metadata(**overrides):
    metadata = {
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 1024,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.weight_names": ["blk.0.attn_q.weight"],
        "prism.hadamard.inverse_weight_names": ["token_embd.weight"],
        "prism.hadamard.sign_widths": [5120],
        "prism.hadamard.sign_values": [1] * 5120,
        "prism.hadamard.gdn_v_grouped": True,
    }
    metadata.update(overrides)
    return metadata


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
    metadata = _hadamard_metadata(
        **{"prism.hadamard.weight_names": ["blk.0.router.weight"]}
    )

    with pytest.raises(NotImplementedError, match="verified Qwen3.5 GGUF paths"):
        parse_gguf_config(_shim(**metadata))


def test_qwen35_metadata_parses_supported_prism_hadamard_contract():
    shim = _shim(**_hadamard_metadata())

    config = parse_gguf_config(shim)

    assert config.gguf_hadamard is not None
    assert config.gguf_hadamard.block_size == 1024
    assert config.gguf_hadamard.weight_names == frozenset({"blk.0.attn_q.weight"})
    assert config.gguf_hadamard.inverse_weight_names == frozenset({"token_embd.weight"})
    assert config.gguf_hadamard.gdn_v_grouped
    assert config.gguf_hadamard.signs_on(5120, "cpu").tolist() == [1] * 5120


def test_qwen35_metadata_rejects_incomplete_prism_hadamard_contract():
    with pytest.raises(ValueError, match="incomplete Prism Hadamard metadata"):
        parse_gguf_config(_shim(**{"prism.hadamard.version": 1}))


def test_qwen35_metadata_rejects_bad_prism_hadamard_sign_vector():
    metadata = _hadamard_metadata()
    metadata["prism.hadamard.sign_values"] = [1] * 5119

    with pytest.raises(ValueError, match="length does not match"):
        parse_gguf_config(_shim(**metadata))


def _small_hadamard_reference(x, signs, *, inverse=False):
    matrix = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=torch.float32,
    ) / 2
    values = x.float()
    if not inverse:
        values = values * signs
    values = values.reshape(-1, 4) @ matrix.T
    if inverse:
        values = values.reshape_as(x) * signs
    return values.reshape_as(x).to(x.dtype)


def test_qwen35_gguf_linear_applies_only_declared_hadamard_weight(monkeypatch):
    from freetoken.layers import gguf as gguf_layers
    from freetoken.layers.gguf import GGUFLinear
    from freetoken.models.config import GGUFHadamardConfig
    from freetoken.models.gguf.dequant import GGML_F32

    signs = (1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1)
    config = GGUFHadamardConfig(
        block_size=4,
        weight_names=frozenset({"blk.0.ffn_down.weight"}),
        inverse_weight_names=frozenset(),
        signs_by_width={12: signs},
    )
    x = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    captured = []
    monkeypatch.setattr(
        gguf_layers,
        "fused_mul_mat_gguf",
        lambda value, _weight, _quant_type: captured.append(value.clone()) or value,
    )
    selected = GGUFLinear(
        12,
        1,
        GGML_F32,
        hadamard_config=config,
        hadamard_weight_name="blk.0.ffn_down.weight",
    )
    unselected = GGUFLinear(
        12,
        1,
        GGML_F32,
        hadamard_config=config,
        hadamard_weight_name="blk.0.ffn_up.weight",
    )

    selected.forward(x)
    unselected.forward(x)

    sign_tensor = torch.tensor(signs, dtype=torch.float32)
    torch.testing.assert_close(
        captured[0], _small_hadamard_reference(x, sign_tensor)
    )
    torch.testing.assert_close(captured[1], x)


def test_qwen35_gguf_linear_applies_grouped_gdn_permutation_before_hadamard(
    monkeypatch,
):
    from freetoken.layers import gguf as gguf_layers
    from freetoken.layers.gguf import GGUFLinear
    from freetoken.models.config import GGUFHadamardConfig
    from freetoken.models.gguf.dequant import GGML_F32

    name = "blk.0.ssm_out.weight"
    signs = (1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1)
    config = GGUFHadamardConfig(
        block_size=4,
        weight_names=frozenset({name}),
        inverse_weight_names=frozenset(),
        signs_by_width={12: signs},
        gdn_v_grouped=True,
    )
    x = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    captured = []
    monkeypatch.setattr(
        gguf_layers,
        "fused_mul_mat_gguf",
        lambda value, _weight, _quant_type: captured.append(value.clone()) or value,
    )
    layer = GGUFLinear(
        12,
        1,
        GGML_F32,
        hadamard_config=config,
        hadamard_weight_name=name,
        hadamard_permutation=(2, 3, 2),
    )

    layer.forward(x)

    grouped = x.reshape(-1, 3, 2, 2).transpose(1, 2).contiguous().reshape(-1, 12)
    expected = _small_hadamard_reference(grouped, torch.tensor(signs).float())
    torch.testing.assert_close(captured[0], expected)


def test_qwen35_gguf_embedding_applies_inverse_hadamard_after_lookup(monkeypatch):
    from freetoken.layers.gguf import GGUFEmbedding
    from freetoken.models.config import GGUFHadamardConfig
    from freetoken.models.gguf import dequant
    from freetoken.models.gguf.dequant import GGML_F32

    signs = (1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1)
    config = GGUFHadamardConfig(
        block_size=4,
        weight_names=frozenset(),
        inverse_weight_names=frozenset({"token_embd.weight"}),
        signs_by_width={12: signs},
    )
    embedding = GGUFEmbedding(
        4,
        12,
        GGML_F32,
        hadamard_config=config,
        hadamard_weight_name="token_embd.weight",
    )
    embedding.qweight[:, 0] = torch.arange(4, dtype=torch.uint8)

    def fake_dequantize(rows, _quant_type, dtype):
        return (
            rows[:, :1].float()
            + torch.arange(12, dtype=torch.float32).unsqueeze(0)
        ).to(dtype)

    monkeypatch.setattr(dequant, "dequantize", fake_dequantize)
    token_ids = torch.tensor([1, 3])

    actual = embedding.forward(token_ids)

    latent = fake_dequantize(embedding.qweight[token_ids], GGML_F32, torch.bfloat16)
    expected = _small_hadamard_reference(
        latent, torch.tensor(signs).float(), inverse=True
    )
    torch.testing.assert_close(actual, expected)


def test_convert_qwen35_gguf_attaches_declared_hadamard_paths():
    from freetoken.models.gguf.dequant import GGML_PQ2_0
    from freetoken.models.qwen3_5_moe.gguf import convert_qwen35_to_gguf

    weight_names = [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_out.weight",
        "blk.0.ffn_down.weight",
        "output.weight",
    ]
    metadata = _hadamard_metadata(
        **{
            "prism.hadamard.weight_names": weight_names,
            "prism.hadamard.sign_widths": [5120, 6144, 17408],
            "prism.hadamard.sign_values": [1] * (5120 + 6144 + 17408),
        }
    )
    config = parse_gguf_config(_shim(**metadata))
    tensor_names = {
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_out.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    }
    config = replace(
        config,
        vocab_size=4,
        gguf_tensor_types={name: GGML_PQ2_0 for name in tensor_names},
    )
    linear = SimpleNamespace(conv_dim=10240, value_dim=6144, num_v_heads=48)
    layer = SimpleNamespace(
        _is_linear=True,
        linear_attn=linear,
        mlp=SimpleNamespace(gate_up_proj=None, down_proj=None),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=None,
            layers=SimpleNamespace(op_list=[layer]),
        ),
        lm_head=None,
    )

    convert_qwen35_to_gguf(model, config)

    assert model.model.embed_tokens._hadamard_weight_name == "token_embd.weight"
    assert linear.in_proj.parts.op_list[0]._hadamard_weight_name == weight_names[0]
    assert linear.in_proj.parts.op_list[1]._hadamard_weight_name == weight_names[1]
    assert linear.out_proj._hadamard_weight_name == weight_names[2]
    assert linear.out_proj._hadamard_permutation == (16, 3, 128)
    assert layer.mlp.down_proj._hadamard_weight_name == weight_names[3]
    assert model.lm_head.proj._hadamard_weight_name == weight_names[4]


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


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")
def test_qwen35_tied_gguf_head_uses_embedding_on_xpu(monkeypatch):
    from types import SimpleNamespace

    from freetoken import core
    from freetoken.layers.gguf import GGUFEmbedding, fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q4_0
    from freetoken.models.qwen3_5_moe.gguf import GGUFTiedLMHead

    monkeypatch.setattr(
        core,
        "_GLOBAL_CTX",
        SimpleNamespace(batch=SimpleNamespace(is_prefill=False)),
    )
    generator = torch.Generator().manual_seed(603)
    embedding = GGUFEmbedding(7, 32, GGML_Q4_0)
    packed = torch.randint(0, 256, (7, 18), dtype=torch.uint8, generator=generator)
    packed[:, :2] = torch.tensor([0.5], dtype=torch.float16).view(torch.uint8)
    embedding.qweight = packed.to("xpu")
    hidden = torch.randn(2, 32, dtype=torch.bfloat16, generator=generator).to("xpu")

    actual = GGUFTiedLMHead(embedding, GGML_Q4_0).forward(hidden)
    expected = fused_mul_mat_gguf(hidden, embedding.qweight, GGML_Q4_0)
    torch.xpu.synchronize()

    assert actual.device.type == "xpu"
    torch.testing.assert_close(actual, expected)


def test_qwen35_tied_gguf_ignores_redundant_output_weight(monkeypatch):
    from types import SimpleNamespace

    from freetoken import utils
    from freetoken.models.gguf import reader
    from freetoken.models.qwen3_5_moe import gguf

    config = SimpleNamespace(
        tie_word_embeddings=True,
        num_layers=0,
    )
    redundant_output = SimpleNamespace(
        name="output.weight",
        packed=lambda: torch.empty((1, 18), dtype=torch.uint8),
    )
    monkeypatch.setattr(utils, "cached_load_hf_config", lambda _path: object())
    monkeypatch.setattr(gguf, "parse_gguf_config", lambda _shim: config)
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _path: [redundant_output])

    assert (
        list(
            gguf.iter_gguf_weights(
                "tied.gguf",
                torch.device("cpu"),
                include_moe_experts=False,
                include_non_moe=True,
            )
        )
        == []
    )
