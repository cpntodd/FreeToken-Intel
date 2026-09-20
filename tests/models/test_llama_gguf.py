from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.accelerator.openvino import OpenVINOIslandResult
from freetoken.models.llama.attention import LlamaAttention
from freetoken.models.llama.gguf import _reverse_rope_permute, parse_gguf_config


def test_parse_llama_gguf_config():
    shim = SimpleNamespace(
        metadata={
            "llama.block_count": 16,
            "llama.embedding_length": 2048,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 64,
            "llama.feed_forward_length": 8192,
            "llama.attention.layer_norm_rms_epsilon": 1e-5,
            "llama.rope.dimension_count": 64,
            "llama.context_length": 131072,
            "llama.rope.freq_base": 500000.0,
        },
        vocab_size=128256,
        tie_word_embeddings=True,
        architectures=["LlamaGGUFForCausalLM"],
    )

    config = parse_gguf_config(shim)

    assert config.num_layers == 16
    assert config.hidden_size == 2048
    assert config.num_qo_heads == 32
    assert config.num_kv_heads == 8
    assert config.rotary_config.max_position == 131072
    assert config.tie_word_embeddings is True
    assert config.architectures == ["LlamaGGUFForCausalLM"]


def test_reverse_rope_permute_restores_hf_row_order():
    gguf_rows = torch.tensor([[0], [2], [1], [3], [4], [6], [5], [7]])

    restored = _reverse_rope_permute(gguf_rows, num_heads=2)

    torch.testing.assert_close(restored, torch.arange(8).reshape(8, 1))


def test_llama_qkv_openvino_island_is_opt_in_and_restores_activation_dtype():
    hidden_states = torch.zeros((2, 8), dtype=torch.bfloat16)
    island_result = OpenVINOIslandResult(
        output=torch.ones((2, 12), dtype=torch.float16),
        input_copy_seconds=0.1,
        inference_seconds=0.2,
        output_copy_seconds=0.3,
    )

    class _Island:
        def __call__(self, value):
            assert value is hidden_states
            return island_result

    attention = object.__new__(LlamaAttention)
    attention.layer_id = 0
    attention._openvino_qkv_island = None
    attention._openvino_fallback_logged = False
    attention.qkv_proj = SimpleNamespace(forward=lambda _value: torch.zeros(2, 12))

    native = attention._project_qkv(hidden_states)
    attention.configure_openvino_qkv_island(_Island())
    accelerated = attention._project_qkv(hidden_states)

    assert native.dtype == torch.float32
    assert accelerated.dtype == hidden_states.dtype
    assert attention._openvino_last_result is island_result


def test_llama_qkv_openvino_island_rejects_other_layers():
    attention = object.__new__(LlamaAttention)
    attention.layer_id = 1

    with pytest.raises(ValueError, match="restricted to Llama layer 0"):
        attention.configure_openvino_qkv_island(object())
