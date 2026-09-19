from __future__ import annotations

from types import SimpleNamespace

import torch
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
