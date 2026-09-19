from __future__ import annotations

import torch
from freetoken.kernel.index import indexing


def test_portable_index_matches_torch_embedding():
    weights = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    indices = torch.tensor([5, 1, 3])

    actual = indexing(weights, indices)

    torch.testing.assert_close(actual, weights[indices])


def test_portable_index_masks_tokens_outside_vocab_shard():
    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    indices = torch.tensor([1, 2, 4, 5])

    actual = indexing(weights, indices, vocab_range=(2, 3))

    expected = torch.stack((torch.zeros(4), weights[0], weights[2], torch.zeros(4)))
    torch.testing.assert_close(actual, expected)
