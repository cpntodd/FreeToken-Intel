from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class TorchAttentionMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    q_positions: torch.Tensor
    swa_indices: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TorchAttentionBackend(BaseAttnBackend):
    """Portable eager paged attention used by the native XPU path.

    This intentionally favors correctness over kernel-level performance. It keeps
    FreeToken's existing KV ownership and batching semantics while expressing the
    attention math as ordinary PyTorch operations supported by CUDA and XPU.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchAttentionMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads, head_dim = k_raw.shape[-2:]
        q_heads = q.shape[1]
        if q_heads % kv_heads:
            raise ValueError(f"query heads ({q_heads}) must be divisible by KV heads ({kv_heads})")
        k_cache = k_raw.view(-1, kv_heads, head_dim)
        v_cache = v_raw.view(-1, kv_heads, head_dim)
        spec = attn_spec or AttentionSpec()
        scale = spec.sm_scale if spec.sm_scale is not None else head_dim**-0.5
        indices = metadata.swa_indices if spec.sliding_window is not None and metadata.swa_indices is not None else metadata.indices

        outputs: list[torch.Tensor] = []
        group = q_heads // kv_heads
        for req_idx in range(metadata.indptr.numel() - 1):
            q_lo = int(metadata.cu_seqlens_q[req_idx].item())
            q_hi = int(metadata.cu_seqlens_q[req_idx + 1].item())
            kv_lo = int(metadata.indptr[req_idx].item())
            kv_hi = int(metadata.indptr[req_idx + 1].item())
            req_indices = indices[kv_lo:kv_hi].long()
            req_k = k_cache.index_select(0, req_indices).repeat_interleave(group, dim=1)
            req_v = v_cache.index_select(0, req_indices).repeat_interleave(group, dim=1)
            req_q = q[q_lo:q_hi]

            logits = torch.einsum("qhd,khd->hqk", req_q.float(), req_k.float()) * scale
            query_positions = metadata.q_positions[q_lo:q_hi].long()
            key_positions = torch.arange(kv_hi - kv_lo, device=q.device)
            mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            if spec.sliding_window is not None:
                mask &= key_positions.unsqueeze(0) > (
                    query_positions.unsqueeze(1) - spec.sliding_window
                )
            logits.masked_fill_(~mask.unsqueeze(0), float("-inf"))

            values = req_v.transpose(0, 1).float()
            if spec.sinks is not None:
                sink_logits = spec.sinks[:q_heads].float().view(q_heads, 1, 1)
                sink_logits = sink_logits.expand(-1, q_hi - q_lo, -1)
                logits = torch.cat((logits, sink_logits), dim=-1)
                values = torch.cat(
                    (values, torch.zeros(q_heads, 1, head_dim, device=q.device)), dim=1
                )
            probs = torch.softmax(logits, dim=-1)
            outputs.append(torch.einsum("hqk,hkd->qhd", probs, values).to(q.dtype))
        return torch.cat(outputs, dim=0)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        page_table = get_global_ctx().page_table
        q_lens = [req.extend_len for req in reqs]
        kv_lens = [req.device_len for req in reqs]
        cu_seqlens_q = torch.tensor([0] + q_lens, dtype=torch.int32, device=self.device).cumsum_(0)
        indptr = torch.tensor([0] + kv_lens, dtype=torch.int32, device=self.device).cumsum_(0)
        indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        swa_indices = None
        if getattr(self.kvcache, "swa_paged", False):
            swa_indices = self.kvcache.translate_loc_from_full_to_swa(indices)
        positions = batch.positions
        if positions is None:
            positions = torch.zeros(sum(q_lens), dtype=torch.int64, device=self.device)
        batch.attn_metadata = TorchAttentionMetadata(
            cu_seqlens_q=cu_seqlens_q,
            indptr=indptr,
            indices=indices,
            q_positions=positions,
            swa_indices=swa_indices,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        if bs_list:
            raise RuntimeError("the portable torch attention backend does not support graph capture")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise RuntimeError("the portable torch attention backend does not support graph capture")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise RuntimeError("the portable torch attention backend does not support graph capture")
