from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearOProj, LinearQKVMerged, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.utils import div_even, init_logger, nvtx_annotate

logger = init_logger(__name__)

if TYPE_CHECKING:
    import torch

    from freetoken.models.config import ModelConfig


class LlamaAttention(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
        prefix: str = "",
    ):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self._openvino_qkv_island = None
        self._openvino_fallback_logged = False
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.head_dim = head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
            quant_config=config.quant,
            prefix=f"{prefix}.qkv_proj",
        )
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.o_proj",
        )

    def configure_openvino_qkv_island(self, island) -> None:
        if self.layer_id != 0:
            raise ValueError("the OpenVINO QKV island is restricted to Llama layer 0")
        self._openvino_qkv_island = island
        self._openvino_fallback_logged = False

    def _project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        island = self._openvino_qkv_island
        if island is None:
            return self.qkv_proj.forward(x)

        result = island(x)
        self._openvino_last_result = result
        logger.debug(
            "OpenVINO Llama layer-0 QKV backend=%s input_copy_s=%.6f "
            "inference_s=%.6f output_copy_s=%.6f fallback_reason=%s",
            result.backend,
            result.input_copy_seconds,
            result.inference_seconds,
            result.output_copy_seconds,
            result.fallback_reason,
        )
        if result.backend == "xpu" and not self._openvino_fallback_logged:
            logger.warning(
                "OpenVINO Llama layer-0 QKV used explicit XPU fallback: %s",
                result.fallback_reason,
            )
            self._openvino_fallback_logged = True
        return result.output.to(dtype=x.dtype)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self._project_qkv(x)
        del x
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self.o_proj.forward(o.view(-1, self.qo_attn_dim))


__all__ = ["LlamaAttention"]
