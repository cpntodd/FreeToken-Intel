from __future__ import annotations

import argparse
import json

import gguf
import numpy as np
import torch

from freetoken.kernel.sycl.causal_conv1d import q6_k_matvec_sycl
from freetoken.models.gguf.dequant import GGML_Q6_K, dequantize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the SYCL Q6_K matvec on bounded rows from a GGUF tensor."
    )
    parser.add_argument("model", help="GGUF checkpoint path")
    parser.add_argument("--tensor", default="blk.64.attn_output.weight")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("an Intel XPU is required")
    if args.rows < 1 or args.tokens < 1:
        raise ValueError("rows and tokens must be positive")

    reader = gguf.GGUFReader(args.model)
    tensor = next((item for item in reader.tensors if item.name == args.tensor), None)
    if tensor is None:
        raise ValueError(f"tensor not found: {args.tensor}")
    if int(tensor.tensor_type) != GGML_Q6_K:
        raise ValueError(f"{args.tensor} is not Q6_K")

    ggml_shape = tuple(int(dim) for dim in tensor.shape)
    width = ggml_shape[0]
    if width % 256:
        raise ValueError(f"Q6_K input width must be divisible by 256, got {width}")
    total_rows = int(np.prod(ggml_shape[1:])) if len(ggml_shape) > 1 else 1
    row_bytes = width // 256 * 210
    if args.rows > total_rows:
        raise ValueError(f"requested {args.rows} rows but tensor has {total_rows}")

    packed_view = np.asarray(tensor.data).reshape(total_rows, row_bytes)
    packed_cpu = torch.from_numpy(np.array(packed_view[: args.rows], copy=True))
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    generator = torch.Generator().manual_seed(args.seed)
    x_cpu = torch.randn((args.tokens, width), generator=generator, dtype=dtype)
    weight = dequantize(packed_cpu, GGML_Q6_K, torch.float32).reshape(args.rows, width)
    expected = (x_cpu.float() @ weight.T).to(dtype)

    actual = q6_k_matvec_sycl(x_cpu.to("xpu"), packed_cpu.to("xpu"))
    torch.xpu.synchronize()
    actual_cpu = actual.cpu()
    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(actual_cpu, expected, rtol=tolerance, atol=tolerance)

    absolute_error = (actual_cpu.float() - expected.float()).abs()
    relative_error = absolute_error / expected.float().abs().clamp_min(1e-12)
    print(
        json.dumps(
            {
                "device": torch.xpu.get_device_name(0),
                "tensor": args.tensor,
                "ggml_shape": ggml_shape,
                "ggml_type": "Q6_K",
                "rows_tested": args.rows,
                "input_width": width,
                "tokens": args.tokens,
                "dtype": args.dtype,
                "xpu_output": str(actual.device),
                "max_abs_error": float(absolute_error.max()),
                "max_rel_error": float(relative_error.max()),
                "parity": True,
            }
        )
    )


if __name__ == "__main__":
    main()
