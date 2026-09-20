"""Compare native SYCL packed GGUF matvecs with the XPU dequant+matmul path.

Run on an Intel XPU with:
    FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python .venv/bin/python \
        benchmarks/bench_sycl_gguf_batches.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from freetoken.kernel.sycl.causal_conv1d import (
    iq4_nl_matvec_sycl,
    q4_0_matvec_sycl,
    q4_1_matvec_sycl,
    q5_0_matvec_sycl,
    q5_1_matvec_sycl,
)
from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_IQ4_NL,
    GGML_Q4_0,
    GGML_Q4_1,
    GGML_Q5_0,
    GGML_Q5_1,
    dequantize,
)


KERNELS = {
    "Q4_0": (GGML_Q4_0, q4_0_matvec_sycl),
    "Q4_1": (GGML_Q4_1, q4_1_matvec_sycl),
    "IQ4_NL": (GGML_IQ4_NL, iq4_nl_matvec_sycl),
    "Q5_0": (GGML_Q5_0, q5_0_matvec_sycl),
    "Q5_1": (GGML_Q5_1, q5_1_matvec_sycl),
}


def _parse_tokens(value: str) -> list[int]:
    tokens = [int(part) for part in value.split(",")]
    if not tokens or any(token <= 0 for token in tokens):
        raise argparse.ArgumentTypeError("token counts must be positive integers")
    return tokens


def _packed_weight(
    name: str, quant_type: int, rows: int, in_features: int, seed: int
) -> torch.Tensor:
    block, block_bytes = BLOCK_SHAPE[quant_type]
    if in_features % block:
        raise ValueError(f"in_features must be divisible by {block} for {name}")

    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0,
        256,
        (rows, in_features // block * block_bytes),
        dtype=torch.uint8,
        generator=generator,
    )
    blocks = packed.view(rows, in_features // block, block_bytes)
    blocks[:, :, 0] = 0
    blocks[:, :, 1] = 56
    if name in {"Q4_1", "Q5_1"}:
        blocks[:, :, 2] = 0
        blocks[:, :, 3] = 52
    return packed


def _xpu_fallback(x: torch.Tensor, packed: torch.Tensor, quant_type: int):
    outputs = []
    for start in range(0, packed.shape[0], 4096):
        chunk = packed[start : start + 4096]
        weight = dequantize(chunk, quant_type, x.dtype).reshape(chunk.shape[0], -1)
        outputs.append(x @ weight.T)
    return torch.cat(outputs, dim=-1)


def _elapsed_ms(fn) -> float:
    torch.xpu.synchronize()
    start = time.perf_counter_ns()
    fn()
    torch.xpu.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def _benchmark_case(
    name: str,
    quant_type: int,
    kernel,
    tokens: int,
    rows: int,
    in_features: int,
    dtype: torch.dtype,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    packed_cpu = _packed_weight(name, quant_type, rows, in_features, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    x_cpu = torch.randn(tokens, in_features, dtype=dtype, generator=generator)
    packed = packed_cpu.to("xpu")
    x = x_cpu.to("xpu")

    direct = lambda: kernel(x, packed)
    fallback = lambda: _xpu_fallback(x, packed, quant_type)
    for _ in range(warmup):
        direct()
        fallback()
    torch.xpu.synchronize()

    direct_output = direct()
    fallback_output = fallback()
    torch.xpu.synchronize()
    tolerance = 5e-5 if dtype == torch.float32 else 6e-2
    torch.testing.assert_close(
        direct_output,
        fallback_output,
        rtol=tolerance,
        atol=tolerance,
    )
    max_abs_error = (
        direct_output.float() - fallback_output.float()
    ).abs().max().item()

    timings = {"direct_sycl_ms": [], "xpu_dequant_matmul_ms": []}
    for iteration in range(iterations):
        paths = [("direct_sycl_ms", direct), ("xpu_dequant_matmul_ms", fallback)]
        if iteration % 2:
            paths.reverse()
        for label, fn in paths:
            timings[label].append(_elapsed_ms(fn))

    direct_ms = statistics.median(timings["direct_sycl_ms"])
    fallback_ms = statistics.median(timings["xpu_dequant_matmul_ms"])
    return {
        "format": name,
        "tokens": tokens,
        "rows": rows,
        "in_features": in_features,
        "dtype": str(dtype).removeprefix("torch."),
        "fallback_median_ms": fallback_ms,
        "direct_sycl_median_ms": direct_ms,
        "fallback_over_direct_speedup": fallback_ms / direct_ms,
        "max_abs_error_vs_fallback": max_abs_error,
        "parity": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_parse_tokens, default=_parse_tokens("1,4,8,16,32"))
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=5120)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("this benchmark requires an Intel XPU")
    if args.rows <= 0 or args.in_features <= 0:
        raise ValueError("rows and in_features must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    results = []
    for name, (quant_type, kernel) in KERNELS.items():
        for index, tokens in enumerate(args.tokens):
            results.append(
                _benchmark_case(
                    name,
                    quant_type,
                    kernel,
                    tokens,
                    args.rows,
                    args.in_features,
                    dtype,
                    seed=1000 + index,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )

    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "device": torch.xpu.get_device_name(0),
                "platform": torch.xpu.get_device_properties(0).platform_name,
                "benchmark": "synthetic packed GGUF matvec; includes XPU dequantization",
                "warmup": args.warmup,
                "iterations": args.iterations,
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
