from __future__ import annotations

import argparse
import json
import statistics

import torch
from freetoken.accelerator.openvino import OpenVINODenseIsland


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVINO GPU compute-island benchmark")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--output-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--source", choices=("cpu", "xpu"), default="xpu")
    args = parser.parse_args()

    weight = torch.randn(args.output_size, args.hidden_size, dtype=torch.float16)
    island = OpenVINODenseIsland(
        weight, max_batch_tokens=args.tokens, activation="silu", device="GPU"
    )
    hidden_states = torch.randn(
        args.tokens, args.hidden_size, dtype=torch.float16, device=args.source
    )

    island(hidden_states)
    measurements = [island(hidden_states) for _ in range(args.iterations)]
    print(
        json.dumps(
            {
                "requested_device": island.execution_info.requested_device,
                "execution_devices": island.execution_info.execution_devices,
                "full_device_name": island.execution_info.full_device_name,
                "source_device": args.source,
                "tokens": args.tokens,
                "hidden_size": args.hidden_size,
                "output_size": args.output_size,
                "iterations": args.iterations,
                "median_input_copy_ms": 1000
                * statistics.median(m.input_copy_seconds for m in measurements),
                "median_inference_ms": 1000
                * statistics.median(m.inference_seconds for m in measurements),
                "median_output_copy_ms": 1000
                * statistics.median(m.output_copy_seconds for m in measurements),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
