from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F
from freetoken.accelerator.openvino import OpenVINODenseIsland


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenVINO GPU compute-island benchmark"
    )
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
    source_device = torch.device(args.source)
    hidden_states = torch.randn(
        args.tokens, args.hidden_size, dtype=torch.float16, device=source_device
    )
    source_weight = weight.to(source_device)

    def sync_source() -> None:
        if source_device.type == "xpu":
            torch.xpu.synchronize(source_device)

    def source_eager() -> torch.Tensor:
        return F.silu(hidden_states @ source_weight.T)

    def median_source_ms(fn) -> tuple[float, torch.Tensor]:
        for _ in range(3):
            fn()
        sync_source()
        samples = []
        result = None
        for _ in range(args.iterations):
            sync_source()
            started = time.perf_counter()
            result = fn()
            sync_source()
            samples.append((time.perf_counter() - started) * 1000)
        assert result is not None
        return statistics.median(samples), result

    for _ in range(3):
        island(hidden_states)
    sync_source()
    openvino_samples = []
    input_copy_samples = []
    inference_samples = []
    output_copy_samples = []
    openvino_result = None
    for _ in range(args.iterations):
        sync_source()
        started = time.perf_counter()
        openvino_result = island(hidden_states)
        sync_source()
        openvino_samples.append((time.perf_counter() - started) * 1000)
        input_copy_samples.append(openvino_result.input_copy_seconds * 1000)
        inference_samples.append(openvino_result.inference_seconds * 1000)
        output_copy_samples.append(openvino_result.output_copy_seconds * 1000)

    assert openvino_result is not None
    openvino_ms = statistics.median(openvino_samples)
    openvino_output = openvino_result.output
    source_eager_ms, eager_output = median_source_ms(source_eager)
    torch.testing.assert_close(openvino_output, eager_output, rtol=2e-3, atol=2e-3)
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
                "median_openvino_pipeline_ms": openvino_ms,
                "median_source_eager_ms": source_eager_ms,
                "openvino_to_source_eager_ratio": openvino_ms / source_eager_ms,
                "output_max_abs_error": (openvino_output - eager_output)
                .abs()
                .max()
                .item(),
                "median_input_copy_ms": statistics.median(input_copy_samples),
                "median_inference_ms": statistics.median(inference_samples),
                "median_output_copy_ms": statistics.median(output_copy_samples),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
