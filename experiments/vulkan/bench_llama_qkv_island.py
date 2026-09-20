from __future__ import annotations

import argparse
import gc
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from freetoken.core import SamplingParams
from freetoken.llm import LLM


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = REPO_ROOT / "experiments" / "vulkan" / "run_probe.sh"


def _sync_xpu() -> None:
    torch.xpu.synchronize()


def _write_f16(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    little_endian = np.asarray(array, dtype="<f2")
    little_endian.tofile(path)


def _write_f32(path: Path, tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().to(torch.float32).cpu().contiguous().numpy()
    little_endian = np.asarray(array, dtype="<f4")
    little_endian.tofile(path)
    return little_endian


def _metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = max(float(np.linalg.norm(reference.astype(np.float64))), 1e-12)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
    }


def _numeric_device_id(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value), 0)
    except ValueError:
        try:
            return int(str(value), 16)
        except ValueError:
            return None


def _capture_one_qkv(args: argparse.Namespace, artifact_dir: Path) -> dict:
    started = time.perf_counter()
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        accelerator="xpu",
        cuda_graph_max_bs=0,
        max_running_req=1,
        max_seq_len_override=args.context,
        num_token_override=args.kv_tokens,
    )
    load_seconds = time.perf_counter() - started
    engine = llm.engine
    if engine.runtime.kind != "xpu" or engine.device.type != "xpu":
        llm.shutdown()
        raise RuntimeError("the model must be loaded on XPU for this experiment")
    capabilities = engine.runtime.capabilities(
        0 if engine.device.index is None else engine.device.index
    )
    attention = engine.model.model.layers.op_list[0].self_attn
    projection = attention.qkv_proj
    weight = projection.weight
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        llm.shutdown()
        raise RuntimeError("layer-0 QKV must expose a dense two-dimensional weight")
    if weight.device.type != "xpu":
        llm.shutdown()
        raise RuntimeError("layer-0 QKV weight is not resident on XPU")
    if getattr(projection, "bias", None) is not None:
        llm.shutdown()
        raise RuntimeError("the Vulkan probe currently requires a bias-free QKV")
    if getattr(attention, "_openvino_qkv_island", None) is not None:
        llm.shutdown()
        raise RuntimeError("disable the OpenVINO QKV island for native-XPU capture")

    captured: dict[str, torch.Tensor] = {}
    original_project_qkv = attention._project_qkv

    def capture_project_qkv(activation: torch.Tensor) -> torch.Tensor:
        output = original_project_qkv(activation)
        if not captured and activation.ndim == 2 and activation.shape[0] >= args.rows:
            captured["activation"] = activation[: args.rows].detach().clone()
            captured["weight"] = projection.weight.detach().clone()
            captured["native_output"] = output[: args.rows].detach().clone()
        return output

    try:
        attention._project_qkv = capture_project_qkv
        input_ids: list[int] | None = None
        for repeat_count in range(args.rows + 8, args.rows - 1, -1):
            prompt = "Hi. " * repeat_count
            candidate_ids = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )["input_ids"]
            if args.rows <= len(candidate_ids) <= args.context:
                input_ids = candidate_ids
                break
        if input_ids is None:
            raise RuntimeError(
                "could not fit enough prefill rows within the configured context"
            )
        generation_started = time.perf_counter()
        llm.generate(
            [input_ids],
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        _sync_xpu()
        generation_seconds = time.perf_counter() - generation_started
    finally:
        attention._project_qkv = original_project_qkv

    if "activation" not in captured:
        llm.shutdown()
        raise RuntimeError(
            f"no layer-0 prefill with at least {args.rows} rows was captured"
        )

    activation = captured["activation"]
    dense_weight = captured["weight"]
    native_output = captured["native_output"]
    _sync_xpu()
    cast_started = time.perf_counter()
    activation_f16 = activation.to(torch.float16).contiguous()
    weight_f16 = dense_weight.to(torch.float16).contiguous()
    _sync_xpu()
    input_cast_ms = (time.perf_counter() - cast_started) * 1000.0

    input_copy_started = time.perf_counter()
    activation_host = activation_f16.cpu().contiguous()
    _sync_xpu()
    input_xpu_to_host_ms = (time.perf_counter() - input_copy_started) * 1000.0
    weight_copy_started = time.perf_counter()
    weight_host = weight_f16.cpu().contiguous()
    _sync_xpu()
    weight_xpu_to_host_ms = (time.perf_counter() - weight_copy_started) * 1000.0

    reference_started = time.perf_counter()
    with torch.inference_mode():
        xpu_reference = F.linear(activation_f16.float(), weight_f16.float())
        _sync_xpu()
    xpu_reference_ms = (time.perf_counter() - reference_started) * 1000.0

    native_copy_started = time.perf_counter()
    native_output_host = native_output.to(torch.float32).cpu().contiguous()
    _sync_xpu()
    native_output_xpu_to_host_ms = (
        time.perf_counter() - native_copy_started
    ) * 1000.0

    input_path = artifact_dir / "activation.f16"
    weight_path = artifact_dir / "weight.f16"
    reference_path = artifact_dir / "xpu_reference.f32"
    native_path = artifact_dir / "native_qkv.f32"
    _write_f16(input_path, activation_host)
    _write_f16(weight_path, weight_host)
    reference_values = _write_f32(reference_path, xpu_reference)
    native_values = _write_f32(native_path, native_output_host)

    manifest = {
        "stage": "LlamaAttention._project_qkv layer-0 prefill",
        "model": args.model,
        "accelerator": capabilities.kind,
        "device": capabilities.name,
        "device_id": capabilities.device_id,
        "driver": capabilities.driver_version,
        "rows": int(activation.shape[0]),
        "input_size": int(dense_weight.shape[1]),
        "output_size": int(dense_weight.shape[0]),
        "qkv_row_split": [
            int(attention.qo_attn_dim),
            int(attention.kv_attn_dim),
            int(attention.kv_attn_dim),
        ],
        "activation_dtype": str(activation.dtype),
        "weight_dtype": str(dense_weight.dtype),
        "native_output_dtype": str(native_output.dtype),
        "probe_input_dtype": "float16",
        "probe_output_dtype": "float32",
        "prompt_token_ids": [int(token) for token in input_ids],
        "timings_ms": {
            "model_load": load_seconds * 1000.0,
            "generation": generation_seconds * 1000.0,
            "xpu_fp16_operand_cast": input_cast_ms,
            "activation_xpu_to_host": input_xpu_to_host_ms,
            "weight_xpu_to_host_static": weight_xpu_to_host_ms,
            "xpu_fp32_reference": xpu_reference_ms,
            "native_output_xpu_to_host": native_output_xpu_to_host_ms,
        },
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    del activation, dense_weight, native_output, activation_f16, weight_f16
    del activation_host, weight_host, native_output_host, xpu_reference
    captured.clear()
    llm.shutdown()
    del llm
    gc.collect()
    torch.xpu.empty_cache()

    return {
        "manifest": manifest,
        "input_path": input_path,
        "weight_path": weight_path,
        "reference_values": reference_values,
        "native_values": native_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the Vulkan dense probe with a real Llama QKV prefill"
    )
    parser.add_argument("model", help="local Llama GGUF path")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--kv-tokens", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.context, args.kv_tokens, args.iterations) <= 0:
        parser.error("rows, context, kv-tokens, and iterations must be positive")
    if args.context < args.rows or args.kv_tokens < args.context:
        parser.error("context must cover rows and kv-tokens must cover context")

    if args.output_dir is None:
        artifact_dir = Path(tempfile.mkdtemp(prefix="freetoken-vulkan-qkv-"))
    else:
        artifact_dir = args.output_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if any(artifact_dir.iterdir()):
            parser.error("output-dir must be empty to avoid overwriting artifacts")

    capture = _capture_one_qkv(args, artifact_dir)
    output_path = artifact_dir / "vulkan_output.f32"
    rows = capture["manifest"]["rows"]
    input_size = capture["manifest"]["input_size"]
    output_size = capture["manifest"]["output_size"]
    probe_started = time.perf_counter()
    probe = subprocess.run(
        [
            "bash",
            str(PROBE_SCRIPT),
            "best",
            str(rows),
            str(input_size),
            str(output_size),
            str(args.iterations),
            str(capture["input_path"]),
            str(capture["weight_path"]),
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    probe_invocation_seconds = time.perf_counter() - probe_started
    probe_result = json.loads(probe.stdout)
    xpu_device_id = _numeric_device_id(capture["manifest"]["device_id"])
    vulkan_device_id = int(probe_result["device_id"])
    if xpu_device_id is not None and xpu_device_id != vulkan_device_id:
        raise RuntimeError(
            f"XPU device id {xpu_device_id:#x} does not match Vulkan device "
            f"id {vulkan_device_id:#x}"
        )
    vulkan_values = np.fromfile(output_path, dtype="<f4")
    expected_count = rows * output_size
    if vulkan_values.size != expected_count:
        raise RuntimeError("Vulkan output file has an invalid element count")
    vulkan_values = vulkan_values.reshape(rows, output_size)
    reference_values = capture["reference_values"]
    native_values = capture["native_values"]

    _sync_xpu()
    handoff_started = time.perf_counter()
    vulkan_on_xpu = torch.from_numpy(vulkan_values.copy()).to(device="xpu")
    vulkan_model_dtype = vulkan_on_xpu.to(dtype=torch.bfloat16)
    _sync_xpu()
    output_h2d_cast_ms = (time.perf_counter() - handoff_started) * 1000.0
    returned_values = vulkan_model_dtype.to(torch.float32).cpu().numpy()

    report = {
        "artifact_dir": str(artifact_dir),
        "capture": capture["manifest"],
        "vulkan": probe_result,
        "timings_ms": {
            "probe_invocation_including_compile_process_and_file_io": (
                probe_invocation_seconds * 1000.0
            ),
            "vulkan_output_cpu_to_xpu_and_bf16_cast": output_h2d_cast_ms,
        },
        "validation_gate": False,
        "device_ids_match": (
            None if xpu_device_id is None else xpu_device_id == vulkan_device_id
        ),
        "comparison": {
            "vulkan_vs_xpu_fp32_on_identical_fp16_operands": _metrics(
                vulkan_values, reference_values
            ),
            "vulkan_output_cast_to_bf16_vs_native_xpu_qkv": _metrics(
                returned_values, native_values
            ),
            "native_comparison_note": (
                "This includes BF16-to-FP16 operand conversion as well as backend "
                "math; use the identical-FP16-operand comparison to isolate Vulkan."
            ),
        },
    }
    (artifact_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
