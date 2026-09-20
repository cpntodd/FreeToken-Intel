from __future__ import annotations

import argparse
import json
from pathlib import Path

import openvino as ov
import torch
from torch.utils.cpp_extension import load


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test whether OpenVINO GPU can consume a PyTorch XPU USM pointer."
    )
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("the OpenVINO USM probe requires a PyTorch XPU device")

    repo_root = Path(__file__).resolve().parents[1]
    openvino_root = Path(ov.__file__).resolve().parent
    openvino_libraries = openvino_root / "libs"
    runtime_candidates = sorted(openvino_libraries.glob("libopenvino.so.*"))
    if not runtime_candidates:
        raise RuntimeError(
            f"OpenVINO C++ runtime library not found in {openvino_libraries}"
        )

    extension = load(
        name="freetoken_openvino_usm_probe",
        sources=[str(repo_root / "experiments/openvino_usm_probe.cpp")],
        extra_include_paths=[str(openvino_root / "include")],
        extra_ldflags=[
            str(runtime_candidates[-1]),
            f"-Wl,-rpath,{openvino_libraries}",
        ],
        extra_cflags=["-O2", "-std=c++17"],
        verbose=args.verbose_build,
    )

    core = ov.Core()
    expected_device = str(core.get_property("GPU", "FULL_DEVICE_NAME"))
    context_type = str(
        core.get_default_context("GPU").get_params()["CONTEXT_TYPE"].astype(str)
    )
    source = torch.arange(16, dtype=torch.float32, device="xpu").reshape(2, 8)
    torch.xpu.synchronize()
    try:
        result = extension.probe_xpu_usm(source)
    except RuntimeError as error:
        print(
            json.dumps(
                {
                    "torch_device": torch.xpu.get_device_name(0),
                    "torch_platform": torch.xpu.get_device_properties(0).platform_name,
                    "openvino_device": expected_device,
                    "openvino_context_type": context_type,
                    "remote_tensor_created": False,
                    "pointer_identity": None,
                    "output_parity": None,
                    "error": str(error),
                },
                indent=2,
            )
        )
        return 2
    torch.xpu.synchronize()

    actual = torch.tensor(result["output"], dtype=torch.float32).reshape(2, 8)
    torch.testing.assert_close(actual, (source * 2).cpu(), rtol=0, atol=0)
    if not result["pointer_identity"]:
        raise RuntimeError(
            "OpenVINO copied instead of wrapping the PyTorch XPU allocation"
        )

    report = {
        "torch_device": torch.xpu.get_device_name(0),
        "torch_platform": torch.xpu.get_device_properties(0).platform_name,
        "openvino_device": expected_device,
        "openvino_context_type": context_type,
        "openvino_execution_devices": result["execution_devices"],
        "remote_tensor_created": True,
        "pointer_identity": result["pointer_identity"],
        "output_parity": True,
        "queue_mode": "serialized with torch.xpu.synchronize before and after infer",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
