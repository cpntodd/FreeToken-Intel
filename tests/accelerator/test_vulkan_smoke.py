from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "experiments" / "vulkan" / "run_probe.sh"


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("glslc", "g++", "vulkaninfo")),
    reason="Vulkan probe toolchain unavailable",
)
def test_vulkan_dense_probe_executes_on_intel_discrete_gpu():
    completed = subprocess.run(
        ["bash", str(PROBE), "best"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)

    assert result["vendor_id"] == 0x8086
    assert result["device_id"] == 0xE20B
    assert "BMG" in result["device"]
    assert result["max_abs_error"] <= 1e-3
    assert result["kernel"] in {"naive_fp32", "tiled_fp32", "cooperative_fp16_fp32_acc"}
    assert result["median_dispatch_ms"] > 0
