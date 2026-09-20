import importlib
import json

from freetoken import accelerator, cli
from freetoken.accelerator import (
    AcceleratorBackendStatus,
    AcceleratorCapabilities,
    AcceleratorDiscovery,
)
from freetoken.cli import main


def _b580_capability() -> AcceleratorCapabilities:
    return AcceleratorCapabilities(
        kind="xpu",
        index=0,
        name="Intel Arc B580",
        total_memory=12 * 1024**3,
        device_id="0xE20B",
        uuid="GPU-test",
        driver_version="1.6.test",
        platform_name="Level Zero",
        graph_capture=False,
        streams=True,
        events=True,
    )


def _b580_discovery() -> AcceleratorDiscovery:
    return AcceleratorDiscovery(
        devices=(_b580_capability(),),
        backends=(
            AcceleratorBackendStatus(
                kind="cuda",
                status="unavailable",
                device_count=0,
                message="PyTorch reports this backend as unavailable",
            ),
            AcceleratorBackendStatus(
                kind="xpu", status="available", device_count=1, message=None
            ),
        ),
    )


def test_top_level_help_lists_devices_command(capsys) -> None:
    assert main(["--help"]) == 0

    assert (
        "devices     Report detected CUDA and XPU device capabilities"
        in capsys.readouterr().out
    )


def test_devices_command_displays_device_and_runtime_features(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(accelerator, "probe_accelerators", _b580_discovery)
    monkeypatch.setattr(
        cli,
        "_probe_native_sycl_extension",
        lambda: {"status": "importable", "check": "python_import", "message": None},
    )

    assert main(["devices"]) == 0

    output = capsys.readouterr().out
    assert "xpu:0 | Intel Arc B580 | 12.0 GiB" in output
    assert "Device ID: 0xE20B" in output
    assert "Driver: 1.6.test" in output
    assert "Platform: Level Zero" in output
    assert "Graph capture: no" in output
    assert "Engine: single-GPU eager dense inference" in output
    assert "Engine attention: torch" in output
    assert "Engine routed MoE: no" in output
    assert "Native SYCL extension: importable" in output


def test_devices_command_json_is_machine_readable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(accelerator, "probe_accelerators", _b580_discovery)
    monkeypatch.setattr(
        cli,
        "_probe_native_sycl_extension",
        lambda: {"status": "importable", "check": "python_import", "message": None},
    )

    assert main(["devices", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    [device] = report["devices"]
    assert device["device"] == "xpu:0"
    assert device["name"] == "Intel Arc B580"
    assert device["total_memory"] == 12 * 1024**3
    assert device["platform_name"] == "Level Zero"
    assert device["engine"] == {
        "attention_backends": ["torch"],
        "cuda_graphs": False,
        "eager_only": True,
        "routed_moe": False,
        "single_gpu_only": True,
    }
    assert report["backends"] == [
        {
            "device_count": 0,
            "kind": "cuda",
            "message": "PyTorch reports this backend as unavailable",
            "status": "unavailable",
        },
        {"device_count": 1, "kind": "xpu", "message": None, "status": "available"},
    ]
    assert report["native_sycl_extension"] == {
        "status": "importable",
        "check": "python_import",
        "message": None,
    }


def test_devices_command_keeps_xpu_available_when_sycl_extension_fails(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(accelerator, "probe_accelerators", _b580_discovery)
    monkeypatch.setattr(
        cli,
        "_probe_native_sycl_extension",
        lambda: {
            "status": "load_failed",
            "check": "python_import",
            "message": "OSError: libsycl.so.8: cannot open shared object file",
        },
    )

    assert main(["devices", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["backends"][1]["status"] == "available"
    assert report["native_sycl_extension"] == {
        "status": "load_failed",
        "check": "python_import",
        "message": "OSError: libsycl.so.8: cannot open shared object file",
    }


def test_native_sycl_extension_probe_reports_importable_module(monkeypatch) -> None:
    def import_module(name: str) -> object:
        assert name == "freetoken.kernel._sycl_kernels"
        return object()

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert cli._probe_native_sycl_extension() == {
        "status": "importable",
        "check": "python_import",
        "message": None,
    }


def test_native_sycl_extension_probe_reports_missing_module(monkeypatch) -> None:
    def import_module(name: str) -> object:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert cli._probe_native_sycl_extension() == {
        "status": "missing",
        "check": "python_import",
        "message": "native SYCL extension is not installed",
    }


def test_native_sycl_extension_probe_reports_load_failure(monkeypatch) -> None:
    def import_module(name: str) -> object:
        raise OSError("libsycl.so.8: cannot open shared object file")

    monkeypatch.setattr(importlib, "import_module", import_module)

    result = cli._probe_native_sycl_extension()

    assert result["status"] == "load_failed"
    assert result["check"] == "python_import"
    assert "libsycl.so.8" in result["message"]


def test_native_sycl_extension_is_not_probed_without_confirmed_xpu(
    monkeypatch,
) -> None:
    def fail_probe() -> dict[str, str | None]:
        raise AssertionError("SYCL import should not run without a confirmed XPU")

    monkeypatch.setattr(cli, "_probe_native_sycl_extension", fail_probe)

    assert cli._native_sycl_extension_status(False) == {
        "status": "not_probed",
        "check": "python_import",
        "message": "PyTorch did not confirm an available XPU device",
    }


def test_devices_command_reports_empty_accelerator_list_without_cpu_fallback(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        accelerator,
        "probe_accelerators",
        lambda: AcceleratorDiscovery(
            devices=(),
            backends=(
                AcceleratorBackendStatus(
                    kind="cuda",
                    status="unavailable",
                    device_count=0,
                    message="PyTorch reports this backend as unavailable",
                ),
                AcceleratorBackendStatus(
                    kind="xpu",
                    status="unavailable",
                    device_count=0,
                    message="No XPU device was found",
                ),
            ),
        ),
    )

    assert main(["devices"]) == 0

    output = capsys.readouterr().out
    assert "No CUDA or XPU accelerator devices detected." in output
    assert "CUDA: unavailable" in output
    assert "XPU: unavailable: No XPU device was found" in output
    assert "Native SYCL extension: not_probed" in output
    assert "CPU" not in output


def test_devices_command_displays_backend_probe_failures(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        accelerator,
        "probe_accelerators",
        lambda: AcceleratorDiscovery(
            devices=(),
            backends=(
                AcceleratorBackendStatus(
                    kind="cuda",
                    status="error",
                    device_count=0,
                    message="RuntimeError: CUDA driver is missing",
                ),
                AcceleratorBackendStatus(
                    kind="xpu",
                    status="unavailable",
                    device_count=0,
                    message="No XPU device was found",
                ),
            ),
        ),
    )

    assert main(["devices"]) == 0

    output = capsys.readouterr().out
    assert "CUDA: probe failed: RuntimeError: CUDA driver is missing" in output
    assert "XPU: unavailable: No XPU device was found" in output


def test_devices_command_rejects_unknown_options(capsys) -> None:
    assert main(["devices", "--cpu"]) == 2

    assert "accepts only --json or --help" in capsys.readouterr().err
