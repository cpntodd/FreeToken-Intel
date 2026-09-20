import json

from freetoken import accelerator
from freetoken.accelerator import AcceleratorCapabilities
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


def test_top_level_help_lists_devices_command(capsys) -> None:
    assert main(["--help"]) == 0

    assert (
        "devices     Report detected CUDA and XPU device capabilities"
        in capsys.readouterr().out
    )


def test_devices_command_displays_device_and_runtime_features(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        accelerator, "discover_accelerators", lambda: (_b580_capability(),)
    )

    assert main(["devices"]) == 0

    output = capsys.readouterr().out
    assert "xpu:0 | Intel Arc B580 | 12.0 GiB" in output
    assert "Device ID: 0xE20B" in output
    assert "Driver: 1.6.test" in output
    assert "Platform: Level Zero" in output
    assert "Graph capture: no" in output


def test_devices_command_json_is_machine_readable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        accelerator, "discover_accelerators", lambda: (_b580_capability(),)
    )

    assert main(["devices", "--json"]) == 0

    [device] = json.loads(capsys.readouterr().out)
    assert device["device"] == "xpu:0"
    assert device["name"] == "Intel Arc B580"
    assert device["total_memory"] == 12 * 1024**3
    assert device["platform_name"] == "Level Zero"


def test_devices_command_reports_empty_accelerator_list_without_cpu_fallback(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(accelerator, "discover_accelerators", lambda: ())

    assert main(["devices"]) == 0

    output = capsys.readouterr().out
    assert output == "No CUDA or XPU accelerator devices detected.\n"
    assert "CPU" not in output


def test_devices_command_rejects_unknown_options(capsys) -> None:
    assert main(["devices", "--cpu"]) == 2

    assert "accepts only --json or --help" in capsys.readouterr().err
