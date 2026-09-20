from __future__ import annotations

import re
from pathlib import Path

import tomllib


def _project() -> dict:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip()
        for requirement in requirements
    }


def test_accelerator_profiles_do_not_force_cuda_dependencies_on_xpu() -> None:
    project = _project()
    dependencies = _names(project["project"]["dependencies"])
    extras = project["project"]["optional-dependencies"]
    xpu = _names(extras["xpu"])
    cuda = _names(extras["cuda"])

    assert {"torch", "torchvision", "triton", "flashlib"}.isdisjoint(dependencies)
    assert xpu == {"torch"}
    assert {"torch", "torchvision", "triton", "flashlib"}.issubset(cuda)
    assert extras["accel"] == ["freetoken[cuda,fi,sgl]"]


def test_xpu_profile_pins_the_tested_local_torch_build() -> None:
    project = _project()
    xpu = project["project"]["optional-dependencies"]["xpu"]

    assert xpu == [
        "torch==2.12.1; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ]
