"""Build-time checks for the XPU extension's SYCL runtime ABI."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOLCHAIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "freetoken"
    / "kernel"
    / "_toolchain.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_freetoken_toolchain_test", _TOOLCHAIN_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_TOOLCHAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOLCHAIN)


def test_readelf_dynamic_extracts_soname_and_dependencies(
    monkeypatch, tmp_path: Path
) -> None:
    output = """
 0x0000000000000001 (NEEDED)             Shared library: [libur_loader.so.0]
 0x000000000000000e (SONAME)             Library soname: [libsycl.so.9]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
"""
    monkeypatch.setattr(_TOOLCHAIN.shutil, "which", lambda _: "/usr/bin/readelf")
    monkeypatch.setattr(
        _TOOLCHAIN.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )

    soname, needed = _TOOLCHAIN._readelf_dynamic(tmp_path / "libsycl.so")

    assert soname == "libsycl.so.9"
    assert needed == ["libur_loader.so.0", "libstdc++.so.6"]


def test_sycl_abi_check_accepts_matching_runtime(tmp_path: Path) -> None:
    _TOOLCHAIN._require_sycl_abi_match(
        "libsycl.so.8",
        ["libc10_xpu.so", "libsycl.so.8", "libtorch_cpu.so"],
        tmp_path / "compiler" / "libsycl.so",
        tmp_path / "torch" / "libtorch_xpu.so",
    )


def test_sycl_abi_check_rejects_mismatched_runtime(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"libsycl\.so\.9.*requires libsycl\.so\.8"):
        _TOOLCHAIN._require_sycl_abi_match(
            "libsycl.so.9",
            ["libsycl.so.8"],
            tmp_path / "compiler" / "libsycl.so",
            tmp_path / "torch" / "libtorch_xpu.so",
        )


def test_sycl_abi_check_fails_closed_when_dependency_is_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="could not verify the SYCL ABI"):
        _TOOLCHAIN._require_sycl_abi_match(
            "libsycl.so.8",
            ["libc10_xpu.so"],
            tmp_path / "compiler" / "libsycl.so",
            tmp_path / "torch" / "libtorch_xpu.so",
        )


def test_readelf_is_required_for_sycl_abi_check(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_TOOLCHAIN.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="readelf from binutils is required"):
        _TOOLCHAIN._readelf_dynamic(tmp_path / "libsycl.so")
