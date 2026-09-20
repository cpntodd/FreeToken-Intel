"""CUDA and SYCL toolchain/torch consistency checks.

Standalone on purpose: setup.py and the kernel-cache build backend load this
file by path, so it must not import the freetoken package.
"""

from __future__ import annotations

import functools
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

ALLOW_MISMATCH_ENV = "FREETOKEN_ALLOW_CUDA_MISMATCH"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _nvcc_path() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME:
        return os.path.join(CUDA_HOME, "bin", "nvcc")
    return shutil.which("nvcc")


def nvcc_release(nvcc: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", proc.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def torch_cuda_major() -> int | None:
    import torch

    cuda = getattr(torch.version, "cuda", None)
    return int(cuda.split(".")[0]) if cuda else None


@functools.cache
def check_nvcc_matches_torch() -> None:
    """Refuse to nvcc-compile kernels across CUDA majors.

    nvcc-built binaries link libcudart.so.<nvcc major>; at runtime only the
    torch wheel's own CUDA runtime is guaranteed to be loadable.
    """
    if os.getenv(ALLOW_MISMATCH_ENV, "").strip().lower() in _TRUE_VALUES:
        return
    torch_major = torch_cuda_major()
    if torch_major is None:
        return
    nvcc = _nvcc_path()
    if nvcc is None:
        return
    release = nvcc_release(nvcc)
    if release is None:
        return
    if release[0] != torch_major:
        import torch

        raise RuntimeError(
            f"nvcc {release[0]}.{release[1]} would build kernels linking "
            f"libcudart.so.{release[0]}, but torch {torch.__version__} ships CUDA "
            f"{torch.version.cuda} (libcudart.so.{torch_major}). Install a CUDA "
            f"{torch_major}.x toolkit, or set {ALLOW_MISMATCH_ENV}=1 to override."
        )


def _readelf_dynamic(path: Path) -> tuple[str | None, list[str]]:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise RuntimeError(
            "readelf from binutils is required to validate the SYCL runtime ABI"
        )

    try:
        proc = subprocess.run(
            [readelf, "-d", str(path)], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(
            f"could not inspect ELF dependencies for {path}: {detail}"
        ) from exc

    soname_match = re.search(r"\(SONAME\).*?\[([^\]]+)\]", proc.stdout)
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", proc.stdout)
    return (soname_match.group(1) if soname_match else None), needed


def _require_sycl_abi_match(
    compiler_soname: str | None,
    torch_needed: list[str],
    compiler_library: Path,
    torch_xpu_library: Path,
) -> None:
    torch_sycl = [
        name for name in torch_needed if re.fullmatch(r"libsycl\.so(?:\.\d+)*", name)
    ]
    if compiler_soname is None or len(torch_sycl) != 1:
        raise RuntimeError(
            "could not verify the SYCL ABI: expected a compiler libsycl SONAME and "
            f"one torch_xpu libsycl dependency (found {torch_sycl!r}); "
            f"compiler={compiler_library}, torch_xpu={torch_xpu_library}"
        )

    if compiler_soname != torch_sycl[0]:
        raise RuntimeError(
            f"SYCL compiler runtime {compiler_soname} ({compiler_library}) is incompatible "
            f"with PyTorch XPU, which requires {torch_sycl[0]} ({torch_xpu_library}). "
            "Select an icpx compiler from the same oneAPI release as the installed "
            "PyTorch XPU runtime; see docs/intel-xpu.md."
        )


def check_sycl_matches_torch() -> None:
    """Refuse to build XPU extensions against an incompatible SYCL runtime ABI."""
    oneapi_root = Path(os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")).expanduser()
    compiler_version = os.environ.get("FREETOKEN_SYCL_COMPILER_VERSION", "latest")
    compiler_root = (oneapi_root / "compiler" / compiler_version).resolve()
    if not compiler_root.is_dir():
        raise RuntimeError(
            f"oneAPI compiler directory was not found: {compiler_root}; set "
            "ONEAPI_ROOT and FREETOKEN_SYCL_COMPILER_VERSION to an installed release"
        )

    cxx = os.environ.get("CXX", "").strip()
    command = shlex.split(cxx)
    if not command:
        raise RuntimeError(
            "CXX must point to the selected oneAPI icpx compiler when building "
            "FREETOKEN_ACCELERATOR=xpu"
        )
    executable = command[0]
    resolved = shutil.which(executable) if not os.path.isabs(executable) else executable
    if resolved is None or not Path(resolved).is_file():
        raise RuntimeError(
            f"the configured CXX compiler could not be found: {executable}"
        )
    compiler_path = Path(resolved).resolve()
    if compiler_root not in compiler_path.parents:
        raise RuntimeError(
            f"CXX resolves to {compiler_path}, outside the selected oneAPI compiler "
            f"directory {compiler_root}; set CXX to that release's icpx"
        )

    compiler_library = compiler_root / "lib" / "libsycl.so"
    if not compiler_library.is_file():
        raise RuntimeError(
            f"the selected compiler's SYCL runtime was not found: {compiler_library}"
        )

    import torch

    torch_xpu_library = (
        Path(torch.__file__).resolve().parent / "lib" / "libtorch_xpu.so"
    )
    if not torch_xpu_library.is_file():
        raise RuntimeError(
            f"the installed PyTorch XPU library was not found: {torch_xpu_library}"
        )

    compiler_soname, _ = _readelf_dynamic(compiler_library)
    _, torch_needed = _readelf_dynamic(torch_xpu_library)
    _require_sycl_abi_match(
        compiler_soname, torch_needed, compiler_library, torch_xpu_library
    )
