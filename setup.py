"""Setuptools entry point.

The default install is lightweight and builds the CUDA extension lazily on first
GPU use. Set MAMBA3_BUILD_CUDA=1 to compile it during installation instead.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup


def cuda_extensions():
    if os.getenv("MAMBA3_BUILD_CUDA", "0") != "1":
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as exc:  # pragma: no cover - build-time diagnostic
        raise RuntimeError(
            "MAMBA3_BUILD_CUDA=1 requires PyTorch in the build environment. "
            "Install PyTorch first, then use `pip install --no-build-isolation .`."
        ) from exc

    root = Path(__file__).parent
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"]
    if os.name == "nt":
        cuda_flags.append("-Xcompiler=/Zc:preprocessor")
    extension = CUDAExtension(
        name="mamba3._C",
        sources=[
            str(root / "mamba3" / "csrc" / "scan.cpp"),
            str(root / "mamba3" / "csrc" / "scan_cuda.cu"),
            str(root / "mamba3" / "csrc" / "scan_row_cuda.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O3"] if os.name != "nt" else ["/O2"],
            "nvcc": cuda_flags,
        },
    )
    return [extension], {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}


ext_modules, cmdclass = cuda_extensions()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
