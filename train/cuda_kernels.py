"""Loader for the MX550 fused CUDA kernels (train/cuda/cvpn_fused.cu).

Compiles once with ``torch.utils.cpp_extension`` (cached by torch) and returns
the extension module, or ``None`` when CUDA / a compiler is unavailable so
callers can fall back to the eager path.  Targeted at sm_75 (MX550) to keep the
build fast; override with ``GITCG_CUDA_ARCH``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_CU_PATH = Path(__file__).with_name("cuda") / "cvpn_fused.cu"


@lru_cache(maxsize=1)
def cvpn_kernels():
    """Return the compiled extension module, or ``None`` if it cannot be built."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in this repo
        return None
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
    except Exception:
        return None

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("GITCG_CUDA_ARCH", "7.5"))
    build_dir = os.environ.get("GITCG_CUDA_BUILD_DIR") or None
    try:
        return load(
            name="cvpn_fused",
            sources=[str(_CU_PATH)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception:
        return None


def kernels_available() -> bool:
    return cvpn_kernels() is not None
