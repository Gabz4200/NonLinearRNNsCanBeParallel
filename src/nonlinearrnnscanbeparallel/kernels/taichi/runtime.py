"""Taichi runtime initialization - called once per process."""

from __future__ import annotations

import os
import threading
from typing import Any

_initialized = False
_lock = threading.Lock()

# Lazy import keeps taichi optional until first use.
_ti: Any = None
_ARCH_MAP: dict[str, Any] = {}


def _import_taichi() -> None:
    """Import taichi and populate arch map."""
    global _ti, _ARCH_MAP
    if _ti is not None:
        return
    try:
        import taichi as ti

        _ti = ti
        _ARCH_MAP = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "metal": ti.metal}
    except ImportError:
        _ti = None
        _ARCH_MAP = {}


def ensure_initialized(arch: str | None = None) -> None:
    """Initialize Taichi exactly once per process."""
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        _import_taichi()
        if _ti is None:
            raise RuntimeError(
                "Taichi not installed. Install with 'uv sync --extra taichi'"
                " or 'pip install taichi'."
            )
        name = arch or os.getenv("PROJECT_TAICHI_ARCH", "cpu")
        if name not in _ARCH_MAP:
            raise ValueError(f"Unknown Taichi arch '{name}'. Valid: {list(_ARCH_MAP)}")
        _ti.init(arch=_ARCH_MAP[name])
        _initialized = True
