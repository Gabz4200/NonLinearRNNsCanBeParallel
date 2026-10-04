"""RNN step implementations - reference and backend."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .rnn_step import RNNStepReference


class ReferenceOp:
    """Reference implementation (always works)."""

    def __call__(
        self, x: torch.Tensor, h_prev: torch.Tensor, rnn_layer: nn.Module, *args, **kwargs
    ) -> torch.Tensor:
        return RNNStepReference()(x, h_prev, rnn_layer)


class BackendOp:
    """Taichi-backed implementation; the kernel lives in ``kernels.taichi.rnn_step``."""

    def __init__(self) -> None:
        from nonlinearrnnscanbeparallel.kernels.taichi.runtime import ensure_initialized

        ensure_initialized()

    def __call__(
        self, x: torch.Tensor, h_prev: torch.Tensor, rnn_layer: nn.Module, *args, **kwargs
    ) -> torch.Tensor:
        from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step

        return rnn_step(x, h_prev, rnn_layer)


def build_op(backend: str) -> Any:
    """Build the op implementation for a named backend.

    Raises:
        ValueError: ``backend`` names no known implementation. Silently
            falling back to the reference would hide a typo behind numbers
            that look like a working kernel run.
    """
    if backend == "taichi":
        return BackendOp()
    if backend == "reference":
        return ReferenceOp()
    raise ValueError(f"Unknown rnn_step backend '{backend}'. Valid: 'taichi', 'reference'")
