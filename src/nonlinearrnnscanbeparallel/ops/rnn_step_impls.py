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
    """Backend implementation (Taichi) - raises until kernel is provided."""

    def __init__(self) -> None:
        from nonlinearrnnscanbeparallel.kernels.taichi.runtime import ensure_initialized

        ensure_initialized()

    def __call__(
        self, x: torch.Tensor, h_prev: torch.Tensor, rnn_layer: nn.Module, *args, **kwargs
    ) -> torch.Tensor:
        from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step

        return rnn_step(x, h_prev, rnn_layer)


def build_op(backend: str) -> Any:
    """Build op implementation for given backend."""
    if backend == "taichi":
        return BackendOp()
    return ReferenceOp()
