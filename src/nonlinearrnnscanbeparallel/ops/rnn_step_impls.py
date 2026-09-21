"""RNN step implementations - reference and backend."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .rnn_step import rnn_step_reference


def _reference_op(
    x: torch.Tensor, h_prev: torch.Tensor, rnn_layer: nn.Module, *args, **kwargs
) -> torch.Tensor:
    return rnn_step_reference(x, h_prev, rnn_layer)


def _backend_op(
    x: torch.Tensor, h_prev: torch.Tensor, rnn_layer: nn.Module, *args, **kwargs
) -> torch.Tensor:
    from nonlinearrnnscanbeparallel.kernels.taichi.rnn_step import rnn_step

    return rnn_step(x, h_prev, rnn_layer)


def build_op(
    backend: str,
) -> Callable[..., torch.Tensor]:
    """Build op implementation for given backend."""
    if backend == "taichi":
        from nonlinearrnnscanbeparallel.kernels.taichi.runtime import ensure_initialized

        ensure_initialized()
        return _backend_op
    return _reference_op
