"""Taichi RNN step kernel scaffold - raises NotImplementedError until explicitly requested."""

from __future__ import annotations

import torch

from .runtime import ensure_initialized


def rnn_step(
    x: torch.Tensor, h_prev: torch.Tensor, rnn_layer: torch.nn.Module, *args, **kwargs
) -> torch.Tensor:
    """
    Taichi RNN step kernel.

    Raises NotImplementedError - implement @ti.kernel body when explicitly requested.
    """
    ensure_initialized()
    raise NotImplementedError(
        "Taichi RNN step kernel not implemented; use reference op or request a specific kernel."
    )
