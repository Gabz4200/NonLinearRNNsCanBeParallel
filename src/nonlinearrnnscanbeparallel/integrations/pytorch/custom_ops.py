"""PyTorch custom ops skeleton for Taichi integration (reference path only)."""

from __future__ import annotations

import torch
from torch import nn


def rnn_step_custom_op(
    x: torch.Tensor,
    h_prev: torch.Tensor,
    rnn_layer: nn.Module,
) -> torch.Tensor:
    """Reference wrapper; module usage excluded from op boundary."""
    from nonlinearrnnscanbeparallel.ops.rnn_step import rnn_step_reference

    return rnn_step_reference(x, h_prev, rnn_layer)
