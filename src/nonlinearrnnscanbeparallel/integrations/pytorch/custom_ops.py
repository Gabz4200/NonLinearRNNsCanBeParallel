"""PyTorch custom ops skeleton for Taichi integration (reference path only)."""

from __future__ import annotations

import torch
from torch import nn


def rnn_step_custom_op(
    x: torch.Tensor,
    h_prev: torch.Tensor,
    rnn_layer: nn.Module,
) -> torch.Tensor:
    """Eager reference wrapper; Taichi backend provided via build_op when installed."""
    from nonlinearrnnscanbeparallel.ops.rnn_step_impls import build_op

    op = build_op("reference")
    return op(x, h_prev, rnn_layer)
