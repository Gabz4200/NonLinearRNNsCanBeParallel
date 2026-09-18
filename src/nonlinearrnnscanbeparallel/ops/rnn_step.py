"""RNN step operation reference implementation.

This is the default PyTorch reference for a single RNN step.
When Taichi backend is requested, the implementation raises NotImplementedError
until a kernel body is explicitly provided.
"""

from __future__ import annotations

import torch
from torch import nn

from nonlinearrnnscanbeparallel.models.base import RNNState


def rnn_step_reference(
    x: torch.Tensor,
    h_prev: torch.Tensor,
    rnn_layer: nn.Module,
) -> torch.Tensor:
    """
    Single RNN step reference implementation.

    Args:
        x: Input tensor [B, D]
        h_prev: Previous hidden state [B, H, D_head]
        rnn_layer: MultiHeadRNNLayer module

    Returns:
        New hidden state [B, H, D_head]
    """
    state = [RNNState(hidden=h_prev)]
    _, new_states = rnn_layer(x.unsqueeze(1), state)
    return new_states[0].hidden


class RNNStepReference:
    """Reference implementation of RNN step."""

    def __call__(self, x: torch.Tensor, h_prev: torch.Tensor, rnn_layer: nn.Module) -> torch.Tensor:
        return rnn_step_reference(x, h_prev, rnn_layer)
