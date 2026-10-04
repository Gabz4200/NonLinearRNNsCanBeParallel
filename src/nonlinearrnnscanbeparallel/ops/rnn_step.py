"""RNN step reference implementation.

The default PyTorch path for a single RNN step. It is the correctness oracle
that any accelerated backend must reproduce: :mod:`ops.rnn_step_impls` selects
between this and the Taichi kernel, and the parity test compares them.

The layer must expose ``step``; this module does not emulate ``forward`` for a
layer that lacks it, because doing so would silently accept a layer whose step
semantics are undefined.
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
    """Advance one token through a multi-head RNN layer.

    Args:
        x: Input token, [B, D].
        h_prev: Previous per-head state, [B, H, D_head].
        rnn_layer: A layer exposing ``step(x_t, RNNState)``.

    Returns:
        The updated per-head state, [B, H, D_head]. This is the recurrent
        state only; the layer's layer-output (``x_t + projected``) is the
        caller's to keep, exactly as in the model protocol.
    """
    step = getattr(rnn_layer, "step", None)
    if not callable(step):
        raise TypeError(
            f"{type(rnn_layer).__name__} has no step(); the single-token op "
            "requires a layer that implements the model protocol"
        )
    new_state = step(x, RNNState(hidden=h_prev))[1]
    return new_state.hidden


class RNNStepReference:
    """Reference implementation of RNN step."""

    def __call__(self, x: torch.Tensor, h_prev: torch.Tensor, rnn_layer: nn.Module) -> torch.Tensor:
        return rnn_step_reference(x, h_prev, rnn_layer)
