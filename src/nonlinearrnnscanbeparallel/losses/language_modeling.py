"""Causal language modeling loss with shift inside.

The data module stores unshifted ``labels`` (``labels == input_ids``
except ``-100`` mask positions); the shift happens here so BPTT and
parallel tasks share one code path.
"""

from __future__ import annotations

import torch
from torch import nn

from .classification import cross_entropy_loss


def causal_lm_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Next-token cross entropy: ``logits[:, :-1]`` vs ``targets[:, 1:]``."""
    return cross_entropy_loss(logits[:, :-1], targets[:, 1:], ignore_index)


class CausalLMLoss(nn.Module):
    """Causal LM loss module with ignore_index."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return causal_lm_loss(logits, targets, self.ignore_index)
