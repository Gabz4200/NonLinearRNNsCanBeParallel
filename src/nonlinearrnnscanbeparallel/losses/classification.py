"""Classification loss functions."""

from __future__ import annotations

import torch
from torch import nn


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Cross entropy loss for sequence classification."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
