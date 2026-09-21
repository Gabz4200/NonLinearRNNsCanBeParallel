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


class SequenceCrossEntropy(nn.Module):
    """Sequence cross entropy with ignore_index."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return cross_entropy_loss(logits, targets, self.ignore_index)
