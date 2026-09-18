"""Plain metric helpers."""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute accuracy ignoring specified index."""
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()


def sequence_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> float:
    """Compute accuracy at last valid token per sequence."""
    batch, seq, _ = logits.shape
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    last_valid = mask.flip(dims=[1]).argmax(dim=1)
    last_valid = seq - 1 - last_valid
    batch_indices = torch.arange(batch, device=logits.device)
    last_logits = logits[batch_indices, last_valid]
    last_targets = targets[batch_indices, last_valid]
    return (last_logits.argmax(dim=-1) == last_targets).float().mean().item()
