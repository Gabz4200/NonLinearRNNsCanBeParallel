"""Plain metric helpers."""

from __future__ import annotations

import math

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute accuracy ignoring specified index."""
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()


def perplexity(loss: float) -> float:
    """Perplexity from a cross-entropy loss, clamped for numerical safety."""
    return math.exp(min(loss, 20.0))


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Mean next-token accuracy ignoring masked positions."""
    return accuracy(logits, targets, ignore_index)


def negative_log_likelihood(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> float:
    """Mean per-token NLL in nats, ignoring masked positions.

    Mathematically identical to mean cross-entropy over the non-ignored
    targets; kept as a named LM metric so training logs carry ``nll``
    alongside ``loss`` (perplexity is ``exp(nll)``).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    return (-gathered[mask]).mean().item()
