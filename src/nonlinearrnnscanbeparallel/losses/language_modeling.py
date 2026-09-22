"""Causal language modeling loss with shift inside.

The data module stores unshifted ``labels`` (``labels == input_ids``
except ``-100`` mask positions); the shift happens here so BPTT and
parallel tasks share one code path.
"""

from __future__ import annotations

import torch

from .classification import cross_entropy_loss


def causal_lm_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Next-token cross entropy: ``logits[:, :-1]`` vs ``targets[:, 1:]``."""
    return cross_entropy_loss(logits[:, :-1], targets[:, 1:], ignore_index)
