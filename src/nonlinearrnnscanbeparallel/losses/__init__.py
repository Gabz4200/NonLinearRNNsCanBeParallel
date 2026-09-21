"""Losses package."""

from .classification import SequenceCrossEntropy, cross_entropy_loss
from .language_modeling import CausalLMLoss, causal_lm_loss

__all__ = [
    "CausalLMLoss",
    "SequenceCrossEntropy",
    "causal_lm_loss",
    "cross_entropy_loss",
]
