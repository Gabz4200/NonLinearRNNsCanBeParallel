"""Losses package."""

from .classification import cross_entropy_loss
from .language_modeling import causal_lm_loss

__all__ = [
    "causal_lm_loss",
    "cross_entropy_loss",
]
