"""Shared numerical helpers for the minimal recurrent models."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def minimal_candidate(x: torch.Tensor) -> torch.Tensor:
    """Positive, continuous candidate activation from Appendix B."""
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def minimal_log_candidate(x: torch.Tensor) -> torch.Tensor:
    """Logarithm of :func:`minimal_candidate` without forming small probabilities."""
    return torch.where(
        x >= 0,
        (functional.relu(x) + 0.5).log(),
        -functional.softplus(-x),
    )


def parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
    """Compute ``h_t = exp(log_coeffs_t) * h_{t-1} + exp(log_values_t)``.

    ``log_coeffs`` has shape ``[B, T, ...]`` and ``log_values`` has shape
    ``[B, T + 1, ...]``. The scan runs independently over every feature.
    """
    a_star = functional.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    return torch.exp(a_star + log_h0_plus_b_star)[:, 1:]
