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


def segmented_parallel_scan_log(
    log_coeffs: torch.Tensor,
    log_values: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Segmented parallel prefix scan for variable-length sequences.

    Args:
        log_coeffs: [B, T, ...] log coefficients
        log_values: [B, T+1, ...] log values (including initial)
        segment_ids: [B, T] integer segment IDs, same segment = same sequence

    Returns:
        Output states [B, T, ...]
    """
    batch, seq_len = log_coeffs.shape[:2]

    # At segment boundaries, we need to reset the scan
    # segment_ids[b, t] != segment_ids[b, t-1] means new sequence
    # For t=0, it's always a boundary
    is_boundary = torch.zeros_like(segment_ids, dtype=torch.bool)
    is_boundary[:, 0] = True
    if seq_len > 1:
        is_boundary[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]

    # For boundaries, we want: h_t = exp(log_values_t) (no carry from previous)
    # This is achieved by setting log_coeffs = -inf at boundary positions
    # and ensuring log_values has the correct value

    # Create modified log_coeffs and log_values
    neg_inf = torch.full_like(log_coeffs, -1e30)
    log_coeffs_mod = torch.where(is_boundary.unsqueeze(-1), neg_inf, log_coeffs)

    # For log_values, at boundaries we use the original log_values (which includes initial state)
    # The initial state log_values[:, 0] is for t=0 boundary
    # For subsequent boundaries, we need to shift
    # Actually, the standard scan already handles this if we set log_coeffs = -inf at boundaries
    # because then a_star won't accumulate across boundaries

    # Run standard scan on modified coefficients
    a_star = functional.pad(torch.cumsum(log_coeffs_mod, dim=1), (0, 0, 0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    return torch.exp(a_star + log_h0_plus_b_star)[:, 1:]
