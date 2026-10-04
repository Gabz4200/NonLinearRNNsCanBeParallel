"""HOPE memory math: retrieval and the in-context update rule.

Pure tensor helpers for the self-modifying associative memories of HOPE
(arXiv:2512.24695 §8.2).  Kept separate from :mod:`.hope` so the layer holds
only recurrence and module wiring.

Memory state.  Each memory is a single square matrix ``M in R^{N x N}`` per
head, retrieved linearly as ``M(z) = M z``.  The square shape is not a
convenience: the paper's update right-multiplies by a key-space factor,

    ``M_t = M_{t-1} (alpha_t I - eta_t k_t k_t^T) - eta_t grad``,

so ``(alpha I - eta k k^T)`` must be ``N x N`` against a ``N x N`` memory.
``retrieve`` would also accept a rectangular ``K x V`` memory, but then the
decay factor would have to act on the value axis instead, which is not the
rule the paper derives.

Update rule (Eq. 93, the L2-regression objective behind the reported LM
results)::

    M_t = M_{t-1} (alpha_t I - eta_t k_t k_t^T) - eta_t (M_{t-1} k_t - v_t) k_t^T

The ``dot`` objective of Eq. 92 replaces the error term with ``v_t k_t^T``.
"""

from __future__ import annotations

import torch

# Internal objectives supported by the cell (paper Eqs. 92-93).
HOPE_OBJECTIVES = ("l2", "dot")

# The five memories of a HOPE layer, in the order they are stacked along dim 1
# of the layer state.  Order is part of the state contract: the parallel
# wrapper builds boundary states in this layout.
HOPE_MEMORIES = ("mem", "k", "v", "eta", "alpha")


def logit(p: float) -> float:
    """Inverse sigmoid, clamped away from the open-interval infinities."""
    eps = 1e-6
    clamped = min(max(p, eps), 1.0 - eps)
    return float(torch.logit(torch.tensor(clamped)))


def retrieve(m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Memory retrieval ``M(z) = M z``.

    Args:
        m: [B, H, N, N] memory per head.
        z: [B, H, N] query per head.

    Returns:
        [B, H, N] retrieved value per head.
    """
    return torch.einsum("bhkv,bhk->bhv", m, z)


def l2_gradient(m_prev: torch.Tensor, k_t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``(M k - v) k^T``: the L2-regression gradient of Eq. 93.

    The factor 2 from ``||.||^2`` is absorbed into the internal learning rate
    and omitted here, matching the paper's dual forms.

    Args:
        m_prev: [B, H, N, N] memory at t-1.
        k_t: [B, H, N] key driving the step.
        target: [B, H, N] regression target (v_t, or the self-generated
            ``v_hat_s`` of Eq. 95).

    Returns:
        [B, H, N, N] gradient with respect to the memory matrix.
    """
    error = retrieve(m_prev, k_t) - target
    return torch.einsum("bhv,bhk->bhkv", error, k_t)


def advance(
    m_prev: torch.Tensor,
    k_t: torch.Tensor,
    target: torch.Tensor,
    eta_t: torch.Tensor,
    alpha_t: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    """One in-context gradient step for a single memory (Eq. 96).

    ``eta_t`` and ``alpha_t`` are per-head scalars broadcast over the matrix.

    Args:
        m_prev: [B, H, N, N] memory at t-1.
        k_t: [B, H, N] key driving the step.
        target: [B, H, N] regression target.
        eta_t, alpha_t: [B, H, 1, 1] internal learning rate and retention.
        objective: "l2" (Eq. 93) or "dot" (Eq. 92).

    Returns:
        [B, H, N, N] updated memory.

    Raises:
        ValueError: if ``objective`` is not a supported objective.
    """
    n = m_prev.shape[-1]
    identity = torch.eye(n, dtype=m_prev.dtype, device=m_prev.device).expand_as(m_prev)
    decay = alpha_t * identity - eta_t * torch.einsum("bhk,bhl->bhkl", k_t, k_t)
    m_decayed = torch.matmul(m_prev, decay)

    if objective == "dot":
        grad = torch.einsum("bhv,bhk->bhkv", target, k_t)
    elif objective == "l2":
        grad = l2_gradient(m_prev, k_t, target)
    else:
        raise ValueError(f"Unknown HOPE objective: {objective}")

    return m_decayed - eta_t * grad
