"""Ops module public API and interfaces."""

from __future__ import annotations

from typing import Protocol

import torch


class ForwardOp(Protocol):
    """Protocol for forward-only operations."""

    def __call__(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class DifferentiableOp(ForwardOp, Protocol):
    """Protocol for differentiable operations (autograd compatible)."""

    ...


class ProfilerOp(DifferentiableOp, Protocol):
    """Protocol for operations with profiling support."""

    def profile(self, inputs: torch.Tensor) -> dict[str, float]: ...
