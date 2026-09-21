"""Ops module public API and interfaces."""

from __future__ import annotations

from typing import Protocol

import torch


class ForwardOp(Protocol):
    """Protocol for forward operations (autograd compatible)."""

    def __call__(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...
