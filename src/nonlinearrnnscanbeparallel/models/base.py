"""Base model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RNNState:
    """Recurrent state container for RNNs."""

    hidden: torch.Tensor  # [B, H, D] or [B, D]
    extra: dict[str, torch.Tensor] | None = None


RNNStateList = list[RNNState]


class BaseRNNModel(nn.Module, ABC):
    """Base class for RNN models. Returns latents only; no loss computation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

    @abstractmethod
    def forward(
        self, x: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, D]
            state: Optional recurrent states (one per RNN layer)

        Returns:
            Tuple of (output [B, T, D], new_states)
        """
        pass

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        """Initialize recurrent states for a batch."""
        pass


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SiLU(nn.Module):
    """SiLU (Swish) activation: x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
