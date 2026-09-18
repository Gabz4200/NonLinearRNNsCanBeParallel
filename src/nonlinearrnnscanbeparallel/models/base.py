"""Base model interface with Protocol and dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@dataclass
class RNNState:
    """Recurrent state container for RNNs."""

    hidden: torch.Tensor  # [B, H, D] or [B, N, K, V] for M2RNN
    extra: dict[str, torch.Tensor] | None = None


@dataclass
class RNNStateList:
    """List of RNN states, one per layer."""

    states: list[RNNState]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> RNNState:
        return self.states[index]

    def __iter__(self):
        return iter(self.states)

    @classmethod
    def from_list(cls, states: list[RNNState]) -> RNNStateList:
        return cls(states)


@dataclass
class RNNOutput:
    """Structured output from RNN forward pass."""

    logits: torch.Tensor  # [B, T, C] or [B, C] for step
    states: RNNStateList


@runtime_checkable
class RNNModule(Protocol):
    """Protocol defining the interface for all RNN models.

    Any class implementing these methods conforms to the RNN interface,
    enabling structural subtyping without inheritance coupling.
    """

    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float

    def forward(
        self, x: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        """
        Forward pass (cumsum-style: processes full sequence).

        Args:
            x: Input tensor [B, T, D]
            state: Optional recurrent states (one per RNN layer)

        Returns:
            Tuple of (output [B, T, D], new_states)
        """
        ...

    def step(
        self, x_t: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        """
        Single-token autoregressive step.

        Args:
            x_t: Input tensor [B, D]
            state: Optional recurrent states (one per RNN layer)

        Returns:
            Tuple of (output [B, D], new_states)
        """
        ...

    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        """Initialize recurrent states for a batch."""
        ...


class BaseRNNModel(nn.Module, ABC):
    """Base class for RNN models. Returns latents only; no loss computation.

    Implements RNNModule protocol via inheritance. Concrete models should
    inherit from this class and implement the abstract methods.
    """

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
        Forward pass (cumsum-style: processes full sequence).

        Args:
            x: Input tensor [B, T, D]
            state: Optional recurrent states (one per RNN layer)

        Returns:
            Tuple of (output [B, T, D], new_states)
        """
        pass

    @abstractmethod
    def step(
        self, x_t: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        """
        Single-token autoregressive step.

        Args:
            x_t: Input tensor [B, D]
            state: Optional recurrent states (one per RNN layer)

        Returns:
            Tuple of (output [B, D], new_states)
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
