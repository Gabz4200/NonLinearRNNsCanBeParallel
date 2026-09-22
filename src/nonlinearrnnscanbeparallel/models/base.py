"""Base model interface with dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass
class RNNOutput:
    """Structured output from RNN forward pass."""

    logits: torch.Tensor  # [B, T, C] or [B, C] for step
    states: RNNStateList


class BaseRNNModel(nn.Module, ABC):
    """Base class for RNN models. Returns latents only; no loss computation.

    Implements RNNModule protocol via inheritance. Concrete models should
    inherit from this class and implement the abstract methods.
    """

    layers: nn.ModuleList
    emb_norm: nn.Module
    final_norm: nn.Module
    classifier: nn.Module

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


class SequentialRNNModel(BaseRNNModel):
    """Shared sequential training loop for layer-pair (rnn, ff) models."""

    def forward(
        self, x: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        x = self.emb_norm(x)
        if state is None:
            state = self.init_state(x.shape[0], x.device)

        new_states_list: list[RNNState] = []
        for index, (rnn_layer, ff_layer) in enumerate(self.layers):
            x, new_state = rnn_layer(x, state[index])
            new_states_list.append(new_state)
            x = ff_layer(x)

        return self.classifier(self.final_norm(x)), RNNStateList(new_states_list)

    def step(
        self, x_t: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        x_t = self.emb_norm(x_t)
        if state is None:
            state = self.init_state(x_t.shape[0], x_t.device)

        new_states_list: list[RNNState] = []
        for index, (rnn_layer, ff_layer) in enumerate(self.layers):
            x_t, new_state = rnn_layer.step(x_t, state[index])
            new_states_list.append(new_state)
            x_t = ff_layer(x_t)

        return self.classifier(self.final_norm(x_t)), RNNStateList(new_states_list)


def chunked_forward_bptt(
    forward_fn: Callable[[torch.Tensor, RNNStateList | None], tuple[torch.Tensor, RNNStateList]],
    input_ids: torch.Tensor,
    state: RNNStateList | None,
    chunk: int,
) -> tuple[torch.Tensor, RNNStateList | None]:
    """Run ``forward_fn`` in chunks, detaching hidden state (and extra) between chunks."""
    _, sequence_len = input_ids.shape
    if chunk <= 0 or sequence_len <= chunk:
        return forward_fn(input_ids, state)

    outputs: list[torch.Tensor] = []
    for start in range(0, sequence_len, chunk):
        end = min(start + chunk, sequence_len)
        logits, state = forward_fn(input_ids[:, start:end], state)
        outputs.append(logits)
        if state is not None and end < sequence_len:
            detached = [
                RNNState(
                    hidden=item.hidden.detach(),
                    extra=(
                        {k: v.detach() for k, v in item.extra.items()}
                        if item.extra is not None
                        else None
                    ),
                )
                for item in state.states
            ]
            state = RNNStateList(detached)
    return torch.cat(outputs, dim=1), state


RNNModule = BaseRNNModel
