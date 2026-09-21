"""MLP-RNN implementation per arXiv:2603.03612 Sections 2.2 and 2.4.

Multilayer RNN with RMSNorm, SiLU, and multi-head structure.  Uses RMSNorm
and SiLU per the modernization note in the task spec; the paper's LayerNorm
and ReLU are the original form.  Architecture follows Definitions 3, 7, and 8.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from torch import nn

from .base import BaseRNNModel, RMSNorm, RNNState, RNNStateList
from .registry import register_model

HeadFactory = Callable[[int], nn.Module]


class MLPHead(nn.Module):
    """Multi-layer feedforward network for the recurrence function f(h_{t-1}, x_t).

    Pure Linear -> SiLU stack with no normalization inside the recurrence.
    Normalization inside f would rescale state magnitudes and destroy the
    poly-precision stack encodings the Siegelmann-Sontag construction
    (Theorem 1, arXiv:2603.03612) relies on; sublayer norms per
    Definitions 7/8 live outside the recurrence and are unaffected.
    """

    def __init__(
        self,
        head_dim: int,
        mlp_hidden_mult: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_dim = head_dim * 2 if i == 0 else head_dim * mlp_hidden_mult
            out_dim = head_dim * mlp_hidden_mult if i < num_layers - 1 else head_dim
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, h_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_prev, x_t], dim=-1)
        return self.net(combined)


def make_mlp_heads(
    head_dim: int,
    mlp_hidden_mult: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Default head factory for MLP-RNN."""
    return MLPHead(head_dim, mlp_hidden_mult, num_layers, dropout)


class MultiHeadRNNLayer(nn.Module):
    """Single multi-head RNN sublayer per Definition 7.

    The recurrent head function f(h_{t-1}, x_t) is constructed per-head by
    ``head_factory``.  MLP-RNN uses MLPHead by default; RKANRNN injects an
    RKANHead with the same surrounding structure.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_factory: HeadFactory = make_mlp_heads,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.norm = RMSNorm(hidden_dim)
        self.heads = nn.ModuleList([head_factory(self.head_dim) for _ in range(num_heads)])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def step(self, x_t: torch.Tensor, state: RNNState) -> tuple[torch.Tensor, RNNState]:
        """Single-token update: x_t [B, D] with state.hidden [B, H, D_head]."""
        batch = x_t.shape[0]
        h_prev = state.hidden
        if h_prev.dim() == 2:
            h_prev = h_prev.view(batch, self.num_heads, self.head_dim)

        x_norm = self.norm(x_t)
        x_heads = x_norm.view(batch, self.num_heads, self.head_dim)

        head_outputs: list[torch.Tensor] = []
        for h in range(self.num_heads):
            head_outputs.append(self.heads[h](h_prev[:, h], x_heads[:, h]))

        aggregated = self.output_proj(torch.cat(head_outputs, dim=-1))
        aggregated = self.dropout(aggregated)

        # Residual connection per Definition 7.
        return x_t + aggregated, RNNState(hidden=torch.stack(head_outputs, dim=1))

    def forward(self, x: torch.Tensor, state: RNNState) -> tuple[torch.Tensor, RNNState]:
        # Inputs carry [B, T, D]; states carry [B, H, D_head].
        h = state.hidden
        if h.dim() == 2:
            h = h.view(x.shape[0], self.num_heads, self.head_dim)
        cur = RNNState(hidden=h)

        outputs: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            x_t, cur = self.step(x[:, t, :], cur)
            outputs.append(x_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), cur


class FeedForwardSublayer(nn.Module):
    """Feedforward sublayer per Definition 8."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.up = nn.Linear(hidden_dim, hidden_dim * mlp_hidden_mult, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim * mlp_hidden_mult, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.up(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.down(x)
        return residual + x


@register_model("mlp_rnn")
class MLPRNN(BaseRNNModel):
    """Multilayer MLP-RNN per arXiv:2603.03612 (Definitions 3, 7, 8).

    Architecture (paper Def 7 / Def 8):
      - Token embedding (provided by the task)
      - emb_norm -> alternating {MultiHeadRNNLayer, FeedForwardSublayer} x num_layers
      - final_norm -> classifier

    Modernization note: uses RMSNorm + SiLU instead of the paper's LayerNorm + ReLU
    (explicitly requested per task spec).  See README.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        mlp_hidden_mult: int = 4,
        mlp_num_layers: int = 2,
        num_classes: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)

        self.emb_norm = RMSNorm(hidden_dim)

        self.layers = nn.ModuleList()
        self.num_rnn_layers = num_layers
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiHeadRNNLayer(
                            hidden_dim,
                            num_heads,
                            head_factory=functools.partial(
                                make_mlp_heads,
                                mlp_hidden_mult=mlp_hidden_mult,
                                num_layers=mlp_num_layers,
                                dropout=dropout,
                            ),
                            dropout=dropout,
                        ),
                        FeedForwardSublayer(hidden_dim, mlp_hidden_mult, dropout),
                    ]
                )
            )

        self.final_norm = RMSNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(
        self, x: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        batch = x.shape[0]
        x = self.emb_norm(x)

        if state is None:
            state = self.init_state(batch, x.device)

        new_states_list: list[RNNState] = []
        for rnn_layer_idx, layer_pair in enumerate(self.layers):
            rnn_layer, ff_layer = layer_pair
            x, new_state = rnn_layer(x, state[rnn_layer_idx])
            new_states_list.append(new_state)
            x = ff_layer(x)

        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits, RNNStateList(new_states_list)

    def step(
        self, x_t: torch.Tensor, state: RNNStateList | None = None
    ) -> tuple[torch.Tensor, RNNStateList]:
        """Single-token decoding: x_t [B, D] -> logits [B, C] plus updated states."""
        x_t = self.emb_norm(x_t)

        if state is None:
            state = self.init_state(x_t.shape[0], x_t.device)

        new_states_list: list[RNNState] = []
        for rnn_layer_idx, layer_pair in enumerate(self.layers):
            rnn_layer, ff_layer = layer_pair
            x_t, new_state = rnn_layer.step(x_t, state[rnn_layer_idx])
            new_states_list.append(new_state)
            x_t = ff_layer(x_t)

        return self.classifier(self.final_norm(x_t)), RNNStateList(new_states_list)

    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        return RNNStateList(
            [
                RNNState(
                    hidden=torch.zeros(
                        batch_size,
                        self.num_heads,
                        self.hidden_dim // self.num_heads,
                        device=device,
                    )
                )
                for _ in range(self.num_rnn_layers)
            ]
        )
