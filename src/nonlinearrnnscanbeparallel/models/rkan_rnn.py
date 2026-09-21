"""rKAN-RNN model (same architecture as MLP-RNN, with rKAN instead of MLP for FF sublayers)."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseRNNModel, RMSNorm, RNNState, RNNStateList
from .registry import register_model
from .rkan import RationalFeedForward, RKANHead


@register_model("rkan_rnn")
class RKANRNN(BaseRNNModel):
    """
    rKAN-RNN: MLP-RNN architecture with Rational Kolmogorov-Arnold Networks
    replacing MLP/FFN sublayers (per arXiv:2406.14495v1).

    Same multi-layer structure with RMSNorm + SiLU, but feedforward
    sublayers use rKAN (Padé or Jacobi rational basis functions) instead
    of standard two-layer MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        rkan_degree: int = 3,
        rkan_alpha: float = 1.0,
        rkan_beta: float = 1.0,
        rkan_iota: float = 1.0,
        rkan_mapping: str = "algebraic_infinite",
        rkan_type: str = "jacobi",
        rkan_num_basis: int = 8,
        num_classes: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.rkan_degree = rkan_degree
        self.rkan_alpha = rkan_alpha
        self.rkan_beta = rkan_beta
        self.rkan_iota = rkan_iota
        self.rkan_mapping = rkan_mapping
        self.rkan_type = rkan_type
        self.rkan_num_basis = rkan_num_basis
        self.num_classes = num_classes

        # No internal embedding; the task embeds token IDs first.
        self.emb_norm = RMSNorm(hidden_dim)

        from .mlp_rnn import MultiHeadRNNLayer

        self.layers = nn.ModuleList()
        self.num_rnn_layers = num_layers
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiHeadRNNLayer(
                            hidden_dim,
                            num_heads,
                            head_factory=lambda h_dim: RKANHead(
                                h_dim,
                                rkan_degree=rkan_degree,
                                rkan_alpha=rkan_alpha,
                                rkan_beta=rkan_beta,
                                rkan_iota=rkan_iota,
                                rkan_mapping=rkan_mapping,
                                rkan_type=rkan_type,
                                rkan_num_basis=rkan_num_basis,
                            ),
                            dropout=dropout,
                        ),
                        RationalFeedForward(
                            hidden_dim,
                            rkan_degree=rkan_degree,
                            rkan_alpha=rkan_alpha,
                            rkan_beta=rkan_beta,
                            rkan_iota=rkan_iota,
                            rkan_mapping=rkan_mapping,
                            rkan_type=rkan_type,
                            rkan_num_basis=rkan_num_basis,
                            dropout=dropout,
                        ),
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
        return logits, RNNStateList.from_list(new_states_list)

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

        return self.classifier(self.final_norm(x_t)), RNNStateList.from_list(new_states_list)

    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        return RNNStateList.from_list(
            [
                RNNState(
                    hidden=torch.zeros(
                        batch_size, self.num_heads, self.hidden_dim // self.num_heads, device=device
                    )
                )
                for _ in range(self.num_rnn_layers)
            ]
        )
