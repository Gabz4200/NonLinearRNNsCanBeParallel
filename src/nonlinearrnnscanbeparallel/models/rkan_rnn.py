"""rKAN-RNN model (same architecture as MLP-RNN, with rKAN instead of MLP for FF sublayers)."""

from __future__ import annotations

import functools

import torch
from torch import nn

from .base import RMSNorm, RNNState, RNNStateList, SequentialRNNModel
from .mlp_rnn import MultiHeadRNNLayer
from .registry import register_model
from .rkan import RationalFeedForward, RKANHead


@register_model("rkan_rnn")
class RKANRNN(SequentialRNNModel):
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
                                RKANHead,
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

    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        return RNNStateList(
            [
                RNNState(
                    hidden=torch.zeros(
                        batch_size, self.num_heads, self.hidden_dim // self.num_heads, device=device
                    )
                )
                for _ in range(self.num_rnn_layers)
            ]
        )
