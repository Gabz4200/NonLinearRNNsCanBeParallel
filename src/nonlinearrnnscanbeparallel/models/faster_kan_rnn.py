"""FasterKanRNN model (MLP-RNN architecture with FasterKAN for recurrence + FFN)."""

from __future__ import annotations

import functools

import torch
from torch import nn

from .base import RMSNorm, RNNState, RNNStateList, SequentialRNNModel
from .faster_kan import FasterKanFeedForward, FasterKANHead
from .mlp_rnn import MultiHeadRNNLayer
from .registry import register_model


@register_model("faster_kan_rnn")
class FasterKanRNN(SequentialRNNModel):
    """
    FasterKan-RNN: MLP-RNN architecture with FasterKAN (RSWAF bases) replacing
    MLP recurrence heads and feedforward sublayers (per AthanasiosDelis/faster-kan,
    Fast KAN arXiv:2405.06721).

    Same multi-layer structure with RMSNorm + SiLU, but both the recurrence head
    and the feedforward sublayers use FasterKAN layers instead of standard
    two-layer MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        fkan_grid_min: float = -1.2,
        fkan_grid_max: float = 0.2,
        fkan_num_grids: int = 8,
        fkan_inv_denominator: float = 0.5,
        fkan_train_grid: bool = False,
        fkan_train_inv_denominator: bool = False,
        fkan_spline_weight_init_scale: float = 0.1,
        num_classes: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.fkan_num_grids = fkan_num_grids

        # No internal embedding; the task embeds token IDs first.
        self.emb_norm = RMSNorm(hidden_dim)

        fkan_kwargs = {
            "fkan_grid_min": fkan_grid_min,
            "fkan_grid_max": fkan_grid_max,
            "fkan_num_grids": fkan_num_grids,
            "fkan_inv_denominator": fkan_inv_denominator,
            "fkan_train_grid": fkan_train_grid,
            "fkan_train_inv_denominator": fkan_train_inv_denominator,
            "fkan_spline_weight_init_scale": fkan_spline_weight_init_scale,
        }
        self.layers = nn.ModuleList()
        self.num_rnn_layers = num_layers
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiHeadRNNLayer(
                            hidden_dim,
                            num_heads,
                            head_factory=functools.partial(FasterKANHead, **fkan_kwargs),
                            dropout=dropout,
                        ),
                        FasterKanFeedForward(hidden_dim, dropout=dropout, **fkan_kwargs),
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
