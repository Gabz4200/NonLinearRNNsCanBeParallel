"""minLSTM implementation per arXiv:2410.01201v3 Appendix B."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import RMSNorm, RNNState, RNNStateList, SequentialRNNModel
from .minimal import minimal_candidate, minimal_log_candidate, parallel_scan_log, per_head_linear
from .mlp_rnn import FeedForwardSublayer
from .registry import register_model


class MinLSTMLayer(nn.Module):
    """One multi-head minLSTM layer with length-independent gate scaling."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm = RMSNorm(hidden_dim)
        self.linear_f = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.linear_i = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.linear_h = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_len = x.shape[:2]
        x_heads = x.reshape(batch, sequence_len, self.num_heads, self.head_dim)
        forget = per_head_linear(self.linear_f, x_heads, stack_dim=2)
        input_gate = per_head_linear(self.linear_i, x_heads, stack_dim=2)
        candidate = per_head_linear(self.linear_h, x_heads, stack_dim=2)
        return forget, input_gate, candidate

    def step(self, x_t: torch.Tensor, state: RNNState) -> tuple[torch.Tensor, RNNState]:
        """Update one token using normalized forget and input gates."""
        batch = x_t.shape[0]
        h_prev = state.hidden
        if h_prev.dim() == 2:
            h_prev = h_prev.view(batch, self.num_heads, self.head_dim)

        x_norm = self.norm(x_t)
        x_heads = x_norm.view(batch, self.num_heads, self.head_dim)
        forget = torch.sigmoid(per_head_linear(self.linear_f, x_heads, stack_dim=1))
        input_gate = torch.sigmoid(per_head_linear(self.linear_i, x_heads, stack_dim=1))
        candidate = minimal_candidate(per_head_linear(self.linear_h, x_heads, stack_dim=1))
        gate_sum = forget + input_gate
        forget_prime = forget / gate_sum
        input_prime = input_gate / gate_sum
        h_t = forget_prime * h_prev + input_prime * candidate

        output = self.output_proj(h_t.reshape(batch, self.hidden_dim))
        output = self.dropout(output)
        return x_t + output, RNNState(hidden=h_t)

    def forward(self, x: torch.Tensor, state: RNNState) -> tuple[torch.Tensor, RNNState]:
        x_norm = self.norm(x)
        forget, input_gate, candidate = self._project(x_norm)

        h_prev = state.hidden
        if h_prev.dim() == 2:
            h_prev = h_prev.view(x.shape[0], self.num_heads, self.head_dim)
        diff = F.softplus(-forget) - F.softplus(-input_gate)
        log_forget = -F.softplus(diff)
        log_input = -F.softplus(-diff)
        log_h0 = h_prev.unsqueeze(1).log()
        log_candidate = minimal_log_candidate(candidate)
        log_values = torch.cat([log_h0, log_input + log_candidate], dim=1)

        h_t = parallel_scan_log(log_forget, log_values)
        output = self.output_proj(h_t.reshape(x.shape[0], x.shape[1], self.hidden_dim))
        output = self.dropout(output)
        return x + output, RNNState(hidden=h_t[:, -1])


@register_model("min_lstm")
class MinLSTM(SequentialRNNModel):
    """Minimal LSTM with parallel log-space training and sequential decoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        mlp_hidden_mult: int = 4,
        num_classes: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.emb_norm = RMSNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MinLSTMLayer(hidden_dim, num_heads, dropout),
                        FeedForwardSublayer(hidden_dim, mlp_hidden_mult, dropout),
                    ]
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        return RNNStateList(
            [
                RNNState(
                    hidden=torch.ones(
                        batch_size,
                        self.num_heads,
                        self.hidden_dim // self.num_heads,
                        device=device,
                    )
                )
                for _ in range(self.num_layers)
            ]
        )
