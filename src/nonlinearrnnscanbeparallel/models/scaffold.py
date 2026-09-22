"""minGRU Scaffold for parallel prefix scan.

This is a lightweight linear recurrence that summarizes the prefix.
It is separate from the target RNN and discarded at inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import RMSNorm
from .minimal import minimal_candidate, minimal_log_candidate, parallel_scan_log, per_head_linear


class ScaffoldBase(nn.Module):
    """Structural protocol for scaffold modules.

    Any scaffold maps a layer input [B, T, D] to boundary summaries
    [B, T, S]. Implementations vary (minGRU vs minLSTM); the wrapper
    only depends on this forward signature.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MinGRUScaffold(ScaffoldBase):
    """Parallel-scannable minGRU scaffold.

    Computes s_t = (1 - z_t) * s_{t-1} + z_t * h_t where z_t, h_t depend only on x_t.
    This is an associative prefix scan in log-space for numerical stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm = RMSNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_z = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim, bias=True) for _ in range(num_heads)]
        )
        self.linear_h = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim, bias=True) for _ in range(num_heads)]
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project input to update gate and candidate per head.

        Args:
            x: Input tensor [B, T, D] or [B, D] for single step
        """
        if x.dim() == 2:
            # Single step: [B, D] -> [B, 1, D]
            x = x.unsqueeze(1)
        batch, sequence_len = x.shape[:2]
        x_heads = x.reshape(batch, sequence_len, self.num_heads, self.head_dim)
        update = per_head_linear(self.linear_z, x_heads, stack_dim=2)
        candidate = per_head_linear(self.linear_h, x_heads, stack_dim=2)
        return update, candidate

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Single-token update for sequential decoding.

        Args:
            x_t: Input tensor [B, D]
            h_prev: Previous hidden state [B, H, D_head] or [B, D]

        Returns:
            New hidden state [B, H, D_head]
        """
        batch = x_t.shape[0]
        if h_prev.dim() == 2:
            h_prev = h_prev.view(batch, self.num_heads, self.head_dim)

        x_norm = self.norm(x_t)
        x_proj = self.input_proj(x_norm)
        x_heads = x_proj.view(batch, self.num_heads, self.head_dim)

        update = torch.sigmoid(per_head_linear(self.linear_z, x_heads, stack_dim=1))
        candidate = minimal_candidate(per_head_linear(self.linear_h, x_heads, stack_dim=1))
        h_t = (1.0 - update) * h_prev + update * candidate
        return h_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward pass using associative prefix scan.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            States for all timesteps [B, T, D]
        """
        input_dtype = x.dtype
        x_norm = self.norm(x)
        x_proj = self.input_proj(x_norm)
        update, candidate = self._project(x_proj)

        # Use log-space scan for numerical stability (matching minGRU implementation)
        # log_coeffs = log(1 - z), log_values = log(z * candidate) + log(h_prev)
        log_update = -F.softplus(-update)  # log(z)
        log_one_minus_update = -F.softplus(update)  # log(1 - z)
        log_candidate = minimal_log_candidate(candidate)  # log(candidate)

        # Initial state is zeros -> approximate with large negative number
        # log(h_0) where h_0 = 0 -> use -1e6 so exp(log_h0) ≈ 0
        batch, seq_len, num_heads, head_dim = update.shape
        log_h0 = torch.full(
            (batch, 1, num_heads, head_dim), -1e6, device=update.device, dtype=update.dtype
        )

        # log_coeffs = log(1 - z) for all T steps
        log_coeffs = log_one_minus_update
        # log_values = [log_h0, log(z_1) + log(candidate_1), ..., log(z_T) + log(candidate_T)]
        log_values = torch.cat([log_h0, log_update + log_candidate], dim=1)

        h_t = parallel_scan_log(log_coeffs, log_values)

        # Project to output dimension
        output = self.output_proj(h_t.reshape(x.shape[0], x.shape[1], self.hidden_dim))
        output = self.dropout(output)

        # Cast output to input dtype
        return output.to(input_dtype)


class MinLSTMScaffold(ScaffoldBase):
    """Parallel-scannable minLSTM scaffold.

    Computes h_t = f'_t * h_{t-1} + i'_t * c_t with normalized gates
    f' = f/(f+i), i' = i/(f+i), mirroring MinLSTMLayer gate math in
    log-space for numerical stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm = RMSNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_f = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim, bias=True) for _ in range(num_heads)]
        )
        self.linear_i = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim, bias=True) for _ in range(num_heads)]
        )
        self.linear_h = nn.ModuleList(
            [nn.Linear(self.head_dim, self.head_dim, bias=True) for _ in range(num_heads)]
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch, sequence_len = x.shape[:2]
        x_heads = x.reshape(batch, sequence_len, self.num_heads, self.head_dim)
        forget = per_head_linear(self.linear_f, x_heads, stack_dim=2)
        input_gate = per_head_linear(self.linear_i, x_heads, stack_dim=2)
        candidate = per_head_linear(self.linear_h, x_heads, stack_dim=2)
        return forget, input_gate, candidate

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        batch = x_t.shape[0]
        if h_prev.dim() == 2:
            h_prev = h_prev.view(batch, self.num_heads, self.head_dim)

        x_norm = self.norm(x_t)
        x_proj = self.input_proj(x_norm)
        x_heads = x_proj.view(batch, self.num_heads, self.head_dim)

        forget = torch.sigmoid(per_head_linear(self.linear_f, x_heads, stack_dim=1))
        input_gate = torch.sigmoid(per_head_linear(self.linear_i, x_heads, stack_dim=1))
        candidate = minimal_candidate(per_head_linear(self.linear_h, x_heads, stack_dim=1))
        gate_sum = forget + input_gate
        h_t = (forget / gate_sum) * h_prev + (input_gate / gate_sum) * candidate
        return h_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_norm = self.norm(x)
        x_proj = self.input_proj(x_norm)
        forget, input_gate, candidate = self._project(x_proj)

        diff = F.softplus(-forget) - F.softplus(-input_gate)
        log_forget = -F.softplus(diff)
        log_input = -F.softplus(-diff)
        log_candidate = minimal_log_candidate(candidate)

        batch, seq_len, num_heads, head_dim = forget.shape
        log_h0 = torch.full(
            (batch, 1, num_heads, head_dim), -1e6, device=forget.device, dtype=forget.dtype
        )

        log_coeffs = log_forget
        log_values = torch.cat([log_h0, log_input + log_candidate], dim=1)

        h_t = parallel_scan_log(log_coeffs, log_values)

        output = self.output_proj(h_t.reshape(x.shape[0], x.shape[1], self.hidden_dim))
        output = self.dropout(output)
        return output.to(input_dtype)


class ScaffoldStack(ScaffoldBase):
    """Stack of 1..N scaffold layers applied per outer RNN layer.

    ``scaffold_type`` selects the cell (min_gru | min_lstm);
    ``num_layers`` is unbounded (typical 1-4). ``num_layers=1``
    reproduces the legacy single-scaffold behavior exactly.
    """

    def __init__(
        self,
        scaffold_type: str = "min_gru",
        input_dim: int = 0,
        hidden_dim: int = 0,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if scaffold_type not in ("min_gru", "min_lstm"):
            raise ValueError(f"Unknown scaffold_type: {scaffold_type}")
        if num_layers < 1:
            raise ValueError("scaffold num_layers must be >= 1")
        self.scaffold_type = scaffold_type
        self.num_scaffold_layers = num_layers
        cell = MinGRUScaffold if scaffold_type == "min_gru" else MinLSTMScaffold
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(cell(in_dim, hidden_dim, num_heads, dropout))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
