"""minGRU Scaffold for parallel prefix scan.

This is a lightweight linear recurrence that summarizes the prefix.
It is separate from the target RNN and discarded at inference.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as functional
from torch import nn

from .base import RMSNorm
from .minimal import minimal_candidate, minimal_log_candidate, parallel_scan_log, per_head_linear


class ScaffoldBase(nn.Module, ABC):
    """Structural protocol for scaffold modules.

    Any scaffold maps a layer input [B, T, D] to prefix-summary states
    [B, T, S]. Implementations vary (minGRU vs minLSTM vs linear_sum vs
    fourier); the wrapper only depends on this forward signature, and
    reads only the states at chunk-boundary indices. Contract: state t
    must depend on inputs <= t only (causality — a boundary state seeds
    the first step of the next chunk, and future tokens must not leak
    into it), and each such state becomes the target RNN's initial hidden
    state for that chunk; within a chunk the target recurrence updates
    the state normally from step 2 onward.
    """

    @staticmethod
    def positional_encoding(
        sequence_len: int, hidden_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sinusoidal absolute positional encoding, length/dim agnostic.

        Same construction as Fourier-Wavelet's additive "X + P" input
        encoding (arXiv:2111.15473 Fig 1); parameter-free.
        """
        positions = torch.arange(sequence_len, device=device, dtype=dtype).unsqueeze(1)
        dims = torch.arange(hidden_dim, device=device, dtype=dtype).unsqueeze(0)
        exponent = torch.floor(dims / 2.0) * 2.0 / hidden_dim
        inverse_freq = torch.pow(torch.full_like(dims, 10000.0), exponent)
        angles = positions / inverse_freq
        return torch.where(dims % 2 == 0, angles.sin(), angles.cos())

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_encoding(x.shape[1], x.shape[2], x.device, x.dtype).unsqueeze(0)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a layer input [B, T, D] to causally-derived prefix states [B, T, S]."""


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
        log_update = -functional.softplus(-update)  # log(z)
        log_one_minus_update = -functional.softplus(update)  # log(1 - z)
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

        diff = functional.softplus(-forget) - functional.softplus(-input_gate)
        log_forget = -functional.softplus(diff)
        log_input = -functional.softplus(-diff)
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


class LinearSumScaffold(ScaffoldBase):
    """Linear projection + position-weighted prefix sum.

    s_t = sum_{i<t} w_i x_i with w = linspace(min_weight, 1.0, t):
    w_i = min_weight + (1 - min_weight) * i / (t - 1) for t > 1 (and
    w_0 = min_weight for t = 1). Every prefix token contributes
    (min_weight > 0), the most recent dominates, older tokens decay
    linearly. Computed with two cumulative sums, fully parallel over T.
    """

    def __init__(self, input_dim: int, hidden_dim: int, min_weight: float = 0.1) -> None:
        super().__init__()
        if not 0.0 < min_weight < 1.0:
            raise ValueError("scaffold min_weight must be in (0, 1)")
        self.min_weight = min_weight
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)  # [B, T, D]
        sequence_len = x.shape[1]
        total = x.cumsum(dim=1)  # sum_{i<t} x_i
        if sequence_len <= 1:
            return self.min_weight * total
        positions = torch.arange(sequence_len, device=x.device, dtype=x.dtype)
        weighted = (x * positions.view(1, -1, 1)).cumsum(dim=1)  # sum_{i<t} i * x_i
        # w_i = min_weight + (1 - min_weight) * i / (t - 1) for t >= 2
        scale = torch.zeros(sequence_len, device=x.device, dtype=x.dtype)
        scale[1:] = (1.0 - self.min_weight) / positions[1:]
        return self.min_weight * total + weighted * scale.view(1, -1, 1)


class FourierScaffold(ScaffoldBase):
    """Causal cosine real-DFT token mixing (FNet/Fourier-Wavelet).

    s_t = (2 / T) * sum_{k <= t} x_k cos(2 pi t k / T)

    This is the causally masked real discrete Fourier transform of the
    prefix (arXiv:2111.15473 Sec 3.2), i.e. FNet-style Fourier token
    mixing (arXiv:2105.03824) made autoregressive: bit t depends only on
    inputs <= t, so chunk-boundary summaries never leak future tokens.
    Positional encoding is applied once at stack input (see
    ``ScaffoldBase.positional_encoding``). Parameter-free apart from the
    per-layer linear projection to hidden_dim; the mixing kernel is a
    fixed cosine basis. Computed as a triangular matrix multiply
    (DFT-matrix form): O(T^2) time and memory per scaffold layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)  # [B, T, D]
        sequence_len = x.shape[1]
        positions = torch.arange(sequence_len, device=x.device, dtype=x.dtype)
        kernel = torch.tril(
            torch.cos(2.0 * math.pi * torch.outer(positions, positions) / sequence_len)
        )
        kernel = kernel * (2.0 / sequence_len)
        return torch.einsum("tk,bkd->btd", kernel, x)


class ScaffoldStack(ScaffoldBase):
    """Stack of 1..N scaffold layers applied per outer RNN layer.

    ``scaffold_type`` selects the cell (min_gru | min_lstm | linear_sum
    | fourier); ``num_layers`` is unbounded (typical 1-4). ``num_layers=1``
    reproduces the legacy single-scaffold behavior exactly. When
    ``positional`` is set (default), a fixed sinusoidal positional encoding
    is added once to the stack input (Fourier-Wavelet "X + P", arXiv:2111.
    15473 Fig 1); off is the FNet ablation (arXiv:2105.03824) where the
    mixing kernel alone encodes position.
    """

    def __init__(
        self,
        scaffold_type: str = "min_gru",
        input_dim: int = 0,
        hidden_dim: int = 0,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        min_weight: float = 0.1,
        positional: bool = True,
    ) -> None:
        super().__init__()
        if scaffold_type not in ("min_gru", "min_lstm", "linear_sum", "fourier"):
            raise ValueError(f"Unknown scaffold_type: {scaffold_type}")
        if num_layers < 1:
            raise ValueError("scaffold num_layers must be >= 1")
        self.scaffold_type = scaffold_type
        self.num_scaffold_layers = num_layers
        self.positional = positional
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            if scaffold_type == "linear_sum":
                layers.append(LinearSumScaffold(in_dim, hidden_dim, min_weight))
            elif scaffold_type == "fourier":
                layers.append(FourierScaffold(in_dim, hidden_dim))
            else:
                cell = MinGRUScaffold if scaffold_type == "min_gru" else MinLSTMScaffold
                layers.append(cell(in_dim, hidden_dim, num_heads, dropout))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional:
            x = self._add_positional(x)
        for layer in self.layers:
            x = layer(x)
        return x
