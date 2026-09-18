"""minGRU Scaffold for parallel prefix scan.

This is a lightweight linear recurrence that summarizes the prefix.
It is separate from the target RNN and discarded at inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .base import RMSNorm
from .minimal import minimal_log_candidate, parallel_scan_log, segmented_parallel_scan_log


class MinGRUScaffold(nn.Module):
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

    def candidate_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Minimal candidate activation: x >= 0 -> x + 0.5, else sigmoid(x)."""
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

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
        update = torch.stack(
            [linear(x_heads[:, :, head]) for head, linear in enumerate(self.linear_z)],
            dim=2,
        )
        candidate = torch.stack(
            [linear(x_heads[:, :, head]) for head, linear in enumerate(self.linear_h)],
            dim=2,
        )
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

        update = torch.stack(
            [torch.sigmoid(linear(x_heads[:, head])) for head, linear in enumerate(self.linear_z)],
            dim=1,
        )
        candidate = torch.stack(
            [
                self.candidate_activation(linear(x_heads[:, head]))
                for head, linear in enumerate(self.linear_h)
            ],
            dim=1,
        )
        h_t = (1.0 - update) * h_prev + update * candidate
        return h_t

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor | None = None) -> torch.Tensor:
        """Parallel forward pass using associative prefix scan.

        Args:
            x: Input tensor [B, T, D]
            seq_lens: Optional tensor of sequence lengths [B]. If provided,
                the scan is reset at each sequence boundary.

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

        # Handle variable-length sequences: reset at sequence boundaries
        if seq_lens is not None:
            # Create segment IDs from sequence lengths
            # segment_ids[b, t] = 0 for t < seq_lens[b], 1 for t >= seq_lens[b] (padding)
            # But we also need to handle multiple sequences per batch if packed
            # For simplicity, assume each batch item is one sequence with length seq_lens[b]
            # and padding after that
            B, T = x.shape[:2]
            device = x.device
            t_idx = torch.arange(T, device=device).view(1, T).expand(B, T)
            # segment_id = 0 for valid positions, 1 for padding
            segment_ids = (t_idx >= seq_lens.view(-1, 1)).long()
            # Run segmented scan
            h_t = segmented_parallel_scan_log(log_coeffs, log_values, segment_ids)
        else:
            # Run standard parallel scan
            h_t = parallel_scan_log(log_coeffs, log_values)

        # Project to output dimension
        output = self.output_proj(h_t.reshape(x.shape[0], x.shape[1], self.hidden_dim))
        output = self.dropout(output)

        # Cast output to input dtype
        return output.to(input_dtype)
