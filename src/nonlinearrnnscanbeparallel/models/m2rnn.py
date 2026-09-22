"""Matrix-to-Matrix RNN (M²RNN) per arXiv:2603.14360 Section 3.

Implements the matrix-valued hidden-state recurrence:
    Z_t = tanh(H_{t-1} W + k_t v_t^T)        (Eq. 19)
    H_t = f_t H_{t-1} + (1 - f_t) Z_t        (Eq. 20)
    y_t = H_t^T q_t + w_r ⊙ v_t             (Eq. 21)
    y_gt = RMSNorm(y_t ⊙ g_t)               (Eq. 22)
    o_t = W_o y_gt                           (Eq. 23)

Pre-recurrence projections (q, k, v via depthwise causal conv1d+SiLU;
f via parameterized forget gate ψ; g via SiLU) are computed in parallel
across the sequence.  The tanh non-linearity forces a sequential loop for
the matrix-state recurrence.

Precision and complexity contract.
The paper (arXiv:2603.14360) proves M²RNN *can represent* all tasks of
non-linear vector-valued RNNs (Theorem 1) but does NOT claim P-completeness
in Section 3.2.  Unlike arXiv:2603.03612 (the MLP-RNN companion), this
repository does not implement or verify polynomial-precision / polynomial-
state scaling that would be required for a P-complete construction.  The
code realizes the *architecture* (matrix-valued states, forget gate,
outer-product write) but the complexity class guarantee remains unverified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RMSNorm, RNNState, RNNStateList, SequentialRNNModel
from .mlp_rnn import FeedForwardSublayer
from .registry import register_model

# Default per-head dimensions following the paper's multi-value formulation
# (Section 3.1, Table 8): K=64, V=16 yields the largest state per parameter.
DEFAULT_KEY_DIM = 64
DEFAULT_VALUE_DIM = 16


def psi(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Parameterized forget-gate activation ψ from Eq. 13.

        f_t = ψ(x_t) = 1 / (1 + e^{x_t + β})^α

    `alpha` is the log of the paper's α (effective α = exp(alpha) > 0);
    `beta` is the paper's β shift directly. When effective α=1 this
    reduces to σ(-x_t - β).
    """
    return 1.0 / (1.0 + torch.exp(x + beta)).pow(torch.exp(alpha))


class DepthwiseConv1d(nn.Module):
    """Depthwise causal 1-D convolution (kernel size 4) with SiLU activation.

    Used for q, k, v projections per Section 3.1.2.  Operates on
    [B, T, D] tensors, internally transposing to [B, D, T] for Conv1d.

    Supports conv cache for BPTT chunking / autoregressive step().
    """

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=0,  # we apply causal left-padding manually
            groups=channels,  # depthwise
            bias=False,
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, conv_cache: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            conv_cache: [B, D, kernel_size-1] or None

        Returns:
            out: [B, T, D]
            new_cache: [B, D, kernel_size-1]
        """
        # Transpose to Conv1d layout: [B, D, T]
        x_t = x.transpose(1, 2)

        # Prepend cache if available, else left-pad with zeros
        if conv_cache is not None:
            x_t = torch.cat([conv_cache, x_t], dim=-1)
        else:
            x_t = F.pad(x_t, (self.kernel_size - 1, 0))

        # New cache = last (kernel_size-1) tokens of the *padded input*
        # (needed for next chunk's causal conv)
        new_cache = x_t[..., -(self.kernel_size - 1) :].detach()

        # Convolution
        x_t = self.conv(x_t)

        return x_t.transpose(1, 2), new_cache


class M2RNNLayer(nn.Module):
    """Single M²RNN layer with matrix-valued hidden states (arXiv:2603.14360 §3.1).

    Per-head state: H ∈ R^{K×V}; output per head: y ∈ R^V.

    Pre-recurrence projections are computed in parallel across all
    time steps; the tanh non-linearity forces a sequential loop for
    the matrix-state recurrence.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        key_dim: int = DEFAULT_KEY_DIM,
        value_dim: int = DEFAULT_VALUE_DIM,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        proj_dim = num_heads * (key_dim + key_dim + value_dim)
        self.qkv_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.qkv_conv = DepthwiseConv1d(proj_dim, kernel_size=conv_kernel)

        self.W_f = nn.Linear(hidden_dim, num_heads, bias=False)
        # Log-alpha keeps init at the sigmoid case.
        self.alpha_raw = nn.Parameter(torch.zeros(num_heads))
        self.beta_raw = nn.Parameter(torch.zeros(num_heads))

        self.W_g = nn.Linear(hidden_dim, num_heads * value_dim, bias=False)
        self.W = nn.Parameter(torch.stack([torch.eye(value_dim) for _ in range(num_heads)]))
        self.w_r = nn.Parameter(torch.ones(num_heads, value_dim) * 0.1)

        self.W_o = nn.Linear(num_heads * value_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Per-head RMSNorm scale.
        self.rms_scale = nn.Parameter(torch.ones(num_heads, value_dim))

    def _project(
        self,
        x: torch.Tensor,
        conv_cache: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Parallel pre-recurrence projections for a [B, T, D] input.

        Args:
            x: [B, T, D]
            conv_cache: [B, proj_dim, kernel_size-1] or None

        Returns:
            q, k [B, T, N, K], v [B, T, N, V], f [B, T, N], g [B, T, N, V], new_conv_cache
        """
        t = x.shape[1]
        num_heads = self.num_heads
        dim_k = self.key_dim
        dim_v = self.value_dim

        qkv_proj = self.qkv_proj(x)
        qkv, new_conv_cache = self.qkv_conv(qkv_proj, conv_cache)
        qkv = nn.SiLU()(qkv)

        q = qkv[:, :, : num_heads * dim_k].reshape(x.shape[0], t, num_heads, dim_k)
        k = qkv[:, :, num_heads * dim_k : 2 * num_heads * dim_k].reshape(
            x.shape[0], t, num_heads, dim_k
        )
        v = qkv[:, :, 2 * num_heads * dim_k :].reshape(x.shape[0], t, num_heads, dim_v)

        f = psi(self.W_f(x), self.alpha_raw, self.beta_raw)

        g = nn.SiLU()(self.W_g(x))
        g = g.reshape(x.shape[0], t, num_heads, dim_v)
        return q, k, v, f, g, new_conv_cache

    def _recurrence_update(
        self,
        h_prev: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        q_t: torch.Tensor,
        g_t: torch.Tensor,
        f_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One matrix-state update (Eqs. 19-22): (H_{t-1}, k, v, q, g, f) -> (H_t, y_gt)."""
        hw = torch.einsum("bnkv,nvw->bnkw", h_prev, self.W)
        outer = torch.einsum("bnk,bnv->bnkv", k_t, v_t)
        z = torch.tanh(hw + outer)

        f_t = f_t.unsqueeze(-1).unsqueeze(-1)
        h = f_t * h_prev + (1.0 - f_t) * z

        ht_tq = torch.einsum("bnkv,bnk->bnv", h, q_t)
        y_t = ht_tq + self.w_r * v_t
        return h, self._rmsnorm(y_t * g_t)

    def step(self, x_t: torch.Tensor, state: RNNState) -> tuple[torch.Tensor, RNNState]:
        """Single-token update: x_t [B, D] with state.hidden [B, N, K, V].

        Uses conv cache from state.extra for causal convolution continuity.
        """
        conv_cache = state.extra.get("conv_cache") if state.extra else None
        q, k, v, f, g, new_conv_cache = self._project(x_t.unsqueeze(1), conv_cache)

        h = state.hidden
        if h.dim() == 3:
            h = h.view(x_t.shape[0], self.num_heads, self.key_dim, self.value_dim)

        h, y_gt = self._recurrence_update(h, k[:, 0], v[:, 0], q[:, 0], g[:, 0], f[:, 0])

        o = self.dropout(self.W_o(y_gt.reshape(x_t.shape[0], self.num_heads * self.value_dim)))
        # Residual connection around the recurrence (paper §3.1.2 Fig. 2).
        extra = {"conv_cache": new_conv_cache}
        return o + x_t, RNNState(hidden=h, extra=extra)

    def forward(self, x: torch.Tensor, state: RNNState) -> tuple[torch.Tensor, RNNState]:
        t = x.shape[1]
        num_heads = self.num_heads
        dim_v = self.value_dim

        conv_cache = state.extra.get("conv_cache") if state.extra else None
        q, k, v, f, g, new_conv_cache = self._project(x, conv_cache)

        h = state.hidden
        if h.dim() == 3:
            h = h.view(x.shape[0], num_heads, self.key_dim, dim_v)

        outputs: list[torch.Tensor] = []
        for t_step in range(t):
            h, y_gt = self._recurrence_update(
                h, k[:, t_step], v[:, t_step], q[:, t_step], g[:, t_step], f[:, t_step]
            )
            outputs.append(y_gt)

        y_out = torch.stack(outputs, dim=1)
        y_out = y_out.reshape(x.shape[0], t, num_heads * dim_v)

        o = self.W_o(y_out)
        o = self.dropout(o)
        # Residual connection around the recurrence (paper §3.1.2 Fig. 2).
        extra = {"conv_cache": new_conv_cache}
        return o + x, RNNState(hidden=h, extra=extra)

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm over the value dimension V for each head.

        x: [B, N, V] → normalized [B, N, V].
        """
        mean_sq = (x * x).mean(dim=-1, keepdim=True).clamp(min=1e-6)
        return x * mean_sq.rsqrt() * self.rms_scale


@register_model("m2rnn")
class M2RNN(SequentialRNNModel):
    """Matrix-to-Matrix RNN per arXiv:2603.14360.

    Architecture (paper §3.1.2, Fig. 2):
      - Token embedding (provided by task, already embedded to hidden_dim)
      - emb_norm -> alternating {M2RNNLayer, FeedForwardSublayer} x num_layers
      - final_norm -> classifier

    The M²RNN layer replaces attention in a Transformer-like block:
    RMSNorm + residual around each M²RNNLayer and FFN.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        key_dim: int = DEFAULT_KEY_DIM,
        value_dim: int = DEFAULT_VALUE_DIM,
        mlp_hidden_mult: int = 4,
        num_classes: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.emb_norm = RMSNorm(hidden_dim)

        self.layers = nn.ModuleList()
        self.num_rnn_layers = num_layers
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        M2RNNLayer(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            key_dim=key_dim,
                            value_dim=value_dim,
                            dropout=dropout,
                        ),
                        FeedForwardSublayer(hidden_dim, mlp_hidden_mult, dropout),
                    ]
                )
            )

        self.final_norm = RMSNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def init_state(self, batch_size: int, device: torch.device) -> RNNStateList:
        # conv_cache shape: [B, proj_dim, kernel_size-1]
        # proj_dim = num_heads * (key_dim + key_dim + value_dim)
        proj_dim = self.num_heads * (self.key_dim + self.key_dim + self.value_dim)
        kernel_size = 4  # default in M2RNNLayer
        return RNNStateList(
            [
                RNNState(
                    hidden=torch.zeros(
                        batch_size,
                        self.num_heads,
                        self.key_dim,
                        self.value_dim,
                        device=device,
                    ),
                    extra={
                        "conv_cache": torch.zeros(
                            batch_size, proj_dim, kernel_size - 1, device=device
                        )
                    },
                )
                for _ in range(self.num_rnn_layers)
            ]
        )
