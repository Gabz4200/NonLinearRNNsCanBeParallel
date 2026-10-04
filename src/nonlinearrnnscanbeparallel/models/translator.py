"""Translator module: maps scaffold summary to target RNN hidden state space.

The translator runs only at chunk boundaries (M evaluations per forward pass),
making it cheap regardless of internal complexity.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import RMSNorm
from .rkan import RKANLayer

# Upper bound on the translator MLP's hidden width. Matrix-state targets emit
# num_heads * prod(memory shape) units per boundary, which can reach tens of
# thousands; expanding that 4x would put almost the entire parameter budget in
# the translator instead of the target RNN.
TRANSLATOR_MAX_HIDDEN = 512


def _init_gdn2_linear(layer: nn.Linear) -> None:
    """Initialize linear layer per GDN-2 recipe: Xavier uniform with gain 2^-2.5, zero bias."""
    gain = 2**-2.5  # ≈ 0.1768
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _build_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class _RKANTranslator(nn.Module):
    """rKAN-based translator for rKAN-RNN targets.

    Maps from scaffold_dim -> hidden_dim using two rKAN layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rkan_degree: int = 3,
        rkan_alpha: float = 1.0,
        rkan_beta: float = 1.0,
        rkan_iota: float = 1.0,
        rkan_mapping: str = "algebraic_infinite",
        rkan_type: str = "jacobi",
        rkan_num_basis: int = 4,
        dropout: float = 0.1,
        norm_type: str = "rmsnorm",
    ) -> None:
        super().__init__()
        self.norm = _build_norm(norm_type, input_dim)
        self.rkan1 = RKANLayer(
            input_dim,
            hidden_dim * 4,
            degree=rkan_degree,
            alpha=rkan_alpha,
            beta=rkan_beta,
            iota=rkan_iota,
            mapping_type=rkan_mapping,
            rkan_type=rkan_type,
            num_basis=rkan_num_basis,
        )
        self.dropout = nn.Dropout(dropout)
        self.rkan2 = RKANLayer(
            hidden_dim * 4,
            hidden_dim,
            degree=rkan_degree,
            alpha=rkan_alpha,
            beta=rkan_beta,
            iota=rkan_iota,
            mapping_type=rkan_mapping,
            rkan_type=rkan_type,
            num_basis=rkan_num_basis,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.rkan1(x)
        x = self.dropout(x)
        x = self.rkan2(x)
        return x


class TranslatorLayer(nn.Module):
    """Single translator layer: scaffold summary -> boundary state.

    ``translator_type`` (mlp | rkan) is a pure hyperparameter, decoupled
    from the target model being trained. ``output_shape`` selects the
    output geometry: "vector" [B, M, D] or "matrix".  The matrix geometry is
    ``[B, M, N, K, V]`` for M2RNN, and ``[B, M, N, S, K, V]`` when
    ``num_memories > 1`` (HOPE stacks five memories per head on axis 2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        translator_type: str = "mlp",
        output_shape: str = "vector",
        dropout: float = 0.1,
        rkan_degree: int = 3,
        rkan_alpha: float = 1.0,
        rkan_beta: float = 1.0,
        rkan_iota: float = 1.0,
        rkan_mapping: str = "algebraic_infinite",
        rkan_type: str = "jacobi",
        rkan_num_basis: int = 4,
        use_gdn2_init: bool = True,
        num_heads: int = 4,
        key_dim: int = 16,
        value_dim: int = 16,
        num_memories: int = 1,
        state_scale: float | None = None,
        pre_norm: bool = True,
        norm_type: str = "rmsnorm",
    ) -> None:
        super().__init__()
        if translator_type not in ("mlp", "rkan"):
            raise ValueError(f"Unknown translator_type: {translator_type}")
        if output_shape not in ("vector", "matrix"):
            raise ValueError(f"Unknown output_shape: {output_shape}")
        if num_memories < 1:
            raise ValueError("num_memories must be >= 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.translator_type = translator_type
        self.output_shape = output_shape
        self.use_gdn2_init = use_gdn2_init
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_memories = num_memories
        matrix_state_dim = num_heads * num_memories * key_dim * value_dim

        # Pre-norm (on scaffold output), RMSNorm by default, LayerNorm opt-in
        self.input_norm = _build_norm(norm_type, input_dim) if pre_norm else nn.Identity()

        if translator_type == "mlp":
            out_dim = matrix_state_dim if output_shape == "matrix" else hidden_dim
            # 2-layer MLP with SiLU (matching MLP-RNN, minGRU, minLSTM).
            # The hidden width is capped: matrix targets can emit tens of
            # thousands of units per boundary, and an uncapped 4x expansion
            # there dominates the whole parameter budget.
            net_hidden = min(out_dim * 4, max(out_dim, TRANSLATOR_MAX_HIDDEN))
            self.net = nn.Sequential(
                nn.Linear(input_dim, net_hidden, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(net_hidden, out_dim, bias=False),
            )
            if use_gdn2_init:
                self._apply_gdn2_init()
        else:
            # 2-layer rKAN (matching rKAN-RNN)
            out_dim = matrix_state_dim if output_shape == "matrix" else hidden_dim
            self.net = _RKANTranslator(
                input_dim,
                out_dim,
                rkan_degree=rkan_degree,
                rkan_alpha=rkan_alpha,
                rkan_beta=rkan_beta,
                rkan_iota=rkan_iota,
                rkan_mapping=rkan_mapping,
                rkan_type=rkan_type,
                rkan_num_basis=rkan_num_basis,
                dropout=dropout,
                norm_type=norm_type,
            )
        # Output normalization (on boundary state before feeding to chunk RNN)
        if output_shape == "matrix":
            self.output_norm = RMSNorm(matrix_state_dim)
        else:
            self.output_norm = RMSNorm(hidden_dim)
        # Targets whose state magnitude differs a lot from unit RMS (HOPE's
        # memories initialize near zero) need the boundary rescaled to match;
        # otherwise chunk 1 starts far outside the trained basin.
        self.state_scale = state_scale

    def _apply_gdn2_init(self) -> None:
        """Apply GDN-2 initialization: Xavier uniform with gain 2^-2.5, zero biases."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                _init_gdn2_linear(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Translate scaffold summary to boundary state.

        Args:
            x: Scaffold states at boundaries [B, M, D_scaffold]

        Returns:
            Boundary hidden states:
            - vector output_shape: [B, M, D_target]
            - matrix output_shape: [B, M, N, K, V], plus an extra
              [B, M, N, S, K, V] axis when ``num_memories > 1``
        """
        x = self.input_norm(x)
        x = self.net(x)
        if self.output_norm is not None:
            x = self.output_norm(x)
        if self.state_scale is not None:
            # The output norm puts every boundary at unit RMS, but a target's
            # state may live at a very different magnitude (HOPE's memories
            # start near zero). Rescale so the boundary matches that scale.
            x = x * self.state_scale
        if self.output_shape == "matrix":
            # Reshape to matrix state; the memory axis is dropped when a target
            # keeps a single matrix per head.
            batch, num_boundaries, _ = x.shape
            shape = [batch, num_boundaries, self.num_heads]
            if self.num_memories > 1:
                shape.append(self.num_memories)
            shape.extend((self.key_dim, self.value_dim))
            x = x.view(*shape)
        return x


_LEGACY_TARGET_TO_TRANSLATOR = {
    "mlp": "mlp",
    "min_gru": "mlp",
    "min_lstm": "mlp",
    "m2rnn": "mlp",
    "hope": "mlp",
    "rkan": "rkan",
}


class TranslatorStack(nn.Module):
    """Stack of 1..N translator layers applied per outer RNN layer.

    Intermediate layers use vector output; only the final layer applies
    ``output_shape`` (matrix reshape for M2RNN targets). ``num_layers=1``
    reproduces the legacy single-translator behavior exactly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        translator_type: str = "mlp",
        num_layers: int = 1,
        output_shape: str = "vector",
        dropout: float = 0.1,
        rkan_degree: int = 3,
        rkan_alpha: float = 1.0,
        rkan_beta: float = 1.0,
        rkan_iota: float = 1.0,
        rkan_mapping: str = "algebraic_infinite",
        rkan_type: str = "jacobi",
        rkan_num_basis: int = 4,
        use_gdn2_init: bool = True,
        num_heads: int = 4,
        key_dim: int = 16,
        value_dim: int = 16,
        num_memories: int = 1,
        state_scale: float | None = None,
        pre_norm: bool = True,
        norm_type: str = "rmsnorm",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("translator num_layers must be >= 1")
        self.num_translator_layers = num_layers
        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers):
            last = i == num_layers - 1
            layers.append(
                TranslatorLayer(
                    in_dim,
                    hidden_dim,
                    translator_type=translator_type,
                    output_shape=output_shape if last else "vector",
                    dropout=dropout,
                    rkan_degree=rkan_degree,
                    rkan_alpha=rkan_alpha,
                    rkan_beta=rkan_beta,
                    rkan_iota=rkan_iota,
                    rkan_mapping=rkan_mapping,
                    rkan_type=rkan_type,
                    rkan_num_basis=rkan_num_basis,
                    use_gdn2_init=use_gdn2_init,
                    num_heads=num_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    num_memories=num_memories,
                    state_scale=state_scale if last else None,
                    pre_norm=pre_norm,
                    norm_type=norm_type,
                )
            )
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
