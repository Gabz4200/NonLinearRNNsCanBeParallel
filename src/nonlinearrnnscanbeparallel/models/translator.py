"""Translator module: maps scaffold summary to target RNN hidden state space.

The translator runs only at chunk boundaries (M evaluations per forward pass),
making it cheap regardless of internal complexity.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import RMSNorm
from .rkan import RKANLayer


def _init_gdn2_linear(layer: nn.Linear) -> None:
    """Initialize linear layer per GDN-2 recipe: Xavier uniform with gain 2^-2.5, zero bias."""
    gain = 2**-2.5  # ≈ 0.1768
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


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
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(input_dim)
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
    output geometry: "vector" [B, M, D] or "matrix" [B, M, N, K, V]
    (M2RNN targets only).
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
    ) -> None:
        super().__init__()
        if translator_type not in ("mlp", "rkan"):
            raise ValueError(f"Unknown translator_type: {translator_type}")
        if output_shape not in ("vector", "matrix"):
            raise ValueError(f"Unknown output_shape: {output_shape}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.translator_type = translator_type
        self.output_shape = output_shape
        self.use_gdn2_init = use_gdn2_init
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        matrix_state_dim = num_heads * key_dim * value_dim

        # Input normalization (on scaffold output)
        self.input_norm = RMSNorm(input_dim)

        if translator_type == "mlp":
            out_dim = matrix_state_dim if output_shape == "matrix" else hidden_dim
            # 2-layer MLP with SiLU (matching MLP-RNN, minGRU, minLSTM)
            self.net = nn.Sequential(
                nn.Linear(input_dim, out_dim * 4, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 4, out_dim, bias=False),
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
            )
        # Output normalization (on boundary state before feeding to chunk RNN)
        if output_shape == "matrix":
            self.output_norm = RMSNorm(matrix_state_dim)
        else:
            self.output_norm = RMSNorm(hidden_dim)

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
            - matrix output_shape: [B, M, N, K, V] (matrix state)
        """
        x = self.input_norm(x)
        x = self.net(x)
        x = self.output_norm(x)
        if self.output_shape == "matrix":
            # Reshape to matrix state [B, M, N, K, V]
            batch, num_boundaries, _ = x.shape
            x = x.view(batch, num_boundaries, self.num_heads, self.key_dim, self.value_dim)
        return x


_LEGACY_TARGET_TO_TRANSLATOR = {
    "mlp": "mlp",
    "min_gru": "mlp",
    "min_lstm": "mlp",
    "m2rnn": "mlp",
    "rkan": "rkan",
}


class Translator(TranslatorLayer):
    """Backward-compatible translator bound to a target model type.

    ``target_type`` (mlp | rkan | min_gru | min_lstm | m2rnn) selects the
    legacy architecture: rkan target -> rkan translator, m2rnn target ->
    matrix output shape, everything else -> mlp translator. Prefer
    :class:`TranslatorLayer` with explicit ``translator_type`` /
    ``output_shape`` for new code.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        target_type: str = "mlp",
        translator_type: str | None = None,
        output_shape: str | None = None,
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
    ) -> None:
        if target_type not in ("mlp", "rkan", "min_gru", "min_lstm", "m2rnn"):
            raise ValueError(f"Unknown target_type: {target_type}")
        resolved_translator = translator_type or _LEGACY_TARGET_TO_TRANSLATOR[target_type]
        resolved_shape = output_shape or ("matrix" if target_type == "m2rnn" else "vector")
        super().__init__(
            input_dim,
            hidden_dim,
            translator_type=resolved_translator,
            output_shape=resolved_shape,
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
        )
        self.target_type = target_type


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
                )
            )
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
