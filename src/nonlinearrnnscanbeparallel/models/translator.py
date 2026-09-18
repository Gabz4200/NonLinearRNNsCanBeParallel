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


class Translator(nn.Module):
    """Maps scaffold summary states to target RNN boundary states.

    Per layer: each target layer has its own translator.
    Architecture matches target's function class:
    - MLP-RNN -> 2-layer MLP translator
    - rKAN-RNN -> 2-layer rKAN translator
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        target_type: str = "mlp",
        dropout: float = 0.1,
        rkan_degree: int = 3,
        rkan_alpha: float = 1.0,
        rkan_beta: float = 1.0,
        rkan_iota: float = 1.0,
        rkan_mapping: str = "algebraic_infinite",
        rkan_type: str = "jacobi",
        rkan_num_basis: int = 4,
        use_gdn2_init: bool = True,
    ) -> None:
        super().__init__()
        if target_type not in ("mlp", "rkan"):
            raise ValueError(f"Unknown target_type: {target_type}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.target_type = target_type
        self.use_gdn2_init = use_gdn2_init

        # Input normalization (on scaffold output)
        self.input_norm = RMSNorm(input_dim)

        if target_type == "mlp":
            # 2-layer MLP with SiLU (matching MLP-RNN)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 4, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
            )
            if use_gdn2_init:
                self._apply_gdn2_init()
        else:
            # 2-layer rKAN (matching rKAN-RNN)
            self.net = _RKANTranslator(
                input_dim,
                hidden_dim,
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
            Boundary hidden states [B, M, D_target]
        """
        x = self.input_norm(x)
        x = self.net(x)
        x = self.output_norm(x)
        return x
