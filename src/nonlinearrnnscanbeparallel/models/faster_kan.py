"""FasterKAN: Fast KAN (arXiv:2405.06721) with RSWAF bases.

Fast KAN replaces B-spline bases with Gaussian radial basis functions
    b_i(u) = exp(-((u - u_i) / h)^2)
evaluated on a fixed uniform grid after an optional LayerNorm, then mixes
with a single bias-free linear layer (optionally plus a SiLU base branch).

FasterKAN (AthanasiosDelis/faster-kan) swaps the Gaussian for the
Reflectional SWitch Activation Function (RSWAF), a sech^2 basis that
approximates cubic B-spline bases on a uniform grid at ~1.5x forward speed:

    b_i(u) = 1 - tanh^2((u - u_i) / h)

with `inv_denominator = 1/h` (default 0.5). FasterKAN also drops the SiLU
base branch entirely and keeps the LayerNorm unconditional so inputs land
inside the grid range without rescaling the grid.

Note: the reference implementation computes `tanh(u - u_i)` (omitting the
`/h`) and only folds `inv_denominator` into its custom backward; its
`spline_weight_init_scale` argument is likewise dead (its
``SplineLinear`` does plain Xavier). This implementation follows the
README formula (the spec) and applies the init scale as an actual
truncated-normal std.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import RMSNorm


class ReflectionalSwitchBasis(nn.Module):
    """RSWAF basis b_i(u) = 1 - tanh^2((u - u_i) * inv_denominator) on a uniform grid.

    ``grid`` is [num_grids]; ``inv_denominator`` (1/h) scales the width.
    ``train_grid`` / ``train_inv_denominator`` make those basis
    hyperparameters learnable (experimental, default off), mirroring the
    reference.
    """

    def __init__(
        self,
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        inv_denominator: float = 0.5,
        train_grid: bool = False,
        train_inv_denominator: bool = False,
    ) -> None:
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        if train_grid:
            self.grid = nn.Parameter(grid)
        else:
            self.register_buffer("grid", grid)
        self.inv_denominator = nn.Parameter(
            torch.tensor(float(inv_denominator)),
            requires_grad=train_inv_denominator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate every basis at every input coordinate.

        Args:
            x: [..., in_dim]

        Returns:
            [..., in_dim, num_grids]
        """
        diff = (x.unsqueeze(-1) - self.grid) * self.inv_denominator
        tanh_diff = torch.tanh(diff)
        return 1.0 - tanh_diff * tanh_diff


class FasterKANLayer(nn.Module):
    """One FasterKAN layer: LayerNorm -> RSWAF basis -> linear mix.

    Maps [..., input_dim] -> [..., output_dim] with the expansion
        out = W * flatten(basis(LN(x))),   W: (input_dim * num_grids, output_dim).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        inv_denominator: float = 0.5,
        train_grid: bool = False,
        train_inv_denominator: bool = False,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.layernorm = nn.LayerNorm(input_dim)
        self.basis = ReflectionalSwitchBasis(
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            inv_denominator=inv_denominator,
            train_grid=train_grid,
            train_inv_denominator=train_inv_denominator,
        )
        self.spline_linear = nn.Linear(input_dim * num_grids, output_dim, bias=False)
        nn.init.trunc_normal_(self.spline_linear.weight, mean=0.0, std=spline_weight_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x = self.layernorm(x)
        basis = self.basis(x)  # [..., in_dim, num_grids]
        basis = basis.reshape(*original_shape, self.input_dim * self.num_grids)
        out = self.spline_linear(basis)
        return out.view(*original_shape, self.output_dim)


class FasterKANHead(nn.Module):
    """FasterKAN recurrence head: f(h_prev, x_t) -> h_t.

    Uses a single FasterKANLayer over the concatenated [h_prev; x_t] input,
    replacing ``MLPHead`` in MLP-RNN-style recurrence heads (per
    arXiv:2603.03612: "MLP-RNN with rKAN replacing MLP/FF", here with
    FasterKAN instead of rKAN).
    """

    def __init__(
        self,
        head_dim: int,
        fkan_grid_min: float = -1.2,
        fkan_grid_max: float = 0.2,
        fkan_num_grids: int = 8,
        fkan_inv_denominator: float = 0.5,
        fkan_train_grid: bool = False,
        fkan_train_inv_denominator: bool = False,
        fkan_spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.faster_kan = FasterKANLayer(
            input_dim=head_dim * 2,
            output_dim=head_dim,
            grid_min=fkan_grid_min,
            grid_max=fkan_grid_max,
            num_grids=fkan_num_grids,
            inv_denominator=fkan_inv_denominator,
            train_grid=fkan_train_grid,
            train_inv_denominator=fkan_train_inv_denominator,
            spline_weight_init_scale=fkan_spline_weight_init_scale,
        )

    def forward(self, h_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_prev, x_t], dim=-1)
        return self.faster_kan(combined)


class FasterKanFeedForward(nn.Module):
    """Feedforward sublayer using FasterKAN instead of an MLP (Definition 8).

    Replaces the two-layer SiLU MLP with two FasterKAN layers, mirroring
    ``RationalFeedForward``'s shape/structure (hidden -> 4*hidden -> hidden).
    """

    def __init__(
        self,
        hidden_dim: int,
        fkan_grid_min: float = -1.2,
        fkan_grid_max: float = 0.2,
        fkan_num_grids: int = 8,
        fkan_inv_denominator: float = 0.5,
        fkan_train_grid: bool = False,
        fkan_train_inv_denominator: bool = False,
        fkan_spline_weight_init_scale: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer_kwargs = {
            "grid_min": fkan_grid_min,
            "grid_max": fkan_grid_max,
            "num_grids": fkan_num_grids,
            "inv_denominator": fkan_inv_denominator,
            "train_grid": fkan_train_grid,
            "train_inv_denominator": fkan_train_inv_denominator,
            "spline_weight_init_scale": fkan_spline_weight_init_scale,
        }
        self.norm = RMSNorm(hidden_dim)
        self.fkan1 = FasterKANLayer(hidden_dim, hidden_dim * 4, **layer_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.fkan2 = FasterKANLayer(hidden_dim * 4, hidden_dim, **layer_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fkan1(x)
        x = self.dropout(x)
        x = self.fkan2(x)
        return residual + x
