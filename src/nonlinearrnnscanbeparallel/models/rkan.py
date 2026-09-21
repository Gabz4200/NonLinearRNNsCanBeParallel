"""rKAN (Rational Kolmogorov-Arnold Networks) per arXiv:2406.14495v1.

Implements rational basis functions using:
  - Padé approximation (Section 3.1, Eq. 9): ratio of two weighted Jacobi-polynomial sums.
  - Rational Jacobi functions (Section 3.2, Eq. 12): J_k^(α,β)(φ(ξ_k; SoftPlus(ι))).

Both follow the KAN representation (Theorem 3.1 / Eq. 10):
    F(ξ) = Σ_k ψ_k( Σ_q φ_{q,k}(σ(ξ_q)) )
where the inner sum Σ_q runs over input coordinates q and the outer sum over K = num_basis
basis functions.

Per the paper (Section 2), α and β are trainable parameters constrained to > -1 via
ELU(·; κ=1), and ι is constrained to > 0 via SoftPlus.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import RMSNorm


def _inv_elu(y: float) -> float:
    """Inverse of ELU(·, κ=1): maps desired effective value to raw parameter.

    For y > 0: ELU(raw) = raw, so raw = y.
    For y ≤ 0: ELU(raw) = e^{raw} - 1, so raw = ln(y + 1).
    """
    if y > 0:
        return y
    return math.log1p(y)


def _inv_softplus(y: float) -> float:
    """Inverse of SoftPlus: raw such that softplus(raw) = y."""
    return math.log(math.expm1(y)) if y > 0 else 0.0


class JacobiPolynomial(nn.Module):
    """Jacobi polynomials J_n^(α,β)(ξ) for n = 0..degree, per arXiv:2406.14495v1 §2.

    Explicit formula:
      J_n^(α,β)(ξ) = [Γ(α+n+1) / (n! Γ(α+β+n+1))]
                     · Σ_{m=0}^{n} C(n,m) · [Γ(α+β+n+m+1) / Γ(α+m+1)] · ((ξ-1)/2)^m

    α, β are trainable raw parameters whose effective values are
        α_eff = ELU(α_raw)  (> -1)
        β_eff = ELU(β_raw)  (> -1)
    following the paper's use of ELU with κ=1 to enforce validity.
    """

    def __init__(
        self,
        degree: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.degree = degree
        # Effective params use ELU with kappa 1; range is (-1, infinity).
        # Initialize raw so effective value equals the config default.
        self.alpha_raw = nn.Parameter(torch.tensor(_inv_elu(float(alpha))))
        self.beta_raw = nn.Parameter(torch.tensor(_inv_elu(float(beta))))

    @property
    def alpha(self) -> torch.Tensor:
        """Effective α = ELU(α_raw), guaranteed > -1."""
        return F.elu(self.alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        """Effective β = ELU(β_raw), guaranteed > -1."""
        return F.elu(self.beta_raw)

    def coefficients(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Coefficients of the Jacobi polynomial expansion.

        Returns [degree+1, degree+1] where [n, m] is the coefficient of
        ((ξ-1)/2)^m in J_n^(α,β)(ξ).  Computed in log-space for stability.
        """
        d = self.degree
        n_vals = torch.arange(d + 1, device=alpha.device, dtype=torch.float32)
        m_vals = torch.arange(d + 1, device=alpha.device, dtype=torch.float32)

        # Broadcast to matrices with shape [d+1, d+1].
        n_mat = n_vals.unsqueeze(1)  # [d+1, 1]
        m_mat = m_vals.unsqueeze(0)  # [1, d+1]

        # Binomial coefficient C(n, m); zero for m > n.
        log_binom = (
            torch.lgamma(n_mat + 1) - torch.lgamma(m_mat + 1) - torch.lgamma(n_mat - m_mat + 1)
        )
        valid = m_mat <= n_mat
        log_binom = torch.where(valid, log_binom, torch.full_like(log_binom, float("-inf")))

        # Constant factor per row n.
        log_const = (
            torch.lgamma(alpha + n_mat + 1)
            - torch.lgamma(n_mat + 1)
            - torch.lgamma(alpha + beta + n_mat + 1)
        )

        # Gamma ratio of the expansion coefficients.
        log_gamma_ratio = torch.lgamma(alpha + beta + n_mat + m_mat + 1) - torch.lgamma(
            alpha + m_mat + 1
        )

        log_coeff = log_const + log_binom + log_gamma_ratio
        coeffs = torch.exp(log_coeff)
        return torch.where(valid, coeffs, torch.zeros_like(coeffs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate J_0..J_degree at points x (already mapped into [-1, 1]).

        Args:
            x: [..., 1] or [...] values in [-1, 1].

        Returns:
            [..., degree+1] with the Jacobi polynomial values for n = 0..degree.
        """
        trailing_one = x.shape[-1] == 1 if x.ndim > 0 else False
        x_flat = x.reshape(-1)  # [N]
        z = (x_flat - 1.0) / 2.0  # ((ξ-1)/2), per the paper

        # Powers of z up to degree.
        z_powers = torch.ones_like(z)
        powers = [z_powers]
        for _ in range(1, self.degree + 1):
            z_powers = z_powers * z
            powers.append(z_powers)
        z_powers = torch.stack(powers, dim=-1)  # [N, degree+1]

        coeffs = self.coefficients(self.alpha, self.beta)  # [degree+1, degree+1]
        out = z_powers @ coeffs.T  # [N, degree+1]

        if trailing_one:
            return out.view(*x.shape[:-1], self.degree + 1)
        return out.view(*x.shape, self.degree + 1)


class RationalMapping(nn.Module):
    """Rational mapping φ : Ω → [-1, 1] per arXiv:2406.14495v1 Definition 1.

    Supports finite, semi-infinite, and infinite domains.  For rational Jacobi
    functions the paper applies SoftPlus(ι) to keep ι > 0 (Section 3.2).
    """

    def __init__(
        self,
        mapping_type: str = "algebraic_infinite",
        iota: float = 1.0,
        d0: float = -1.0,
        d1: float = 1.0,
    ) -> None:
        super().__init__()
        if mapping_type not in (
            "linear",
            "logarithmic_semi",
            "algebraic_semi",
            "exponential_semi",
            "logarithmic_infinite",
            "algebraic_infinite",
        ):
            raise ValueError(f"Unknown mapping type: {mapping_type}")
        self.mapping_type = mapping_type
        self.iota_raw = nn.Parameter(torch.tensor(_inv_softplus(iota)))
        self.register_buffer("d0", torch.tensor(d0))
        self.register_buffer("d1", torch.tensor(d1))

    @property
    def iota(self) -> torch.Tensor:
        """Positive ι via SoftPlus, per the paper."""
        return F.softplus(self.iota_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map x ∈ Ω to [-1, 1]."""
        iota = self.iota
        d0 = self.d0
        d1 = self.d1
        mappings = {
            "linear": lambda: (2 * x - d0 - d1) / (d1 - d0),
            "logarithmic_semi": lambda: 2 * torch.tanh(x / iota) - 1,
            "algebraic_semi": lambda: (x - iota) / (x + iota),
            "exponential_semi": lambda: 1 - 2 * torch.exp(-x / iota),
            "logarithmic_infinite": lambda: torch.tanh(x / iota),
            "algebraic_infinite": lambda: x / torch.sqrt(x * x + iota * iota),
        }
        try:
            return mappings[self.mapping_type]()
        except KeyError:
            raise ValueError(f"Unknown mapping type: {self.mapping_type}") from None


class RKANLayer(nn.Module):
    """Coordinate-wise rational KAN layer per arXiv:2406.14495v1 §3.

    Implements the KAN representation (Eq. 10):
        F(ξ) = Σ_k ψ_k( Σ_q φ_{q,k}(σ(ξ_q)) )

    Two rational basis variants:
      - Jacobi-rKAN (Eq. 12): for output channel k,
            φ_{q,k}(ξ_q) = J_k^(α,β)( φ(σ(ξ_q); SoftPlus(ι)) )
        i.e. the k-th Jacobi polynomial degree (requires degree >= num_basis-1).
      - Padé-rKAN   (Eq. 9):
            φ_{q,k}(ξ_q) = [Σ_n θ^e_{q,k,n} R_n(ξ_q)] / [Σ_n θ^d_{q,k,n} R_n(ξ_q)]
        with learnable numerator/denominator coefficients.

    The inner sum Σ_q runs over the original `in_dim` input coordinates; the
    outer sum Σ_k over K = num_basis basis channels, followed by a linear ψ
    projection from `num_basis` to `out_dim`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        degree: int = 3,
        alpha: float = 1.0,
        beta: float = 1.0,
        iota: float = 1.0,
        mapping_type: str = "algebraic_infinite",
        rkan_type: str = "jacobi",
        num_basis: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rkan_type not in ("jacobi", "pade"):
            raise ValueError(f"Unknown rkan_type: {rkan_type}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.rkan_type = rkan_type
        self.num_basis = num_basis

        # Both modes share the Jacobi basis; the Pade variant weights its sums.
        self._jacobi_degree = max(degree, num_basis)
        self.jacobi = JacobiPolynomial(self._jacobi_degree, alpha, beta)

        if rkan_type == "pade":
            # Pade coefficients indexed by input, basis, and Jacobi count.
            # Last axis matches the Jacobi basis count.
            basis_count = self._jacobi_degree + 1
            self.theta_e = nn.Parameter(torch.randn(in_dim, num_basis, basis_count) * 0.1)
            self.theta_d = nn.Parameter(torch.randn(in_dim, num_basis, basis_count) * 0.05)
            with torch.no_grad():
                self.theta_d[..., 0].fill_(1.0)  # stabilise denominator

        self.mapping = RationalMapping(
            mapping_type=mapping_type,
            iota=iota,
        )

        # Outer linear projection to output dim.
        # Initialize with smaller scale to account for Jacobi polynomial magnitudes
        # Jacobi polynomials J_k can have magnitude up to ~k at boundaries.
        # Use scale ~ 1/sqrt(num_basis * degree) for stability.
        init_scale = 0.1 / max(1.0, (self.num_basis * self.degree) ** 0.5)
        self.output_weight = nn.Parameter(torch.randn(out_dim, num_basis) * init_scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [..., in_dim]

        Returns:
            [..., out_dim]
        """
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_dim)  # [N, in_dim]

        # Map inputs into [-1, 1] using rational mapping.
        # The mapping handles unbounded inputs naturally (e.g., algebraic_infinite maps R -> [-1,1]).
        x_mapped = self.mapping(x)  # [N, in_dim]

        # Evaluate Jacobi basis functions at each coordinate.
        jacobi_vals = self.jacobi(x_mapped.unsqueeze(-1))  # [N, in_dim, degree+1]

        if self.rkan_type == "jacobi":
            # Skip J_0; it is constant so it adds no signal.
            phi = jacobi_vals[..., 1 : self.num_basis + 1]  # [N, in_dim, num_basis]
            # Clamp to prevent numerical instability from high-degree polynomials at boundaries
            phi = torch.clamp(phi, min=-10.0, max=10.0)
        else:
            phi = self._pade_basis(jacobi_vals)  # [N, in_dim, num_basis]

        # Inner sum over input coordinates.
        # Normalize by sqrt(in_dim) to prevent magnitude explosion with many input coordinates.
        phi_agg = phi.sum(dim=1) / (self.in_dim**0.5)

        # Outer linear projection.
        out = phi_agg @ self.output_weight.T  # [N, out_dim]
        if self.bias is not None:
            out = out + self.bias

        return out.view(*original_shape, self.out_dim)

    def _pade_basis(self, jacobi_vals: torch.Tensor) -> torch.Tensor:
        """Padé-rKAN basis (Eq. 9): ratio of weighted Jacobi sums.

        φ_{q,k} = Σ_n θ^e_{q,k,n} R_n(ξ_q) / Σ_n θ^d_{q,k,n} R_n(ξ_q)
        """
        num = torch.einsum("qkd,nqd->nqk", self.theta_e, jacobi_vals)  # [N, in_dim, num_basis]
        den = torch.einsum("qkd,nqd->nqk", self.theta_d, jacobi_vals)
        return num / (den.abs() + 1e-8)


class RKANHead(nn.Module):
    """rKAN-based recurrence head: f(h_prev, x_t) -> h_t.

    Uses RKANLayer over the concatenated [h_prev; x_t] input, replacing the
    MLPHead recurrence function when rKAN-RNN replaces the MLP recurrence
    heads (per arXiv:2603.03612: "MLP-RNN with rKAN replacing MLP/FFN").
    """

    def __init__(
        self,
        head_dim: int,
        rkan_degree: int = 3,
        rkan_alpha: float = 1.0,
        rkan_beta: float = 1.0,
        rkan_iota: float = 1.0,
        rkan_mapping: str = "algebraic_infinite",
        rkan_type: str = "jacobi",
        rkan_num_basis: int = 4,
    ) -> None:
        super().__init__()
        self.rkan = RKANLayer(
            in_dim=head_dim * 2,
            out_dim=head_dim,
            degree=rkan_degree,
            alpha=rkan_alpha,
            beta=rkan_beta,
            iota=rkan_iota,
            mapping_type=rkan_mapping,
            rkan_type=rkan_type,
            num_basis=rkan_num_basis,
            bias=False,
        )

    def forward(self, h_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_prev, x_t], dim=-1)
        return self.rkan(combined)


class RationalFeedForward(nn.Module):
    """Feedforward sublayer using rKAN instead of a standard MLP (Definition 8).

    Replaces the two-layer ReLU network with rKAN layers.
    """

    def __init__(
        self,
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
        self.norm = RMSNorm(hidden_dim)
        self.rkan1 = RKANLayer(
            hidden_dim,
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
        residual = x
        x = self.norm(x)
        x = self.rkan1(x)
        x = self.dropout(x)
        x = self.rkan2(x)
        return residual + x
