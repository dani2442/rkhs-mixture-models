"""
Temporal mixture weights for Gaussian mixtures on L²([0,1]^2) data.

This module supports a shared Gaussian component model with time-varying
mixture weights pi_k(t), where each time slice is treated as an empirical
measure on L²([0,1]).
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .kernel.base import Kernel
from .mixture import GaussianMixtureModel
from .spaces.L2 import L2Basis


class BasisLogitsTimeWeights(nn.Module):
    """
    Time-varying mixture weights via basis expansion of component logits.

    logits_k(t_l) = sum_r a_{k,r} * phi_r(t_l) + b_k
    pi(t_l) = softmax(logits(t_l)).
    """

    def __init__(
        self,
        basis_matrix: torch.Tensor,
        num_components: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        if basis_matrix.ndim != 2:
            raise ValueError("basis_matrix must have shape (L_t, R_pi)")

        L_t, R_pi = basis_matrix.shape
        self.num_components = num_components
        self.num_times = L_t
        self.num_basis = R_pi

        self.register_buffer(
            "basis_matrix",
            basis_matrix.to(device=device, dtype=dtype),
            persistent=False,
        )
        self.coeffs = nn.Parameter(
            torch.zeros(num_components, R_pi, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.zeros(num_components, device=device, dtype=dtype))

    def logits(self) -> torch.Tensor:
        return self.basis_matrix @ self.coeffs.T + self.bias.unsqueeze(0)

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits(), dim=-1)


class _NeuralODELogitDynamics(nn.Module):
    """RHS of dz/dt = f_theta(t, z)."""

    def __init__(self, num_components: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_components + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_components),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z_batch = z.unsqueeze(0)
            squeezed = True
        else:
            z_batch = z
            squeezed = False

        t_col = torch.full(
            (z_batch.shape[0], 1),
            fill_value=float(t),
            device=z_batch.device,
            dtype=z_batch.dtype,
        )
        out = self.net(torch.cat([z_batch, t_col], dim=-1))
        if squeezed:
            return out.squeeze(0)
        return out


class NeuralODETimeWeights(nn.Module):
    """
    Time-varying mixture weights via Neural ODE over component logits.

    z'(t) = f_theta(t, z(t)),   z(0) = z0
    pi(t_l) = softmax(z(t_l)).
    """

    def __init__(
        self,
        t_grid: torch.Tensor,
        num_components: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        if t_grid.ndim != 1:
            raise ValueError("t_grid must have shape (L_t,)")

        self.num_components = num_components
        self.num_times = t_grid.shape[0]
        self.temp = 1.0

        self.register_buffer(
            "t_grid", t_grid.to(device=device, dtype=dtype), persistent=False
        )
        self.z0 = nn.Parameter(torch.zeros(num_components, device=device, dtype=dtype))
        self.dynamics = _NeuralODELogitDynamics(num_components, hidden_dim)
        self.dynamics.to(device=device, dtype=dtype)

    def logits(self) -> torch.Tensor:
        return odeint(self.dynamics, self.z0, self.t_grid, method="rk4")

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits()/self.temp, dim=-1,)


class TemporalGaussianMixtureModel(nn.Module):
    """
    Gaussian mixture with shared component parameters and time-varying pi(t).

    Data is represented as X_time[l] with shape (n, M), i.e., one empirical
    distribution per time index l.
    """

    def __init__(
        self,
        num_components: int,
        coeff_dim: int,
        time_weight_model: nn.Module,
        covariance_type: Literal["diagonal", "spherical", "full"] = "diagonal",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.num_components = num_components
        self.coeff_dim = coeff_dim
        self.device = device
        self.dtype = dtype

        self.components = GaussianMixtureModel(
            num_components=num_components,
            coeff_dim=coeff_dim,
            covariance_type=covariance_type,
            device=device,
            dtype=dtype,
        )
        self.components._logits.requires_grad_(False)
        self.time_weight_model = time_weight_model

    @property
    def mean(self) -> torch.Tensor:
        return self.components.mean

    @property
    def covariance(self) -> torch.Tensor:
        return self.components.covariance

    @property
    def variance(self) -> torch.Tensor:
        return self.components.variance

    def initialize_from_data(
        self,
        X_time: torch.Tensor,
        method: Literal["random", "kmeans", "kmeans++"] = "kmeans++",
    ) -> None:
        """Initialize shared Gaussian components using all time slices."""
        if X_time.ndim != 3:
            raise ValueError("X_time must have shape (L_t, n, M)")
        L_t, n, M = X_time.shape
        if M != self.coeff_dim:
            raise ValueError(f"X_time coeff dim {M} != model coeff dim {self.coeff_dim}")

        stacked = X_time.permute(1, 0, 2).reshape(n * L_t, M)
        self.components.initialize_from_data(stacked, method=method)

    def compute_mmd2(
        self,
        X_time: torch.Tensor,
        kernel: Kernel,
        compute_const_term: bool = True,
        const_terms: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute mean_t MMD²(P_t, Q_t), with Q_t using time-varying pi(t).
        """
        if X_time.ndim != 3:
            raise ValueError("X_time must have shape (L_t, n, M)")

        L_t, _, M = X_time.shape
        if M != self.coeff_dim:
            raise ValueError(f"X_time coeff dim {M} != model coeff dim {self.coeff_dim}")

        pi_t = self.time_weight_model()  # (L_t, K)
        if pi_t.shape != (L_t, self.num_components):
            raise ValueError(
                f"time weight shape {tuple(pi_t.shape)} != ({L_t}, {self.num_components})"
            )

        m = self.mean
        Kcov = self.covariance
        I = kernel.compute_I(m, Kcov)

        if compute_const_term:
            if const_terms is None:
                const_terms = torch.stack(
                    [kernel.compute_gram_matrix(X_time[l]).mean() for l in range(L_t)],
                    dim=0,
                )
            else:
                const_terms = const_terms.to(device=self.device, dtype=self.dtype)
        else:
            const_terms = torch.zeros(L_t, device=self.device, dtype=self.dtype)

        cross_terms = torch.empty(L_t, device=self.device, dtype=self.dtype)
        mixmix_terms = torch.empty(L_t, device=self.device, dtype=self.dtype)
        Jbar_time = torch.empty(
            L_t, self.num_components, device=self.device, dtype=self.dtype
        )

        for l in range(L_t):
            Jbar_l = kernel.compute_Jbar(X_time[l], m, Kcov)
            Jbar_time[l] = Jbar_l
            cross_terms[l] = (pi_t[l] * Jbar_l).sum()
            mixmix_terms[l] = pi_t[l] @ I @ pi_t[l]

        mmd_per_t = const_terms - 2.0 * cross_terms + mixmix_terms
        mmd2 = mmd_per_t.mean()

        stats = {
            "mmd_per_t": mmd_per_t,
            "const_terms": const_terms,
            "cross_terms": cross_terms,
            "mixmix_terms": mixmix_terms,
            "Jbar_time": Jbar_time,
            "I": I,
            "pi_t": pi_t.detach(),
            "mean": m.detach(),
            "variance": self.variance.detach(),
        }
        return mmd2, stats


def project_l2_2d_to_space_slices(X: torch.Tensor, space_basis: L2Basis) -> torch.Tensor:
    """
    Project X(s,t) onto a 1D L²([0,1]) basis along s for each fixed t.

    Args:
        X: Tensor with shape (n, L_s, L_t, d)
        space_basis: Basis over s with grid size L_s and dimension d

    Returns:
        X_time: Tensor of shape (L_t, n, M_s), where M_s = d * R_s
    """
    if X.ndim != 4:
        raise ValueError("X must have shape (n, L_s, L_t, d)")

    n, L_s, L_t, d = X.shape
    if L_s != space_basis.grid_size:
        raise ValueError(f"L_s={L_s} does not match space basis grid_size={space_basis.grid_size}")
    if d != space_basis.d:
        raise ValueError(f"d={d} does not match space basis d={space_basis.d}")

    weighted_phi = space_basis.Phi * space_basis.w[:, None]  # (L_s, R_s)
    coeffs = torch.einsum("nstd,sr->ntdr", X, weighted_phi)
    coeffs = coeffs.reshape(n, L_t, -1)
    return coeffs.permute(1, 0, 2).contiguous()


def precompute_temporal_const_terms(X_time: torch.Tensor, kernel: Kernel) -> torch.Tensor:
    """Precompute const_l = E_{x,x'~P_l}[k(x,x')] for each time slice l."""
    if X_time.ndim != 3:
        raise ValueError("X_time must have shape (L_t, n, M)")

    L_t = X_time.shape[0]
    return torch.stack(
        [kernel.compute_gram_matrix(X_time[l]).mean() for l in range(L_t)], dim=0
    )


def fit_temporal_gaussian_mixture_mmd(
    model: TemporalGaussianMixtureModel,
    X_time: torch.Tensor,
    kernel: Kernel,
    num_epochs: int = 200,
    lr: float = 0.05,
    init_method: Literal["random", "kmeans", "kmeans++"] = "kmeans++",
    verbose: bool = True,
    log_interval: int = 20,
) -> list[float]:
    """
    Fit temporal mixture by minimizing average_t MMD²(P_t, Q_t).

    The constant data-data term is precomputed and excluded from gradients.
    """
    model.initialize_from_data(X_time, method=init_method)
    const_terms = precompute_temporal_const_terms(X_time, kernel)
    const_mean = const_terms.mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        mmd2_no_const, _ = model.compute_mmd2(
            X_time, kernel, compute_const_term=False, const_terms=None
        )
        mmd2_no_const.backward()
        optimizer.step()

        full_mmd2 = const_mean + mmd2_no_const.detach()
        history.append(full_mmd2.item())

        if verbose and (epoch + 1) % log_interval == 0:
            print(
                f"Epoch {epoch + 1:4d}/{num_epochs}: "
                f"MMD²(avg_t) = {full_mmd2.item():.6f}"
            )

    return history
