"""
Gaussian Mixture Model in Hilbert Spaces for MMD-based fitting.

This module provides a trainable Gaussian mixture model that can be
optimized by minimizing MMD² to an empirical distribution.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal

from .kernel.base import Kernel
from .kernel.gaussian import GaussianKernel
from .spaces.base import HilbertBasis


class GaussianMixtureModel(nn.Module):
    """
    Trainable Gaussian Mixture Model in a Hilbert space.

    The mixture Q = Σ_k π_k N(m_k, K_k) is parameterized by:
      - π: Mixture weights (via softmax of unconstrained logits)
      - m: Component means (coefficients in the basis)
      - K: Component covariances (diagonal, parameterized by log-variances)

    The model can be trained by minimizing MMD²(P, Q) where P is an
    empirical distribution of observed data.
    """

    def __init__(
        self,
        num_components: int,
        coeff_dim: int,
        basis: Optional[HilbertBasis] = None,
        covariance_type: Literal["diagonal", "spherical", "full"] = "diagonal",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            num_components: Number of mixture components K.
            coeff_dim: Dimension M of the coefficient space.
            basis: Optional HilbertBasis for projecting/reconstructing functions.
            covariance_type: Type of covariance parameterization:
                - "diagonal": Per-component, per-dimension variances (K × M params)
                - "spherical": Per-component scalar variance (K params)
                - "full": Full covariance matrices (K × M × M params) [not implemented]
            device: Torch device.
            dtype: Torch dtype.
        """
        super().__init__()
        self.num_components = num_components
        self.coeff_dim = coeff_dim
        self.basis = basis
        self.covariance_type = covariance_type
        self.device = device
        self.dtype = dtype

        # Mixture weights: use logits and softmax for simplex constraint
        # π_k = softmax(logits)_k
        self._logits = nn.Parameter(
            torch.zeros(num_components, device=device, dtype=dtype)
        )

        # Component means: coefficients in R^M
        self._mean_coeffs = nn.Parameter(
            torch.randn(num_components, coeff_dim, device=device, dtype=dtype) * 0.1
        )

        # Component covariances: parameterized by log-variances for positivity
        if covariance_type == "diagonal":
            # (K, M) log-variances
            self._log_var = nn.Parameter(
                torch.zeros(num_components, coeff_dim, device=device, dtype=dtype)
            )
        elif covariance_type == "spherical":
            # (K,) log-variances (one per component)
            self._log_var = nn.Parameter(
                torch.zeros(num_components, device=device, dtype=dtype)
            )
        elif covariance_type == "full":
            # Full covariance via Cholesky factor L: K = L @ L^T
            self._chol_factor = nn.Parameter(
                torch.eye(coeff_dim, device=device, dtype=dtype)
                .unsqueeze(0)
                .repeat(num_components, 1, 1)
            )
        else:
            raise ValueError(f"Unknown covariance_type: {covariance_type}")

    @property
    def pi(self) -> torch.Tensor:
        """Mixture weights π, shape (K,), sums to 1."""
        return torch.softmax(self._logits, dim=0)

    @property
    def mean(self) -> torch.Tensor:
        """Component means in coefficient space, shape (K, M)."""
        return self._mean_coeffs

    @property
    def covariance(self) -> torch.Tensor:
        """
        Component covariance matrices, shape (K, M, M).

        Constructed from the parameterization to ensure positive definiteness.
        """
        if self.covariance_type == "diagonal":
            # Variance = exp(log_var), then embed as diagonal
            var = torch.exp(self._log_var)  # (K, M)
            return torch.diag_embed(var)  # (K, M, M)

        elif self.covariance_type == "spherical":
            # Single variance per component, scaled identity
            var = torch.exp(self._log_var)  # (K,)
            I_M = torch.eye(self.coeff_dim, device=self.device, dtype=self.dtype)
            return var[:, None, None] * I_M.unsqueeze(0)  # (K, M, M)

        elif self.covariance_type == "full":
            # K = L @ L^T
            L = self._chol_factor  # (K, M, M)
            return L @ L.transpose(-1, -2)

    @property
    def variance(self) -> torch.Tensor:
        """
        Component variances (diagonal elements), shape (K, M).
        """
        if self.covariance_type == "diagonal":
            return torch.exp(self._log_var)
        elif self.covariance_type == "spherical":
            var = torch.exp(self._log_var)  # (K,)
            return var.unsqueeze(1).expand(-1, self.coeff_dim)
        elif self.covariance_type == "full":
            cov = self.covariance
            return torch.diagonal(cov, dim1=-2, dim2=-1)

    def initialize_from_data(
        self,
        X: torch.Tensor,
        method: Literal["random", "kmeans", "kmeans++"] = "random",
    ):
        """
        Initialize mixture parameters from data.

        Args:
            X: Data coefficients, shape (n, M)
            method: Initialization method:
                - "random": Random subset of data as means
                - "kmeans": K-means clustering
                - "kmeans++": K-means++ initialization
        """
        n = X.shape[0]

        if method == "random":
            # Randomly select K data points as initial means
            indices = torch.randperm(n, device=self.device)[: self.num_components]
            with torch.no_grad():
                self._mean_coeffs.copy_(X[indices])

        elif method in ["kmeans", "kmeans++"]:
            # Simple k-means++ initialization
            means = []
            # First center: random
            idx = torch.randint(n, (1,), device=self.device).item()
            means.append(X[idx])

            for _ in range(1, self.num_components):
                # Compute distances to nearest center
                centers = torch.stack(means, dim=0)  # (k, M)
                dists = torch.cdist(X, centers)  # (n, k)
                min_dists = dists.min(dim=1).values ** 2  # (n,)

                # Sample proportional to squared distance
                probs = min_dists / min_dists.sum()
                idx = torch.multinomial(probs, 1).item()
                means.append(X[idx])

            with torch.no_grad():
                self._mean_coeffs.copy_(torch.stack(means, dim=0))

        # Initialize variance from data variance
        data_var = X.var(dim=0)  # (M,)
        with torch.no_grad():
            if self.covariance_type == "diagonal":
                self._log_var.copy_(
                    torch.log(data_var + 1e-6).unsqueeze(0).expand(self.num_components, -1)
                )
            elif self.covariance_type == "spherical":
                self._log_var.copy_(
                    torch.log(data_var.mean() + 1e-6).expand(self.num_components)
                )

    def compute_mmd2(
        self,
        X: torch.Tensor,
        kernel: Kernel,
        compute_const_term: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute MMD²(P, Q) where P is the empirical distribution of X
        and Q is this Gaussian mixture.

        Args:
            X: Data coefficients, shape (n, M)
            kernel: Kernel for MMD computation
            compute_const_term: Whether to include E_{x,x'~P}[κ(x,x')]

        Returns:
            mmd2: Scalar tensor
            stats: Dictionary with diagnostic information
        """
        pi = self.pi
        m = self.mean
        Kcov = self.covariance

        # Constant term (data-data)
        if compute_const_term:
            gram = kernel.compute_gram_matrix(X)
            const = gram.mean()
        else:
            const = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Cross term (data-mixture)
        J = kernel.compute_J(X, m, Kcov)  # (n, K)
        Jbar = J.mean(dim=0)  # (K,)

        # Mixture-mixture term
        I = kernel.compute_I(m, Kcov)  # (K, K)

        cross = (pi * Jbar).sum()
        mixmix = pi @ I @ pi

        mmd2 = const - 2.0 * cross + mixmix

        stats = {
            "const": const,
            "cross": cross,
            "mixmix": mixmix,
            "Jbar": Jbar,
            "I": I,
            "pi": pi.detach(),
            "mean": m.detach(),
            "variance": self.variance.detach(),
        }
        return mmd2, stats

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the Gaussian mixture in coefficient space.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            samples: Coefficient samples, shape (num_samples, M)
            assignments: Component assignments, shape (num_samples,)
        """
        pi = self.pi
        m = self.mean
        cov = self.covariance

        # Sample component assignments
        assignments = torch.multinomial(
            pi.expand(num_samples, -1), num_samples=1
        ).squeeze(-1)  # (num_samples,)

        # Sample from each component
        samples = torch.empty(
            num_samples, self.coeff_dim, device=self.device, dtype=self.dtype
        )

        for k in range(self.num_components):
            mask = assignments == k
            n_k = mask.sum().item()
            if n_k > 0:
                # Sample from N(m_k, K_k)
                mean_k = m[k]
                cov_k = cov[k]

                # Use Cholesky for sampling
                L = torch.linalg.cholesky(cov_k + 1e-6 * torch.eye(self.coeff_dim, device=self.device, dtype=self.dtype))
                z = torch.randn(n_k, self.coeff_dim, device=self.device, dtype=self.dtype)
                samples[mask] = mean_k + z @ L.T

        return samples, assignments

    def sample_functions(
        self, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample functions by first sampling coefficients, then reconstructing.

        Requires that a basis was provided at initialization.

        Args:
            num_samples: Number of function samples.

        Returns:
            functions: Reconstructed functions in the basis representation
            coeffs: Coefficient samples, shape (num_samples, M)
            assignments: Component assignments
        """
        if self.basis is None:
            raise ValueError("No basis provided for function reconstruction")

        coeffs, assignments = self.sample(num_samples)
        functions = self.basis.reconstruct(coeffs)

        return functions, coeffs, assignments

    def forward(
        self,
        X: torch.Tensor,
        kernel: Kernel,
        compute_const_term: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass: compute MMD² loss.

        Args:
            X: Data coefficients, shape (n, M)
            kernel: Kernel for MMD
            compute_const_term: Include constant term

        Returns:
            mmd2: MMD² value (scalar)
        """
        mmd2, _ = self.compute_mmd2(X, kernel, compute_const_term)
        return mmd2

    def log_component_densities(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log p_k^{(M)}(x) for each component k and data point x.

        For the projected Gaussian N(m_k, K_k) with positive-definite K_k,
        the density w.r.t. Lebesgue measure on R^M is:
            p_k(x) = (2π)^{-M/2} det(K_k)^{-1/2}
                     exp{-1/2 (x - m_k)^T K_k^{-1} (x - m_k)}

        We compute this in log-domain for numerical stability.

        Args:
            X: Data coefficients, shape (n, M)

        Returns:
            log_densities: shape (n, K), where entry [i,k] = log p_k(X[i])
        """
        n = X.shape[0]
        K = self.num_components
        M = self.coeff_dim

        pi = self.pi  # (K,)
        m = self.mean  # (K, M)
        cov = self.covariance  # (K, M, M)

        log_densities = torch.empty(n, K, device=self.device, dtype=self.dtype)

        for k in range(K):
            # Mahalanobis distance: (x - m_k)^T K_k^{-1} (x - m_k)
            diff = X - m[k]  # (n, M)
            cov_k = cov[k]  # (M, M)

            # Add small regularization for numerical stability
            cov_k_reg = cov_k + 1e-6 * torch.eye(M, device=self.device, dtype=self.dtype)

            # Cholesky for stable inverse and log-determinant
            L = torch.linalg.cholesky(cov_k_reg)  # K_k = L L^T
            # Solve L z = diff^T, then mahal^2 = ||z||^2
            z = torch.linalg.solve_triangular(L, diff.T, upper=False)  # (M, n)
            mahal_sq = (z ** 2).sum(dim=0)  # (n,)

            # log det(K_k) = 2 * sum(log(diag(L)))
            log_det = 2.0 * torch.log(torch.diag(L)).sum()

            # log p_k(x) = -M/2 log(2π) - 1/2 log det(K_k) - 1/2 mahal^2
            log_densities[:, k] = (
                -0.5 * M * torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=self.dtype))
                - 0.5 * log_det
                - 0.5 * mahal_sq
            )

        return log_densities

    def responsibilities(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute responsibilities γ_k(x) = P(Z=k | X=x) for each component.

        The responsibility is the posterior probability that observation x
        came from component k, given by Bayes' rule:
            γ_k(x) = π_k p_k(x) / Σ_s π_s p_s(x)
                   = d(π_k ν_k)/dQ (x)

        This is the Radon-Nikodym derivative of the weighted component
        measure π_k ν_k with respect to the mixture Q.

        Uses log-sum-exp for numerical stability.

        Args:
            X: Data coefficients, shape (n, M)

        Returns:
            gamma: Responsibilities, shape (n, K), where γ[i,k] = P(Z=k | X=X_i)
                   Each row sums to 1.
        """
        pi = self.pi  # (K,)
        log_pi = torch.log(pi + 1e-10)  # (K,)

        log_densities = self.log_component_densities(X)  # (n, K)

        # log(π_k p_k(x)) = log π_k + log p_k(x)
        log_weighted = log_pi.unsqueeze(0) + log_densities  # (n, K)

        # γ_k(x) = exp(log π_k + log p_k(x) - log Σ_s π_s p_s(x))
        # Use log-sum-exp for the denominator
        log_sum = torch.logsumexp(log_weighted, dim=1, keepdim=True)  # (n, 1)
        log_gamma = log_weighted - log_sum  # (n, K)

        return torch.exp(log_gamma)

    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood of the data under the mixture model.

        For the projected observations x_1, ..., x_n in R^M, the log-likelihood is:
            ℓ^{(M)}(π, m, K) = Σ_i log(Σ_k π_k p_k^{(M)}(x_i))

        This converges to the infinite-dimensional log-likelihood as M → ∞
        under mild integrability conditions (martingale convergence argument).

        Uses log-sum-exp for numerical stability.

        Args:
            X: Data coefficients, shape (n, M)

        Returns:
            log_lik: Scalar tensor, the total log-likelihood Σ_i log q(x_i)
        """
        pi = self.pi  # (K,)
        log_pi = torch.log(pi + 1e-10)  # (K,)

        log_densities = self.log_component_densities(X)  # (n, K)

        # log(Σ_k π_k p_k(x)) = logsumexp_k(log π_k + log p_k(x))
        log_weighted = log_pi.unsqueeze(0) + log_densities  # (n, K)
        log_mixture_density = torch.logsumexp(log_weighted, dim=1)  # (n,)

        return log_mixture_density.sum()

    def per_sample_log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample log-likelihood log q(x_i) for each data point.

        Args:
            X: Data coefficients, shape (n, M)

        Returns:
            log_lik: shape (n,), log q(X[i]) for each sample
        """
        pi = self.pi  # (K,)
        log_pi = torch.log(pi + 1e-10)  # (K,)

        log_densities = self.log_component_densities(X)  # (n, K)
        log_weighted = log_pi.unsqueeze(0) + log_densities  # (n, K)

        return torch.logsumexp(log_weighted, dim=1)  # (n,)

    def extra_repr(self) -> str:
        return (
            f"num_components={self.num_components}, "
            f"coeff_dim={self.coeff_dim}, "
            f"covariance_type={self.covariance_type}"
        )


def fit_gaussian_mixture_mmd(
    X: torch.Tensor,
    num_components: int,
    kernel: Kernel,
    num_epochs: int = 100,
    lr: float = 0.01,
    covariance_type: Literal["diagonal", "spherical"] = "diagonal",
    init_method: Literal["random", "kmeans++"] = "kmeans++",
    basis: Optional[HilbertBasis] = None,
    verbose: bool = True,
    log_interval: int = 10,
) -> Tuple[GaussianMixtureModel, list]:
    """
    Fit a Gaussian mixture model to data by minimizing MMD².

    Args:
        X: Data coefficients, shape (n, M)
        num_components: Number of mixture components K
        kernel: Kernel for MMD computation
        num_epochs: Number of optimization epochs
        lr: Learning rate
        covariance_type: Covariance parameterization type
        init_method: Initialization method for means
        basis: Optional HilbertBasis for function reconstruction
        verbose: Whether to print progress
        log_interval: Epochs between progress prints

    Returns:
        model: Fitted GaussianMixtureModel
        history: List of MMD² values per epoch
    """
    n, M = X.shape
    device = X.device
    dtype = X.dtype

    # Create model
    model = GaussianMixtureModel(
        num_components=num_components,
        coeff_dim=M,
        basis=basis,
        covariance_type=covariance_type,
        device=device,
        dtype=dtype,
    )

    # Initialize from data
    model.initialize_from_data(X, method=init_method)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Precompute constant term (doesn't depend on model)
    gram = kernel.compute_gram_matrix(X)
    const_term = gram.mean()

    history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute MMD² (skip constant term in loss, add back for logging)
        mmd2, stats = model.compute_mmd2(X, kernel, compute_const_term=False)
        loss = mmd2  # Minimize cross + mixmix terms

        loss.backward()
        optimizer.step()

        # Log with full MMD²
        full_mmd2 = const_term + mmd2.detach()
        history.append(full_mmd2.item())

        if verbose and (epoch + 1) % log_interval == 0:
            print(
                f"Epoch {epoch+1:4d}/{num_epochs}: "
                f"MMD² = {full_mmd2.item():.6f}, "
                f"π = {model.pi.detach().cpu().numpy()}"
            )

    return model, history
