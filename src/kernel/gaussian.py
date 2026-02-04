"""
Gaussian (RBF) kernel for MMD computation on Hilbert spaces.

The Gaussian radial kernel is defined as:
    κ(x, y) = exp(-||x - y||²_X / (2σ²))

This kernel is universal and characteristic, meaning MMD = 0 iff P = Q.
"""
import torch
from typing import Optional

from .base import Kernel, _log_det_from_cholesky


class GaussianKernel(Kernel):
    """
    Gaussian (RBF) kernel: κ(x, y) = exp(-||x - y||² / (2σ²)).

    Provides closed-form MMD expectations for Gaussian components:
      - J_{i,k} = E_{y~N(m,K)}[κ(X_i, y)]
      - I_{k,s} = E_{y~N(m_k,K_k), y'~N(m_s,K_s)}[κ(y, y')]

    The closed forms involve Fredholm determinants (finite-dim matrix dets)
    and quadratic forms with matrix inversions.
    """

    def __init__(self, sigma: float, eps_jitter: float = 1e-8):
        """
        Args:
            sigma: Bandwidth parameter σ > 0.
            eps_jitter: Small constant for numerical stability.
        """
        super().__init__(eps_jitter=eps_jitter)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self.alpha = -1.0 / (2.0 * sigma * sigma)  # α = -1/(2σ²) < 0

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate κ(x, y) = exp(-||x - y||² / (2σ²)).

        Args:
            x: First input, shape (..., M)
            y: Second input, shape (..., M)

        Returns:
            Kernel values, shape (...)
        """
        diff = x - y
        sq_dist = (diff * diff).sum(dim=-1)
        return torch.exp(-sq_dist / (2.0 * self.sigma * self.sigma))

    def compute_gram_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Efficiently compute Gram matrix using pairwise distances.

        Args:
            X: Data points, shape (n, M)

        Returns:
            Gram matrix, shape (n, n)
        """
        D2 = torch.cdist(X, X, p=2.0) ** 2  # (n, n)
        return torch.exp(-D2 / (2.0 * self.sigma * self.sigma))

    def compute_J(
        self,
        X: torch.Tensor,
        m: torch.Tensor,
        Kcov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute J_{i,k} = E_{y ~ N(m_k, K_k)}[exp(-||X_i - y||² / (2σ²))].

        Closed form (Proposition 2):
            J_{i,k} = det(I - 2αK_k)^{-1/2}
                    × exp(α(X_i - m_k)^T (I - 2αK_k)^{-1} (X_i - m_k))

        where α = -1/(2σ²) < 0, so I - 2αK_k = I + K_k/σ² is SPD.

        Args:
            X: Data points, shape (n, M)
            m: Component means, shape (K, M)
            Kcov: Component covariances, shape (K, M, M)

        Returns:
            J: shape (n, K)
        """
        device = X.device
        dtype = X.dtype
        n, M = X.shape
        K = m.shape[0]

        J = torch.empty((n, K), device=device, dtype=dtype)
        I_M = torch.eye(M, device=device, dtype=dtype)

        for k in range(K):
            # A_k = I - 2αK_k = I + (1/σ²)K_k
            A = I_M - 2.0 * self.alpha * Kcov[k]
            A = A + self.eps_jitter * I_M  # Numerical stability

            L = torch.linalg.cholesky(A)  # (M, M)
            logdetA = _log_det_from_cholesky(L)

            diff = X - m[k].unsqueeze(0)  # (n, M)

            # Solve A^{-1} @ diff^T via Cholesky
            sol = torch.cholesky_solve(diff.T, L)  # (M, n)
            quad = (diff.T * sol).sum(dim=0)  # (n,)

            logJ = -0.5 * logdetA + self.alpha * quad
            J[:, k] = torch.exp(logJ)

        return J

    def compute_I(
        self,
        m: torch.Tensor,
        Kcov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute I_{k,s} = E_{y~N(m_k,K_k), y'~N(m_s,K_s)}[exp(-||y-y'||²/(2σ²))].

        Closed form (Proposition 2):
            I_{k,s} = det(I - 2α(K_k + K_s))^{-1/2}
                    × exp(α(m_k - m_s)^T (I - 2α(K_k + K_s))^{-1} (m_k - m_s))

        Args:
            m: Component means, shape (K, M)
            Kcov: Component covariances, shape (K, M, M)

        Returns:
            I: shape (K, K)
        """
        device = m.device
        dtype = m.dtype
        K = m.shape[0]
        M = m.shape[1]

        Imat = torch.empty((K, K), device=device, dtype=dtype)
        I_M = torch.eye(M, device=device, dtype=dtype)

        for k in range(K):
            for s in range(K):
                A = I_M - 2.0 * self.alpha * (Kcov[k] + Kcov[s])
                A = A + self.eps_jitter * I_M

                L = torch.linalg.cholesky(A)
                logdetA = _log_det_from_cholesky(L)

                dms = (m[k] - m[s]).unsqueeze(1)  # (M, 1)
                sol = torch.cholesky_solve(dms, L)  # (M, 1)
                quad = (dms * sol).sum()

                logI = -0.5 * logdetA + self.alpha * quad
                Imat[k, s] = torch.exp(logI)

        return Imat
