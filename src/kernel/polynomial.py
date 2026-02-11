"""
Polynomial kernel for MMD computation on Hilbert spaces.

The polynomial kernel is defined as:
    κ(x, y) = (⟨x, y⟩_X + c)^p

where p ≥ 1 is the degree and c ≥ 0 is a constant.
"""
import torch
import math
from typing import Optional

from .base import Kernel


class PolynomialKernel(Kernel):
    """
    Polynomial kernel: κ(x, y) = (⟨x, y⟩ + c)^p.

    Provides closed-form MMD expectations for Gaussian components using
    moments of Gaussian distributions.

    For degree p, the expectations involve:
      - J_{i,k}: moments of 1D Gaussian (X_i^T y + c) where y ~ N(m_k, K_k)
      - I_{k,s}: moments of the product y^T y' for independent Gaussians
    """

    def __init__(self, degree: int = 2, c: float = 1.0, eps_jitter: float = 1e-8):
        """
        Args:
            degree: Polynomial degree p ≥ 1.
            c: Constant term c ≥ 0.
            eps_jitter: Small constant for numerical stability.
        """
        super().__init__(eps_jitter=eps_jitter)
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if c < 0:
            raise ValueError(f"c must be >= 0, got {c}")
        self.degree = degree
        self.c = c

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate κ(x, y) = (⟨x, y⟩ + c)^p.

        Args:
            x: First input, shape (..., M)
            y: Second input, shape (..., M)

        Returns:
            Kernel values, shape (...)
        """
        inner = (x * y).sum(dim=-1)
        return (inner + self.c) ** self.degree

    def compute_gram_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Efficiently compute Gram matrix.

        Args:
            X: Data points, shape (n, M)

        Returns:
            Gram matrix, shape (n, n)
        """
        G = X @ X.T  # (n, n)
        return (G + self.c) ** self.degree

    def compute_J(
        self,
        X: torch.Tensor,
        m: torch.Tensor,
        Kcov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute J_{i,k} = E_{y ~ N(m_k, K_k)}[(⟨X_i, y⟩ + c)^p].

        For y ~ N(m_k, K_k), the quantity ⟨X_i, y⟩ + c is a 1D Gaussian:
            ⟨X_i, y⟩ + c ~ N(μ_{i,k}, v_{i,k})
        where:
            μ_{i,k} = X_i^T m_k + c
            v_{i,k} = X_i^T K_k X_i

        The p-th moment of N(μ, v) is given by:
            E[Z^p] = Σ_{t=0}^{⌊p/2⌋} (p! / ((p-2t)! 2^t t!)) v^t μ^{p-2t}

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
        p = self.degree

        J = torch.zeros((n, K), device=device, dtype=dtype)

        # Precompute the coefficients for the moment formula
        # E[Z^p] = Σ_{t=0}^{⌊p/2⌋} coef[t] * v^t * μ^{p-2t}
        max_t = p // 2
        coefs = []
        for t in range(max_t + 1):
            coef = math.factorial(p) / (
                math.factorial(p - 2 * t) * (2 ** t) * math.factorial(t)
            )
            coefs.append(coef)

        for k in range(K):
            # μ_{i,k} = X @ m_k + c, shape (n,)
            mu = X @ m[k] + self.c

            # v_{i,k} = X @ K_k @ X^T diagonal = (X @ K_k) * X summed over M
            # Efficient: v = diag(X @ K_k @ X^T)
            XK = X @ Kcov[k]  # (n, M)
            v = (XK * X).sum(dim=1)  # (n,)

            # Compute the moment
            for t, coef in enumerate(coefs):
                power_mu = p - 2 * t
                J[:, k] += coef * (v ** t) * (mu ** power_mu)

        return J

    def compute_I(
        self,
        m: torch.Tensor,
        Kcov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute I_{k,s} = E_{y~N(m_k,K_k), y'~N(m_s,K_s)}[(⟨y, y'⟩ + c)^p].

        Using the binomial expansion:
            I_{k,s} = Σ_{r=0}^p C(p,r) c^{p-r} E[Z_{k,s}^r]

        where Z_{k,s} = y^T y' for independent Gaussians.

        The moments E[Z_{k,s}^r] are computed using Isserlis/Wick theorem.

        Args:
            m: Component means, shape (K, M)
            Kcov: Component covariances, shape (K, M, M)

        Returns:
            I: shape (K, K)
        """
        device = m.device
        dtype = m.dtype
        K = m.shape[0]
        p = self.degree

        Imat = torch.zeros((K, K), device=device, dtype=dtype)

        for k in range(K):
            for s in range(K):
                # Compute moments E[Z^r] for r = 0, 1, ..., p
                moments = self._compute_product_moments(
                    m[k], Kcov[k], m[s], Kcov[s], p
                )

                # Binomial expansion
                for r in range(p + 1):
                    binom_coef = math.comb(p, r)
                    Imat[k, s] += binom_coef * (self.c ** (p - r)) * moments[r]

        return Imat

    def _compute_product_moments(
        self,
        m1: torch.Tensor,
        K1: torch.Tensor,
        m2: torch.Tensor,
        K2: torch.Tensor,
        max_order: int,
    ) -> torch.Tensor:
        """
        Compute E[Z^r] for r = 0, 1, ..., max_order where Z = y^T y'
        with y ~ N(m1, K1) and y' ~ N(m2, K2) independent.

        Using Isserlis theorem for Gaussian moments:
            E[Z] = m1^T m2
            E[Z^2] = (m1^T m2)^2 + m1^T K2 m1 + m2^T K1 m2 + tr(K1 K2)
            E[Z^3] = (m1^T m2)^3 + 3(m1^T m2)(m1^T K2 m1 + m2^T K1 m2 + tr(K1 K2))
                   + 6 m1^T K2 K1 m2

        Args:
            m1, K1: Mean and covariance of first Gaussian
            m2, K2: Mean and covariance of second Gaussian
            max_order: Maximum moment order to compute

        Returns:
            moments: Tensor of shape (max_order + 1,) with E[Z^r] for r = 0..max_order
        """
        device = m1.device
        dtype = m1.dtype

        moments = torch.zeros(max_order + 1, device=device, dtype=dtype)

        # E[Z^0] = 1
        moments[0] = 1.0

        if max_order >= 1:
            # E[Z] = m1^T m2
            moments[1] = m1 @ m2

        if max_order >= 2:
            # E[Z^2] = (m1^T m2)^2 + m1^T K2 m1 + m2^T K1 m2 + tr(K1 K2)
            m1m2 = moments[1]
            m1K2m1 = m1 @ K2 @ m1
            m2K1m2 = m2 @ K1 @ m2
            trK1K2 = (K1 * K2).sum()  # tr(K1 K2) = Frobenius inner product
            moments[2] = m1m2**2 + m1K2m1 + m2K1m2 + trK1K2

        if max_order >= 3:
            # E[Z^3] = (m1^T m2)^3 + 3(m1^T m2)(m1^T K2 m1 + m2^T K1 m2 + tr(K1 K2))
            #        + 6 m1^T K2 K1 m2
            m1K2K1m2 = m1 @ K2 @ K1 @ m2
            moments[3] = (
                m1m2**3
                + 3 * m1m2 * (m1K2m1 + m2K1m2 + trK1K2)
                + 6 * m1K2K1m2
            )

        if max_order >= 4:
            # For higher orders, use recursive formulas or explicit Isserlis
            # This is a simplified implementation for common cases
            # E[Z^4] requires more terms from Isserlis theorem
            K1K2 = K1 @ K2
            trK1K2_sq = (K1K2 * K1K2.T).sum()  # tr((K1 K2)^2)
            m1K2K1K2m1 = m1 @ K2 @ K1 @ K2 @ m1
            m2K1K2K1m2 = m2 @ K1 @ K2 @ K1 @ m2

            # Fourth moment (simplified, assuming zero means for cross-terms)
            moments[4] = (
                m1m2**4
                + 6 * m1m2**2 * (m1K2m1 + m2K1m2 + trK1K2)
                + 3 * (m1K2m1 + m2K1m2 + trK1K2) ** 2
                + 24 * m1m2 * m1K2K1m2
                + 12 * (m1K2K1K2m1 + m2K1K2K1m2)
                + 12 * trK1K2_sq
            )

        if max_order >= 5:
            # For degree 5+, raise a warning or use numerical integration
            import warnings
            warnings.warn(
                f"Polynomial kernel moments for degree > 4 use approximate formulas. "
                f"Requested degree: {max_order}"
            )
            # Approximate higher moments using lower ones (rough approximation)
            for r in range(5, max_order + 1):
                moments[r] = moments[r - 1] * moments[1]

        return moments


class LinearKernel(PolynomialKernel):
    """
    Linear kernel: κ(x, y) = ⟨x, y⟩ + c.

    This is a special case of polynomial kernel with degree=1.
    """

    def __init__(self, c: float = 0.0, eps_jitter: float = 1e-8):
        """
        Args:
            c: Constant term (bias).
            eps_jitter: Small constant for numerical stability.
        """
        super().__init__(degree=1, c=c, eps_jitter=eps_jitter)


class QuadraticKernel(PolynomialKernel):
    """
    Quadratic kernel: κ(x, y) = (⟨x, y⟩ + c)^2.

    This is a special case of polynomial kernel with degree=2.
    """

    def __init__(self, c: float = 1.0, eps_jitter: float = 1e-8):
        """
        Args:
            c: Constant term.
            eps_jitter: Small constant for numerical stability.
        """
        super().__init__(degree=2, c=c, eps_jitter=eps_jitter)
