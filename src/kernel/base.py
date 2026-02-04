"""
Base class for kernel functions on Hilbert spaces.

Provides infrastructure for computing MMD expectations with
Gaussian mixture components.
"""
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class Kernel(ABC):
    """
    Abstract base class for positive definite kernels on Hilbert spaces.

    A kernel κ: X × X → R must implement:
      - evaluate(x, y): compute κ(x, y)
      - J(X, m, Kcov): E_{y~N(m,K)}[κ(x, y)] for data X
      - I(m, Kcov): E_{y~N(m_k,K_k), y'~N(m_s,K_s)}[κ(y, y')]

    These expectations are used in the MMD computation:
      MMD²(P,Q) = E_{x,x'~P}[κ(x,x')] - 2E_{x~P,y~Q}[κ(x,y)] + E_{y,y'~Q}[κ(y,y')]
    """

    def __init__(self, eps_jitter: float = 1e-8):
        """
        Args:
            eps_jitter: Small constant for numerical stability.
        """
        self.eps_jitter = eps_jitter

    @abstractmethod
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the kernel κ(x, y).

        Args:
            x: First input, shape (..., M)
            y: Second input, shape (..., M)

        Returns:
            Kernel values, shape (...)
        """
        pass

    @abstractmethod
    def compute_J(
        self,
        X: torch.Tensor,
        m: torch.Tensor,
        Kcov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute J_{i,k} = E_{y ~ N(m_k, K_k)}[κ(X_i, y)].

        Args:
            X: Data points, shape (n, M)
            m: Component means, shape (K, M)
            Kcov: Component covariances, shape (K, M, M)

        Returns:
            J: Expectation matrix, shape (n, K)
        """
        pass

    @abstractmethod
    def compute_I(
        self,
        m: torch.Tensor,
        Kcov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute I_{k,s} = E_{y~N(m_k,K_k), y'~N(m_s,K_s)}[κ(y, y')].

        Args:
            m: Component means, shape (K, M)
            Kcov: Component covariances, shape (K, M, M)

        Returns:
            I: Cross-expectation matrix, shape (K, K)
        """
        pass

    def compute_gram_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix K[i,j] = κ(X_i, X_j).

        Args:
            X: Data points, shape (n, M)

        Returns:
            Gram matrix, shape (n, n)
        """
        n = X.shape[0]
        K = torch.empty((n, n), device=X.device, dtype=X.dtype)
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self.evaluate(X[i], X[j])
                K[j, i] = K[i, j]
        return K

    def compute_Jbar(
        self,
        X: torch.Tensor,
        m: torch.Tensor,
        Kcov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute J̄_k = (1/n) Σ_i J_{i,k}.

        Args:
            X: Data points, shape (n, M)
            m: Component means, shape (K, M)
            Kcov: Component covariances, shape (K, M, M)

        Returns:
            Jbar: Mean expectations, shape (K,)
        """
        J = self.compute_J(X, m, Kcov)  # (n, K)
        return J.mean(dim=0)


def _log_det_from_cholesky(L: torch.Tensor) -> torch.Tensor:
    """
    Compute log det(A) from lower Cholesky factor L where A = L @ L^T.

    Args:
        L: Cholesky factor, shape (..., M, M)

    Returns:
        log det(A), shape (...)
    """
    return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
