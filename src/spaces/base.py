"""
Base class for Hilbert space basis representations.

A HilbertBasis provides an orthonormal basis for a finite-dimensional
subspace of a separable Hilbert space X, enabling projection of
elements x ∈ X onto coefficient vectors in R^M.
"""
import torch
from abc import ABC, abstractmethod
from typing import Optional


class HilbertBasis(ABC):
    """
    Abstract base class for finite-dimensional Hilbert basis representations.

    A HilbertBasis defines:
      - An orthonormal basis (e_1, ..., e_M) of a subspace X_M ⊂ X
      - A projection operator Π_M: X → X_M
      - Methods to project data and reconstruct from coefficients

    Subclasses must implement:
      - project(X): project data onto the basis
      - coeff_dim: dimension M of the coefficient space
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project data X onto the orthonormal basis.

        Args:
            X: Input data in the Hilbert space representation.
               Shape depends on the specific basis.

        Returns:
            coeffs: Coefficient vectors in R^M, shape (n, M) where n is
                    the number of samples.
        """
        pass

    @property
    @abstractmethod
    def coeff_dim(self) -> int:
        """
        Return the dimension M of the coefficient space.
        """
        pass

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct data from coefficients (optional, not all bases support this).

        Args:
            coeffs: Coefficient vectors, shape (n, M)

        Returns:
            X: Reconstructed data in the original Hilbert space representation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support reconstruction."
        )

    def inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the inner product ⟨x, y⟩_X in the Hilbert space.

        Default implementation: standard Euclidean inner product on coefficients.
        Override for non-standard inner products (e.g., weighted graphs).

        Args:
            x: Coefficient vectors, shape (..., M)
            y: Coefficient vectors, shape (..., M)

        Returns:
            Inner products, shape (...)
        """
        return (x * y).sum(dim=-1)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm ||x||_X in the Hilbert space.

        Args:
            x: Coefficient vectors, shape (..., M)

        Returns:
            Norms, shape (...)
        """
        return torch.sqrt(self.inner_product(x, x))

    def squared_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute ||x - y||^2_X.

        Args:
            x: Coefficient vectors, shape (..., M)
            y: Coefficient vectors, shape (..., M)

        Returns:
            Squared distances, shape (...)
        """
        diff = x - y
        return self.inner_product(diff, diff)
