"""
R^n space basis implementations.

Provides orthonormal bases for finite-dimensional vector spaces R^n,
useful for re-expressing embeddings (e.g. from WLHashFingerprint) in
different coordinate systems with optional dimensionality reduction.
"""
import torch
import math
from typing import Optional

from .base import HilbertBasis


class CanonicalBasis(HilbertBasis):
    """
    Canonical (identity) basis for R^n.

    project(X) returns X unchanged (or truncated to first R coordinates).
    This is the "no change of basis" option.
    """

    def __init__(
        self,
        n: int,
        R: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            n: Ambient dimension of input vectors.
            R: Number of coordinates to keep (default: n, i.e. no truncation).
            device: Torch device.
            dtype: Torch dtype.
        """
        super().__init__(device=device, dtype=dtype)
        self.n = n
        self.R = R if R is not None else n
        assert 1 <= self.R <= self.n, f"R must be in [1, {self.n}], got {self.R}"
        self._M = self.R

    @property
    def coeff_dim(self) -> int:
        return self._M

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project vectors onto the canonical basis (identity / truncation).

        Args:
            X: Input vectors, shape (n_samples, n) or (n,).

        Returns:
            Coefficients, shape (n_samples, R) or (R,).
        """
        return X[..., : self.R]

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct vectors from canonical coefficients (zero-pad if truncated).

        Args:
            coeffs: Coefficient vectors, shape (n_samples, R) or (R,).

        Returns:
            Reconstructed vectors, shape (n_samples, n) or (n,).
        """
        if self.R == self.n:
            return coeffs
        # Zero-pad to full dimension
        pad_shape = list(coeffs.shape)
        pad_shape[-1] = self.n - self.R
        padding = torch.zeros(pad_shape, device=coeffs.device, dtype=coeffs.dtype)
        return torch.cat([coeffs, padding], dim=-1)


class DiscreteCosineBasis(HilbertBasis):
    """
    Orthonormal Discrete Cosine Transform (DCT-II) basis for R^n.

    Applies the orthonormal DCT-II matrix to re-express vectors in R^n.
    The basis matrix Phi is n×n orthogonal with entries:

        Phi[i, 0] = 1 / sqrt(n)
        Phi[i, k] = sqrt(2/n) * cos(pi * k * (2i + 1) / (2n))   for k >= 1

    With optional truncation to the first R coefficients (keeping the
    lowest-frequency components), this provides dimensionality reduction.

    This is the discrete analogue of the L² cosine basis in L2.py.
    """

    def __init__(
        self,
        n: int,
        R: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            n: Ambient dimension of input vectors.
            R: Number of DCT coefficients to keep (default: n, no truncation).
            device: Torch device.
            dtype: Torch dtype.
        """
        super().__init__(device=device, dtype=dtype)
        self.n = n
        self.R = R if R is not None else n
        assert 1 <= self.R <= self.n, f"R must be in [1, {self.n}], got {self.R}"
        self._M = self.R

        # Precompute the DCT-II basis matrix: shape (n, R)
        self.Phi = self._compute_dct_matrix()

    def _compute_dct_matrix(self) -> torch.Tensor:
        """
        Compute the orthonormal DCT-II basis matrix.

        Returns:
            Phi: shape (n, R), where Phi[:, k] is the k-th DCT basis vector.
        """
        n = self.n
        R = self.R
        Phi = torch.empty((n, R), device=self.device, dtype=self.dtype)

        i = torch.arange(n, device=self.device, dtype=self.dtype)

        # k = 0: constant vector
        Phi[:, 0] = 1.0 / math.sqrt(n)

        # k >= 1: cosine vectors
        for k in range(1, R):
            Phi[:, k] = math.sqrt(2.0 / n) * torch.cos(
                math.pi * k * (2.0 * i + 1.0) / (2.0 * n)
            )

        return Phi

    @property
    def coeff_dim(self) -> int:
        return self._M

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project vectors onto the DCT basis.

        Args:
            X: Input vectors, shape (n_samples, n).

        Returns:
            DCT coefficients, shape (n_samples, R).
        """
        # X @ Phi: (n_samples, n) @ (n, R) -> (n_samples, R)
        Phi = self.Phi.to(dtype=X.dtype, device=X.device)
        return X @ Phi

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct vectors from DCT coefficients.

        Args:
            coeffs: DCT coefficients, shape (n_samples, R).

        Returns:
            Reconstructed vectors, shape (n_samples, n).
        """
        # coeffs @ Phi^T: (n_samples, R) @ (R, n) -> (n_samples, n)
        Phi = self.Phi.to(dtype=coeffs.dtype, device=coeffs.device)
        return coeffs @ Phi.T
