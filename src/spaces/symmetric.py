"""
Symmetric matrix Hilbert space basis with Frobenius inner product.

The space Sym(n) = {A in R^{n x n} : A^T = A} is a Hilbert space
under the Frobenius inner product: <A, B> = tr(A B).

Dimension: n(n+1)/2.

The isometric embedding into R^{n(n+1)/2} is the scaled
half-vectorization (vech):
    vech(A)_k = A[i,j]        if i == j  (diagonal)
    vech(A)_k = sqrt(2) A[i,j] if i < j   (upper triangle)

This preserves the Frobenius inner product:
    <vech(A), vech(B)>_{R^M} = tr(AB) = <A, B>_F

Optional dimensionality reduction via PCA on the vech representation.
"""

import math
from typing import Optional

import torch

from .base import HilbertBasis


class SymmetricMatrixBasis(HilbertBasis):
    """
    Hilbert space basis for Sym(n) with Frobenius inner product.

    Uses the scaled vech mapping for isometric embedding.
    Optional PCA truncation via the ``from_data`` class method.
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
            n: Matrix size (n x n symmetric matrices).
            R: Number of coefficients to keep.  Default ``n(n+1)/2``
               (full vech, no truncation).  If ``R < n(n+1)/2``, call
               ``from_data`` instead to set up PCA.
            device: Torch device.
            dtype: Torch dtype.
        """
        super().__init__(device=device, dtype=dtype)
        self.n = n
        self.full_dim = n * (n + 1) // 2
        self._M = R if R is not None else self.full_dim

        # Upper triangle indices
        rows, cols = torch.triu_indices(n, n, device=device)
        self._triu_rows = rows
        self._triu_cols = cols

        # Scaling: 1 for diagonal, sqrt(2) for off-diagonal
        scale = torch.ones(self.full_dim, device=device, dtype=dtype)
        off_diag = rows != cols
        scale[off_diag] = math.sqrt(2.0)
        self._scale = scale
        self._inv_scale = 1.0 / scale

        # PCA (populated by from_data)
        self._pca_components: Optional[torch.Tensor] = None  # (R, full_dim)
        self._pca_mean: Optional[torch.Tensor] = None  # (full_dim,)

    @classmethod
    def from_data(
        cls,
        n: int,
        R: int,
        training_matrices: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> "SymmetricMatrixBasis":
        """
        Create a PCA-truncated basis fitted on training data.

        Args:
            n: Matrix size.
            R: Number of PCA components to retain.
            training_matrices: Symmetric matrices, shape ``(N, n, n)``.
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            SymmetricMatrixBasis with PCA truncation.
        """
        basis = cls(n=n, R=R, device=device, dtype=dtype)

        vech_data = basis._vech(training_matrices.to(device=device, dtype=dtype))
        mean = vech_data.mean(dim=0)
        centered = vech_data - mean

        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Cap R at available non-trivial components
        n_available = int((S > 1e-12).sum().item())
        R_eff = min(R, n_available, Vh.shape[0])
        basis._M = R_eff
        basis._pca_components = Vh[:R_eff].to(device=device, dtype=dtype)
        basis._pca_mean = mean.to(device=device, dtype=dtype)
        return basis

    # ------------------------------------------------------------------

    @property
    def coeff_dim(self) -> int:
        return self._M

    def _vech(self, X: torch.Tensor) -> torch.Tensor:
        """Scaled half-vectorization.  X: (..., n, n) -> (..., full_dim)."""
        vals = X[..., self._triu_rows, self._triu_cols]
        return vals * self._scale

    def _inverse_vech(self, v: torch.Tensor) -> torch.Tensor:
        """Inverse scaled vech.  v: (..., full_dim) -> (..., n, n)."""
        v_unscaled = v * self._inv_scale
        batch_shape = v.shape[:-1]
        M = torch.zeros(*batch_shape, self.n, self.n,
                         device=v.device, dtype=v.dtype)
        M[..., self._triu_rows, self._triu_cols] = v_unscaled
        M[..., self._triu_cols, self._triu_rows] = v_unscaled
        return M

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project symmetric matrices to coefficient vectors.

        Args:
            X: Symmetric matrices, shape ``(batch, n, n)``.

        Returns:
            Coefficients, shape ``(batch, M)``.
        """
        vech = self._vech(X)
        if self._pca_components is not None:
            return (vech - self._pca_mean) @ self._pca_components.T
        return vech

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct symmetric matrices from coefficients.

        Args:
            coeffs: Coefficient vectors, shape ``(batch, M)``.

        Returns:
            Symmetric matrices, shape ``(batch, n, n)``.
        """
        if self._pca_components is not None:
            vech = coeffs @ self._pca_components + self._pca_mean
        else:
            vech = coeffs
        return self._inverse_vech(vech)
