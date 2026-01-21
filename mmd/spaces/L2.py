"""
L^2 space basis implementations for functional data.
"""
import torch
import math
from typing import Optional

from .base import HilbertBasis


class L2Basis(HilbertBasis):
    """
    Base class for L^2([a,b]; R^d) basis representations.

    Provides infrastructure for:
      - Time grid discretization
      - Quadrature weights for inner product approximation
      - Projection and reconstruction of functional data
    """

    def __init__(
        self,
        T: float,
        R: int,
        grid_size: int,
        d: int = 1,
        a: float = 0.0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            T: End of interval [a, T]
            R: Number of basis functions per spatial dimension
            grid_size: Number of discretization points
            d: Spatial dimension (R^d valued functions)
            a: Start of interval [a, T]
            device: Torch device
            dtype: Torch dtype
        """
        super().__init__(device=device, dtype=dtype)
        self.T = T
        self.a = a
        self.R = R
        self.grid_size = grid_size
        self.d = d
        self._M = R * d

        # Uniform grid on [a,T]
        self.t = torch.linspace(a, T, grid_size, device=device, dtype=dtype)
        # Simple Riemann weights
        self.dt = (T - a) / (grid_size - 1)
        self.w = self.dt * torch.ones_like(self.t)

        # Precompute basis matrix: shape (grid_size, R)
        self.Phi = self._compute_basis_matrix()

    def _compute_basis_matrix(self) -> torch.Tensor:
        """
        Compute the basis function matrix Phi[l, r] = phi_r(t_l).

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def coeff_dim(self) -> int:
        """Return M = R * d."""
        return self._M

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project trajectories X(t) onto the basis.

        Args:
            X: (n, L, d) sampled values at grid points, L = grid_size

        Returns:
            coeffs: (n, M) where M = R*d
                    Ordering: [dim1 basis0..R-1, dim2 basis0..R-1, ...]
        """
        assert X.ndim == 3, "X must be (n, L, d)"
        n, L, d = X.shape
        assert L == self.grid_size, "X grid_size mismatch with basis.grid_size"
        assert d == self.d, f"X spatial dim {d} != basis dim {self.d}"

        # Approximate inner product integral:
        # c[n, r, q] = ∫ X_q(t) phi_r(t) dt ≈ Σ_l X[n,l,q] * phi[l,r] * w[l]
        weighted_Phi = self.Phi * self.w[:, None]  # (L, R)
        # (n, L, d) x (L, R) -> (n, d, R)
        C = torch.einsum("nld,lr->ndr", X, weighted_Phi)

        # Flatten to (n, d*R)
        return C.reshape(n, d * self.R)

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct trajectories from coefficients.

        Args:
            coeffs: (n, M) coefficient vectors

        Returns:
            X: (n, L, d) reconstructed trajectories
        """
        n, M = coeffs.shape
        assert M == self._M, f"coeffs dim {M} != {self._M}"

        # Reshape to (n, d, R)
        C = coeffs.reshape(n, self.d, self.R)

        # Reconstruct: X[n, l, q] = Σ_r C[n, q, r] * Phi[l, r]
        # (n, d, R) x (L, R)^T -> (n, d, L) -> (n, L, d)
        X = torch.einsum("ndr,lr->ndl", C, self.Phi)
        return X.permute(0, 2, 1)


class L2CosineBasis(L2Basis):
    """
    Orthonormal cosine basis on L^2([0,T]; R^d).

    Basis functions:
      phi_0(t) = 1/sqrt(T)
      phi_r(t) = sqrt(2/T) * cos(r*pi*t/T),  r >= 1

    For X in L^2([0,T]; R^d), we use the product basis:
      e_(q,r)(t) = phi_r(t) * e_q,  q=1..d

    Total coefficient dimension: M = R * d
    """

    def __init__(
        self,
        T: float,
        R: int,
        grid_size: int,
        d: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        # For cosine basis, interval starts at 0
        super().__init__(T=T, R=R, grid_size=grid_size, d=d, a=0.0, device=device, dtype=dtype)

    def _compute_basis_matrix(self) -> torch.Tensor:
        """Compute cosine basis functions at grid points."""
        L = self.grid_size
        Phi = torch.empty((L, self.R), device=self.device, dtype=self.dtype)

        # r=0: constant function
        Phi[:, 0] = 1.0 / math.sqrt(self.T)

        # r>=1: cosine functions
        for r in range(1, self.R):
            Phi[:, r] = math.sqrt(2.0 / self.T) * torch.cos(
                r * math.pi * self.t / self.T
            )
        return Phi


class L2FourierBasis(L2Basis):
    """
    Orthonormal Fourier basis on L^2([0,T]; R^d).

    Real-valued basis functions using sin/cos pairs:
      phi_0(t) = 1/sqrt(T)                           (DC component)
      phi_{2k-1}(t) = sqrt(2/T) * cos(2*pi*k*t/T)    (cosine terms)
      phi_{2k}(t) = sqrt(2/T) * sin(2*pi*k*t/T)      (sine terms)

    For periodic functions on [0,T].
    """

    def __init__(
        self,
        T: float,
        R: int,
        grid_size: int,
        d: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(T=T, R=R, grid_size=grid_size, d=d, a=0.0, device=device, dtype=dtype)

    def _compute_basis_matrix(self) -> torch.Tensor:
        """Compute Fourier basis functions at grid points."""
        L = self.grid_size
        Phi = torch.empty((L, self.R), device=self.device, dtype=self.dtype)

        # r=0: DC component
        Phi[:, 0] = 1.0 / math.sqrt(self.T)

        # Alternating cos/sin pairs
        r = 1
        k = 1
        while r < self.R:
            freq = 2.0 * math.pi * k / self.T
            # Cosine term
            Phi[:, r] = math.sqrt(2.0 / self.T) * torch.cos(freq * self.t)
            r += 1
            if r < self.R:
                # Sine term
                Phi[:, r] = math.sqrt(2.0 / self.T) * torch.sin(freq * self.t)
                r += 1
            k += 1

        return Phi