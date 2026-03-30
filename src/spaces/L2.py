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


class H1CosineBasis(L2CosineBasis):
    """
    Cosine basis reweighted to be orthonormal for H^1([0,T]; R^d).

    We use the standard Sobolev inner product

        <f, g>_{H^1} = <f, g>_{L^2} + <f', g'>_{L^2}.

    If f = sum_r c_r phi_r in the L^2-orthonormal cosine basis, then

        ||f||_{H^1}^2 = sum_r (1 + (r*pi/T)^2) c_r^2.

    Returning coefficients scaled by sqrt(1 + (r*pi/T)^2) therefore makes the
    Euclidean geometry in coefficient space match the H^1 norm.
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
        super().__init__(T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype)
        mode_idx = torch.arange(self.R, device=self.device, dtype=self.dtype)
        mode_scale = torch.sqrt(1.0 + (mode_idx * math.pi / self.T) ** 2)
        self.h1_mode_scale = mode_scale
        self.h1_coeff_scale = mode_scale.repeat(self.d)

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """Project sampled curves to H^1-orthonormal cosine coefficients."""
        coeffs_l2 = super().project(X)
        return coeffs_l2 * self.h1_coeff_scale.unsqueeze(0)

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Reconstruct curves from H^1-orthonormal cosine coefficients."""
        coeffs_l2 = coeffs / self.h1_coeff_scale.unsqueeze(0)
        return super().reconstruct(coeffs_l2)


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


class L2TensorBasis2D(HilbertBasis):
    """
    Orthonormal tensor product basis on L^2([0,T]×[0,S]; R^d).

    For the space L^2([0,T]×[0,S]; R^d), we use tensor products of 1D cosine bases:

    Step 1: L^2([0,T]×[0,S]; R^d) ≅ (L^2([0,T]×[0,S]))^d

    Step 2: For scalar L^2([0,T]×[0,S]), use tensor product of 1D orthonormal bases:
        Φ_{m,n}(s,t) = φ_m(s) · φ_n(t),   m,n ≥ 0

    Step 3: For R^d-valued functions, the basis is:
        Ψ_{j,m,n}(s,t) = Φ_{m,n}(s,t) · e_j = φ_m(s) · φ_n(t) · e_j

    where (φ_r)_{r≥0} is the 1D cosine basis:
        φ_0(t) = 1/√T
        φ_r(t) = √(2/T) · cos(r·π·t/T),  r ≥ 1

    Total coefficient dimension: M = R_s × R_t × d

    Expansion: f = Σ_{j=1}^d Σ_{m=0}^{R_s-1} Σ_{n=0}^{R_t-1} c_{j,m,n} · Ψ_{j,m,n}

    Parseval: ||f||² = Σ_{j,m,n} |c_{j,m,n}|²
    """

    def __init__(
        self,
        T: float,
        S: float,
        R_t: int,
        R_s: int,
        grid_size_t: int,
        grid_size_s: int,
        d: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            T: End of first interval [0, T] (typically time)
            S: End of second interval [0, S] (typically space)
            R_t: Number of basis functions in t-direction
            R_s: Number of basis functions in s-direction
            grid_size_t: Number of discretization points in t
            grid_size_s: Number of discretization points in s
            d: Spatial dimension (R^d valued functions)
            device: Torch device
            dtype: Torch dtype
        """
        super().__init__(device=device, dtype=dtype)
        self.T = T
        self.S = S
        self.R_t = R_t
        self.R_s = R_s
        self.grid_size_t = grid_size_t
        self.grid_size_s = grid_size_s
        self.d = d
        self._M = R_t * R_s * d

        # Uniform grids on [0,T] and [0,S]
        self.t = torch.linspace(0, T, grid_size_t, device=device, dtype=dtype)
        self.s = torch.linspace(0, S, grid_size_s, device=device, dtype=dtype)

        # Simple Riemann weights for double integral
        self.dt = T / (grid_size_t - 1) if grid_size_t > 1 else T
        self.ds = S / (grid_size_s - 1) if grid_size_s > 1 else S

        # Precompute 1D basis matrices
        self.Phi_t = self._compute_1d_cosine_basis(self.t, self.T, self.R_t)  # (grid_size_t, R_t)
        self.Phi_s = self._compute_1d_cosine_basis(self.s, self.S, self.R_s)  # (grid_size_s, R_s)

    def _compute_1d_cosine_basis(
        self, grid: torch.Tensor, length: float, R: int
    ) -> torch.Tensor:
        """
        Compute 1D cosine basis functions at grid points.

        Args:
            grid: Grid points, shape (L,)
            length: Interval length (T or S)
            R: Number of basis functions

        Returns:
            Phi: Basis matrix, shape (L, R)
        """
        L = grid.shape[0]
        Phi = torch.empty((L, R), device=self.device, dtype=self.dtype)

        # r=0: constant function φ_0(x) = 1/√length
        Phi[:, 0] = 1.0 / math.sqrt(length)

        # r>=1: cosine functions φ_r(x) = √(2/length) · cos(r·π·x/length)
        for r in range(1, R):
            Phi[:, r] = math.sqrt(2.0 / length) * torch.cos(
                r * math.pi * grid / length
            )
        return Phi

    @property
    def coeff_dim(self) -> int:
        """Return M = R_t × R_s × d."""
        return self._M

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project 2D functions X(s,t) onto the tensor product basis.

        Args:
            X: (n, grid_size_s, grid_size_t, d) sampled values on the 2D grid
               Ordering: X[i, l_s, l_t, j] = X_i^j(s_{l_s}, t_{l_t})

        Returns:
            coeffs: (n, M) where M = R_s × R_t × d
                    Ordering: [dim1 (s0,t0)..(s0,t_{R_t-1})..(s_{R_s-1},t_{R_t-1}), dim2 ..., ...]
                    i.e., coeffs[n, j*R_s*R_t + m*R_t + k] = c_{j,m,k} for j-th component
        """
        assert X.ndim == 4, "X must be (n, grid_size_s, grid_size_t, d)"
        n, L_s, L_t, d = X.shape
        assert L_s == self.grid_size_s, f"X grid_size_s {L_s} != {self.grid_size_s}"
        assert L_t == self.grid_size_t, f"X grid_size_t {L_t} != {self.grid_size_t}"
        assert d == self.d, f"X spatial dim {d} != basis dim {self.d}"

        # Approximate double integral using tensor product quadrature:
        # c_{j,m,k} = ∫∫ X_j(s,t) φ_m(s) φ_k(t) ds dt
        #           ≈ Σ_{l_s,l_t} X[n,l_s,l_t,j] · Φ_s[l_s,m] · Φ_t[l_t,k] · ds · dt

        # Weight the basis matrices
        weighted_Phi_s = self.Phi_s * self.ds  # (L_s, R_s)
        weighted_Phi_t = self.Phi_t * self.dt  # (L_t, R_t)

        # Compute coefficients via einsum:
        # C[n, j, m, k] = Σ_{l_s, l_t} X[n, l_s, l_t, j] · Φ_s[l_s, m] · Φ_t[l_t, k]
        C = torch.einsum("nstd,sm,tk->ndmk", X, weighted_Phi_s, weighted_Phi_t)

        # Flatten to (n, d * R_s * R_t)
        return C.reshape(n, d * self.R_s * self.R_t)

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct 2D functions from coefficients.

        Args:
            coeffs: (n, M) coefficient vectors

        Returns:
            X: (n, grid_size_s, grid_size_t, d) reconstructed 2D functions
        """
        n, M = coeffs.shape
        assert M == self._M, f"coeffs dim {M} != {self._M}"

        # Reshape to (n, d, R_s, R_t)
        C = coeffs.reshape(n, self.d, self.R_s, self.R_t)

        # Reconstruct: X[n, l_s, l_t, j] = Σ_{m,k} C[n, j, m, k] · Φ_s[l_s, m] · Φ_t[l_t, k]
        X = torch.einsum("ndmk,sm,tk->nstd", C, self.Phi_s, self.Phi_t)

        return X

    def get_meshgrid(self) -> tuple:
        """
        Return the 2D meshgrid for visualization.

        Returns:
            S_grid, T_grid: Meshgrid arrays, each shape (grid_size_s, grid_size_t)
        """
        S_grid, T_grid = torch.meshgrid(self.s, self.t, indexing="ij")
        return S_grid, T_grid
