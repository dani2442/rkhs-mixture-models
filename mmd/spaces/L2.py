import torch
from dataclasses import dataclass
import math

@dataclass
class L2CosineBasis:
    """
    Orthonormal cosine basis on L^2([0,T]):

      phi_0(t) = 1/sqrt(T)
      phi_r(t) = sqrt(2/T) * cos(r*pi*t/T), r>=1

    For X in L^2([0,T];R^d), we use product basis:
      e_(q,r)(t) = phi_r(t) * e_q,  q=1..d
    Total coefficients M = R * d.
    """
    T: float
    R: int                     # number of time-basis functions
    grid_size: int             # number of discretization points on [0,T]
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        # Uniform grid on [0,T]
        self.t = torch.linspace(0.0, self.T, self.grid_size, device=self.device, dtype=self.dtype)
        # Simple Riemann weights
        self.dt = self.T / (self.grid_size - 1)
        self.w = self.dt * torch.ones_like(self.t)

        # Precompute phi(t_l) matrix of shape (L, R)
        self.Phi = self._eval_basis(self.t, self.R)  # (L, R)

    def _eval_basis(self, t: torch.Tensor, R: int) -> torch.Tensor:
        L = t.shape[0]
        Phi = torch.empty((L, R), device=t.device, dtype=t.dtype)

        # r=0
        Phi[:, 0] = 1.0 / math.sqrt(self.T)

        # r>=1
        for r in range(1, R):
            Phi[:, r] = math.sqrt(2.0 / self.T) * torch.cos(r * math.pi * t / self.T)
        return Phi

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project trajectories X(t) onto basis.

        Input:
            X: (n, L, d) sampled values, L = grid_size

        Output:
            coeffs: (n, M) where M = R*d
                    ordering: [dim1 basis0..R-1, dim2 basis0..R-1, ..., dim d ...]
        """
        assert X.ndim == 3, "X must be (n, L, d)"
        n, L, d = X.shape
        assert L == self.grid_size, "X grid_size mismatch with basis.grid_size"

        # Approximate inner product integral:
        # c[n, r, q] = ∫ X_q(t) phi_r(t) dt  ~ sum_l X[n,l,q]*phi[l,r]*w[l]
        # Use einsum with weights
        weighted_Phi = self.Phi * self.w[:, None]  # (L, R)
        # (n, L, d) x (L, R) -> (n, d, R)
        C = torch.einsum("nld,lr->ndr", X, weighted_Phi)

        # Flatten to (n, d*R)
        return C.reshape(n, d * self.R)

    def coeff_dim(self, d: int) -> int:
        return d * self.R