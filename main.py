import torch
import math

from mmd.spaces.L2 import L2CosineBasis
from mmd import mmd2_empirical_vs_gaussian_mixture

torch.manual_seed(0)

# Problem setup: L^2([0,T];R^d)
T = 1.0
d = 2
grid_size = 200
R = 20                 # number of time basis functions
sigma_kernel = 1.0

device = torch.device("cpu")
dtype = torch.float64

basis = L2CosineBasis(T=T, R=R, grid_size=grid_size, device=device, dtype=dtype)

# Fake data trajectories X_i(t) on grid: shape (n,L,d)
n = 128
t = basis.t
X_raw = torch.zeros((n, grid_size, d), device=device, dtype=dtype)
# simple synthetic signal + noise
X_raw[..., 0] = torch.sin(2.0 * math.pi * t)[None, :] + 0.1 * torch.randn(n, grid_size, device=device, dtype=dtype)
X_raw[..., 1] = torch.cos(4.0 * math.pi * t)[None, :] + 0.1 * torch.randn(n, grid_size, device=device, dtype=dtype)

# Project to coefficients in R^{M}, M = R*d
X = basis.project(X_raw)   # (n, M)
M = X.shape[1]

# Define a Gaussian mixture in coefficient space
Kmix = 3
pi = torch.tensor([0.4, 0.3, 0.3], device=device, dtype=dtype)

# Means (K,M)
m = 0.2 * torch.randn((Kmix, M), device=device, dtype=dtype)

# Covariances (K,M,M) -- minimal: diagonal covariances in coefficient space
# Ensure positive variances; interpret as covariance operator restricted to the basis.
var = 0.5 + 0.1 * torch.rand((Kmix, M), device=device, dtype=dtype)  # (K,M)
Kcov = torch.diag_embed(var)  # (K,M,M)

# Compute MMD^2
mmd2, stats = mmd2_empirical_vs_gaussian_mixture(
    X=X, pi=pi, m=m, Kcov=Kcov, sigma=sigma_kernel, compute_const_term=True
)
print("MMD^2(P_n, Q) =", float(mmd2))