import torch
import math

from mmd.spaces import L2CosineBasis
from mmd.kernel import GaussianKernel
from mmd import mmd2_empirical_vs_gaussian_mixture, GaussianMixtureModel, fit_gaussian_mixture_mmd

torch.manual_seed(0)

# Problem setup: L^2([0,T];R^d)
T = 1.0
d = 2
grid_size = 200
R = 20                 # number of time basis functions
sigma_kernel = 1.0

device = torch.device("cpu")
dtype = torch.float64

# Create the L2 cosine basis for projecting functional data
basis = L2CosineBasis(T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype)

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

# Create the kernel
kernel = GaussianKernel(sigma=sigma_kernel)

print("=" * 60)
print("Example 1: Manual MMD computation with fixed mixture")
print("=" * 60)

# Define a Gaussian mixture in coefficient space
Kmix = 3
pi = torch.tensor([0.4, 0.3, 0.3], device=device, dtype=dtype)

# Means (K,M)
m = 0.2 * torch.randn((Kmix, M), device=device, dtype=dtype)

# Covariances (K,M,M) -- minimal: diagonal covariances in coefficient space
# Ensure positive variances; interpret as covariance operator restricted to the basis.
var = 0.5 + 0.1 * torch.rand((Kmix, M), device=device, dtype=dtype)  # (K,M)
Kcov = torch.diag_embed(var)  # (K,M,M)

# Compute MMD^2 using the kernel object
mmd2, stats = mmd2_empirical_vs_gaussian_mixture(
    X=X, pi=pi, m=m, Kcov=Kcov, kernel=kernel, compute_const_term=True
)
print("MMD^2(P_n, Q) =", float(mmd2))

print("\n" + "=" * 60)
print("Example 2: Training a Gaussian Mixture Model with MMD")
print("=" * 60)

# Create a trainable mixture model
model = GaussianMixtureModel(
    num_components=3,
    coeff_dim=M,
    basis=basis,
    covariance_type="diagonal",
    device=device,
    dtype=dtype,
)

# Initialize from data
model.initialize_from_data(X, method="kmeans++")

print(f"\nModel: {model}")
print(f"Initial weights: {model.pi.detach().numpy()}")

# Manual training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
num_epochs = 100

print("\nTraining...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    mmd2, stats = model.compute_mmd2(X, kernel, compute_const_term=True)
    mmd2.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}: MMD² = {mmd2.item():.6f}, π = {model.pi.detach().numpy()}")

print(f"\nFinal MMD² = {mmd2.item():.6f}")
print(f"Final weights: {model.pi.detach().numpy()}")
print(f"Final variances (mean per component): {model.variance.mean(dim=1).detach().numpy()}")

print("\n" + "=" * 60)
print("Example 3: Using the convenience function")
print("=" * 60)

# Use the convenience function for fitting
model2, history = fit_gaussian_mixture_mmd(
    X=X,
    num_components=3,
    kernel=kernel,
    num_epochs=100,
    lr=0.05,
    covariance_type="diagonal",
    init_method="kmeans++",
    basis=basis,  # Pass basis for function reconstruction
    verbose=True,
    log_interval=25,
)

print(f"\nFinal MMD² = {history[-1]:.6f}")
print(f"Final weights: {model2.pi.detach().numpy()}")

# Sample from the fitted model
print("\n" + "=" * 60)
print("Example 4: Sampling from the fitted model")
print("=" * 60)

samples, assignments = model2.sample(10)
print(f"Sampled coefficients shape: {samples.shape}")
print(f"Component assignments: {assignments.numpy()}")

# If we have a basis, we can reconstruct functions
functions, coeffs, assignments = model2.sample_functions(5)
print(f"Sampled functions shape: {functions.shape}")  # (5, grid_size, d)