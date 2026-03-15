import torch
import numpy as np
from src.competitors.clustering import ScikitFDAKMeans
from src import L2CosineBasis, GaussianKernel, GaussianMixtureModel
from sklearn.metrics import adjusted_rand_score
device = torch.device('cpu')
dtype = torch.float64

def generate_l2_variance_data(n_samples=500, n_components=5, grid_size=100, R=15, T=1.0, d=2, device=device, dtype=dtype, seed=42):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    M = R * d
    t = torch.linspace(0, T, grid_size, device=device, dtype=dtype)
    basis = L2CosineBasis(T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype)
    true_weights = torch.ones(n_components, device=device, dtype=dtype) / n_components
    assignments = torch.multinomial(true_weights, n_samples, replacement=True)
    counts = torch.bincount(assignments, minlength=n_components)
    
    true_mean_coeffs = torch.zeros(n_components, M, device=device, dtype=dtype)
    true_variances = torch.zeros(n_components, M, device=device, dtype=dtype)
    for k in range(n_components):
        for dim in range(d):
            base_idx = dim * R
            for r in range(R):
                freq_factor = (r + 1) / R
                if k == 0:
                    var = 0.5 * np.exp(-4.0 * freq_factor) + 0.01
                elif k == 1:
                    var = 2.0 * np.exp(-1.0 * freq_factor) + 0.05
                elif k == 2:
                    var = 0.1 * np.abs(np.sin(10 * np.pi * freq_factor)) + 0.02
                elif k == 3:
                     var = 1.0 * freq_factor + 0.01
                else:
                     var = 0.8 * (1 - freq_factor) + 0.01
                true_variances[k, base_idx + r] = var
                
    X_coeffs = torch.zeros(n_samples, M, device=device, dtype=dtype)
    for i in range(n_samples):
        k = assignments[i]
        std = torch.sqrt(true_variances[k])
        X_coeffs[i] = true_mean_coeffs[k] + std * torch.randn(M, device=device, dtype=dtype)
    X_raw = basis.reconstruct(X_coeffs)
    return X_raw, X_coeffs, assignments, {}

X_raw, X_coeffs, assigns, info = generate_l2_variance_data()
basis = L2CosineBasis(T=1.0, R=15, grid_size=100, d=2, device=device, dtype=dtype)
X = basis.project(X_raw)

print("FDA KMeans:", adjusted_rand_score(assigns.numpy(), ScikitFDAKMeans(5).fit_predict(X, basis)))

mod = GaussianMixtureModel(5, X.shape[1], basis, "diagonal", device, dtype)
mod.initialize_from_data(X, "kmeans++")
opt = torch.optim.Adam(mod.parameters(), lr=0.1)
kernel = GaussianKernel(sigma=1.2)
for ep in range(401):
    opt.zero_grad()
    loss, _ = mod.compute_mmd2(X, kernel, compute_const_term=False)
    loss.backward()
    opt.step()
    if ep % 50 == 0:
        resp = mod.responsibilities(X).argmax(1).numpy()
        print(f"Ours ep {ep}:", adjusted_rand_score(assigns.numpy(), resp))
