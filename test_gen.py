import torch
from src.data import generate_l2_mixture_data, generate_l2_sine_cosine_data
from src.competitors.clustering import KMedoidsClustering
from src import L2CosineBasis, GaussianKernel, GaussianMixtureModel
import numpy as np
import sys
import os
from sklearn.metrics import adjusted_rand_score

device = torch.device('cpu')
dtype = torch.float64
X_raw, assigns, info = generate_l2_mixture_data(
    n_samples=500, n_components=5, grid_size=100, noise_std=0.8,
    device=device, dtype=dtype
)
basis = L2CosineBasis(T=1.0, R=15, grid_size=100, d=2, device=device, dtype=dtype)
X = basis.project(X_raw)

mod = GaussianMixtureModel(5, X.shape[1], basis, "diagonal", device, dtype)
mod.initialize_from_data(X, "kmeans++")
opt = torch.optim.Adam(mod.parameters(), lr=0.1)
kernel = GaussianKernel(sigma=1.2)
for _ in range(50):
    opt.zero_grad()
    loss, _ = mod.compute_mmd2(X, kernel, compute_const_term=False)
    loss.backward()
    opt.step()

resp = mod.responsibilities(X).argmax(1).numpy()
print("Ours:", adjusted_rand_score(assigns.numpy(), resp))
kmed = KMedoidsClustering(5).fit_predict(X)
print("KMed:", adjusted_rand_score(assigns.numpy(), kmed))

