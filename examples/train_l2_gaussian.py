"""
Synthetic L² functional data generators used by paper notebooks and benchmarks.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import L2CosineBasis


def generate_l2_gaussian_data(
    n_samples: int,
    n_components: int,
    grid_size: int,
    R: int,
    T: float = 1.0,
    d: int = 2,
    component_weights: torch.Tensor = None,
    seed: int = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
):
    if seed is not None:
        torch.manual_seed(seed)

    M = R * d
    t = torch.linspace(0, T, grid_size, device=device, dtype=dtype)

    basis = L2CosineBasis(T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype)

    if component_weights is None:
        component_weights = torch.ones(n_components, device=device, dtype=dtype) / n_components

    assignments = torch.multinomial(
        component_weights.expand(n_samples, -1), num_samples=1
    ).squeeze(-1)

    true_mean_coeffs = torch.zeros(n_components, M, device=device, dtype=dtype)

    for k in range(n_components):
        for dim in range(d):
            base_idx = dim * R
            for r in range(R):
                freq_factor = (r + 1) / R

                if k == 0:
                    amplitude = 4.0 * np.exp(-5.0 * freq_factor) + 0.1
                elif k == 1:
                    amplitude = 2.0 * (np.exp(-10.0 * freq_factor) + np.exp(-10.0 * (1 - freq_factor)))
                elif k == 2:
                    amplitude = 2.5 * np.abs(np.sin(4 * np.pi * freq_factor)) * np.exp(-1.5 * freq_factor)
                elif k == 3:
                    amplitude = 3.0 * (freq_factor % 0.3) / 0.3 * np.exp(-2.0 * freq_factor)
                elif k == 4:
                    amplitude = 3.5 * np.exp(-25.0 * (freq_factor - 0.3) ** 2)
                elif k == 5:
                    amplitude = 2.0 * freq_factor ** 1.5 + 0.3
                elif k == 6:
                    amplitude = 2.5 if freq_factor < 0.5 else 0.5
                else:
                    amplitude = 2.0 * np.abs(np.sin(7 * np.pi * freq_factor + k)) * (1 + 0.5 * np.cos(3 * np.pi * freq_factor))

                if k % 2 == 0:
                    phase_shift = np.pi * dim / (d + 1) + np.pi * k / 4
                else:
                    phase_shift = -np.pi * dim / 2 + np.pi * (k + r) / 6

                sign = 1 if (k + dim) % 3 != 0 else -1
                true_mean_coeffs[k, base_idx + r] = 0.35 * sign * amplitude * np.cos(phase_shift + r * np.pi / (k + 2))

    true_variances = torch.zeros(n_components, M, device=device, dtype=dtype)

    for k in range(n_components):
        for dim in range(d):
            base_idx = dim * R
            for r in range(R):
                freq_factor = (r + 1) / R

                if k == 0:
                    var = 0.02 + 0.01 * freq_factor
                elif k == 1:
                    var = 0.5 * np.exp(-4.0 * freq_factor) + 0.01
                elif k == 2:
                    var = 0.15 * (1 + 0.8 * np.abs(np.sin(3 * np.pi * freq_factor)))
                elif k == 3:
                    var = 0.4 + 0.1 * freq_factor
                elif k == 4:
                    var = 0.05 + 0.35 * np.exp(-15.0 * (freq_factor - 0.5) ** 2)
                elif k == 5:
                    var = 0.03 + 0.3 * freq_factor ** 2
                elif k == 6:
                    var = 0.1 if freq_factor < 0.4 else 0.35
                else:
                    var = 0.1 + 0.2 * np.abs(np.sin(5 * np.pi * freq_factor)) * (1 - 0.5 * freq_factor)

                dim_scale = 1.0 + 0.5 * dim + 0.2 * np.sin(np.pi * k * dim / d)
                true_variances[k, base_idx + r] = var * dim_scale

    X_coeffs = torch.zeros(n_samples, M, device=device, dtype=dtype)
    for i in range(n_samples):
        k = assignments[i].item()
        std = torch.sqrt(true_variances[k])
        X_coeffs[i] = true_mean_coeffs[k] + std * torch.randn(M, device=device, dtype=dtype)

    X_raw = basis.reconstruct(X_coeffs)
    true_means = basis.reconstruct(true_mean_coeffs)

    info = {
        "component_weights": component_weights,
        "true_mean_coeffs": true_mean_coeffs,
        "true_variances": true_variances,
        "true_means": true_means,
        "t": t,
        "T": T,
        "basis": basis,
    }

    return X_raw, X_coeffs, assignments, info


def generate_n_datasets(
    n_datasets: int,
    n_samples: int,
    n_components: int,
    grid_size: int,
    R: int,
    T: float = 1.0,
    d: int = 2,
    component_weights: torch.Tensor = None,
    base_seed: int = 42,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
):
    datasets = []
    for i in range(n_datasets):
        seed = base_seed + i if base_seed is not None else None
        dataset = generate_l2_gaussian_data(
            n_samples=n_samples,
            n_components=n_components,
            grid_size=grid_size,
            R=R,
            T=T,
            d=d,
            component_weights=component_weights,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        datasets.append(dataset)
    return datasets
