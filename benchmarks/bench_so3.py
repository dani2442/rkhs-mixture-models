#!/usr/bin/env python
"""
Benchmark: L²(SO(3)) synthetic rotation clustering.

Generates SO(3) rotation data from a 3-component mixture, projects
onto the Wigner D-matrix basis, and evaluates clustering quality via
ARI against true component assignments.
"""

import os
import sys

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    GaussianKernel,
    GaussianMixtureModel,
    PolynomialKernel,
    SO3Basis,
    generate_so3_mixture_data,
)
from src.competitors import (
    DBSCANClustering,
    HDBSCANClustering,
    HierarchicalClustering,
    KCenterClustering,
    KernelKGroupsClustering,
    KMedoidsClustering,
    ProjectedGMMEMFixedCovarianceClustering,
)
from benchmarks.runner import save_benchmark_results

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
N_SAMPLES = 200
N_COMPONENTS = 3
NOISE_CONCENTRATION = 8.0   # noise_std ≈ 0.35 rad — creates genuine cluster overlap
L_MAX = 3
# Sigma is computed per-dataset via the median heuristic (see main loop)
NUM_EPOCHS = 400
LR = 0.1
SEED = 42
N_DATASETS = 10  # multiple datasets for mean/std


def _train_mmd_gmm(X, basis, n_components, kernel, num_epochs, lr):
    """Train MMD GMM, return hard labels."""
    M = X.shape[1]
    model = GaussianMixtureModel(
        num_components=n_components,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",
        device=X.device,
        dtype=X.dtype,
    )
    model.initialize_from_data(X, method="kmeans++")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(num_epochs):
        optimizer.zero_grad()
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()

    with torch.no_grad():
        return model.responsibilities(X).argmax(dim=1).cpu().numpy()


def main():
    print("=" * 60)
    print(f"Benchmark: SO(3) ({N_DATASETS} datasets)")
    print("=" * 60)

    device = torch.device("cpu")
    dtype = torch.float64

    ari_results: dict[str, list[float]] = {
        "MMD GMM (Gaussian)": [],
        "MMD GMM (Polynomial)": [],
        "Projected GMM-EM": [],
        "K-Medoids": [],
        "Hierarchical (Avg)": [],
        "DBSCAN": [],
        "HDBSCAN": [],
        "K-Center": [],
        "Kernel k-Groups": [],
    }

    for i in range(N_DATASETS):
        ds_seed = SEED + i
        torch.manual_seed(ds_seed)
        np.random.seed(ds_seed)

        print(f"\n--- Dataset {i + 1}/{N_DATASETS} (seed={ds_seed}) ---")

        X_euler, true_assignments, info = generate_so3_mixture_data(
            n_samples=N_SAMPLES,
            n_components=N_COMPONENTS,
            noise_concentration=NOISE_CONCENTRATION,
            seed=ds_seed,
            device=device,
            dtype=dtype,
        )
        true_labels = true_assignments.cpu().numpy()

        basis = SO3Basis(
            L_max=L_MAX, use_real_basis=True, device=device, dtype=dtype
        )
        X = basis.project(X_euler)

        # Adaptive kernel bandwidth: median pairwise Euclidean distance (median heuristic)
        from scipy.spatial.distance import pdist
        sigma = float(np.median(pdist(X.cpu().numpy(), metric="euclidean")))

        # MMD GMM (Gaussian)
        torch.manual_seed(ds_seed)
        kernel_g = GaussianKernel(sigma=sigma)
        labels = _train_mmd_gmm(X, basis, N_COMPONENTS, kernel_g, NUM_EPOCHS, LR)
        ari = adjusted_rand_score(true_labels, labels)
        ari_results["MMD GMM (Gaussian)"].append(ari)
        print(f"  MMD GMM (Gaussian)    ARI={ari:.4f}")

        # MMD GMM (Polynomial)
        torch.manual_seed(ds_seed)
        kernel_p = PolynomialKernel(degree=2, c=1.0)
        labels = _train_mmd_gmm(X, basis, N_COMPONENTS, kernel_p, NUM_EPOCHS, LR)
        ari = adjusted_rand_score(true_labels, labels)
        ari_results["MMD GMM (Polynomial)"].append(ari)
        print(f"  MMD GMM (Polynomial)  ARI={ari:.4f}")

        em_labels = ProjectedGMMEMFixedCovarianceClustering(
            n_clusters=N_COMPONENTS,
            basis=basis,
            n_init=3,
            max_iter=100,
            random_state=ds_seed,
        ).fit_predict(X_euler)
        ari = adjusted_rand_score(true_labels, em_labels)
        ari_results["Projected GMM-EM"].append(ari)
        print(f"  Projected GMM-EM      ARI={ari:.4f}")

        # Competitors
        X_np = X.cpu().numpy()
        dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        dbscan_min_samples = max(5, N_SAMPLES // 100)
        from examples.compare_clustering_glucodensity import estimate_dbscan_eps

        dbscan_eps = estimate_dbscan_eps(X_np, min_samples=dbscan_min_samples)
        hdbscan_min_cluster_size = max(5, N_SAMPLES // 50)

        competitors = [
            ("K-Medoids", KMedoidsClustering(n_clusters=N_COMPONENTS)),
            (
                "Hierarchical (Avg)",
                HierarchicalClustering(n_clusters=N_COMPONENTS, linkage="average"),
            ),
            (
                "DBSCAN",
                DBSCANClustering(eps=dbscan_eps, min_samples=dbscan_min_samples),
            ),
            ("HDBSCAN", HDBSCANClustering(min_cluster_size=hdbscan_min_cluster_size)),
            ("K-Center", KCenterClustering(n_clusters=N_COMPONENTS)),
            ("Kernel k-Groups", KernelKGroupsClustering(n_clusters=N_COMPONENTS)),
        ]

        _uses_dist = {"K-Medoids", "Hierarchical (Avg)", "DBSCAN", "HDBSCAN", "Kernel k-Groups"}

        for name, method in competitors:
            try:
                if name in _uses_dist:
                    pred = method.fit_predict(X, dist_matrix=dist_matrix)
                else:
                    pred = method.fit_predict(X)

                ari = adjusted_rand_score(true_labels, pred)
                ari_results[name].append(ari)
                print(f"  {name:<22} ARI={ari:.4f}")
            except Exception as e:
                print(f"  {name} failed: {e}")
                ari_results[name].append(float("nan"))

    # ── Aggregate and save ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    results = {}
    for method_name, aris in ari_results.items():
        valid = [a for a in aris if not np.isnan(a)]
        if valid:
            mean_ari = float(np.mean(valid))
            std_ari = float(np.std(valid))
        else:
            mean_ari = float("nan")
            std_ari = float("nan")
        results[method_name] = {
            "ari_scores": aris,
            "mean_ari": mean_ari,
            "std_ari": std_ari,
        }
        print(f"  {method_name:<22} {mean_ari:.3f} ± {std_ari:.3f}")

    save_benchmark_results(
        benchmark_id="so3",
        config={
            "n_samples": N_SAMPLES,
            "n_components": N_COMPONENTS,
            "noise_concentration": NOISE_CONCENTRATION,
            "L_max": L_MAX,
            "sigma_kernel": "adaptive_median",
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "n_datasets": N_DATASETS,
            "seed": SEED,
        },
        results=results,
    )


if __name__ == "__main__":
    main()
