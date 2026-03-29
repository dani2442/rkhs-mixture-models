#!/usr/bin/env python
"""
Benchmark: L² synthetic functional data clustering.

Generates 10 datasets of L²([0,1]; R²) trajectories from a 5-component
Gaussian mixture, evaluates all applicable clustering methods, and reports
ARI statistics.
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    GaussianKernel,
    GaussianMixtureModel,
    L2CosineBasis,
    PolynomialKernel,
)
from src.competitors import (
    DBSCANClustering,
    HDBSCANClustering,
    HierarchicalClustering,
    KCenterClustering,
    KernelKGroupsClustering,
    KMedoidsClustering,
)
from examples.train_l2_gaussian import generate_n_datasets
from benchmarks.runner import save_benchmark_results

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
N_DATASETS = 10
N_SAMPLES = 500
N_COMPONENTS = 5
GRID_SIZE = 100
T = 1.0
D = 2
R = 15
SIGMA_KERNEL = 1.2
NUM_EPOCHS = 500
LR = 0.1
SEED = 42


def _train_mmd_gmm(X, basis, n_components, kernel, num_epochs, lr):
    """Train an MMD GMM and return hard cluster labels."""
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 60)
    print(f"Benchmark: L² Synthetic ({N_DATASETS} datasets)")
    print("=" * 60)

    # ── Data generation ─────────────────────────────────────────────
    true_weights = torch.tensor(
        [0.30, 0.25, 0.20, 0.15, 0.10], device=device, dtype=dtype
    )
    datasets = generate_n_datasets(
        n_datasets=N_DATASETS,
        n_samples=N_SAMPLES,
        n_components=N_COMPONENTS,
        grid_size=GRID_SIZE,
        R=R,
        T=T,
        d=D,
        component_weights=true_weights,
        base_seed=SEED,
        device=device,
        dtype=dtype,
    )

    # ── Method list ─────────────────────────────────────────────────
    ari_results: dict[str, list[float]] = {
        "MMD GMM (Gaussian)": [],
        "MMD GMM (Polynomial)": [],
        "K-Medoids": [],
        "Hierarchical (Avg)": [],
        "DBSCAN": [],
        "HDBSCAN": [],
        "K-Center": [],
        "Kernel k-Groups": [],
    }

    # FDA methods (optional)
    use_fda = False
    try:
        from src.competitors import (
            ScikitFDAAgglomerative,
            ScikitFDAFuzzyCMeans,
            ScikitFDAKMeans,
        )

        ScikitFDAKMeans(n_clusters=N_COMPONENTS)
        ari_results.update(
            {
                "FDA K-Means": [],
                "FDA Fuzzy C-Means": [],
                "FDA Agglomerative": [],
            }
        )
        use_fda = True
    except Exception:
        print("scikit-fda not available, skipping FDA methods.")

    # ── Evaluate over datasets ──────────────────────────────────────
    for i, (X_raw, X_coeffs, true_assignments, info) in enumerate(datasets):
        print(f"\n--- Dataset {i + 1}/{N_DATASETS} ---")
        true_labels = true_assignments.cpu().numpy()

        basis = L2CosineBasis(
            T=T, R=R, grid_size=GRID_SIZE, d=D, device=device, dtype=dtype
        )
        X = basis.project(X_raw)

        # MMD GMM (Gaussian)
        torch.manual_seed(SEED + i)
        kernel_g = GaussianKernel(sigma=SIGMA_KERNEL)
        labels = _train_mmd_gmm(X, basis, N_COMPONENTS, kernel_g, NUM_EPOCHS, LR)
        ari = adjusted_rand_score(true_labels, labels)
        ari_results["MMD GMM (Gaussian)"].append(ari)
        print(f"  MMD GMM (Gaussian)    ARI={ari:.4f}")

        # MMD GMM (Polynomial)
        torch.manual_seed(SEED + i)
        kernel_p = PolynomialKernel(degree=2, c=1.0)
        labels = _train_mmd_gmm(X, basis, N_COMPONENTS, kernel_p, NUM_EPOCHS, LR)
        ari = adjusted_rand_score(true_labels, labels)
        ari_results["MMD GMM (Polynomial)"].append(ari)
        print(f"  MMD GMM (Polynomial)  ARI={ari:.4f}")

        # Metric-space competitors
        from scipy.spatial.distance import pdist, squareform

        X_np = X.cpu().numpy()
        dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        competitors = [
            ("K-Medoids", KMedoidsClustering(n_clusters=N_COMPONENTS)),
            (
                "Hierarchical (Avg)",
                HierarchicalClustering(n_clusters=N_COMPONENTS, linkage="average"),
            ),
            ("DBSCAN", DBSCANClustering(eps=1.5, min_samples=5)),
            ("HDBSCAN", HDBSCANClustering(min_cluster_size=5)),
            ("K-Center", KCenterClustering(n_clusters=N_COMPONENTS)),
            ("Kernel k-Groups", KernelKGroupsClustering(n_clusters=N_COMPONENTS)),
        ]

        if use_fda:
            competitors.extend(
                [
                    ("FDA K-Means", ScikitFDAKMeans(n_clusters=N_COMPONENTS)),
                    (
                        "FDA Fuzzy C-Means",
                        ScikitFDAFuzzyCMeans(n_clusters=N_COMPONENTS),
                    ),
                    (
                        "FDA Agglomerative",
                        ScikitFDAAgglomerative(
                            n_clusters=N_COMPONENTS, linkage="average"
                        ),
                    ),
                ]
            )

        _uses_dist = {"K-Medoids", "Hierarchical (Avg)", "DBSCAN", "HDBSCAN", "Kernel k-Groups"}

        for name, method in competitors:
            try:
                if "FDA" in name:
                    pred = method.fit_predict(X, basis=basis)
                elif name in _uses_dist:
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
        benchmark_id="l2",
        config={
            "n_datasets": N_DATASETS,
            "n_samples": N_SAMPLES,
            "n_components": N_COMPONENTS,
            "grid_size": GRID_SIZE,
            "R": R,
            "sigma_kernel": SIGMA_KERNEL,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "seed": SEED,
        },
        results=results,
    )


if __name__ == "__main__":
    main()
