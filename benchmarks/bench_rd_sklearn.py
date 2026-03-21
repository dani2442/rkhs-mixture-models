#!/usr/bin/env python
"""
Benchmark: R^d sklearn toy datasets.

Mirrors the scikit-learn clustering gallery (noisy circles, moons, blobs, …)
and compares sklearn baselines, metric-space competitors, and both MMD GMM
variants.  ARI is averaged over the 5 labelled datasets.
"""

import os
import sys
import time
import warnings

import numpy as np
import torch
from sklearn import cluster, datasets, mixture
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import GaussianKernel, GaussianMixtureModel, PolynomialKernel, fit_gaussian_mixture_mmd
from src.competitors import (
    DBSCANClustering,
    HDBSCANClustering,
    HierarchicalClustering,
    KCenterClustering,
    KMedoidsClustering,
)
from benchmarks.runner import save_benchmark_results

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
N_SAMPLES = 500
SEED = 30
MMD_SEED = 42
MMD_EPOCHS = 300
MMD_LR = 0.05
KERNEL_C = 1.0
GAUSSIAN_SIGMA = 0.0  # <= 0 → median heuristic
GAUSSIAN_SIGMA_SCALE = 0.5
COV_TYPE = "full"


def _make_toy_datasets():
    """Build the 6 toy datasets (5 with labels)."""
    noisy_circles = datasets.make_circles(n_samples=N_SAMPLES, factor=0.5, noise=0.05, random_state=SEED)
    noisy_moons = datasets.make_moons(n_samples=N_SAMPLES, noise=0.05, random_state=SEED)
    blobs = datasets.make_blobs(n_samples=N_SAMPLES, random_state=SEED)
    rng = np.random.RandomState(SEED)
    no_structure = rng.rand(N_SAMPLES, 2), None

    rs2 = 170
    X_aniso, y_aniso = datasets.make_blobs(n_samples=N_SAMPLES, random_state=rs2)
    X_aniso = np.dot(X_aniso, [[0.6, -0.6], [-0.4, 0.8]])
    aniso = (X_aniso, y_aniso)

    varied = datasets.make_blobs(n_samples=N_SAMPLES, cluster_std=[1.0, 2.5, 0.5], random_state=rs2)

    default_base = {
        "quantile": 0.3, "eps": 0.3, "damping": 0.9, "preference": -200,
        "n_neighbors": 3, "n_clusters": 3, "min_samples": 7, "xi": 0.05,
        "min_cluster_size": 0.1, "allow_single_cluster": True,
        "hdbscan_min_cluster_size": 15, "hdbscan_min_samples": 3,
        "random_state": 42,
    }

    specs = [
        ("Noisy circles", noisy_circles,  {"damping": 0.77, "preference": -240, "quantile": 0.2, "n_clusters": 2, "min_samples": 7, "xi": 0.08}),
        ("Noisy moons",   noisy_moons,    {"damping": 0.75, "preference": -220, "n_clusters": 2, "min_samples": 7, "xi": 0.1}),
        ("Varied",        varied,         {"eps": 0.18, "n_neighbors": 2, "min_samples": 7, "xi": 0.01, "min_cluster_size": 0.2}),
        ("Anisotropic",   aniso,          {"eps": 0.15, "n_neighbors": 2, "min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
        ("Blobs",         blobs,          {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
        ("No structure",  no_structure,   {}),
    ]

    merged = []
    for name, data, overrides in specs:
        params = default_base.copy()
        params.update(overrides)
        merged.append((name, data, params))
    return merged


def _run_mmd(X_np, n_clusters, kernel_name):
    """Train MMD GMM and return hard labels."""
    torch.manual_seed(MMD_SEED)
    X_t = torch.as_tensor(X_np, dtype=torch.float64)

    if kernel_name == "polynomial":
        kernel = PolynomialKernel(degree=2, c=KERNEL_C)
    else:
        if GAUSSIAN_SIGMA > 0:
            sigma = GAUSSIAN_SIGMA
        else:
            pw = torch.cdist(X_t, X_t)
            iu = torch.triu_indices(pw.shape[0], pw.shape[1], offset=1)
            upper = pw[iu[0], iu[1]]
            positive = upper[upper > 0]
            sigma = GAUSSIAN_SIGMA_SCALE * (torch.median(positive).item() if positive.numel() > 0 else 1.0)
        kernel = GaussianKernel(sigma=max(float(sigma), 1e-6))

    model, _ = fit_gaussian_mixture_mmd(
        X=X_t, num_components=n_clusters, kernel=kernel,
        num_epochs=MMD_EPOCHS, lr=MMD_LR, covariance_type=COV_TYPE,
        init_method="kmeans++", verbose=False,
    )

    with torch.no_grad():
        if kernel_name == "gaussian":
            return model.responsibilities(X_t).argmax(dim=1).cpu().numpy()
        else:
            dists = torch.cdist(X_t, model.mean.detach())
            return dists.argmin(dim=1).cpu().numpy()


def main():
    np.random.seed(SEED)

    print("=" * 60)
    print("Benchmark: R^d sklearn toy datasets")
    print("=" * 60)

    dataset_specs = _make_toy_datasets()
    has_hdbscan = hasattr(cluster, "HDBSCAN")

    # Collect per-dataset ARI for each method
    ari_per_method: dict[str, list[float]] = {}

    for ds_name, (X, y_true), params in dataset_specs:
        print(f"\n--- {ds_name} ---")
        X = StandardScaler().fit_transform(X)

        bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])
        connectivity = kneighbors_graph(X, n_neighbors=params["n_neighbors"], include_self=False)
        connectivity = 0.5 * (connectivity + connectivity.T)

        n_cl = params["n_clusters"]

        algorithms = [
            ("MiniBatch KMeans", cluster.MiniBatchKMeans(n_clusters=n_cl, n_init="auto", random_state=params["random_state"])),
            ("Affinity Propagation", cluster.AffinityPropagation(damping=params["damping"], preference=params["preference"], random_state=params["random_state"])),
            ("MeanShift", cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)),
            ("Spectral", cluster.SpectralClustering(n_clusters=n_cl, eigen_solver="arpack", affinity="nearest_neighbors", random_state=params["random_state"])),
            ("Ward", cluster.AgglomerativeClustering(n_clusters=n_cl, linkage="ward", connectivity=connectivity)),
            ("Agglomerative (Avg)", cluster.AgglomerativeClustering(linkage="average", metric="cityblock", n_clusters=n_cl, connectivity=connectivity)),
            ("DBSCAN", cluster.DBSCAN(eps=params["eps"])),
            ("OPTICS", cluster.OPTICS(min_samples=params["min_samples"], xi=params["xi"], min_cluster_size=params["min_cluster_size"])),
            ("BIRCH", cluster.Birch(n_clusters=n_cl)),
            ("Gaussian Mixture (EM)", mixture.GaussianMixture(n_components=n_cl, covariance_type="full", random_state=params["random_state"])),
        ]

        if has_hdbscan:
            algorithms.append(("HDBSCAN", cluster.HDBSCAN(
                min_samples=params["hdbscan_min_samples"],
                min_cluster_size=params["hdbscan_min_cluster_size"],
                allow_single_cluster=params["allow_single_cluster"],
            )))

        # Metric-space competitors on raw features
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(X, metric="euclidean"))

        algorithms.extend([
            ("K-Medoids", KMedoidsClustering(n_clusters=n_cl)),
            ("Hierarchical (Avg)", HierarchicalClustering(n_clusters=n_cl, linkage="average")),
            ("K-Center", KCenterClustering(n_clusters=n_cl)),
        ])

        # MMD variants
        algorithms.extend([
            ("MMD GMM (Polynomial)", None),
            ("MMD GMM (Gaussian)", None),
        ])

        for alg_name, alg in algorithms:
            try:
                if alg_name == "MMD GMM (Polynomial)":
                    y_pred = _run_mmd(X, n_cl, "polynomial")
                elif alg_name == "MMD GMM (Gaussian)":
                    y_pred = _run_mmd(X, n_cl, "gaussian")
                elif alg_name == "K-Medoids":
                    y_pred = KMedoidsClustering(n_clusters=n_cl).fit_predict(
                        torch.as_tensor(X, dtype=torch.float64), dist_matrix=dist_matrix
                    )
                elif alg_name == "Hierarchical (Avg)":
                    y_pred = HierarchicalClustering(n_clusters=n_cl, linkage="average").fit_predict(
                        torch.as_tensor(X, dtype=torch.float64), dist_matrix=dist_matrix
                    )
                elif alg_name == "K-Center":
                    y_pred = KCenterClustering(n_clusters=n_cl).fit_predict(
                        torch.as_tensor(X, dtype=torch.float64)
                    )
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        alg.fit(X)
                    y_pred = alg.labels_.astype(int) if hasattr(alg, "labels_") else alg.predict(X)
            except Exception as e:
                print(f"  {alg_name} failed: {e}")
                continue

            if y_true is not None:
                ari = float(adjusted_rand_score(y_true, y_pred))
                ari_per_method.setdefault(alg_name, []).append(ari)
                print(f"  {alg_name:<26} ARI={ari:.4f}")
            else:
                print(f"  {alg_name:<26} (no labels)")

    # ── Aggregate and save ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary (mean ARI over labelled datasets)")
    print("=" * 60)

    results = {}
    for method_name, aris in ari_per_method.items():
        mean_ari = float(np.mean(aris))
        std_ari = float(np.std(aris))
        results[method_name] = {
            "ari_scores": aris,
            "mean_ari": mean_ari,
            "std_ari": std_ari,
        }
        print(f"  {method_name:<26} {mean_ari:.3f} ± {std_ari:.3f}")

    save_benchmark_results(
        benchmark_id="rd",
        config={
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "mmd_seed": MMD_SEED,
            "mmd_epochs": MMD_EPOCHS,
            "mmd_lr": MMD_LR,
            "cov_type": COV_TYPE,
        },
        results=results,
    )


if __name__ == "__main__":
    main()
