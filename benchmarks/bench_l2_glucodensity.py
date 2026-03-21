#!/usr/bin/env python
"""
Benchmark: L²([0,T]; L²) glucodensity (CGM) clustering.

Uses real CGM data, sliding-window glucodensity curves, and K=2
(control vs treatment).  ARI is computed against patient group labels.
Multiple random restarts for stochastic methods give mean/std.
"""

import gc
import os
import sys

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_glucodensity_temporal import (
    CONTROL_IDS,
    TREATMENT_IDS,
    build_training_representation,
    compute_sliding_windows,
    load_and_preprocess_cgm,
)
from examples.compare_clustering_glucodensity import (
    build_patient_level_features,
    compute_patient_posteriors_from_labels,
    compute_patient_posteriors_from_soft,
    estimate_dbscan_eps,
    expand_patient_soft_labels,
    resolve_gaussian_sigma,
)
from src import GaussianKernel, GaussianMixtureModel, PolynomialKernel
from src.competitors.clustering import (
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
DATA_PATH = "data/glucodensities/cgm_all_patients.csv"
K = 2
R_S = 8
N_TIME_BINS = 16
WINDOW_SIZE = 4
WINDOW_STRIDE = 1
BLOCK_SIZE = 4
EPOCHS = 600
LR = 0.01
N_RESTARTS = 5
SEED = 42
OURS_UNIT = "patient_mean"
ASSIGNMENT_MODE = "responsibility"


def _train_mmd_gmm_restart(X, basis, K, kernel, epochs, lr, seed, assignment_mode):
    """Single restart of MMD GMM training.  Returns (objective, soft_labels)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GaussianMixtureModel(
        num_components=K,
        coeff_dim=X.shape[1],
        basis=basis,
        covariance_type="diagonal",
        device=X.device,
        dtype=X.dtype,
    )
    model.initialize_from_data(X, method="kmeans++")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        obj, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        if assignment_mode == "responsibility":
            soft = model.responsibilities(X).cpu().numpy()
        else:
            dists = torch.cdist(X, model.mean.detach())
            labels = dists.argmin(dim=1)
            soft = (
                torch.nn.functional.one_hot(labels, num_classes=K)
                .to(dtype=X.dtype)
                .cpu()
                .numpy()
            )
    return obj.item(), soft


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cpu")
    dtype = torch.float64

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, DATA_PATH)

    print("=" * 60)
    print(f"Benchmark: Glucodensity (K={K})")
    print("=" * 60)

    # ── Data loading ────────────────────────────────────────────────
    patient_data = load_and_preprocess_cgm(
        csv_path=csv_path,
        max_prop_missing=0.20,
        block_size=BLOCK_SIZE,
        verbose=True,
    )

    curves, t_indices, patient_ids, window_days, _ = compute_sliding_windows(
        patient_data=patient_data,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        verbose=True,
    )

    rep = build_training_representation(
        curves=curves,
        t_indices=t_indices,
        n_time_bins=N_TIME_BINS,
        r_s=R_S,
        device=device,
        dtype=dtype,
        verbose=True,
    )
    basis = rep["space_basis"]
    coeff_mean = rep["coeff_mean"]
    coeff_std = rep["coeff_std"]

    curves_tensor = torch.tensor(curves, device=device, dtype=dtype).unsqueeze(-1)
    X_raw_coeffs = basis.project(curves_tensor)
    X = (X_raw_coeffs - coeff_mean) / coeff_std

    N, M = X.shape
    print(f"  Total windows: {N}, coeff dim: {M}")

    control_set = set(CONTROL_IDS)
    true_window_labels = np.array(
        [0 if pid in control_set else 1 for pid in patient_ids], dtype=int
    )

    ari_results: dict[str, list[float]] = {}

    # ── MMD GMM (Gaussian) — multiple restarts ──────────────────────
    for kernel_name, kernel_cls in [
        ("MMD GMM (Gaussian)", lambda sigma: GaussianKernel(sigma=sigma)),
        ("MMD GMM (Polynomial)", lambda _: PolynomialKernel(degree=2, c=1.0)),
    ]:
        aris = []
        for restart in range(N_RESTARTS):
            X_ours, ours_index_ids = build_patient_level_features(
                X, patient_ids, OURS_UNIT
            )
            sigma = resolve_gaussian_sigma(
                X_ours, sigma=0.0, sigma_scale=1.0, max_samples=2048
            )
            kernel = kernel_cls(sigma)

            obj, soft = _train_mmd_gmm_restart(
                X_ours,
                basis=None,
                K=K,
                kernel=kernel,
                epochs=EPOCHS,
                lr=LR,
                seed=SEED + restart,
                assignment_mode=ASSIGNMENT_MODE,
            )
            resp = expand_patient_soft_labels(soft, ours_index_ids, patient_ids)
            labels = resp.argmax(axis=1)
            ari = adjusted_rand_score(true_window_labels, labels)
            aris.append(ari)
            print(f"  {kernel_name} restart {restart + 1}/{N_RESTARTS}  ARI={ari:.4f}")

        ari_results[kernel_name] = aris

    # ── Competitors ─────────────────────────────────────────────────
    X_np = X.cpu().numpy()
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(X_np, metric="euclidean"))

    dbscan_min_samples = max(5, N // 100)
    # Try decreasing percentiles until DBSCAN produces ≥ 2 clusters (90th often collapses to 1)
    dbscan_eps = None
    for _pct in (50, 60, 70, 80, 90):
        _candidate = estimate_dbscan_eps(X_np, min_samples=dbscan_min_samples, percentile=_pct)
        _test_labels = DBSCANClustering(eps=_candidate, min_samples=dbscan_min_samples).fit_predict(
            torch.as_tensor(X_np, dtype=torch.float64), dist_matrix=dist_matrix
        )
        _valid = _test_labels[_test_labels >= 0]
        if len(_valid) >= 2 and len(set(_valid)) >= 2:
            dbscan_eps = _candidate
            print(f"  DBSCAN eps={_candidate:.4f} (percentile={_pct})")
            break
    if dbscan_eps is None:
        dbscan_eps = estimate_dbscan_eps(X_np, min_samples=dbscan_min_samples, percentile=50)
        print(f"  DBSCAN: no valid eps found; falling back to 50th percentile eps={dbscan_eps:.4f}")
    hdbscan_min_cluster_size = max(2, N // 100)

    competitors = [
        ("K-Medoids", KMedoidsClustering(n_clusters=K)),
        (
            "Hierarchical (Avg)",
            HierarchicalClustering(n_clusters=K, linkage="average"),
        ),
        ("DBSCAN", DBSCANClustering(eps=dbscan_eps, min_samples=dbscan_min_samples)),
        ("HDBSCAN", HDBSCANClustering(min_cluster_size=hdbscan_min_cluster_size)),
        ("K-Center", KCenterClustering(n_clusters=K)),
    ]

    use_fda = False
    try:
        from src.competitors.clustering import (
            ScikitFDAAgglomerative,
            ScikitFDAFuzzyCMeans,
            ScikitFDAKMeans,
        )

        ScikitFDAKMeans(n_clusters=K)
        competitors.extend(
            [
                ("FDA K-Means", ScikitFDAKMeans(n_clusters=K)),
                ("FDA Fuzzy C-Means", ScikitFDAFuzzyCMeans(n_clusters=K)),
                (
                    "FDA Agglomerative",
                    ScikitFDAAgglomerative(n_clusters=K, linkage="average"),
                ),
            ]
        )
        use_fda = True
    except Exception:
        print("  scikit-fda not available, skipping FDA methods.")

    _uses_dist = {"K-Medoids", "Hierarchical (Avg)", "DBSCAN", "HDBSCAN"}

    for name, method in competitors:
        try:
            if "FDA" in name:
                labels = method.fit_predict(X, basis=basis)
            elif name in _uses_dist:
                labels = method.fit_predict(X, dist_matrix=dist_matrix)
            else:
                labels = method.fit_predict(X)

            # For density-based methods, noise points (label=-1) get reassigned to
            # nearest non-noise cluster so ARI can be computed on all points.
            labels = np.array(labels)
            noise_mask = labels < 0
            if noise_mask.any():
                non_noise_idx = np.where(~noise_mask)[0]
                if len(non_noise_idx) >= 2 and len(set(labels[~noise_mask])) >= 2:
                    from scipy.spatial.distance import cdist
                    X_np_arr = X.cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
                    dists = cdist(X_np_arr[noise_mask], X_np_arr[non_noise_idx])
                    labels[noise_mask] = labels[non_noise_idx[dists.argmin(axis=1)]]
                else:
                    # All noise or single cluster — assign everyone to cluster 0
                    labels[:] = 0

            ari = adjusted_rand_score(true_window_labels, labels)
            ari_results[name] = [ari]
            print(f"  {name:<22} ARI={ari:.4f}")
        except Exception as e:
            print(f"  {name} failed: {e}")
            ari_results[name] = [float("nan")]

    del dist_matrix
    gc.collect()

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
        benchmark_id="glucodensity",
        config={
            "K": K,
            "r_s": R_S,
            "n_time_bins": N_TIME_BINS,
            "window_size": WINDOW_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "n_restarts": N_RESTARTS,
            "seed": SEED,
        },
        results=results,
    )


if __name__ == "__main__":
    main()
