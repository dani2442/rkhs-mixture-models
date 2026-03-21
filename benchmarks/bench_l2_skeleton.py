#!/usr/bin/env python
"""
Benchmark: L²([0,1]; R^75) NTU skeleton clustering.

Loads NTU RGB+D skeleton sequences, filters to 5 action classes,
resamples to a fixed grid, projects onto L² cosine basis, and
evaluates clustering quality via ARI against action labels.
"""

import glob
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_ntu_skeleton import (
    resample_trajectory,
    skeleton_to_trajectory,
)
from examples.visualize_ntu_skeleton import (
    get_action_label,
    parse_skeleton_file,
)
from src import GaussianKernel, GaussianMixtureModel, L2CosineBasis, PolynomialKernel
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
DATA_ROOT = Path("data/ntu_rgbd_skeleton")
N_COMPONENTS = 5
GRID_SIZE = 64
D = 75  # 25 joints × 3 coords
T = 1.0
R = 10  # basis functions per spatial dim
SIGMA_KERNEL = 2.0
NUM_EPOCHS = 300
LR = 0.05
SEED = 42
N_RUNS = 5  # multiple subsample runs for mean/std
SAMPLES_PER_CLASS = 60

# 5 well-separated NTU action classes (diverse movement types):
# 010 = clapping, 023 = hand waving, 031 = pointing to something,
# 042 = salute, 050 = punching/slapping
TARGET_ACTIONS = ["010", "023", "031", "042", "050"]


def _load_by_action(skeleton_files: list[str]) -> dict[str, list]:
    """Group skeleton files by their action class code."""
    by_action: dict[str, list] = {}
    for fpath in skeleton_files:
        label = get_action_label(fpath)
        code = label.replace("Action ", "")
        by_action.setdefault(code, []).append(fpath)
    return by_action


def _load_subsample(by_action, target_actions, samples_per_class, grid_size, rng):
    """Load and resample a balanced subsample of trajectories."""
    trajectories = []
    labels = []

    for class_idx, action_code in enumerate(target_actions):
        files = by_action.get(action_code, [])
        if len(files) < samples_per_class:
            chosen = files
        else:
            chosen = rng.sample(files, samples_per_class)

        for fpath in chosen:
            try:
                frames = parse_skeleton_file(fpath)
                if len(frames) < 2:
                    continue
                traj = skeleton_to_trajectory(frames)
                traj = resample_trajectory(traj, grid_size)
                trajectories.append(traj)
                labels.append(class_idx)
            except Exception:
                continue

    return np.stack(trajectories, axis=0), np.array(labels)


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
    print(f"Benchmark: NTU Skeleton (K={N_COMPONENTS}, {N_RUNS} runs)")
    print("=" * 60)

    skeleton_files = sorted(
        glob.glob(str(DATA_ROOT / "**/*.skeleton"), recursive=True)
    )
    if not skeleton_files:
        print(f"No .skeleton files found under {DATA_ROOT}.")
        print("Run  examples/download_ntu_skeleton.py  first.")
        raise SystemExit(1)

    by_action = _load_by_action(skeleton_files)
    available = [a for a in TARGET_ACTIONS if a in by_action]
    if len(available) < N_COMPONENTS:
        print(
            f"Only {len(available)}/{N_COMPONENTS} target actions found. "
            f"Available: {sorted(by_action.keys())}"
        )
        raise SystemExit(1)

    print(f"  Target actions: {available}")
    for a in available:
        print(f"    Action {a}: {len(by_action[a])} files")

    device = torch.device("cpu")
    dtype = torch.float64

    ari_results: dict[str, list[float]] = {
        "MMD GMM (Gaussian)": [],
        "MMD GMM (Polynomial)": [],
        "K-Medoids": [],
        "Hierarchical (Avg)": [],
        "DBSCAN": [],
        "HDBSCAN": [],
        "K-Center": [],
    }

    use_fda = False
    try:
        from src.competitors import (
            ScikitFDAAgglomerative,
            ScikitFDAFuzzyCMeans,
            ScikitFDAKMeans,
        )
        ScikitFDAKMeans(n_clusters=N_COMPONENTS)
        ari_results.update({
            "FDA K-Means": [],
            "FDA Fuzzy C-Means": [],
            "FDA Agglomerative": [],
        })
        use_fda = True
    except Exception:
        print("  scikit-fda not available, skipping FDA methods.")

    for run in range(N_RUNS):
        run_seed = SEED + run
        rng = random.Random(run_seed)
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

        print(f"\n--- Run {run + 1}/{N_RUNS} (seed={run_seed}) ---")

        X_raw_np, true_labels = _load_subsample(
            by_action, available, SAMPLES_PER_CLASS, GRID_SIZE, rng
        )
        n_actual = X_raw_np.shape[0]
        print(f"  Loaded {n_actual} trajectories")

        # Center
        mean_traj = X_raw_np.mean(axis=0)
        X_centered = X_raw_np - mean_traj[None, :, :]
        X_raw = torch.tensor(X_centered, device=device, dtype=dtype)

        # Project
        basis = L2CosineBasis(T=T, R=R, grid_size=GRID_SIZE, d=D, device=device, dtype=dtype)
        X = basis.project(X_raw)

        # MMD GMM (Gaussian)
        kernel_g = GaussianKernel(sigma=SIGMA_KERNEL)
        labels = _train_mmd_gmm(X, basis, N_COMPONENTS, kernel_g, NUM_EPOCHS, LR)
        ari = adjusted_rand_score(true_labels, labels)
        ari_results["MMD GMM (Gaussian)"].append(ari)
        print(f"  MMD GMM (Gaussian)    ARI={ari:.4f}")

        # MMD GMM (Polynomial)
        torch.manual_seed(run_seed)
        kernel_p = PolynomialKernel(degree=2, c=1.0)
        labels = _train_mmd_gmm(X, basis, N_COMPONENTS, kernel_p, NUM_EPOCHS, LR)
        ari = adjusted_rand_score(true_labels, labels)
        ari_results["MMD GMM (Polynomial)"].append(ari)
        print(f"  MMD GMM (Polynomial)  ARI={ari:.4f}")

        # Competitors
        from scipy.spatial.distance import pdist, squareform
        X_np = X.cpu().numpy()
        dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        dbscan_min_samples = max(5, n_actual // 100)
        from examples.compare_clustering_glucodensity import estimate_dbscan_eps
        dbscan_eps = estimate_dbscan_eps(X_np, min_samples=dbscan_min_samples)
        hdbscan_min_cluster_size = max(5, n_actual // 50)

        competitors = [
            ("K-Medoids", KMedoidsClustering(n_clusters=N_COMPONENTS)),
            ("Hierarchical (Avg)", HierarchicalClustering(n_clusters=N_COMPONENTS, linkage="average")),
            ("DBSCAN", DBSCANClustering(eps=dbscan_eps, min_samples=dbscan_min_samples)),
            ("HDBSCAN", HDBSCANClustering(min_cluster_size=hdbscan_min_cluster_size)),
            ("K-Center", KCenterClustering(n_clusters=N_COMPONENTS)),
        ]

        if use_fda:
            competitors.extend([
                ("FDA K-Means", ScikitFDAKMeans(n_clusters=N_COMPONENTS)),
                ("FDA Fuzzy C-Means", ScikitFDAFuzzyCMeans(n_clusters=N_COMPONENTS)),
                ("FDA Agglomerative", ScikitFDAAgglomerative(n_clusters=N_COMPONENTS, linkage="average")),
            ])

        _uses_dist = {"K-Medoids", "Hierarchical (Avg)", "DBSCAN", "HDBSCAN"}

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
        benchmark_id="skeleton",
        config={
            "n_components": N_COMPONENTS,
            "grid_size": GRID_SIZE,
            "R": R,
            "sigma_kernel": SIGMA_KERNEL,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "n_runs": N_RUNS,
            "samples_per_class": SAMPLES_PER_CLASS,
            "target_actions": TARGET_ACTIONS,
            "seed": SEED,
        },
        results=results,
    )


if __name__ == "__main__":
    main()
