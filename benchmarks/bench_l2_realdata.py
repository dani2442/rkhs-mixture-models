#!/usr/bin/env python
"""
Benchmark: L² real-world functional data clustering.

Loads 6 real/classic functional datasets, projects each onto an L² cosine
basis, runs all applicable clustering methods, and reports ARI statistics
(mean ± std across the datasets).
"""

import os
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rdata
import torch
from scipy.spatial.distance import pdist, squareform
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
    GPmixClustering,
    HDBSCANClustering,
    HierarchicalClustering,
    KCenterClustering,
    KernelKGroupsClustering,
    KMedoidsClustering,
    ProjectedGMMEMFixedCovarianceClustering,
)
from examples.compare_clustering_glucodensity import estimate_dbscan_eps
from benchmarks.runner import save_benchmark_results

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
NUM_EPOCHS = 400
LR = 0.1
SEED = 42
DATA_DIR = Path("data")


# ──────────────────────────────────────────────────────────────────────
# Dataset loaders — each returns (X_np, y, K, name)
#   X_np: (n_samples, grid_size) or (n_samples, grid_size, d) float64
#   y:    (n_samples,)            int labels
#   K:    int                     number of clusters
# ──────────────────────────────────────────────────────────────────────

def _load_growth():
    from skfda.datasets import fetch_growth

    fd, y = fetch_growth(return_X_y=True)
    X = np.asarray(fd.data_matrix, dtype=float)
    if X.ndim == 3 and X.shape[-1] == 1:
        X = X[..., 0]
    y = np.asarray(y, dtype=int)
    return X, y, 2, "Growth"


def _load_weather():
    """Load the vector-valued Canadian Weather functional dataset."""
    from skfda.datasets import fetch_weather

    fd, y = fetch_weather(return_X_y=True)
    X = np.asarray(fd.data_matrix, dtype=float)
    y = np.asarray(y, dtype=int)
    return X, y, len(np.unique(y)), "Canadian Weather"


def _load_waveform():
    rng = np.random.default_rng(SEED)
    n_samples = 500
    t = np.arange(1, 22, dtype=float)

    def h1(x):
        return np.maximum(6.0 - np.abs(x - 11.0), 0.0)

    h1v, h2v, h3v = h1(t), h1(t - 4.0), h1(t + 4.0)

    y = rng.integers(0, 3, size=n_samples)
    u = rng.uniform(0.0, 1.0, size=n_samples)
    eps = rng.normal(0.0, 1.0, size=(n_samples, t.size))

    X = np.empty((n_samples, t.size), dtype=float)
    bases = [(h1v, h2v), (h1v, h3v), (h2v, h3v)]
    for i in range(n_samples):
        a, b = bases[y[i]]
        X[i] = u[i] * a + (1.0 - u[i]) * b + eps[i]

    return X, y, 3, "Waveform"


def _load_phoneme():
    url = "https://www.math.univ-toulouse.fr/~ferraty/SOFTWARES/NPFDA/npfda-phoneme.dat"
    df = pd.read_csv(url, sep=r"\s+", header=None, engine="python")
    X = df.iloc[:, :150].to_numpy(dtype=float)
    y = df.iloc[:, 150].to_numpy(dtype=int) - 1  # 0-indexed
    return X, y, 5, "Phoneme"


def _load_kneading():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "cfda_0.12.1.tar.gz"
    url = "https://cran.r-project.org/src/contrib/cfda_0.12.1.tar.gz"

    if not tar_path.exists():
        urllib.request.urlretrieve(url, tar_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("data/flours.rda")]
        extracted = tar.extractfile(members[0])
        rda_bytes = extracted.read()

    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as tmp:
        tmp.write(rda_bytes)
        tmp_path = Path(tmp.name)

    try:
        obj = rdata.read_rda(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    flours = obj["flours"]
    X = np.asarray(flours["data"], dtype=float)
    quality = flours["quality"]
    if isinstance(quality, pd.Categorical):
        y = quality.codes.astype(int)
    else:
        y = np.asarray(quality, dtype=int) - 1  # 0-indexed

    # R layout: (time, samples) → transpose to (samples, time)
    if X.shape[0] == len(np.asarray(flours["time"])):
        X = X.T

    return X, y, 3, "Kneading"


def _load_ecg200():
    from aeon.datasets import load_classification

    X, y = load_classification(name="ECG200", split=None)
    # aeon univariate: (n, 1, T) → (n, T)
    X = np.asarray(X, dtype=float)
    if X.ndim == 3 and X.shape[1] == 1:
        X = X[:, 0, :]
    y = np.asarray(y)
    # ECG200 labels are strings "-1"/"1"; map to 0/1
    unique_labels = np.unique(y)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in y], dtype=int)
    return X, y, 2, "ECG200"


# ──────────────────────────────────────────────────────────────────────
# Training helper (same pattern as bench_l2_synthetic.py)
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Noise reassignment for density-based methods
# ──────────────────────────────────────────────────────────────────────

def _reassign_noise(labels, X_np):
    """Reassign noise points (label < 0) to nearest non-noise cluster."""
    labels = np.array(labels)
    noise_mask = labels < 0
    if noise_mask.any():
        non_noise_idx = np.where(~noise_mask)[0]
        if len(non_noise_idx) >= 2 and len(set(labels[~noise_mask])) >= 2:
            from scipy.spatial.distance import cdist
            dists = cdist(X_np[noise_mask], X_np[non_noise_idx])
            labels[noise_mask] = labels[non_noise_idx[dists.argmin(axis=1)]]
        else:
            labels[:] = 0
    return labels


# ──────────────────────────────────────────────────────────────────────
# Dataset specs: (loader_fn, R)
# ──────────────────────────────────────────────────────────────────────

DATASET_SPECS = [
    (_load_growth,   15),
    (_load_weather,  15),
    (_load_waveform, 10),
    (_load_phoneme,  15),
    (_load_kneading, 15),
    (_load_ecg200,   15),
]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 60)
    print(f"Benchmark: L² Real Data ({len(DATASET_SPECS)} datasets)")
    print("=" * 60)

    # ── Method list ──────────────────────────────────────────────────
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
        "GPmix": [],
    }

    use_functional = False
    try:
        from src.competitors import (
            CurvclustClustering,
            FclustClustering,
            FunHDDCClustering,
            FunclustClustering,
            KCentresClustering,
            ScikitFDAAgglomerative,
            ScikitFDAFuzzyCMeans,
            ScikitFDAKMeans,
        )
        ScikitFDAKMeans(n_clusters=2)
        ari_results.update({
            "FDA K-Means": [],
            "FDA Fuzzy C-Means": [],
            "FDA Agglomerative": [],
            "Funclust": [],
            "FunHDDC": [],
            "fclust": [],
            "K-Centres": [],
            "Curvclust": [],
        })
        use_functional = True
    except Exception:
        print("functional-data baselines not available, skipping FDA/L2-specific methods.")

    # ── Evaluate over datasets ───────────────────────────────────────
    for loader_fn, R in DATASET_SPECS:
        t_ds_start = time.perf_counter()
        try:
            X_np, y_true, K, ds_name = loader_fn()
        except Exception as e:
            print(f"\n--- Skipping dataset (load failed: {e}) ---")
            continue

        if X_np.ndim == 2:
            X_np = X_np[..., None]
        if X_np.ndim != 3:
            print(
                f"\n--- Skipping {ds_name} "
                f"(expected shape (n, grid[, d]), got {X_np.shape}) ---"
            )
            continue

        grid_size = X_np.shape[1]
        d = X_np.shape[2]
        N = X_np.shape[0]
        print(
            f"\n--- {ds_name} "
            f"(n={N}, grid={grid_size}, d={d}, K={K}, R={R}) ---"
        )

        # Project onto L2 cosine basis
        basis = L2CosineBasis(
            T=1.0, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype
        )
        X_tensor = torch.tensor(X_np, device=device, dtype=dtype)
        X_coeffs = basis.project(X_tensor)

        # Normalize coefficients
        coeff_mean = X_coeffs.mean(dim=0, keepdim=True)
        coeff_std = X_coeffs.std(dim=0, keepdim=True).clamp(min=1e-8)
        X = (X_coeffs - coeff_mean) / coeff_std

        # Adaptive kernel sigma (median heuristic)
        X_np_coeffs = X.cpu().numpy()
        sigma = float(np.median(pdist(X_np_coeffs, metric="euclidean")))
        sigma = max(sigma, 1e-6)

        # MMD GMM (Gaussian)
        torch.manual_seed(SEED)
        kernel_g = GaussianKernel(sigma=sigma)
        labels = _train_mmd_gmm(X, basis, K, kernel_g, NUM_EPOCHS, LR)
        ari = adjusted_rand_score(y_true, labels)
        ari_results["MMD GMM (Gaussian)"].append(ari)
        print(f"  MMD GMM (Gaussian)    ARI={ari:.4f}")

        # MMD GMM (Polynomial)
        torch.manual_seed(SEED)
        kernel_p = PolynomialKernel(degree=2, c=1.0)
        labels = _train_mmd_gmm(X, basis, K, kernel_p, NUM_EPOCHS, LR)
        ari = adjusted_rand_score(y_true, labels)
        ari_results["MMD GMM (Polynomial)"].append(ari)
        print(f"  MMD GMM (Polynomial)  ARI={ari:.4f}")

        em_labels = ProjectedGMMEMFixedCovarianceClustering(
            n_clusters=K,
            n_init=3,
            max_iter=100,
            random_state=SEED,
        ).fit_predict(X)
        ari = adjusted_rand_score(y_true, em_labels)
        ari_results["Projected GMM-EM"].append(ari)
        print(f"  Projected GMM-EM      ARI={ari:.4f}")

        # Metric-space competitors
        dist_matrix = squareform(pdist(X_np_coeffs, metric="euclidean"))

        dbscan_min_samples = max(5, N // 100)
        dbscan_eps = estimate_dbscan_eps(X_np_coeffs, min_samples=dbscan_min_samples)
        hdbscan_min_cluster_size = max(5, N // 50)

        competitors = [
            ("K-Medoids", KMedoidsClustering(n_clusters=K)),
            ("Hierarchical (Avg)", HierarchicalClustering(n_clusters=K, linkage="average")),
            ("DBSCAN", DBSCANClustering(eps=dbscan_eps, min_samples=dbscan_min_samples)),
            ("HDBSCAN", HDBSCANClustering(min_cluster_size=hdbscan_min_cluster_size)),
            ("K-Center", KCenterClustering(n_clusters=K)),
            ("Kernel k-Groups", KernelKGroupsClustering(n_clusters=K)),
            ("GPmix", GPmixClustering(n_clusters=K, random_state=SEED)),
        ]

        if use_functional:
            competitors.extend([
                ("FDA K-Means", ScikitFDAKMeans(n_clusters=K)),
                ("FDA Fuzzy C-Means", ScikitFDAFuzzyCMeans(n_clusters=K)),
                ("FDA Agglomerative", ScikitFDAAgglomerative(n_clusters=K, linkage="average")),
                ("Funclust", FunclustClustering(n_clusters=K)),
                ("FunHDDC", FunHDDCClustering(n_clusters=K)),
                ("fclust", FclustClustering(n_clusters=K)),
                ("K-Centres", KCentresClustering(n_clusters=K)),
                ("Curvclust", CurvclustClustering(n_clusters=K)),
            ])

        _uses_dist = {"K-Medoids", "Hierarchical (Avg)", "DBSCAN", "HDBSCAN", "Kernel k-Groups"}

        for name, method in competitors:
            try:
                if name in {
                    "FDA K-Means",
                    "FDA Fuzzy C-Means",
                    "FDA Agglomerative",
                    "Funclust",
                    "FunHDDC",
                    "fclust",
                    "K-Centres",
                    "Curvclust",
                }:
                    pred = method.fit_predict(X, basis=basis)
                elif name in _uses_dist:
                    pred = method.fit_predict(X, dist_matrix=dist_matrix)
                elif name == "GPmix":
                    # GPmix does its own smoothing/projection, so it takes the
                    # raw curves on the grid rather than the cosine coefficients.
                    # Our re-implementation also handles vector-valued (d>1) curves.
                    pred = method.fit_predict(X_np)
                else:
                    pred = method.fit_predict(X)

                pred = _reassign_noise(pred, X_np_coeffs)
                ari = adjusted_rand_score(y_true, pred)
                ari_results[name].append(ari)
                print(f"  {name:<22} ARI={ari:.4f}")
            except Exception as e:
                print(f"  {name} failed: {e}")
                ari_results[name].append(float("nan"))

        t_ds_elapsed = time.perf_counter() - t_ds_start
        print(f"  [Dataset total: {t_ds_elapsed:.1f}s]")

    # ── Aggregate and save ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary (mean ARI over datasets)")
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
        benchmark_id="l2_real",
        config={
            "datasets": [fn.__name__ for fn, _ in DATASET_SPECS],
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "seed": SEED,
        },
        results=results,
    )


if __name__ == "__main__":
    main()
