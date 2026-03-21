#!/usr/bin/env python
"""
Benchmark: L²(Graphs) molecular clustering.

Uses the ENZYMES TUDataset (600 graphs, 6 classes) from PyTorch Geometric.
Graphs are embedded via WLHashFingerprint, projected onto a DCT basis,
and clustered.  ARI is computed against true enzyme class labels.
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset

from src import DiscreteCosineBasis, GaussianKernel, GaussianMixtureModel, PolynomialKernel
from src.spaces.graph_embedding import WLHashFingerprint
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
DATASET_NAME = "ENZYMES"
N_COMPONENTS = 6  # 6 enzyme classes
WL_DIM = 128
WL_RADIUS = 2
DCT_R = 32  # keep top-32 DCT coefficients
SIGMA_KERNEL = 2.0
NUM_EPOCHS = 400
LR = 0.1
SEED = 42
N_RUNS = 5
BATCH_SIZE = 64


def _embed_dataset(dataset, encoder, batch_size):
    """Embed all graphs using WLHashFingerprint in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="WLHash embedding"):
        batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batch = Batch.from_data_list(batch_data)
        emb = encoder(batch)
        all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)


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
    print(f"Benchmark: Graph ({DATASET_NAME}, K={N_COMPONENTS}, {N_RUNS} runs)")
    print("=" * 60)

    device = torch.device("cpu")
    dtype = torch.float64

    # ── Load dataset ────────────────────────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "TU")

    print(f"\n[1] Loading {DATASET_NAME} dataset...")
    dataset = TUDataset(root=data_dir, name=DATASET_NAME, use_node_attr=True)
    print(f"  {len(dataset)} graphs, {dataset.num_classes} classes")

    true_labels = np.array([dataset[i].y.item() for i in range(len(dataset))])
    print(f"  Class distribution: {dict(zip(*np.unique(true_labels, return_counts=True)))}")

    # ── Embed graphs ────────────────────────────────────────────────
    print(f"\n[2] Embedding graphs via WLHash (dim={WL_DIM}, radius={WL_RADIUS})...")
    encoder = WLHashFingerprint(dim=WL_DIM, radius=WL_RADIUS, l2_normalize=True)
    embeddings = _embed_dataset(dataset, encoder, BATCH_SIZE)
    print(f"  Embedding shape: {embeddings.shape}")

    # ── Project onto DCT basis ──────────────────────────────────────
    n_raw = embeddings.shape[1]
    R = min(DCT_R, n_raw)
    basis = DiscreteCosineBasis(n=n_raw, R=R, device=device, dtype=dtype)

    X_all = basis.project(embeddings.to(device=device, dtype=dtype))
    print(f"  DCT coefficients: {X_all.shape}")

    # ── Run benchmarks ──────────────────────────────────────────────
    ari_results: dict[str, list[float]] = {
        "MMD GMM (Gaussian)": [],
        "MMD GMM (Polynomial)": [],
        "K-Medoids": [],
        "Hierarchical (Avg)": [],
        "DBSCAN": [],
        "HDBSCAN": [],
        "K-Center": [],
    }

    for run in range(N_RUNS):
        run_seed = SEED + run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

        print(f"\n--- Run {run + 1}/{N_RUNS} (seed={run_seed}) ---")

        X = X_all.clone()

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

        dbscan_min_samples = max(5, len(dataset) // 100)
        from examples.compare_clustering_glucodensity import estimate_dbscan_eps

        dbscan_eps = estimate_dbscan_eps(X_np, min_samples=dbscan_min_samples)
        hdbscan_min_cluster_size = max(5, len(dataset) // 50)

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
        ]

        _uses_dist = {"K-Medoids", "Hierarchical (Avg)", "DBSCAN", "HDBSCAN"}

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
        benchmark_id="graph",
        config={
            "dataset": DATASET_NAME,
            "n_components": N_COMPONENTS,
            "wl_dim": WL_DIM,
            "wl_radius": WL_RADIUS,
            "dct_R": DCT_R,
            "sigma_kernel": SIGMA_KERNEL,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "n_runs": N_RUNS,
            "seed": SEED,
        },
        results=results,
    )


if __name__ == "__main__":
    main()
