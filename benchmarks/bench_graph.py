#!/usr/bin/env python
"""
Benchmark: L²(Graphs) molecular clustering on MUTAG.

Pipeline:
  1. Embed graphs with WLHashFingerprint (L2-normalised).
  2. Build the WL Gram matrix (cosine similarity = WL subtree kernel).
  3. Kernel-PCA to a fixed-dimensional Euclidean space that preserves WL kernel
     distances.  Euclidean distances in this space equal WL-kernel distances.
  4. Select Gaussian kernel bandwidth via an MMD-proxy sweep:
     - For each candidate σ, train MMD-GMM for a short burst and record the
       relative decrease in the MMD² proxy loss.  The σ that gives the greatest
       relative improvement captures the most cluster structure.
  5. Run all clustering methods.  Competitors receive the precomputed cosine
     distance matrix on raw WL fingerprints (the natural metric for molecular
     similarity) so they operate in the same kernel space.
"""

import os
import sys

import numpy as np
import torch
# Limit PyTorch thread pool to avoid busy-wait spinning on small tensors
torch.set_num_threads(2)

from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset

from src import CanonicalBasis, GaussianKernel, GaussianMixtureModel, PolynomialKernel
from src.spaces.graph_embedding import WLHashFingerprint
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
DATASET_NAME = "MUTAG"
N_COMPONENTS = 2          # mutagenic vs non-mutagenic
WL_DIM = 512              # WL hash fingerprint dimension
WL_RADIUS = 3             # WL iterations (captures 3-hop substructures)
KPCA_DIM = 32             # kernel-PCA output dimension
NUM_EPOCHS = 500
N_RESTARTS = 5            # multiple restarts; pick best by final MMD² loss
LR = 0.05
SEED = 42
N_RUNS = 5
BATCH_SIZE = 64

# MMD-proxy sigma search
SIGMA_CANDIDATES = [0.1, 0.5, 1.0, 2.0, 4.0]
MMD_PROXY_EPOCHS = 60


def _embed_dataset(dataset, encoder, batch_size):
    """Embed all graphs using WLHashFingerprint in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="WLHash embedding"):
        batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batch = Batch.from_data_list(batch_data)
        emb = encoder(batch)
        all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)


def _kernel_pca(gram: torch.Tensor, R: int) -> torch.Tensor:
    """
    Kernel PCA: map the N×N Gram matrix to N×R Euclidean features.

    Euclidean distances in the output space equal kernel distances,
    so this gives the "canonical" low-dim embedding for the kernel.
    """
    n = gram.shape[0]
    # Centre the Gram matrix: K_c = H K H,  H = I - (1/n) 11^T
    one_n = torch.ones(n, n, device=gram.device, dtype=gram.dtype) / n
    H = torch.eye(n, device=gram.device, dtype=gram.dtype) - one_n
    K_c = H @ gram @ H

    # Eigendecomposition (eigh returns ascending order)
    vals, vecs = torch.linalg.eigh(K_c)      # vals (n,), vecs (n, n)
    top_vals = vals[-R:].flip(0).clamp(min=0) # (R,) descending, clamp negatives
    top_vecs = vecs[:, -R:].flip(-1)          # (n, R)
    features = top_vecs * top_vals.sqrt().unsqueeze(0)  # (n, R)
    return features


def _median_heuristic(X_np: np.ndarray) -> float:
    """Median pairwise L2 distance heuristic for kernel bandwidth."""
    from scipy.spatial.distance import pdist
    n = X_np.shape[0]
    if n > 300:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 300, replace=False)
        X_sub = X_np[idx]
    else:
        X_sub = X_np
    dists = pdist(X_sub, metric="euclidean")
    return float(np.median(dists))


def _train_mmd_gmm(X, basis, n_components, kernel, num_epochs, lr, base_seed, n_restarts):
    """
    Train with multiple restarts; return labels from the run with highest
    assignment crispness (mean max-responsibility), skipping collapsed solutions.

    Each restart initialises from a fresh sklearn K-means run (better than
    pure k-means++ seed selection because it gives per-cluster covariance init).
    """
    from sklearn.cluster import KMeans

    X_np = X.cpu().numpy()
    best_labels = None
    best_crispness = -1.0

    for r in range(n_restarts):
        seed = base_seed + r * 100
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GaussianMixtureModel(
            num_components=n_components,
            coeff_dim=X.shape[1],
            basis=basis,
            covariance_type="diagonal",
            device=X.device,
            dtype=X.dtype,
        )

        # Better init: run K-means, use per-cluster centers and variances
        km = KMeans(n_clusters=n_components, n_init=5, max_iter=100, random_state=seed)
        km.fit(X_np)
        centers = torch.tensor(km.cluster_centers_, device=X.device, dtype=X.dtype)

        with torch.no_grad():
            model._mean_coeffs.copy_(centers)
            # Per-cluster variance (tighter than global)
            for k in range(n_components):
                mask = torch.tensor(km.labels_ == k, device=X.device)
                if mask.sum() > 1:
                    cv = X[mask].var(dim=0).clamp(min=1e-6)
                else:
                    cv = X.var(dim=0).clamp(min=1e-6) * 0.5
                model._log_var[k] = torch.log(cv)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        for _ in range(num_epochs):
            optimizer.zero_grad()
            mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
            mmd2.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            resp = model.responsibilities(X)          # (n, K)
            labels = resp.argmax(dim=1).cpu().numpy()
            pi = model.pi
            # Exclude collapsed solutions (any component with weight < 5%)
            if pi.min().item() > 0.05:
                crispness = resp.max(dim=1).values.mean().item()
                if crispness > best_crispness:
                    best_crispness = crispness
                    best_labels = labels

    if best_labels is None:
        # All restarts collapsed — fall back to most recent labels
        best_labels = labels

    return best_labels


def _mmd_proxy_score(X, basis, n_components, sigma, n_epochs, lr, seed):
    """
    Train MMD-GMM for `n_epochs` and return the **negative relative improvement**
    in the MMD² proxy loss (lower returned value = better bandwidth).

    Relative improvement = (loss_before - loss_after) / max(|loss_before|, ε).
    - Too-small σ: kernel too noisy, little improvement → score ≈ 0
    - Too-large σ: kernel too flat, nothing to learn → score ≈ 0
    - Right σ: kernel sees cluster structure → large improvement
    """
    torch.manual_seed(seed)
    kernel = GaussianKernel(sigma=sigma)
    model = GaussianMixtureModel(
        num_components=n_components,
        coeff_dim=X.shape[1],
        basis=basis,
        covariance_type="diagonal",
        device=X.device,
        dtype=X.dtype,
    )
    model.initialize_from_data(X, method="kmeans++")

    with torch.no_grad():
        loss_init, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        loss_init = loss_init.item()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()

    with torch.no_grad():
        loss_final, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        loss_final = loss_final.item()

    # Absolute improvement: how much MMD² actually decreased.
    # Relative improvement (/ |loss_init|) can blow up when |loss_init| ≈ 0
    # (small σ → cross and mix terms both vanish → loss_init ≈ 0 → degenerate ratio).
    absolute_improvement = max(0.0, loss_init - loss_final)
    return -absolute_improvement  # negate so that min-selection picks best


def _select_sigma_mmd_proxy(X, basis, n_components, candidates, n_epochs, lr, seed):
    """Use relative MMD² improvement as a proxy to select bandwidth."""
    print("  [MMD proxy] sweeping sigma candidates …", flush=True)
    median_sigma = _median_heuristic(X.cpu().numpy())
    print(f"  [MMD proxy] median heuristic σ = {median_sigma:.4f}", flush=True)

    all_candidates = sorted(set(candidates + [round(median_sigma, 4)]))

    best_sigma = all_candidates[0]
    best_score = float("inf")
    for sigma in all_candidates:
        score = _mmd_proxy_score(X, basis, n_components, sigma, n_epochs, lr, seed)
        abs_improvement = -score  # negate back to "higher = better"
        print(f"  [MMD proxy]   σ={sigma:.4f}  abs_improvement={abs_improvement:.6f}", flush=True)
        if score < best_score:
            best_score = score
            best_sigma = sigma

    print(f"  [MMD proxy] → selected σ={best_sigma:.4f}", flush=True)
    return best_sigma


def main():
    print("=" * 60)
    print(f"Benchmark: Graph ({DATASET_NAME}, K={N_COMPONENTS}, {N_RUNS} runs)")
    print("=" * 60)

    device = torch.device("cpu")
    dtype = torch.float64

    # ── Load dataset ────────────────────────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "TU")

    print(f"\n[1] Loading {DATASET_NAME} dataset...", flush=True)
    dataset = TUDataset(root=data_dir, name=DATASET_NAME, use_node_attr=True)
    print(f"  {len(dataset)} graphs, {dataset.num_classes} classes", flush=True)

    true_labels = np.array([dataset[i].y.item() for i in range(len(dataset))])
    print(f"  Class distribution: {dict(zip(*np.unique(true_labels, return_counts=True)))}", flush=True)

    # ── Embed graphs ────────────────────────────────────────────────
    print(f"\n[2] Embedding graphs via WLHash (dim={WL_DIM}, radius={WL_RADIUS})...", flush=True)
    encoder = WLHashFingerprint(dim=WL_DIM, radius=WL_RADIUS, l2_normalize=True)
    embeddings = _embed_dataset(dataset, encoder, BATCH_SIZE)  # (N, WL_DIM), L2-normalised
    print(f"  Embedding shape: {embeddings.shape}", flush=True)

    # ── Competitor distance: cosine on raw L2-normalised fingerprints ──
    # Cosine similarity of L2-normalised WL fingerprints IS the WL subtree kernel.
    print("\n[3] Pre-computing WL kernel (cosine) distance matrix...", flush=True)
    emb_np = embeddings.cpu().numpy()
    cos_sim = emb_np @ emb_np.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    dist_matrix_cos = 1.0 - cos_sim   # (N, N) cosine distance ∈ [0, 2]

    # ── Kernel PCA (Gram matrix → Euclidean features) ──────────────
    # Euclidean distance in KPCA space = WL kernel distance.
    print(f"\n[4] Kernel PCA → {KPCA_DIM}-dim Euclidean features...", flush=True)
    gram = torch.from_numpy(cos_sim + 1.0).to(device=device, dtype=dtype)
    # Convert cosine *distance* to cosine *similarity* Gram: sim = 1 - dist → gram = cos_sim
    # Use the similarity Gram (not distance) for KPCA
    gram_sim = torch.from_numpy(cos_sim).to(device=device, dtype=dtype)
    R = min(KPCA_DIM, len(dataset) - 1)
    X_all = _kernel_pca(gram_sim, R)         # (N, R)
    basis = CanonicalBasis(n=R, R=R, device=device, dtype=dtype)
    print(f"  KPCA features: {X_all.shape}", flush=True)

    # ── MMD-proxy sigma search (done once) ──────────────────────────
    print("\n[5] Selecting Gaussian kernel bandwidth via MMD proxy...", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    best_sigma = _select_sigma_mmd_proxy(
        X_all, basis, N_COMPONENTS,
        SIGMA_CANDIDATES, MMD_PROXY_EPOCHS, LR, SEED
    )

    # ── DBSCAN eps from cosine distance matrix ──────────────────────
    dbscan_min_samples = max(5, len(dataset) // 50)
    k = dbscan_min_samples
    knn_dists = np.sort(np.partition(dist_matrix_cos, k, axis=1)[:, k])
    dbscan_eps = float(np.percentile(knn_dists, 90.0))
    hdbscan_min_cluster_size = max(5, len(dataset) // 30)

    # ── Run benchmarks ──────────────────────────────────────────────
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

    for run in range(N_RUNS):
        run_seed = SEED + run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

        print(f"\n--- Run {run + 1}/{N_RUNS} (seed={run_seed}) ---", flush=True)

        X = X_all.clone()

        # MMD GMM (Gaussian) with proxy-selected sigma + multiple restarts
        kernel_g = GaussianKernel(sigma=best_sigma)
        labels = _train_mmd_gmm(X, basis, N_COMPONENTS, kernel_g,
                                 NUM_EPOCHS, LR, run_seed, N_RESTARTS)
        ari = adjusted_rand_score(true_labels, labels)
        ari_results["MMD GMM (Gaussian)"].append(ari)
        print(f"  MMD GMM (Gaussian)    ARI={ari:.4f}", flush=True)

        # MMD GMM (Polynomial)
        kernel_p = PolynomialKernel(degree=2, c=1.0)
        labels = _train_mmd_gmm(X, basis, N_COMPONENTS, kernel_p,
                                 NUM_EPOCHS, LR, run_seed, N_RESTARTS)
        ari = adjusted_rand_score(true_labels, labels)
        ari_results["MMD GMM (Polynomial)"].append(ari)
        print(f"  MMD GMM (Polynomial)  ARI={ari:.4f}", flush=True)

        em_labels = ProjectedGMMEMFixedCovarianceClustering(
            n_clusters=N_COMPONENTS,
            n_init=3,
            max_iter=100,
            random_state=run_seed,
        ).fit_predict(X)
        ari = adjusted_rand_score(true_labels, em_labels)
        ari_results["Projected GMM-EM"].append(ari)
        print(f"  Projected GMM-EM      ARI={ari:.4f}", flush=True)

        # ── Competitors (cosine distance on raw WL fingerprints) ────
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
                    pred = method.fit_predict(X, dist_matrix=dist_matrix_cos)
                else:
                    pred = method.fit_predict(X)  # K-Center: Euclidean on KPCA features
                ari = adjusted_rand_score(true_labels, pred)
                ari_results[name].append(ari)
                print(f"  {name:<22} ARI={ari:.4f}", flush=True)
            except Exception as e:
                print(f"  {name} failed: {e}", flush=True)
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
            "kpca_dim": KPCA_DIM,
            "sigma_kernel": best_sigma,
            "sigma_selection": "mmd_proxy_relative_improvement",
            "num_epochs": NUM_EPOCHS,
            "n_restarts": N_RESTARTS,
            "lr": LR,
            "n_runs": N_RUNS,
            "seed": SEED,
            "competitor_distance": "cosine_wl_fingerprints",
        },
        results=results,
    )


if __name__ == "__main__":
    main()
