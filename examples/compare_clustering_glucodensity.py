#!/usr/bin/env python
"""
Example: Comparing clustering methods on real glucodensity (CGM) data.

Compares our MMD Gaussian Mixture Model against competitor clustering methods.
Since ground-truth labels are unavailable, evaluation uses:
  1. Silhouette score (internal clustering quality)
  2. Total Variation divergence between control and treatment groups (clinical relevance)
"""

import argparse
import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_glucodensity_temporal import (
    CONTROL_IDS,
    TREATMENT_IDS,
    build_training_representation,
    compute_sliding_windows,
    load_and_preprocess_cgm,
)
from src import GaussianKernel, GaussianMixtureModel
from src.competitors import (
    DBSCANClustering,
    HDBSCANClustering,
    HierarchicalClustering,
    KCenterClustering,
    KMedoidsClustering,
    ScikitFDAAgglomerative,
    ScikitFDAFuzzyCMeans,
    ScikitFDAKMeans,
)


# ---------------------------------------------------------------------------
# Patient-level posterior and group divergence utilities
# ---------------------------------------------------------------------------

def compute_patient_posteriors_from_labels(
    labels: np.ndarray,
    patient_ids: list,
    n_clusters: int,
) -> dict:
    """Convert hard cluster labels to per-patient posterior distributions."""
    posteriors = {}
    unique_patients = sorted(set(patient_ids))
    for pid in unique_patients:
        idx = [i for i, p in enumerate(patient_ids) if p == pid]
        if not idx:
            continue
        counts = np.bincount([labels[i] for i in idx], minlength=n_clusters).astype(float)
        counts /= counts.sum()
        posteriors[pid] = counts
    return posteriors


def compute_patient_posteriors_from_soft(
    soft_labels: np.ndarray,
    patient_ids: list,
) -> dict:
    """Average soft cluster posteriors per patient."""
    posteriors = {}
    unique_patients = sorted(set(patient_ids))
    for pid in unique_patients:
        idx = [i for i, p in enumerate(patient_ids) if p == pid]
        if not idx:
            continue
        posteriors[pid] = soft_labels[idx].mean(axis=0)
    return posteriors


def build_patient_level_features(
    X: torch.Tensor,
    patient_ids: list,
    mode: str,
) -> tuple[torch.Tensor, list]:
    """Aggregate window features into one feature vector per patient."""
    if mode == "window":
        return X, list(patient_ids)

    X_np = X.detach().cpu().numpy()
    unique_patients = sorted(set(patient_ids))
    patient_features = []

    for pid in unique_patients:
        idx = [i for i, p in enumerate(patient_ids) if p == pid]
        X_pid = X_np[idx]
        mean_feat = X_pid.mean(axis=0)

        if mode == "patient_mean":
            feat = mean_feat
        elif mode == "patient_meanstd":
            feat = np.concatenate([mean_feat, X_pid.std(axis=0)])
        else:
            raise ValueError(f"Unknown patient feature mode: {mode}")

        patient_features.append(feat)

    X_patient = torch.tensor(
        np.asarray(patient_features),
        device=X.device,
        dtype=X.dtype,
    )
    return X_patient, unique_patients


def expand_patient_soft_labels(
    patient_soft_labels: np.ndarray,
    patient_index_ids: list,
    window_patient_ids: list,
) -> np.ndarray:
    """Broadcast patient-level soft assignments back to all windows."""
    soft_by_patient = {
        pid: patient_soft_labels[i]
        for i, pid in enumerate(patient_index_ids)
    }
    return np.asarray([soft_by_patient[pid] for pid in window_patient_ids])


def compute_tv_divergence(posteriors: dict, control_ids: list, treatment_ids: list) -> float:
    """TV distance between mean control and treatment group posteriors."""
    ctrl = [posteriors[pid] for pid in control_ids if pid in posteriors]
    treat = [posteriors[pid] for pid in treatment_ids if pid in posteriors]
    if not ctrl or not treat:
        return float("nan")
    ctrl_mean = np.mean(ctrl, axis=0)
    treat_mean = np.mean(treat, axis=0)
    return 0.5 * float(np.sum(np.abs(ctrl_mean - treat_mean)))


def estimate_dbscan_eps(X_np: np.ndarray, min_samples: int, percentile: float = 90.0) -> float:
    """
    Estimate a good DBSCAN eps from the k-NN distance distribution.
    Sorts distances to the (min_samples)-th nearest neighbor and returns
    the given percentile — a standard data-adaptive heuristic for eps.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_np)
    distances, _ = nbrs.kneighbors(X_np)
    knn_dists = np.sort(distances[:, -1])  # distance to k-th neighbor, sorted
    return float(np.percentile(knn_dists, percentile))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare clustering methods on glucodensity data.")
    p.add_argument("--data-path", type=str, default="data/glucodensities/cgm_all_patients.csv")
    p.add_argument("--max-prop-missing", type=float, default=0.20)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--r-s", type=int, default=8, help="Cosine basis functions for intraday")
    p.add_argument("--n-time-bins", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.0, help="Gaussian kernel bandwidth; <= 0 uses median heuristic.")
    p.add_argument("--sigma-scale", type=float, default=1.0, help="Scale factor applied to the auto-selected bandwidth.")
    p.add_argument(
        "--sigma-max-samples",
        type=int,
        default=2048,
        help="Maximum number of windows used for the median-heuristic bandwidth estimate.",
    )
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-restarts", type=int, default=5, help="Random restarts; keep best by true MMD²")
    p.add_argument(
        "--assignment-mode",
        type=str,
        default="responsibility",
        choices=["kernel", "responsibility", "nearest_mean"],
        help="How to convert the fitted MMD mixture into hard/soft cluster assignments.",
    )
    p.add_argument(
        "--ours-unit",
        type=str,
        default="patient_mean",
        choices=["window", "patient_mean", "patient_meanstd"],
        help="Feature granularity used to fit the MMD method.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="all",
        choices=["all", "ours"],
        help="Run the full benchmark or only the MMD method for faster iteration.",
    )
    p.add_argument("--out-dir", type=str, default="paper/images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def resolve_gaussian_sigma(
    X: torch.Tensor,
    sigma: float,
    sigma_scale: float,
    max_samples: int,
) -> float:
    """Return the user-provided sigma or a median-heuristic bandwidth."""
    if sigma > 0:
        return float(sigma)

    with torch.no_grad():
        if X.shape[0] > max_samples:
            sample_idx = torch.randperm(X.shape[0], device=X.device)[:max_samples]
            X_sigma = X[sample_idx]
        else:
            X_sigma = X

        pairwise_dist = torch.cdist(X_sigma, X_sigma)
        iu = torch.triu_indices(pairwise_dist.shape[0], pairwise_dist.shape[1], offset=1)
        upper = pairwise_dist[iu[0], iu[1]]
        positive = upper[upper > 0]
        median_sigma = torch.median(positive).item() if positive.numel() > 0 else 1.0
    return max(float(sigma_scale) * float(median_sigma), 1e-6)


def get_mmd_assignments(
    model: GaussianMixtureModel,
    X: torch.Tensor,
    kernel: GaussianKernel,
    assignment_mode: str,
) -> np.ndarray:
    """Return soft assignments for the fitted MMD model."""
    with torch.no_grad():
        if assignment_mode == "kernel":
            soft = model.kernel_responsibilities(X, kernel)
        elif assignment_mode == "responsibility":
            soft = model.responsibilities(X)
        elif assignment_mode == "nearest_mean":
            distances = torch.cdist(X, model.mean.detach())
            labels = distances.argmin(dim=1)
            soft = torch.nn.functional.one_hot(labels, num_classes=model.num_components).to(dtype=X.dtype)
        else:
            raise ValueError(f"Unknown assignment mode: {assignment_mode}")
    return soft.cpu().numpy()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    dtype = torch.float64

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, args.data_path)
    out_dir = os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sections_dir = os.path.join(project_root, "paper", "sections")
    os.makedirs(sections_dir, exist_ok=True)

    K = args.n_components

    print("=" * 70)
    print(f"Glucodensity Clustering Comparison  (K={K})")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and preprocess
    # ------------------------------------------------------------------
    print("\n[1] Loading and preprocessing CGM data...")
    patient_data = load_and_preprocess_cgm(
        csv_path=csv_path,
        max_prop_missing=args.max_prop_missing,
        block_size=args.block_size,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 2. Sliding windows
    # ------------------------------------------------------------------
    print("\n[2] Building sliding-window curves...")
    curves, t_indices, patient_ids, window_days, _ = compute_sliding_windows(
        patient_data=patient_data,
        window_size=args.window_size,
        stride=args.window_stride,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 3. Basis projection (flat; no temporal binning for static clustering)
    # ------------------------------------------------------------------
    print("\n[3] Building representation...")
    rep = build_training_representation(
        curves=curves,
        t_indices=t_indices,
        n_time_bins=args.n_time_bins,
        r_s=args.r_s,
        device=device,
        dtype=dtype,
        verbose=True,
    )
    basis = rep["space_basis"]
    coeff_mean = rep["coeff_mean"]
    coeff_std = rep["coeff_std"]

    # Project all windows to coefficient space  (N, M_s)
    curves_tensor = torch.tensor(curves, device=device, dtype=dtype).unsqueeze(-1)  # (N, 288, 1)
    X_raw_coeffs = basis.project(curves_tensor)  # (N, M_s)
    X = (X_raw_coeffs - coeff_mean) / coeff_std  # normalized

    N, M = X.shape
    print(f"  Total windows: {N}, coeff dim: {M}")

    # ------------------------------------------------------------------
    # 4. Results containers
    # ------------------------------------------------------------------
    results = {}  # name -> {"silhouette": float, "tv": float, "ari": float, "labels": array}

    # When K=2, build per-window true binary labels (0=control, 1=treatment)
    control_set = set(CONTROL_IDS)
    true_window_labels = None
    if K == 2:
        true_window_labels = np.array(
            [0 if pid in control_set else 1 for pid in patient_ids], dtype=int
        )
        print(f"  K=2: ARI will be computed using control/treatment as ground truth.")

    # ------------------------------------------------------------------
    # 5. Our method: MMD Gaussian Mixture Model (multiple restarts)
    # ------------------------------------------------------------------
    X_ours, ours_index_ids = build_patient_level_features(X, patient_ids, args.ours_unit)
    resolved_sigma = resolve_gaussian_sigma(
        X_ours,
        sigma=args.sigma,
        sigma_scale=args.sigma_scale,
        max_samples=args.sigma_max_samples,
    )
    print(
        f"\n[4] Training MMD Gaussian Mixture Model "
        f"(K={K}, units={args.ours_unit}, N_train={X_ours.shape[0]}, "
        f"{args.epochs} epochs, {args.n_restarts} restarts, "
        f"sigma={resolved_sigma:.4f}, assignment={args.assignment_mode})..."
    )
    kernel = GaussianKernel(sigma=resolved_sigma)

    best_objective_val = float("inf")
    best_resp = None

    for restart in range(args.n_restarts):
        restart_seed = args.seed + restart
        torch.manual_seed(restart_seed)
        np.random.seed(restart_seed)

        model = GaussianMixtureModel(
            num_components=K,
            coeff_dim=X_ours.shape[1],
            basis=basis if args.ours_unit == "window" else None,
            covariance_type="diagonal",
            device=device,
            dtype=dtype,
        )
        model.initialize_from_data(X_ours, method="kmeans++")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

        for epoch in range(args.epochs):
            optimizer.zero_grad()
            mmd2, _ = model.compute_mmd2(X_ours, kernel, compute_const_term=False)
            mmd2.backward()
            optimizer.step()
            scheduler.step()
            if (epoch + 1) % 200 == 0:
                print(f"  Restart {restart+1}/{args.n_restarts}  Epoch {epoch+1}/{args.epochs}  MMD²={mmd2.item():.6f}")

        with torch.no_grad():
            objective_val, _ = model.compute_mmd2(X_ours, kernel, compute_const_term=False)
            objective_val = objective_val.item()
            resp_candidate = get_mmd_assignments(model, X_ours, kernel, args.assignment_mode)

        print(f"  Restart {restart+1}/{args.n_restarts}  MMD²(no const)={objective_val:.6f}")
        if objective_val < best_objective_val:
            best_objective_val = objective_val
            best_resp = resp_candidate

    print(f"  Best MMD²(no const)={best_objective_val:.6f}")
    if args.ours_unit == "window":
        resp = best_resp
        posteriors_ours = compute_patient_posteriors_from_soft(resp, patient_ids)
    else:
        resp = expand_patient_soft_labels(best_resp, ours_index_ids, patient_ids)
        posteriors_ours = {
            pid: best_resp[i]
            for i, pid in enumerate(ours_index_ids)
        }
    our_labels = resp.argmax(axis=1)

    sil = silhouette_score(X.cpu().numpy(), our_labels) if len(set(our_labels)) > 1 else float("nan")
    tv_ours = compute_tv_divergence(posteriors_ours, CONTROL_IDS, TREATMENT_IDS)
    ari_ours = adjusted_rand_score(true_window_labels, our_labels) if true_window_labels is not None else float("nan")
    results["Ours (MMD GMM)"] = {"silhouette": sil, "tv": tv_ours, "ari": ari_ours, "labels": our_labels}
    ari_str = f"  ARI={ari_ours:.4f}" if not np.isnan(ari_ours) else ""
    print(f"  Ours (MMD GMM)  silhouette={sil:.4f}  TV={tv_ours:.4f}{ari_str}")

    # ------------------------------------------------------------------
    # 6. Competitor methods
    # ------------------------------------------------------------------
    if args.methods == "all":
        print("\n[5] Evaluating competitor methods...")
        X_np = X.cpu().numpy()
        dbscan_min_samples = max(5, N // 100)  # at least 5, ~1% of data
        dbscan_eps = estimate_dbscan_eps(X_np, min_samples=dbscan_min_samples, percentile=90.0)
        hdbscan_min_cluster_size = max(5, N // 50)  # at least 5, ~2% of data
        print(f"  DBSCAN auto eps={dbscan_eps:.4f} (min_samples={dbscan_min_samples})")
        print(f"  HDBSCAN min_cluster_size={hdbscan_min_cluster_size}")

        # Share one pairwise distance matrix across precomputed-distance methods.
        print("  Pre-computing pairwise distance matrix...")
        from scipy.spatial.distance import pdist, squareform

        dist_matrix = squareform(pdist(X_np, metric="euclidean"))
        print(f"  Distance matrix shape: {dist_matrix.shape}  ({dist_matrix.nbytes / 1e6:.1f} MB)")

        competitors = [
            ("K-Medoids", KMedoidsClustering(n_clusters=K)),
            ("Hierarchical", HierarchicalClustering(n_clusters=K, linkage="average")),
            ("DBSCAN", DBSCANClustering(eps=dbscan_eps, min_samples=dbscan_min_samples)),
            ("HDBSCAN", HDBSCANClustering(min_cluster_size=hdbscan_min_cluster_size)),
            ("K-Center", KCenterClustering(n_clusters=K)),
        ]

        use_fda = False
        try:
            _ = ScikitFDAKMeans(n_clusters=K)
            competitors.extend([
                ("FDA K-Means", ScikitFDAKMeans(n_clusters=K)),
                ("FDA Fuzzy C-Means", ScikitFDAFuzzyCMeans(n_clusters=K)),
                ("FDA Agglomerative", ScikitFDAAgglomerative(n_clusters=K, linkage="average")),
            ])
            use_fda = True
        except Exception:
            print("  scikit-fda not available, skipping FDA methods.")

        _uses_dist_matrix = {"K-Medoids", "Hierarchical", "DBSCAN", "HDBSCAN"}

        for name, method in competitors:
            try:
                if "FDA" in name:
                    labels = method.fit_predict(X, basis=basis)
                elif name in _uses_dist_matrix:
                    labels = method.fit_predict(X, dist_matrix=dist_matrix)
                else:
                    labels = method.fit_predict(X)

                unique_labels = set(labels)
                # DBSCAN/HDBSCAN may assign -1 (noise); exclude noise for metrics
                valid_mask = labels >= 0
                n_valid = valid_mask.sum()
                n_clusters_found = len(unique_labels - {-1})

                if n_clusters_found < 2 or n_valid < 2:
                    print(f"  {name}: only {n_clusters_found} cluster(s) found, skipping metrics.")
                    results[name] = {"silhouette": float("nan"), "tv": float("nan"), "ari": float("nan"), "labels": labels}
                    continue

                sil = silhouette_score(X_np[valid_mask], labels[valid_mask])
                posteriors = compute_patient_posteriors_from_labels(labels, patient_ids, n_clusters_found)
                tv = compute_tv_divergence(posteriors, CONTROL_IDS, TREATMENT_IDS)
                ari = adjusted_rand_score(true_window_labels, labels) if true_window_labels is not None else float("nan")
                results[name] = {"silhouette": sil, "tv": tv, "ari": ari, "labels": labels}
                ari_str = f"  ARI={ari:.4f}" if not np.isnan(ari) else ""
                print(f"  {name}  silhouette={sil:.4f}  TV={tv:.4f}{ari_str}  ({n_clusters_found} clusters)")
            except Exception as e:
                print(f"  {name} failed: {e}")
                results[name] = {"silhouette": float("nan"), "tv": float("nan"), "ari": float("nan"), "labels": np.zeros(N)}

        del dist_matrix
        gc.collect()

    # ------------------------------------------------------------------
    # 7. Summary table
    # ------------------------------------------------------------------
    print("\n[6] Results summary:")
    show_ari = K == 2
    header = f"{'Method':<25} {'Silhouette':>12} {'TV Divergence':>14}"
    if show_ari:
        header += f" {'ARI':>10}"
    print(header)
    print("-" * (55 + (10 if show_ari else 0)))
    sorted_results = sorted(results.items(), key=lambda x: x[1]["tv"], reverse=True)
    for name, r in sorted_results:
        sil_str = f"{r['silhouette']:.4f}" if not np.isnan(r["silhouette"]) else "  N/A "
        tv_str = f"{r['tv']:.4f}" if not np.isnan(r["tv"]) else "  N/A "
        line = f"  {name:<23} {sil_str:>12} {tv_str:>14}"
        if show_ari:
            ari_str = f"{r['ari']:.4f}" if not np.isnan(r["ari"]) else "  N/A "
            line += f" {ari_str:>10}"
        print(line)

    # ------------------------------------------------------------------
    # 8. Plot: bar chart comparing methods
    # ------------------------------------------------------------------
    print("\n[7] Generating plots...")
    method_names = [n for n, _ in sorted_results]
    sil_vals = [r["silhouette"] for _, r in sorted_results]
    tv_vals = [r["tv"] for _, r in sorted_results]
    ari_vals = [r["ari"] for _, r in sorted_results] if K == 2 else None

    x = np.arange(len(method_names))
    width = 0.35

    n_panels = 3 if K == 2 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels + 1, 5))

    ax = axes[0]
    ax.bar(x, sil_vals, width * 2, color="steelblue", edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Internal Clustering Quality")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, tv_vals, width * 2, color="tomato", edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("TV Divergence (Control vs Treatment)")
    ax.set_title("Clinical Group Separation")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    if K == 2:
        ax = axes[2]
        ax.bar(x, ari_vals, width * 2, color="mediumseagreen", edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Adjusted Rand Index")
        ax.set_title("ARI vs Control/Treatment Labels")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Glucodensity Clustering Comparison (K={K}, N_windows={N})", fontsize=12
    )
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "glucodensity_clustering_comparison.pdf")
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    print(f"  Saved plot to {plot_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    # ------------------------------------------------------------------
    # 9. LaTeX table
    # ------------------------------------------------------------------
    table_str = "\\begin{table}[htpb]\n\\centering\n"
    table_str += "\\caption{Clustering comparison on glucodensity data. "
    table_str += "Silhouette measures internal cohesion; "
    table_str += "TV Divergence measures separation between control and treatment groups"
    if K == 2:
        table_str += "; ARI is computed against control/treatment binary labels (only valid for $K=2$)"
    table_str += ".}\n"
    table_str += "\\label{tab:glucodensity_clustering_comparison}\n"
    col_spec = "lccc" if K == 2 else "lcc"
    table_str += f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
    header_row = "Method & Silhouette & TV Divergence"
    if K == 2:
        header_row += " & ARI"
    table_str += header_row + " \\\\\n\\midrule\n"
    for name, r in sorted_results:
        sil_str = f"{r['silhouette']:.3f}" if not np.isnan(r["silhouette"]) else "--"
        tv_str = f"{r['tv']:.3f}" if not np.isnan(r["tv"]) else "--"
        row = f"{name} & {sil_str} & {tv_str}"
        if K == 2:
            ari_str = f"{r['ari']:.3f}" if not np.isnan(r["ari"]) else "--"
            row += f" & {ari_str}"
        table_str += row + " \\\\\n"
    table_str += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    table_path = os.path.join(sections_dir, "glucodensity_clustering_comparison_table.tex")
    with open(table_path, "w") as f:
        f.write(table_str)
    print(f"  Saved LaTeX table to {table_path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
