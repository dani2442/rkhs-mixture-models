#!/usr/bin/env python
"""
Example: Comparing clustering methods on real glucodensity (CGM) data.

Compares our MMD Gaussian Mixture Model against competitor clustering methods.
Since ground-truth labels are unavailable, evaluation uses:
  1. Silhouette score (internal clustering quality)
  2. Total Variation divergence between control and treatment groups (clinical relevance)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_glucodensity_temporal import (
    CONTROL_IDS,
    TREATMENT_IDS,
    build_training_representation,
    compute_sliding_windows,
    load_and_preprocess_cgm,
)
from src import GaussianKernel, GaussianMixtureModel
from src.competitors.clustering import (
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


def compute_tv_divergence(posteriors: dict, control_ids: list, treatment_ids: list) -> float:
    """TV distance between mean control and treatment group posteriors."""
    ctrl = [posteriors[pid] for pid in control_ids if pid in posteriors]
    treat = [posteriors[pid] for pid in treatment_ids if pid in posteriors]
    if not ctrl or not treat:
        return float("nan")
    ctrl_mean = np.mean(ctrl, axis=0)
    treat_mean = np.mean(treat, axis=0)
    return 0.5 * float(np.sum(np.abs(ctrl_mean - treat_mean)))


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
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--r-s", type=int, default=8, help="Cosine basis functions for intraday")
    p.add_argument("--n-time-bins", type=int, default=16)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out-dir", type=str, default="paper/images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


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
    results = {}  # name -> {"silhouette": float, "tv": float, "labels": array}

    # ------------------------------------------------------------------
    # 5. Our method: MMD Gaussian Mixture Model
    # ------------------------------------------------------------------
    print(f"\n[4] Training MMD Gaussian Mixture Model (K={K}, {args.epochs} epochs)...")
    kernel = GaussianKernel(sigma=args.sigma)
    model = GaussianMixtureModel(
        num_components=K,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )
    model.initialize_from_data(X, method="kmeans++")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  MMD²={mmd2.item():.6f}")

    with torch.no_grad():
        resp = model.responsibilities(X).cpu().numpy()  # (N, K)
    our_labels = resp.argmax(axis=1)

    sil = silhouette_score(X.cpu().numpy(), our_labels) if len(set(our_labels)) > 1 else float("nan")
    posteriors_ours = compute_patient_posteriors_from_soft(resp, patient_ids)
    tv_ours = compute_tv_divergence(posteriors_ours, CONTROL_IDS, TREATMENT_IDS)
    results["Ours (MMD GMM)"] = {"silhouette": sil, "tv": tv_ours, "labels": our_labels}
    print(f"  Ours (MMD GMM)  silhouette={sil:.4f}  TV={tv_ours:.4f}")

    # ------------------------------------------------------------------
    # 6. Competitor methods
    # ------------------------------------------------------------------
    print("\n[5] Evaluating competitor methods...")
    competitors = [
        ("K-Medoids", KMedoidsClustering(n_clusters=K)),
        ("Hierarchical", HierarchicalClustering(n_clusters=K, linkage="average")),
        ("DBSCAN", DBSCANClustering(eps=1.5, min_samples=5)),
        ("HDBSCAN", HDBSCANClustering(min_cluster_size=5)),
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

    for name, method in competitors:
        try:
            if "FDA" in name:
                labels = method.fit_predict(X, basis=basis)
            else:
                labels = method.fit_predict(X)

            unique_labels = set(labels)
            # DBSCAN/HDBSCAN may assign -1 (noise); exclude noise for metrics
            valid_mask = labels >= 0
            n_valid = valid_mask.sum()
            n_clusters_found = len(unique_labels - {-1})

            if n_clusters_found < 2 or n_valid < 2:
                print(f"  {name}: only {n_clusters_found} cluster(s) found, skipping metrics.")
                results[name] = {"silhouette": float("nan"), "tv": float("nan"), "labels": labels}
                continue

            sil = silhouette_score(X.cpu().numpy()[valid_mask], labels[valid_mask])
            posteriors = compute_patient_posteriors_from_labels(labels, patient_ids, n_clusters_found)
            tv = compute_tv_divergence(posteriors, CONTROL_IDS, TREATMENT_IDS)
            results[name] = {"silhouette": sil, "tv": tv, "labels": labels}
            print(f"  {name}  silhouette={sil:.4f}  TV={tv:.4f}  ({n_clusters_found} clusters)")
        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = {"silhouette": float("nan"), "tv": float("nan"), "labels": np.zeros(N)}

    # ------------------------------------------------------------------
    # 7. Summary table
    # ------------------------------------------------------------------
    print("\n[6] Results summary:")
    print(f"{'Method':<25} {'Silhouette':>12} {'TV Divergence':>14}")
    print("-" * 55)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["tv"], reverse=True)
    for name, r in sorted_results:
        sil_str = f"{r['silhouette']:.4f}" if not np.isnan(r["silhouette"]) else "  N/A "
        tv_str = f"{r['tv']:.4f}" if not np.isnan(r["tv"]) else "  N/A "
        print(f"  {name:<23} {sil_str:>12} {tv_str:>14}")

    # ------------------------------------------------------------------
    # 8. Plot: bar chart comparing methods
    # ------------------------------------------------------------------
    print("\n[7] Generating plots...")
    method_names = [n for n, _ in sorted_results]
    sil_vals = [r["silhouette"] for _, r in sorted_results]
    tv_vals = [r["tv"] for _, r in sorted_results]

    x = np.arange(len(method_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(x, sil_vals, width * 2, color="steelblue", edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Internal Clustering Quality")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, tv_vals, width * 2, color="tomato", edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("TV Divergence (Control vs Treatment)")
    ax.set_title("Clinical Group Separation")
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
    table_str += "TV Divergence measures separation between control and treatment groups.}\n"
    table_str += "\\label{tab:glucodensity_clustering_comparison}\n"
    table_str += "\\begin{tabular}{lcc}\n\\toprule\n"
    table_str += "Method & Silhouette & TV Divergence \\\\\n\\midrule\n"
    for name, r in sorted_results:
        sil_str = f"{r['silhouette']:.3f}" if not np.isnan(r["silhouette"]) else "--"
        tv_str = f"{r['tv']:.3f}" if not np.isnan(r["tv"]) else "--"
        table_str += f"{name} & {sil_str} & {tv_str} \\\\\n"
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
