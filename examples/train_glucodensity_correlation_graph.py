#!/usr/bin/env python
"""
Temporal mixture model on glucose data using a *graph* representation.

The 24-hour day is modelled as a graph whose nodes are hours (0–23)
and whose edges reflect inter-hour correlations.  Each sliding-window
observation is a 24-dimensional *graph signal* (mean hourly glucose)
projected onto the Graph Laplacian eigenbasis.

Pipeline:
  1. Load CGM data (reuse preprocessing from train_glucodensity_temporal).
  2. Sliding windows → per-window hourly glucose means (24-dim signals)
     **and** the overall 24×24 correlation matrix.
  3. Threshold the mean correlation → adjacency → GraphLaplacianBasis.
  4. Project hourly signals onto graph Laplacian eigenbasis.
  5. Bin into temporal slices, normalise coefficients.
  6. Fit temporal Gaussian mixture (shared components, time-varying π(t)).
  7. Compute patient posteriors, divergence analysis, plots.
"""

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_glucodensity_temporal import (
    CONTROL_IDS,
    TREATMENT_IDS,
    compute_group_divergence_stats,
    load_and_preprocess_cgm,
    print_group_divergence_summary,
    run_experiment,
    interpolate_pi,
)
from src import GaussianKernel, GraphLaplacianBasis
from src.spaces import L2CosineBasis
from src.temporal_mixture import (
    BasisLogitsTimeWeights,
    NeuralODETimeWeights,
    TemporalGaussianMixtureModel,
)
from src.visualization import plot_cluster_probabilities_by_group


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

N_HOURS = 24
SLOTS_PER_HOUR = 12  # 288 / 24


def slots_to_hourly(day_matrix: np.ndarray) -> np.ndarray:
    """Average 288-slot daily curves into 24 hourly values.

    Args:
        day_matrix: (n_days, 288), may contain NaN.

    Returns:
        hourly: (n_days, 24).
    """
    n_days, n_slots = day_matrix.shape
    assert n_slots == 288
    reshaped = day_matrix.reshape(n_days, N_HOURS, SLOTS_PER_HOUR)
    return np.nanmean(reshaped, axis=2)


def compute_sliding_window_hourly(
    patient_data: dict,
    window_size: int = 7,
    stride: int = 1,
    min_valid_frac: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray, dict]:
    """Compute per-window mean hourly glucose (24-dim signals).

    Returns:
        hourly_signals: (N_windows, 24) mean hourly glucose per window
        t_indices: (N_windows,) centre-day of each window
        patient_ids: list of PtID per window
        window_days: (N_windows,) start day index
        patient_n_days: dict PtID → total valid days
    """
    all_signals = []
    all_t = []
    all_pid = []
    all_wday = []
    patient_n_days: dict = {}

    for pt_id, days_matrix in patient_data.items():
        n_days = days_matrix.shape[0]
        n_windows = n_days - window_size + 1
        if n_windows < 1:
            continue
        patient_n_days[pt_id] = n_days

        for w in range(0, n_windows, stride):
            window = days_matrix[w: w + window_size]
            hourly = slots_to_hourly(window)  # (window_size, 24)

            # Average across days (nanmean over valid days)
            with np.errstate(all="ignore"):
                avg_hourly = np.nanmean(hourly, axis=0)  # (24,)

            valid_count = np.sum(~np.isnan(hourly), axis=0)
            min_count = int(np.ceil(min_valid_frac * window_size))
            avg_hourly[valid_count < min_count] = np.nan

            nan_mask = np.isnan(avg_hourly)
            if nan_mask.all():
                continue
            if nan_mask.any():
                valid_idx = np.where(~nan_mask)[0]
                avg_hourly = np.interp(
                    np.arange(N_HOURS), valid_idx, avg_hourly[valid_idx],
                )

            t_val = w + (window_size - 1) / 2.0
            all_signals.append(avg_hourly)
            all_t.append(t_val)
            all_pid.append(pt_id)
            all_wday.append(w)

    hourly_signals = np.stack(all_signals, axis=0)
    t_indices = np.array(all_t)
    window_days = np.array(all_wday)

    if verbose:
        print(f"  Sliding windows (size={window_size}, stride={stride}):")
        print(f"    Total windows: {len(hourly_signals)}")
        print(f"    Patients contributing: {len(set(all_pid))}")
        print(f"    Glucose range: [{np.nanmin(hourly_signals):.1f}, "
              f"{np.nanmax(hourly_signals):.1f}]")

    return hourly_signals, t_indices, all_pid, window_days, patient_n_days


def compute_mean_correlation(
    patient_data: dict,
    window_size: int = 14,
    stride: int = 1,
    shrinkage: float = 0.1,
    min_valid_days: int = 3,
    verbose: bool = True,
) -> np.ndarray:
    """Compute the mean hourly correlation across all sliding windows.

    Used to build a single fixed graph for the GraphLaplacianBasis.

    Returns:
        mean_corr: (24, 24) correlation matrix.
    """
    all_corr = []

    for pt_id, days_matrix in patient_data.items():
        n_days = days_matrix.shape[0]
        n_windows = n_days - window_size + 1
        if n_windows < 1:
            continue

        for w in range(0, n_windows, stride):
            window = days_matrix[w: w + window_size]
            hourly = slots_to_hourly(window)

            valid_days = np.sum(~np.all(np.isnan(hourly), axis=1))
            if valid_days < min_valid_days:
                continue

            # Drop all-NaN rows, impute remaining NaNs
            valid_mask = ~np.all(np.isnan(hourly), axis=1)
            data = hourly[valid_mask].copy()
            if data.shape[0] < 2:
                continue
            col_means = np.nanmean(data, axis=0)
            for j in range(data.shape[1]):
                m = np.isnan(data[:, j])
                if m.any():
                    data[m, j] = col_means[j]

            corr = np.corrcoef(data, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            all_corr.append(corr)

    if not all_corr:
        if verbose:
            print("  Warning: no valid windows for correlation; using identity.")
        return np.eye(N_HOURS)

    mean_corr = np.mean(all_corr, axis=0)

    # Shrinkage
    I = np.eye(N_HOURS)
    mean_corr = (1.0 - shrinkage) * mean_corr + shrinkage * I

    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(mean_corr)
    eigvals = np.maximum(eigvals, 0.0)
    mean_corr = (eigvecs * eigvals) @ eigvecs.T

    if verbose:
        n_edges_50 = int((np.abs(mean_corr) > 0.5).sum() - N_HOURS) // 2
        print(f"  Mean correlation computed from {len(all_corr)} windows")
        print(f"  Edges (|corr| > 0.5): {n_edges_50}")

    return mean_corr


def threshold_to_adjacency(
    corr: np.ndarray,
    threshold: float = 0.3,
    absolute: bool = True,
) -> np.ndarray:
    """Threshold a correlation matrix to produce a binary adjacency matrix.

    Args:
        corr: (n, n) correlation matrix.
        threshold: Minimum |correlation| to keep an edge.
        absolute: If True, use absolute correlation for thresholding.

    Returns:
        adjacency: (n, n) binary symmetric matrix with zero diagonal.
    """
    vals = np.abs(corr) if absolute else corr
    adj = (vals >= threshold).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


# ---------------------------------------------------------------------------
# Temporal binning (reused from train_glucodensity_correlation.py idea)
# ---------------------------------------------------------------------------


def build_temporal_coefficients(
    coeffs: np.ndarray,
    t_indices: np.ndarray,
    n_time_bins: int,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = True,
) -> dict:
    """Bin pre-projected coefficient vectors into temporal slices.

    Returns a dict with keys:
        X_time, t_grid, mask, t_min_days, t_max_days,
        coeff_mean, coeff_std, coeff_dim, L_t, n_samples
    """
    N, M = coeffs.shape

    t_min = float(t_indices.min())
    t_max = float(t_indices.max())
    if t_max - t_min < 1e-9:
        t_max = t_min + 1.0

    bin_edges = np.linspace(t_min, t_max + 1e-9, n_time_bins + 1)
    bin_assignments = np.clip(
        np.digitize(t_indices, bin_edges) - 1, 0, n_time_bins - 1
    )

    bins: dict = {}
    for b in range(n_time_bins):
        idx = np.where(bin_assignments == b)[0]
        if len(idx) > 0:
            bins[b] = coeffs[idx]

    valid_bins = sorted(bins.keys())
    if len(valid_bins) < 2:
        raise ValueError("Need at least 2 non-empty temporal bins")

    bin_sizes = [len(bins[b]) for b in valid_bins]
    max_size = max(bin_sizes)

    slices, mask_rows, centers = [], [], []
    for b in valid_bins:
        n_b = len(bins[b])
        rng = np.random.RandomState(42 + b)
        perm = rng.permutation(n_b)
        data_b = bins[b][perm]
        padded = np.zeros((max_size, M), dtype=np.float64)
        padded[:n_b] = data_b
        slices.append(padded)
        m = np.zeros(max_size, dtype=bool)
        m[:n_b] = True
        mask_rows.append(m)
        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2.0)

    X_time_np = np.stack(slices, axis=0)
    mask_np = np.stack(mask_rows, axis=0)
    t_grid_days = np.array(centers)
    t_grid_norm = (t_grid_days - t_min) / (t_max - t_min)

    X_time = torch.tensor(X_time_np, device=device, dtype=dtype)
    t_grid = torch.tensor(t_grid_norm, device=device, dtype=dtype)
    mask = torch.tensor(mask_np, device=device)

    valid_coeffs = torch.cat(
        [X_time[l][mask[l]] for l in range(X_time.shape[0])], dim=0
    )
    coeff_mean = valid_coeffs.mean(dim=0, keepdim=True)
    coeff_std = valid_coeffs.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_time = (X_time - coeff_mean) / coeff_std

    if verbose:
        L_t = X_time.shape[0]
        print(f"  Temporal binning: {n_time_bins} requested, {L_t} non-empty")
        print(f"  Day range: [{t_min:.1f}, {t_max:.1f}]")
        print(f"  Bin sizes: min={min(bin_sizes)}, max={max(bin_sizes)}, "
              f"median={int(np.median(bin_sizes))}")
        print(f"  Padded to n_max={max_size}, coeff_dim={M}")

    return {
        "X_time": X_time,
        "t_grid": t_grid,
        "mask": mask,
        "t_min_days": t_min,
        "t_max_days": t_max,
        "coeff_mean": coeff_mean.squeeze(0),
        "coeff_std": coeff_std.squeeze(0),
        "coeff_dim": M,
        "L_t": int(X_time.shape[0]),
        "n_samples": int(X_time.shape[1]),
    }


def estimate_sigma_median_heuristic(
    X_time: torch.Tensor,
    mask: torch.Tensor,
    max_points: int = 2000,
) -> float:
    all_coeffs = torch.cat(
        [X_time[l][mask[l]] for l in range(X_time.shape[0])], dim=0
    )
    if all_coeffs.shape[0] < 2:
        return 1.0
    n_sub = min(max_points, all_coeffs.shape[0])
    idx = torch.randperm(all_coeffs.shape[0])[:n_sub]
    sub = all_coeffs[idx]
    dists = torch.cdist(sub, sub)
    positive = dists[dists > 0]
    if positive.numel() == 0:
        return 1.0
    return float(torch.median(positive).item())


# ---------------------------------------------------------------------------
# Patient posteriors (graph-signal version)
# ---------------------------------------------------------------------------


def compute_all_patient_posteriors_graph(
    patient_data: dict,
    t_indices_all: np.ndarray,
    patient_ids_all: list,
    window_days_all: np.ndarray,
    window_size: int,
    min_valid_frac: float,
    model: TemporalGaussianMixtureModel,
    basis: GraphLaplacianBasis,
    coeff_mean: torch.Tensor,
    coeff_std: torch.Tensor,
    t_grid: torch.Tensor,
    t_min_days: float,
    t_max_days: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[dict, dict, dict]:
    """Compute per-patient posteriors from hourly graph signals."""

    with torch.no_grad():
        pi_t = model.time_weight_model().cpu().numpy()
    t_grid_np = t_grid.cpu().numpy()
    day_range = max(t_max_days - t_min_days, 1e-9)

    patient_indices: dict = {}
    for i, pid in enumerate(patient_ids_all):
        patient_indices.setdefault(pid, []).append(i)

    patient_posteriors: dict = {}
    patient_time_norm: dict = {}
    patient_time_days: dict = {}

    for pid, idx_list in patient_indices.items():
        idx = np.array(idx_list)
        t_vals_abs = t_indices_all[idx]
        days = window_days_all[idx]

        # Recompute hourly signals for this patient's windows
        days_matrix = patient_data[pid]
        signals = []
        for w_day in days:
            w = int(w_day)
            window = days_matrix[w: w + window_size]
            hourly = slots_to_hourly(window)
            with np.errstate(all="ignore"):
                avg = np.nanmean(hourly, axis=0)
            nan_mask = np.isnan(avg)
            if nan_mask.any():
                valid_idx_h = np.where(~nan_mask)[0]
                if valid_idx_h.size > 0:
                    avg = np.interp(np.arange(N_HOURS), valid_idx_h, avg[valid_idx_h])
                else:
                    avg = np.zeros(N_HOURS)
            signals.append(avg)

        signals_t = torch.tensor(np.stack(signals), device=device, dtype=dtype)
        x_coeffs = basis.project(signals_t)
        x_coeffs = (x_coeffs - coeff_mean.unsqueeze(0)) / coeff_std.unsqueeze(0)

        t_vals_norm = np.clip((t_vals_abs - t_min_days) / day_range, 0.0, 1.0)
        pi_at_t_np = interpolate_pi(pi_t, t_grid_np, t_vals_norm)
        pi_at_t = torch.tensor(pi_at_t_np, device=device, dtype=dtype)

        with torch.no_grad():
            posteriors = model.responsibilities(x_coeffs, pi_at_t)

        patient_posteriors[pid] = posteriors.cpu().numpy()
        patient_time_norm[pid] = t_vals_norm
        patient_time_days[pid] = days

    return patient_posteriors, patient_time_norm, patient_time_days


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_graph_model_summary(
    history: list[float],
    pi_t_np: np.ndarray,
    recon_means_np: np.ndarray,
    t_grid_days_np: np.ndarray,
    adjacency_np: np.ndarray,
    out_dir: str,
    model_name: str,
    show: bool,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    K = pi_t_np.shape[1]
    use_weeks = float(np.max(t_grid_days_np)) > 30.0
    t_axis = t_grid_days_np / 7.0 if use_weeks else t_grid_days_np
    t_label = "Treatment week" if use_weeks else "Treatment day"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Loss
    axes[0].plot(history, lw=2, color="tab:blue")
    axes[0].set_title(f"{model_name}: MMD² training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MMD²(avg_t)")
    axes[0].grid(alpha=0.3)

    # (2) π(t)
    for k in range(K):
        axes[1].plot(t_axis, pi_t_np[:, k], lw=2, label=f"Cluster {k+1}")
    axes[1].set_title(f"{model_name}: cluster weights over time")
    axes[1].set_xlabel(t_label)
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # (3) Mean hourly glucose per cluster
    hours = np.arange(N_HOURS)
    for k in range(K):
        axes[2].plot(hours, recon_means_np[k], lw=2, label=f"Cluster {k+1}")
    axes[2].set_title(f"{model_name}: mean hourly glucose")
    axes[2].set_xlabel("Hour of day")
    axes[2].set_ylabel("Glucose (mg/dL)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Graph-based temporal glucodensity mixture", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "glucodensity_graph_summary.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_graph_adjacency_and_spectrum(
    adjacency_np: np.ndarray,
    eigenvalues_np: np.ndarray,
    corr_np: np.ndarray,
    out_dir: str,
    show: bool,
) -> str:
    """Plot the graph adjacency, correlation heatmap, and eigenvalue spectrum."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Correlation heatmap
    im0 = axes[0].imshow(corr_np, cmap="RdBu_r", vmin=-1, vmax=1,
                          aspect="equal", origin="lower")
    axes[0].set_title("Mean hourly correlation")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Hour")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Adjacency
    axes[1].imshow(adjacency_np, cmap="Greys", aspect="equal", origin="lower")
    n_edges = int(adjacency_np.sum()) // 2
    axes[1].set_title(f"Thresholded adjacency ({n_edges} edges)")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Hour")

    # Eigenvalue spectrum
    axes[2].bar(range(len(eigenvalues_np)), eigenvalues_np,
                color="steelblue", alpha=0.8)
    axes[2].set_title("Laplacian eigenvalue spectrum")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("λ")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "glucodensity_graph_structure.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temporal mixture model on hourly glucose graph signals."
    )
    p.add_argument("--data-path", type=str,
                    default="data/glucodensities/cgm_all_patients.csv")
    p.add_argument("--max-prop-missing", type=float, default=0.20)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--window-size", type=int, default=7,
                    help="Sliding window size (days) for hourly signals")
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--corr-window-size", type=int, default=14,
                    help="Window size for computing the mean correlation (graph)")
    p.add_argument("--corr-shrinkage", type=float, default=0.1)
    p.add_argument("--corr-threshold", type=float, default=0.3,
                    help="Threshold |corr| for edge creation")
    p.add_argument("--model-type", choices=["basis", "ode"], default="ode")
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--num-eigenvectors", type=int, default=15,
                    help="Number of Laplacian eigenvectors for graph basis")
    p.add_argument("--graph-alpha", type=float, default=0.1,
                    help="Regularisation α for graph inner product")
    p.add_argument("--n-time-bins", type=int, default=16)
    p.add_argument("--r-pi", type=int, default=6)
    p.add_argument("--ode-hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.,
                    help="Kernel sigma (0 = median heuristic)")
    p.add_argument("--sigma-mult", type=float, default=1.0)
    p.add_argument("--analysis-time-bins", type=int, default=12)
    p.add_argument("--out-dir", type=str, default="paper/images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    dtype = torch.float64

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, args.data_path)
    out_dir = os.path.join(project_root, args.out_dir)

    print("=" * 72)
    print("Glucodensity Temporal Mixture: Graph Laplacian on Hourly Signals")
    print("=" * 72)

    # 1) Data
    print("\n[Step 1] Loading and preprocessing data...")
    patient_data = load_and_preprocess_cgm(
        csv_path=csv_path,
        max_prop_missing=args.max_prop_missing,
        block_size=args.block_size,
        verbose=True,
    )

    # 2) Mean correlation → graph
    print("\n[Step 2] Computing mean hourly correlation for graph construction...")
    mean_corr = compute_mean_correlation(
        patient_data=patient_data,
        window_size=args.corr_window_size,
        stride=args.window_stride,
        shrinkage=args.corr_shrinkage,
        verbose=True,
    )
    adjacency_np = threshold_to_adjacency(
        mean_corr, threshold=args.corr_threshold, absolute=True,
    )
    n_edges = int(adjacency_np.sum()) // 2
    print(f"  Threshold={args.corr_threshold} → {n_edges} edges "
          f"(density={2*n_edges/(N_HOURS*(N_HOURS-1)):.2%})")

    # 3) GraphLaplacianBasis
    print("\n[Step 3] Building GraphLaplacianBasis...")
    adjacency_t = torch.tensor(adjacency_np, device=device, dtype=dtype)
    basis = GraphLaplacianBasis.from_adjacency(
        adjacency=adjacency_t,
        alpha=args.graph_alpha,
        num_eigenvectors=args.num_eigenvectors,
        device=device,
        dtype=dtype,
    )
    print(f"  Eigenvectors: {args.num_eigenvectors}, coeff_dim={basis.coeff_dim}")
    print(f"  Eigenvalue range: [{basis.eigenvalues.min():.4f}, "
          f"{basis.eigenvalues.max():.4f}]")

    # 4) Sliding-window hourly signals
    print("\n[Step 4] Computing sliding-window hourly glucose signals...")
    hourly_signals, t_indices, patient_ids, window_days, _ = \
        compute_sliding_window_hourly(
            patient_data=patient_data,
            window_size=args.window_size,
            stride=args.window_stride,
            verbose=True,
        )

    # 5) Project onto graph basis
    print("\n[Step 5] Projecting hourly signals onto graph Laplacian basis...")
    signals_t = torch.tensor(hourly_signals, device=device, dtype=dtype)
    coeffs_all = basis.project(signals_t).detach().cpu().numpy()  # (N, R)
    print(f"  Projected shape: {coeffs_all.shape}")

    # 6) Temporal binning + normalisation
    print("\n[Step 6] Temporal binning...")
    rep = build_temporal_coefficients(
        coeffs=coeffs_all,
        t_indices=t_indices,
        n_time_bins=args.n_time_bins,
        device=device,
        dtype=dtype,
        verbose=True,
    )

    X_time = rep["X_time"]
    t_grid = rep["t_grid"]
    mask = rep["mask"]

    # 7) Kernel
    sigma_auto = estimate_sigma_median_heuristic(X_time, mask)
    sigma_base = args.sigma if args.sigma > 0 else sigma_auto
    sigma = max(1e-8, sigma_base * args.sigma_mult)
    print(f"  Sigma: base={sigma_base:.4f}, final={sigma:.4f}")
    kernel = GaussianKernel(sigma=sigma)

    # 8) Time-weight model
    if args.model_type == "basis":
        time_basis = L2CosineBasis(
            T=1.0, R=args.r_pi, grid_size=rep["L_t"],
            d=1, device=device, dtype=dtype,
        )
        time_weight_model = BasisLogitsTimeWeights(
            basis_matrix=time_basis.Phi,
            num_components=args.n_components,
            device=device, dtype=dtype,
        )
        model_name = "GraphBasis"
    else:
        time_weight_model = NeuralODETimeWeights(
            t_grid=t_grid,
            num_components=args.n_components,
            hidden_dim=args.ode_hidden,
            device=device, dtype=dtype,
        )
        model_name = "GraphNeuralODE"

    # 9) Train
    print(f"\n[Step 7] Training {model_name}...")
    model, history, pi_t = run_experiment(
        name=model_name,
        time_weight_model=time_weight_model,
        X_time=X_time,
        kernel=kernel,
        n_components=args.n_components,
        coeff_dim=rep["coeff_dim"],
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
        mask=mask,
        verbose=True,
    )

    # 10) Patient posteriors + divergence
    print("\n[Step 8] Computing patient posteriors and group divergence...")
    patient_posteriors, patient_time_norm, patient_time_days = \
        compute_all_patient_posteriors_graph(
            patient_data=patient_data,
            t_indices_all=t_indices,
            patient_ids_all=patient_ids,
            window_days_all=window_days,
            window_size=args.window_size,
            min_valid_frac=0.5,
            model=model,
            basis=basis,
            coeff_mean=rep["coeff_mean"],
            coeff_std=rep["coeff_std"],
            t_grid=t_grid,
            t_min_days=rep["t_min_days"],
            t_max_days=rep["t_max_days"],
            device=device,
            dtype=dtype,
        )

    div_stats = compute_group_divergence_stats(
        patient_posteriors=patient_posteriors,
        patient_time_days=patient_time_days,
        control_ids=CONTROL_IDS,
        treatment_ids=TREATMENT_IDS,
        n_time_bins=args.analysis_time_bins,
    )
    print_group_divergence_summary(model_name, div_stats)

    # 11) Reconstruct mean signals for plotting
    with torch.no_grad():
        means_coeff = model.mean.cpu()
    means_orig = means_coeff * rep["coeff_std"].cpu() + rep["coeff_mean"].cpu()
    recon_means = basis.reconstruct(means_orig)  # (K, 24)
    recon_means_np = recon_means.detach().numpy()

    t_np = t_grid.detach().cpu().numpy()
    day_range = rep["t_max_days"] - rep["t_min_days"]
    t_grid_days_np = t_np * day_range + rep["t_min_days"]

    # 12) Plots
    print("\n[Step 9] Generating plots...")
    plot_graph_adjacency_and_spectrum(
        adjacency_np=adjacency_np,
        eigenvalues_np=basis.eigenvalues.detach().cpu().numpy(),
        corr_np=mean_corr,
        out_dir=out_dir,
        show=not args.no_show,
    )

    plot_graph_model_summary(
        history=history,
        pi_t_np=pi_t.cpu().numpy(),
        recon_means_np=recon_means_np,
        t_grid_days_np=t_grid_days_np,
        adjacency_np=adjacency_np,
        out_dir=out_dir,
        model_name=model_name,
        show=not args.no_show,
    )

    plot_cluster_probabilities_by_group(
        patient_posteriors=patient_posteriors,
        patient_time_days=patient_time_days,
        control_ids=CONTROL_IDS,
        treatment_ids=TREATMENT_IDS,
        n_time_bins=args.analysis_time_bins,
        include_difference_panel=False,
        out_dir=out_dir,
        show=not args.no_show,
    )

    n_ctrl = sum(1 for pid in patient_posteriors if pid in CONTROL_IDS)
    n_treat = sum(1 for pid in patient_posteriors if pid in TREATMENT_IDS)
    print("\n" + "=" * 72)
    print("Done")
    print("=" * 72)
    print(f"  Model: {model_name}")
    print(f"  Graph: {N_HOURS} nodes, {n_edges} edges, "
          f"{args.num_eigenvectors} eigenvectors")
    print(f"  Final MMD²(avg_t): {history[-1]:.6f}")
    print(
        f"  Divergence score: {div_stats['increasing_score']:.6f} | "
        f"final_TV={div_stats['final_sep']:.6f} | delta_TV={div_stats['delta_sep']:.6f}"
    )
    print(f"  Patients with posteriors: {len(patient_posteriors)} "
          f"(control: {n_ctrl}, treatment: {n_treat})")


if __name__ == "__main__":
    main()
