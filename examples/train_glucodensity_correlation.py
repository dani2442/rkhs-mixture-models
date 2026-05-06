#!/usr/bin/env python
"""
Temporal mixture model on hourly glucose *correlation matrices*.

Instead of clustering intraday glucose curves in L^2, this script
clusters the running 24x24 correlation matrices between hours of the
day in the Hilbert space Sym(24) equipped with the Frobenius inner
product.

Pipeline:
  1. Load CGM data (reuse preprocessing from train_glucodensity_temporal).
  2. Sliding windows → per-window 24x24 correlation matrices.
  3. Project to Sym(24) via SymmetricMatrixBasis (scaled vech, opt. PCA).
  4. Bin into temporal slices, normalize coefficients.
  5. Fit temporal Gaussian mixture (shared components, time-varying π(t)).
  6. Compute patient posteriors, divergence analysis, plots.
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
from src import GaussianKernel, SymmetricMatrixBasis
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
    with np.errstate(all="ignore"):
        return np.nanmean(reshaped, axis=2)


def _correlation_from_hourly(
    hourly: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Shrinkage-regularised correlation matrix from (n_days, 24).

    Returns a PSD 24x24 matrix.
    """
    # Drop all-NaN rows
    valid = ~np.all(np.isnan(hourly), axis=1)
    data = hourly[valid].copy()

    if data.shape[0] < 2:
        return np.eye(N_HOURS)

    # Impute remaining NaN with column means
    col_means = np.nanmean(data, axis=0)
    for j in range(data.shape[1]):
        mask = np.isnan(data[:, j])
        if mask.any():
            data[mask, j] = col_means[j]

    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    # Shrinkage towards identity
    I = np.eye(N_HOURS)
    corr_reg = (1.0 - shrinkage) * corr + shrinkage * I

    # Project to PSD (clip negative eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(corr_reg)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs * eigvals) @ eigvecs.T


def compute_sliding_window_correlations(
    patient_data: dict,
    window_size: int = 14,
    stride: int = 1,
    shrinkage: float = 0.1,
    min_valid_days: int = 3,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray, dict]:
    """Compute per-window 24x24 correlation matrices.

    For each sliding window of ``window_size`` days:
      1. Convert each day's 288 slots → 24 hourly averages.
      2. Compute sample correlation (with shrinkage) from the daily vectors.

    Returns:
        corr_matrices: (N_windows, 24, 24)
        t_indices: (N_windows,) centre-day of each window
        patient_ids: list of patient IDs per window
        window_days: (N_windows,) start day index
        patient_n_days: dict PtID → number of valid days
    """
    all_corr = []
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
            hourly = slots_to_hourly(window)

            # Require enough non-all-NaN days
            valid_days = np.sum(~np.all(np.isnan(hourly), axis=1))
            if valid_days < min_valid_days:
                continue

            corr = _correlation_from_hourly(hourly, shrinkage=shrinkage)
            t_val = w + (window_size - 1) / 2.0

            all_corr.append(corr)
            all_t.append(t_val)
            all_pid.append(pt_id)
            all_wday.append(w)

    corr_matrices = np.stack(all_corr, axis=0)
    t_indices = np.array(all_t)
    window_days = np.array(all_wday)

    if verbose:
        print(f"  Sliding-window correlations (size={window_size}, stride={stride}):")
        print(f"    Total windows: {len(corr_matrices)}")
        print(f"    Patients contributing: {len(set(all_pid))}")

    return corr_matrices, t_indices, all_pid, window_days, patient_n_days


# ---------------------------------------------------------------------------
# Temporal binning (generic for coefficient vectors)
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

    # Normalize (valid entries only)
    valid_coeffs = torch.cat([X_time[l][mask[l]] for l in range(X_time.shape[0])], dim=0)
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
    all_coeffs = torch.cat([X_time[l][mask[l]] for l in range(X_time.shape[0])], dim=0)
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
# Patient posteriors (correlation version)
# ---------------------------------------------------------------------------


def compute_all_patient_posteriors_corr(
    patient_data: dict,
    t_indices_all: np.ndarray,
    patient_ids_all: list,
    window_days_all: np.ndarray,
    window_size: int,
    shrinkage: float,
    model: TemporalGaussianMixtureModel,
    basis: SymmetricMatrixBasis,
    coeff_mean: torch.Tensor,
    coeff_std: torch.Tensor,
    t_grid: torch.Tensor,
    t_min_days: float,
    t_max_days: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[dict, dict, dict]:
    """Compute per-patient posteriors from correlation matrices."""

    with torch.no_grad():
        pi_t = model.time_weight_model().cpu().numpy()
    t_grid_np = t_grid.cpu().numpy()
    day_range = max(t_max_days - t_min_days, 1e-9)

    # Group indices by patient
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

        # Recompute correlation matrices for this patient's windows
        days_matrix = patient_data[pid]
        n_days = days_matrix.shape[0]
        corr_list = []
        for w_day in days:
            w = int(w_day)
            window = days_matrix[w: w + window_size]
            hourly = slots_to_hourly(window)
            corr = _correlation_from_hourly(hourly, shrinkage=shrinkage)
            corr_list.append(corr)
        corr_np = np.stack(corr_list, axis=0)
        corr_t = torch.tensor(corr_np, device=device, dtype=dtype)

        x_coeffs = basis.project(corr_t)
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


def plot_correlation_model_summary(
    history: list[float],
    pi_t_np: np.ndarray,
    recon_means_np: np.ndarray,
    t_grid_days_np: np.ndarray,
    out_dir: str,
    model_name: str,
    show: bool,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    K = pi_t_np.shape[1]
    use_weeks = float(np.max(t_grid_days_np)) > 30.0
    t_axis = t_grid_days_np / 7.0 if use_weeks else t_grid_days_np
    t_label = "Treatment week" if use_weeks else "Treatment day"

    n_hours = recon_means_np.shape[-1]  # should be 24

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Loss curve
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

    # (3) Mean correlation matrices as heatmaps
    # recon_means_np has shape (K, n_hours, n_hours)
    for k in range(K):
        ax_inset = fig.add_axes(
            [0.68 + k * 0.11, 0.15, 0.10, 0.70]
        ) if K <= 3 else axes[2]
        if K <= 3:
            im = ax_inset.imshow(recon_means_np[k], cmap="RdBu_r",
                                 vmin=-1, vmax=1, aspect="equal")
            ax_inset.set_title(f"μ_{k+1}", fontsize=9)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
        else:
            break
    if K <= 3:
        axes[2].set_visible(False)
    else:
        im = axes[2].imshow(recon_means_np[0], cmap="RdBu_r",
                            vmin=-1, vmax=1, aspect="equal")
        axes[2].set_title(f"{model_name}: mean corr cluster 1")

    fig.suptitle("Correlation-based temporal mixture", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "glucodensity_correlation_summary.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_mean_correlation_heatmaps(
    recon_means_np: np.ndarray,
    out_dir: str,
    model_name: str,
    show: bool,
) -> str:
    """One heatmap per cluster mean correlation matrix."""
    K = recon_means_np.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4.5))
    if K == 1:
        axes = [axes]
    hours = np.arange(24)

    for k in range(K):
        ax = axes[k]
        im = ax.imshow(recon_means_np[k], cmap="RdBu_r", vmin=-1, vmax=1,
                        aspect="equal", origin="lower")
        ax.set_title(f"Cluster {k+1} mean correlation")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Hour")
        ax.set_xticks(hours[::4])
        ax.set_yticks(hours[::4])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_name}: cluster mean correlation matrices", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "glucodensity_correlation_means.pdf")
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
        description="Temporal mixture model on hourly glucose correlation matrices."
    )
    p.add_argument("--data-path", type=str,
                    default="data/glucodensities/cgm_all_patients.csv")
    p.add_argument("--max-prop-missing", type=float, default=0.20)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--window-size", type=int, default=14,
                    help="Sliding window size in days (larger → better correlation estimate)")
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--shrinkage", type=float, default=0.1,
                    help="Shrinkage towards identity for correlation estimation")
    p.add_argument("--model-type", choices=["basis", "ode"], default="ode")
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--R", type=int, default=None,
                    help="PCA truncation dim (default: full vech = 300)")
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
    print("Glucodensity Temporal Mixture: Correlation Matrices in Sym(24)")
    print("=" * 72)

    # 1) Data
    print("\n[Step 1] Loading and preprocessing data...")
    patient_data = load_and_preprocess_cgm(
        csv_path=csv_path,
        max_prop_missing=args.max_prop_missing,
        block_size=args.block_size,
        verbose=True,
    )

    # 2) Sliding-window correlations
    print("\n[Step 2] Computing sliding-window correlation matrices...")
    corr_matrices, t_indices, patient_ids, window_days, _ = \
        compute_sliding_window_correlations(
            patient_data=patient_data,
            window_size=args.window_size,
            stride=args.window_stride,
            shrinkage=args.shrinkage,
            verbose=True,
        )

    # 3) Build basis
    print("\n[Step 3] Building SymmetricMatrixBasis...")
    corr_t = torch.tensor(corr_matrices, device=device, dtype=dtype)
    if args.R is not None and args.R < N_HOURS * (N_HOURS + 1) // 2:
        basis = SymmetricMatrixBasis.from_data(
            n=N_HOURS, R=args.R,
            training_matrices=corr_t,
            device=device, dtype=dtype,
        )
        print(f"  PCA truncation: {basis.full_dim} → {args.R}")
    else:
        basis = SymmetricMatrixBasis(n=N_HOURS, device=device, dtype=dtype)
        print(f"  Full vech: coeff_dim = {basis.coeff_dim}")

    coeffs_all = basis.project(corr_t).detach().cpu().numpy()  # (N, M)
    print(f"  Projected shape: {coeffs_all.shape}")

    # 4) Temporal binning + normalisation
    print("\n[Step 4] Temporal binning...")
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

    # 5) Kernel
    sigma_auto = estimate_sigma_median_heuristic(X_time, mask)
    sigma_base = args.sigma if args.sigma > 0 else sigma_auto
    sigma = max(1e-8, sigma_base * args.sigma_mult)
    print(f"  Sigma: base={sigma_base:.4f}, final={sigma:.4f}")
    kernel = GaussianKernel(sigma=sigma)

    # 6) Time-weight model
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
        model_name = "CorrBasis"
    else:
        time_weight_model = NeuralODETimeWeights(
            t_grid=t_grid,
            num_components=args.n_components,
            hidden_dim=args.ode_hidden,
            device=device, dtype=dtype,
        )
        model_name = "CorrNeuralODE"

    # 7) Train
    print(f"\n[Step 5] Training {model_name}...")
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

    # 8) Patient posteriors + divergence
    print("\n[Step 6] Computing patient posteriors and group divergence...")
    patient_posteriors, patient_time_norm, patient_time_days = \
        compute_all_patient_posteriors_corr(
            patient_data=patient_data,
            t_indices_all=t_indices,
            patient_ids_all=patient_ids,
            window_days_all=window_days,
            window_size=args.window_size,
            shrinkage=args.shrinkage,
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

    # 9) Reconstruct mean correlation matrices for plotting
    with torch.no_grad():
        means_coeff = model.mean.cpu()
    means_orig = means_coeff * rep["coeff_std"].cpu() + rep["coeff_mean"].cpu()
    recon_means = basis.reconstruct(means_orig)          # (K, 24, 24)
    recon_means_np = recon_means.detach().numpy()

    t_np = t_grid.detach().cpu().numpy()
    day_range = rep["t_max_days"] - rep["t_min_days"]
    t_grid_days_np = t_np * day_range + rep["t_min_days"]

    # 10) Plots
    print("\n[Step 7] Generating plots...")
    plot_correlation_model_summary(
        history=history,
        pi_t_np=pi_t.cpu().numpy(),
        recon_means_np=recon_means_np,
        t_grid_days_np=t_grid_days_np,
        out_dir=out_dir,
        model_name=model_name,
        show=not args.no_show,
    )

    plot_mean_correlation_heatmaps(
        recon_means_np=recon_means_np,
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
    print(f"  Final MMD²(avg_t): {history[-1]:.6f}")
    print(
        f"  Divergence score: {div_stats['increasing_score']:.6f} | "
        f"final_TV={div_stats['final_sep']:.6f} | delta_TV={div_stats['delta_sep']:.6f}"
    )
    print(f"  Patients with posteriors: {len(patient_posteriors)} "
          f"(control: {n_ctrl}, treatment: {n_treat})")


if __name__ == "__main__":
    main()
