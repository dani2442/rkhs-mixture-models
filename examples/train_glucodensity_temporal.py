#!/usr/bin/env python
"""
Glucodensity temporal mixture fitting on real CGM data.

Preprocessing (Antonio's rules):
  1. Discard days with prop_missing > 20%
  2. Discard patients without enough valid days for at least 1 block (7 days)
  3. Values <= 40  → invalid (NaN)
  4. Values >= 400 → invalid (NaN)
  5. NA values     → invalid (NaN)

Temporal structure:
  - s ∈ [0, 24h]: intraday glucose curve (space dimension, 288 slots → L²)
  - t: sliding 7-day window index (temporal dimension)
  Each window averages the valid glucose curves in that 7-day block.

We compare two direct temporal-weight parameterizations:
  [1] Basis logits (cosine basis expansion of pi(t))
  [2] Neural ODE logits (dz/dt = f_theta(t, z), pi(t) = softmax(z(t)))
"""

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import GaussianKernel
from src.spaces import L2CosineBasis
from src.temporal_mixture import (
    BasisLogitsTimeWeights,
    NeuralODETimeWeights,
    TemporalGaussianMixtureModel,
    fit_temporal_gaussian_mixture_mmd,
)

# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------


def load_and_preprocess_cgm(
    csv_path: str,
    max_prop_missing: float = 0.20,
    block_size: int = 7,
    glucose_low: float = 40.0,
    glucose_high: float = 400.0,
    verbose: bool = True,
) -> dict:
    """
    Load the CGM CSV and apply Antonio's preprocessing rules.

    Returns a dict mapping PtID → DataFrame of valid days (288 glucose columns
    with invalid values set to NaN).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    time_cols = [
        c
        for c in df.columns
        if c not in ["PtID", "date", "n_obs", "n_slots", "n_missing", "prop_missing"]
    ]
    assert len(time_cols) == 288, f"Expected 288 time slots, got {len(time_cols)}"

    # Convert glucose columns to numeric (some cells contain "NA" strings or .5 values)
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_total_days = len(df)
    n_total_patients = df["PtID"].nunique()

    # Rule 1: discard days with prop_missing > 20%
    df = df[df["prop_missing"] <= max_prop_missing].copy()
    n_after_rule1 = len(df)

    # Rule 3 & 4 & 5: values <=40, >=400, NA → NaN
    glucose = df[time_cols].values.astype(np.float64)
    glucose[glucose <= glucose_low] = np.nan
    glucose[glucose >= glucose_high] = np.nan
    df[time_cols] = glucose

    # Rule 2: discard patients without enough valid days for ≥1 block
    day_counts = df.groupby("PtID").size()
    valid_patients = day_counts[day_counts >= block_size].index
    df = df[df["PtID"].isin(valid_patients)].copy()

    # Sort by patient and date
    df = df.sort_values(["PtID", "date"]).reset_index(drop=True)

    n_after_all = len(df)
    n_patients_after = df["PtID"].nunique()

    if verbose:
        print(f"  Total days: {n_total_days} → after rule 1 (prop_missing ≤ {max_prop_missing}): {n_after_rule1}")
        print(f"  After value clipping & patient block filter: {n_after_all} days, {n_patients_after} patients")
        print(f"  Patients removed: {n_total_patients - n_patients_after}")

    patient_data = {}
    for pt_id, group in df.groupby("PtID"):
        patient_data[pt_id] = group[time_cols].values.astype(np.float64)

    return patient_data


# ---------------------------------------------------------------------------
# 2. Sliding-window averaging
# ---------------------------------------------------------------------------


def compute_sliding_windows(
    patient_data: dict,
    window_size: int = 7,
    stride: int = 1,
    min_valid_frac: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Compute sliding-window averages across days for each patient.

    For each patient with D valid days, we form windows of `window_size` days.
    Within each window, for each of the 288 time slots, we average the valid
    (non-NaN) glucose values across the days in the window.

    Slots that have no valid value across the window remain NaN and are
    linearly interpolated from neighboring slots.

    Returns:
        curves: np.ndarray of shape (total_windows, 288) — the averaged curves
        window_indices: np.ndarray of shape (total_windows,) — normalized time
                        index t ∈ [0, 1] representing the window's temporal
                        position within that patient's recording.
        patient_ids: list of patient IDs for each window (for reference)
    """
    all_curves = []
    all_t_indices = []
    all_patient_ids = []

    n_slots = 288

    for pt_id, days_matrix in patient_data.items():
        n_days = days_matrix.shape[0]
        n_windows = n_days - window_size + 1

        if n_windows < 1:
            continue

        for w in range(0, n_windows, stride):
            window = days_matrix[w : w + window_size]  # (window_size, 288)

            # Average over days, ignoring NaN
            with np.errstate(all="ignore"):
                avg_curve = np.nanmean(window, axis=0)  # (288,)

            # Count valid fraction per slot
            valid_count = np.sum(~np.isnan(window), axis=0)
            min_count = int(np.ceil(min_valid_frac * window_size))

            # Slots with too few valid values → NaN
            avg_curve[valid_count < min_count] = np.nan

            # Linear interpolation for remaining NaN slots
            nan_mask = np.isnan(avg_curve)
            if nan_mask.all():
                continue  # skip entirely empty windows

            if nan_mask.any():
                valid_idx = np.where(~nan_mask)[0]
                avg_curve = np.interp(
                    np.arange(n_slots),
                    valid_idx,
                    avg_curve[valid_idx],
                )

            # Normalize t in [0, 1] based on position within the patient
            if n_windows > 1:
                t_val = w / (n_windows - 1)
            else:
                t_val = 0.5

            all_curves.append(avg_curve)
            all_t_indices.append(t_val)
            all_patient_ids.append(pt_id)

    curves = np.stack(all_curves, axis=0)       # (N_total, 288)
    t_indices = np.array(all_t_indices)          # (N_total,)

    if verbose:
        print(f"  Sliding windows (size={window_size}, stride={stride}):")
        print(f"    Total windows: {len(curves)}")
        print(f"    Patients contributing: {len(set(all_patient_ids))}")
        print(f"    Glucose range: [{np.nanmin(curves):.1f}, {np.nanmax(curves):.1f}]")

    return curves, t_indices, all_patient_ids


# ---------------------------------------------------------------------------
# 3. Build temporal slices for the model
# ---------------------------------------------------------------------------


def build_temporal_data(
    curves: np.ndarray,
    t_indices: np.ndarray,
    n_time_bins: int,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bin the sliding-window curves into ``n_time_bins`` temporal slices.

    Each curve has a normalized temporal position t ∈ [0,1]. We partition
    [0, 1] into ``n_time_bins`` bins and assign each curve to its bin.

    Because patients have different recording lengths the bins may have
    unequal numbers of samples.  We subsample to the size of the smallest
    non-empty bin so that X_time has regular shape (L_t, n, 288).

    Returns:
        X_time_raw: (L_t, n_min, 288) — raw glucose curves per time bin
        t_grid: (L_t,) — bin center positions in [0, 1]
    """
    bin_edges = np.linspace(0.0, 1.0 + 1e-9, n_time_bins + 1)
    bin_assignments = np.digitize(t_indices, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_time_bins - 1)

    bins = {}
    for b in range(n_time_bins):
        idx = np.where(bin_assignments == b)[0]
        if len(idx) > 0:
            bins[b] = curves[idx]

    # Remove empty bins
    valid_bins = sorted(bins.keys())
    if len(valid_bins) < 2:
        raise ValueError("Need at least 2 non-empty temporal bins")

    # Subsample to smallest bin size for rectangular tensor
    min_size = min(len(bins[b]) for b in valid_bins)

    slices = []
    centers = []
    for b in valid_bins:
        # Random subsample without replacement
        rng = np.random.RandomState(42 + b)
        chosen = rng.choice(len(bins[b]), size=min_size, replace=False)
        slices.append(bins[b][chosen])
        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2.0)

    X_time_raw = np.stack(slices, axis=0)  # (L_t, n_min, 288)
    t_grid_np = np.array(centers)

    if verbose:
        print(f"  Temporal binning: {n_time_bins} requested, {len(valid_bins)} non-empty")
        print(f"  Samples per bin (after subsampling): {min_size}")
        print(f"  X_time_raw shape: {X_time_raw.shape}")

    X_time_raw_t = torch.tensor(X_time_raw, device=device, dtype=dtype)
    t_grid = torch.tensor(t_grid_np, device=device, dtype=dtype)
    return X_time_raw_t, t_grid


# ---------------------------------------------------------------------------
# 4. Project intraday curves to L² cosine basis
# ---------------------------------------------------------------------------


def project_intraday_to_l2(
    X_time_raw: torch.Tensor,
    R_s: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, L2CosineBasis]:
    """
    Project the raw 288-slot intraday curves to an L² cosine basis.

    Args:
        X_time_raw: (L_t, n, 288) raw glucose values
        R_s: number of cosine basis functions

    Returns:
        X_time: (L_t, n, M_s) coefficient tensor (normalized)
        space_basis: the L2CosineBasis object used
        coeff_mean: (M_s,) mean used for normalization
        coeff_std: (M_s,) std used for normalization
    """
    L_t, n, n_slots = X_time_raw.shape
    assert n_slots == 288, f"Expected 288 slots, got {n_slots}"

    # Treat 24h period as T=1 (normalized domain)
    space_basis = L2CosineBasis(
        T=1.0,
        R=R_s,
        grid_size=n_slots,
        d=1,
        device=device,
        dtype=dtype,
    )

    # Project each (L_t, n) slice: X_time_raw[l] has shape (n, 288)
    # L2CosineBasis.project expects (n, L, d) → we need (n, 288, 1)
    coeffs_list = []
    for l in range(L_t):
        X_l = X_time_raw[l].unsqueeze(-1)  # (n, 288, 1)
        c_l = space_basis.project(X_l)      # (n, M_s)
        coeffs_list.append(c_l)

    X_time = torch.stack(coeffs_list, dim=0)  # (L_t, n, M_s)

    # Normalize coefficients (zero mean, unit variance across all data)
    # so that kernel bandwidth is meaningful
    all_coeffs = X_time.reshape(-1, X_time.shape[-1])  # (L_t * n, M_s)
    coeff_mean = all_coeffs.mean(dim=0, keepdim=True)
    coeff_std = all_coeffs.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_time = (X_time - coeff_mean) / coeff_std

    return X_time, space_basis, coeff_mean.squeeze(0), coeff_std.squeeze(0)


# ---------------------------------------------------------------------------
# 5. Experiment runner (same structure as train_l2_2d_temporal_pi.py)
# ---------------------------------------------------------------------------


def run_experiment(
    name: str,
    time_weight_model: torch.nn.Module,
    X_time: torch.Tensor,
    kernel: GaussianKernel,
    n_components: int,
    coeff_dim: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Train a TemporalGaussianMixtureModel and return model + history."""
    model = TemporalGaussianMixtureModel(
        num_components=n_components,
        coeff_dim=coeff_dim,
        time_weight_model=time_weight_model,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )

    history = fit_temporal_gaussian_mixture_mmd(
        model=model,
        X_time=X_time,
        kernel=kernel,
        num_epochs=num_epochs,
        lr=lr,
        init_method="kmeans++",
        verbose=True,
        log_interval=max(1, num_epochs // 8),
    )

    with torch.no_grad():
        pi_t = model.time_weight_model()

    print(f"{name} final MMD²(avg_t): {history[-1]:.6f}")
    print(f"{name} mean pi over t: {pi_t.mean(dim=0).detach().cpu().numpy()}")

    return model, history, pi_t.detach()


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------


def plot_results(
    t_grid: torch.Tensor,
    history_basis: list,
    history_ode: list,
    pi_basis: torch.Tensor,
    pi_ode: torch.Tensor,
    model_basis: TemporalGaussianMixtureModel,
    model_ode: TemporalGaussianMixtureModel,
    space_basis: L2CosineBasis,
    coeff_mean: torch.Tensor,
    coeff_std: torch.Tensor,
    out_dir: str,
    show: bool = True,
) -> None:
    """Create and save comparison plots."""
    t_np = t_grid.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 0, Col 0: Training curves ---
    axes[0, 0].plot(history_basis, label="Basis", lw=2.0)
    axes[0, 0].plot(history_ode, label="NeuralODE", lw=2.0)
    axes[0, 0].set_title("MMD² Training History")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("MMD²(avg_t)")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # --- Row 0, Col 1: Final MMD² bar chart ---
    methods = ["Basis", "NeuralODE"]
    final_vals = [history_basis[-1], history_ode[-1]]
    axes[0, 1].bar(methods, final_vals, color=["C0", "C1"], alpha=0.85)
    axes[0, 1].set_title("Final MMD²(avg_t)")
    axes[0, 1].set_ylabel("MMD²")
    axes[0, 1].grid(alpha=0.3, axis="y")

    # --- Row 0, Col 2: pi(t) for Basis ---
    pi_basis_np = pi_basis.cpu().numpy()
    K = pi_basis_np.shape[1]
    for k in range(K):
        axes[0, 2].plot(t_np, pi_basis_np[:, k], lw=1.5, label=f"$\\pi_{{{k+1}}}$")
    axes[0, 2].set_title("Basis: $\\pi_k(t)$")
    axes[0, 2].set_xlabel("t (normalized window position)")
    axes[0, 2].set_ylabel("weight")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend(fontsize=7, ncol=2)
    axes[0, 2].grid(alpha=0.3)

    # --- Row 1, Col 0: pi(t) for NeuralODE ---
    pi_ode_np = pi_ode.cpu().numpy()
    for k in range(K):
        axes[1, 0].plot(t_np, pi_ode_np[:, k], lw=1.5, label=f"$\\pi_{{{k+1}}}$")
    axes[1, 0].set_title("NeuralODE: $\\pi_k(t)$")
    axes[1, 0].set_xlabel("t (normalized window position)")
    axes[1, 0].set_ylabel("weight")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].grid(alpha=0.3)

    # --- Row 1, Col 1–2: Reconstructed mean components ---
    with torch.no_grad():
        means_basis = model_basis.mean.cpu()  # (K, M_s) — normalized
        means_ode = model_ode.mean.cpu()

    # De-normalize means back to original scale before reconstruction
    means_basis_orig = means_basis * coeff_std.cpu() + coeff_mean.cpu()
    means_ode_orig = means_ode * coeff_std.cpu() + coeff_mean.cpu()

    # L2Basis.reconstruct expects (n, M) and returns (n, L, d); d=1 here
    recon_basis = space_basis.reconstruct(means_basis_orig)   # (K, 288, 1)
    recon_ode = space_basis.reconstruct(means_ode_orig)

    recon_basis_np = recon_basis.detach().squeeze(-1).numpy()     # (K, 288)
    recon_ode_np = recon_ode.detach().squeeze(-1).numpy()
    hours = np.linspace(0, 24, recon_basis_np.shape[1])

    for k in range(K):
        axes[1, 1].plot(hours, recon_basis_np[k], lw=1.3, label=f"comp {k+1}")
    axes[1, 1].set_title("Basis: Reconstructed mean curves")
    axes[1, 1].set_xlabel("Hour of day")
    axes[1, 1].set_ylabel("Glucose (mg/dL)")
    axes[1, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].grid(alpha=0.3)

    for k in range(K):
        axes[1, 2].plot(hours, recon_ode_np[k], lw=1.3, label=f"comp {k+1}")
    axes[1, 2].set_title("NeuralODE: Reconstructed mean curves")
    axes[1, 2].set_xlabel("Hour of day")
    axes[1, 2].set_ylabel("Glucose (mg/dL)")
    axes[1, 2].legend(fontsize=7, ncol=2)
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "glucodensity_temporal_comparison.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"\nSaved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- Additional figure: variance components ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    with torch.no_grad():
        var_basis = model_basis.variance.cpu().numpy()  # (K, M_s)
        var_ode = model_ode.variance.cpu().numpy()

    for k in range(K):
        axes2[0].plot(var_basis[k], lw=1.2, label=f"comp {k+1}")
    axes2[0].set_title("Basis: Variance per coefficient")
    axes2[0].set_xlabel("Coefficient index")
    axes2[0].set_ylabel("Variance")
    axes2[0].legend(fontsize=7, ncol=2)
    axes2[0].grid(alpha=0.3)

    for k in range(K):
        axes2[1].plot(var_ode[k], lw=1.2, label=f"comp {k+1}")
    axes2[1].set_title("NeuralODE: Variance per coefficient")
    axes2[1].set_xlabel("Coefficient index")
    axes2[1].set_ylabel("Variance")
    axes2[1].legend(fontsize=7, ncol=2)
    axes2[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "glucodensity_temporal_variance.pdf")
    fig2.savefig(out_path2, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path2}")

    if show:
        plt.show()
    else:
        plt.close(fig2)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Glucodensity temporal mixture fitting (Basis vs NeuralODE)."
    )
    # Data / preprocessing
    p.add_argument(
        "--data-path",
        type=str,
        default="data/glucodensities/cgm_all_patients.csv",
        help="Path to CGM CSV relative to project root.",
    )
    p.add_argument("--max-prop-missing", type=float, default=0.20)
    p.add_argument("--block-size", type=int, default=7, help="Min days for patient inclusion")
    p.add_argument("--window-size", type=int, default=7, help="Sliding window size (days)")
    p.add_argument("--window-stride", type=int, default=1, help="Sliding window stride (days)")

    # Model
    p.add_argument("--n-components", type=int, default=4, help="Mixture components K")
    p.add_argument("--r-s", type=int, default=12, help="Cosine basis functions for intraday")
    p.add_argument("--n-time-bins", type=int, default=20, help="Number of temporal bins L_t")
    p.add_argument("--r-pi", type=int, default=6, help="Basis functions for pi(t)")
    p.add_argument("--ode-hidden", type=int, default=64, help="NeuralODE hidden dim")

    # Optimization
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.0, help="Gaussian kernel bandwidth (0=auto median heuristic)")

    # Misc
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

    print("=" * 72)
    print("Glucodensity Temporal Mixture: Basis vs NeuralODE")
    print("=" * 72)

    # ----- Step 1: Load & preprocess -----
    print("\n[Step 1] Loading and preprocessing CGM data...")
    patient_data = load_and_preprocess_cgm(
        csv_path=csv_path,
        max_prop_missing=args.max_prop_missing,
        block_size=args.block_size,
        verbose=True,
    )

    # ----- Step 2: Sliding windows -----
    print("\n[Step 2] Computing sliding-window averages...")
    curves, t_indices, patient_ids = compute_sliding_windows(
        patient_data=patient_data,
        window_size=args.window_size,
        stride=args.window_stride,
        verbose=True,
    )

    # ----- Step 3: Bin into temporal slices -----
    print("\n[Step 3] Binning into temporal slices...")
    X_time_raw, t_grid = build_temporal_data(
        curves=curves,
        t_indices=t_indices,
        n_time_bins=args.n_time_bins,
        device=device,
        dtype=dtype,
        verbose=True,
    )

    # ----- Step 4: Project intraday curves to L² cosine basis -----
    print("\n[Step 4] Projecting intraday curves to L² cosine basis...")
    X_time, space_basis, coeff_mean, coeff_std = project_intraday_to_l2(
        X_time_raw=X_time_raw,
        R_s=args.r_s,
        device=device,
        dtype=dtype,
    )
    coeff_dim = X_time.shape[-1]
    L_t = X_time.shape[0]
    n_samples = X_time.shape[1]

    print(f"  X_time shape: (L_t={L_t}, n={n_samples}, M_s={coeff_dim})")

    # ----- Step 5: Set up kernel and time basis -----
    # Median heuristic for sigma if set to 0 (auto)
    sigma = args.sigma
    if sigma <= 0:
        all_coeffs = X_time.reshape(-1, coeff_dim)
        # Subsample for speed
        n_sub = min(2000, all_coeffs.shape[0])
        idx = torch.randperm(all_coeffs.shape[0])[:n_sub]
        sub = all_coeffs[idx]
        dists = torch.cdist(sub, sub)
        sigma = float(torch.median(dists[dists > 0]).item())
        print(f"  Auto sigma (median heuristic): {sigma:.4f}")

    kernel = GaussianKernel(sigma=sigma)

    time_basis = L2CosineBasis(
        T=1.0,
        R=args.r_pi,
        grid_size=L_t,
        d=1,
        device=device,
        dtype=dtype,
    )

    n_components = args.n_components

    # ----- Step 6: Direct MMD with Basis temporal weights -----
    print(f"\n[Experiment 1] Direct MMD: Basis temporal weights (K={n_components})...")
    basis_weight_model = BasisLogitsTimeWeights(
        basis_matrix=time_basis.Phi,
        num_components=n_components,
        device=device,
        dtype=dtype,
    )
    model_basis, history_basis, pi_basis = run_experiment(
        name="Direct-Basis",
        time_weight_model=basis_weight_model,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
    )

    # ----- Step 7: Direct MMD with NeuralODE temporal weights -----
    print(f"\n[Experiment 2] Direct MMD: NeuralODE temporal weights (K={n_components})...")
    ode_weight_model = NeuralODETimeWeights(
        t_grid=t_grid,
        num_components=n_components,
        hidden_dim=args.ode_hidden,
        device=device,
        dtype=dtype,
    )
    model_ode, history_ode, pi_ode = run_experiment(
        name="Direct-NeuralODE",
        time_weight_model=ode_weight_model,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
    )

    # ----- Step 8: Summary -----
    print("\n" + "=" * 72)
    print("Final comparison")
    print("=" * 72)
    print(f"  Direct-Basis    MMD²(avg_t): {history_basis[-1]:.6f}")
    print(f"  Direct-NeuralODE MMD²(avg_t): {history_ode[-1]:.6f}")
    print(f"  Improvement (Basis→ODE): {history_basis[-1] - history_ode[-1]:.6f}")

    # ----- Step 9: Visualization -----
    out_dir = os.path.join(project_root, "paper", "images")
    plot_results(
        t_grid=t_grid,
        history_basis=history_basis,
        history_ode=history_ode,
        pi_basis=pi_basis,
        pi_ode=pi_ode,
        model_basis=model_basis,
        model_ode=model_ode,
        space_basis=space_basis,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        out_dir=out_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
