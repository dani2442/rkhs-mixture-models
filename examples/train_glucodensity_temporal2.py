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

We compare four temporal-weight parameterizations in two paradigms:

  Direct MMD (end-to-end):
    [1] Basis logits (cosine basis expansion of pi(t))
    [2] Neural ODE logits (dz/dt = f_theta(t, z), pi(t) = softmax(z(t)))

  Two-Stage (distillation):
    [3] Stage 1: free discrete pi(t_i) + Gaussian params via MMD
        Stage 2: fit Basis logits to stage-1 pi(t) with L² loss
    [4] Stage 1: same
        Stage 2: fit Neural ODE to stage-1 pi(t) with L² loss
"""

import argparse
import os
import sys
from typing import Tuple

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
from src.visualization import (
    plot_glucodensity_temporal_4model_comparison,
    plot_glucodensity_variance,
    plot_cluster_probabilities_by_group,
    plot_ternary_simplex_evolution,
)

# Patient group labels from the clinical study
CONTROL_IDS = [1, 2, 3, 5, 8, 11, 13, 14, 18, 19, 26, 32, 33, 34, 36,
               43, 44, 53, 55, 57, 59, 66, 75, 77, 82, 86, 87, 89, 90,
               91, 94, 98, 101, 102, 104]
TREATMENT_IDS = [4, 7, 9, 10, 12, 15, 16, 17, 20, 21, 23, 24, 25, 27,
                 28, 29, 30, 31, 35, 37, 38, 39, 40, 42, 46, 47, 48, 49,
                 50, 51, 52, 56, 58, 60, 61, 62, 63, 64, 65, 67, 68, 69,
                 70, 72, 73, 74, 78, 79, 81, 83, 84, 85, 88, 93, 95, 97,
                 99, 100, 103, 105, 106, 107, 108, 109]

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
    Load the CGM CSV.

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
        window_indices: np.ndarray of shape (total_windows,) — the center day
                        of each sliding window (absolute day position).
        patient_ids: list of patient IDs for each window (for reference)
    """
    all_curves = []
    all_t_indices = []
    all_patient_ids = []
    all_window_days = []
    patient_n_days = {}

    n_slots = 288

    for pt_id, days_matrix in patient_data.items():
        n_days = days_matrix.shape[0]
        n_windows = n_days - window_size + 1

        if n_windows < 1:
            continue

        patient_n_days[pt_id] = n_days

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

            # Use absolute center-day of the sliding window
            t_val = w + (window_size - 1) / 2.0

            all_curves.append(avg_curve)
            all_t_indices.append(t_val)
            all_patient_ids.append(pt_id)
            all_window_days.append(w)

    curves = np.stack(all_curves, axis=0)       # (N_total, 288)
    t_indices = np.array(all_t_indices)          # (N_total,)
    window_days = np.array(all_window_days)      # (N_total,)

    if verbose:
        print(f"  Sliding windows (size={window_size}, stride={stride}):")
        print(f"    Total windows: {len(curves)}")
        print(f"    Patients contributing: {len(set(all_patient_ids))}")
        print(f"    Glucose range: [{np.nanmin(curves):.1f}, {np.nanmax(curves):.1f}]")
        median_days = np.median(list(patient_n_days.values()))
        print(f"    Median patient days: {median_days:.0f}")

    return curves, t_indices, all_patient_ids, window_days, patient_n_days


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Bin the sliding-window curves into ``n_time_bins`` temporal slices.

    Each curve has an absolute temporal position (center day of its sliding
    window).  We partition the observed day range [t_min, t_max] into
    ``n_time_bins`` bins and assign each curve to its bin.

    Because patients have different recording lengths, later bins will
    naturally contain fewer samples (only long-recording patients).  We
    pad every bin to ``n_max = max(bin_sizes)`` with zero vectors and
    return a boolean mask indicating the valid (non-padding) entries.

    Bin centers are globally normalized to [0, 1] so that the temporal
    model always operates on a compact domain.

    Returns:
        X_time_raw: (L_t, n_max, 288) — raw glucose curves per time bin
                    (padding entries are zero-filled)
        t_grid: (L_t,) — globally-normalized bin centers in [0, 1]
        mask: (L_t, n_max) — boolean tensor, True = valid sample
        t_min_days: float — minimum day value (for converting back)
        t_max_days: float — maximum day value (for converting back)
    """
    t_min = float(t_indices.min())
    t_max = float(t_indices.max())
    # Guard against degenerate case where all samples share the same day
    if t_max - t_min < 1e-9:
        t_max = t_min + 1.0

    bin_edges = np.linspace(t_min, t_max + 1e-9, n_time_bins + 1)
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

    # Pad to max bin size and create mask
    bin_sizes = [len(bins[b]) for b in valid_bins]
    max_size = max(bin_sizes)
    n_feat = curves.shape[1]  # 288

    slices = []
    mask_rows = []
    centers = []
    for b in valid_bins:
        n_b = len(bins[b])
        # Random shuffle for variety, but keep all samples
        rng = np.random.RandomState(42 + b)
        perm = rng.permutation(n_b)
        data_b = bins[b][perm]
        # Pad with zeros to max_size
        padded = np.zeros((max_size, n_feat), dtype=np.float64)
        padded[:n_b] = data_b
        slices.append(padded)
        # Mask: True for real samples, False for padding
        m = np.zeros(max_size, dtype=bool)
        m[:n_b] = True
        mask_rows.append(m)
        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2.0)

    X_time_raw = np.stack(slices, axis=0)       # (L_t, n_max, 288)
    mask_np = np.stack(mask_rows, axis=0)        # (L_t, n_max)
    t_grid_days = np.array(centers)              # (L_t,) in absolute days

    # Globally normalize bin centers to [0, 1]
    t_grid_norm = (t_grid_days - t_min) / (t_max - t_min)

    if verbose:
        print(f"  Temporal binning: {n_time_bins} requested, {len(valid_bins)} non-empty")
        print(f"  Day range: [{t_min:.1f}, {t_max:.1f}]")
        print(f"  Bin sizes (valid samples): min={min(bin_sizes)}, "
              f"max={max(bin_sizes)}, median={int(np.median(bin_sizes))}")
        print(f"  Padded to n_max={max_size}")
        print(f"  X_time_raw shape: {X_time_raw.shape}")

    X_time_raw_t = torch.tensor(X_time_raw, device=device, dtype=dtype)
    t_grid = torch.tensor(t_grid_norm, device=device, dtype=dtype)
    mask_t = torch.tensor(mask_np, device=device)
    return X_time_raw_t, t_grid, mask_t, t_min, t_max


# ---------------------------------------------------------------------------
# 4. Project intraday curves to L² cosine basis
# ---------------------------------------------------------------------------


def project_intraday_to_l2(
    X_time_raw: torch.Tensor,
    R_s: int,
    device: torch.device,
    dtype: torch.dtype,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, L2CosineBasis]:
    """
    Project the raw 288-slot intraday curves to an L² cosine basis.

    Args:
        X_time_raw: (L_t, n, 288) raw glucose values (may contain padding)
        R_s: number of cosine basis functions
        mask: Optional boolean tensor (L_t, n). True = valid sample.
              When provided, normalization statistics are computed only
              over valid (non-padding) entries.

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

    # Normalize coefficients (zero mean, unit variance across valid data)
    # so that kernel bandwidth is meaningful
    if mask is not None:
        # Gather only valid (non-padding) coefficients for statistics
        valid_coeffs = torch.cat(
            [X_time[l][mask[l]] for l in range(L_t)], dim=0
        )  # (N_valid, M_s)
    else:
        valid_coeffs = X_time.reshape(-1, X_time.shape[-1])  # (L_t * n, M_s)

    coeff_mean = valid_coeffs.mean(dim=0, keepdim=True)
    coeff_std = valid_coeffs.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_time = (X_time - coeff_mean) / coeff_std

    return X_time, space_basis, coeff_mean.squeeze(0), coeff_std.squeeze(0)


# ---------------------------------------------------------------------------
# 5. DiscreteLogitsTimeWeights + helpers for two-stage training
# ---------------------------------------------------------------------------


class DiscreteLogitsTimeWeights(torch.nn.Module):
    """Trainable per-time logits: pi(t_i) = softmax(z_i)."""

    def __init__(
        self,
        num_times: int,
        num_components: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.zeros(num_times, num_components, device=device, dtype=dtype)
        )

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)


def fit_time_weights_l2(
    name: str,
    time_weight_model: torch.nn.Module,
    target_pi_t: torch.Tensor,
    num_epochs: int,
    lr: float,
) -> Tuple[list, torch.Tensor]:
    """Fit a time-weight model to a target pi(t) via L² regression."""
    optimizer = torch.optim.Adam(time_weight_model.parameters(), lr=lr)
    history: list = []
    log_interval = max(1, num_epochs // 8)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pi_pred = time_weight_model()
        loss = torch.mean((pi_pred - target_pi_t) ** 2)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch + 1:4d}/{num_epochs}: {name} L² = {loss.item():.8f}")

    with torch.no_grad():
        pi_fit = time_weight_model()
    print(f"{name} final L²: {history[-1]:.8f}")
    return history, pi_fit.detach()


def copy_component_params(
    dst_model: TemporalGaussianMixtureModel,
    src_model: TemporalGaussianMixtureModel,
) -> None:
    """Copy Gaussian component parameters (means, covariance) from src to dst."""
    with torch.no_grad():
        dst_model.components._mean_coeffs.copy_(src_model.components._mean_coeffs)
        if hasattr(dst_model.components, "_log_var") and hasattr(
            src_model.components, "_log_var"
        ):
            dst_model.components._log_var.copy_(src_model.components._log_var)
        if hasattr(dst_model.components, "_chol_factor") and hasattr(
            src_model.components, "_chol_factor"
        ):
            dst_model.components._chol_factor.copy_(src_model.components._chol_factor)


def evaluate_mmd_with_fixed_components(
    source_model: TemporalGaussianMixtureModel,
    time_weight_model: torch.nn.Module,
    X_time: torch.Tensor,
    kernel: GaussianKernel,
    n_components: int,
    coeff_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    mask: torch.Tensor | None = None,
) -> float:
    """Evaluate MMD² using Gaussian params from source_model + given time_weight_model."""
    eval_model = TemporalGaussianMixtureModel(
        num_components=n_components,
        coeff_dim=coeff_dim,
        time_weight_model=time_weight_model,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )
    copy_component_params(eval_model, source_model)
    with torch.no_grad():
        mmd2, _ = eval_model.compute_mmd2(
            X_time, kernel, compute_const_term=True, mask=mask,
        )
    return float(mmd2.item())


# ---------------------------------------------------------------------------
# 6. Experiment runner (same structure as train_l2_2d_temporal_pi.py)
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
    mask: torch.Tensor | None = None,
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
        mask=mask,
    )

    with torch.no_grad():
        pi_t = model.time_weight_model()

    print(f"{name} final MMD²(avg_t): {history[-1]:.6f}")
    print(f"{name} mean pi over t: {pi_t.mean(dim=0).detach().cpu().numpy()}")

    return model, history, pi_t.detach()


# ---------------------------------------------------------------------------
# 7. Per-patient posterior computation
# ---------------------------------------------------------------------------


def interpolate_pi(
    pi_t_np: np.ndarray,
    t_grid_np: np.ndarray,
    t_query_np: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of pi(t) at arbitrary query time points."""
    K = pi_t_np.shape[1]
    result = np.zeros((len(t_query_np), K))
    for k in range(K):
        result[:, k] = np.interp(t_query_np, t_grid_np, pi_t_np[:, k])
    # Re-normalize to ensure rows sum to 1
    row_sums = result.sum(axis=1, keepdims=True)
    result = result / np.maximum(row_sums, 1e-10)
    return result


def compute_all_patient_posteriors(
    curves: np.ndarray,
    t_indices: np.ndarray,
    patient_ids: list,
    window_days: np.ndarray,
    model: TemporalGaussianMixtureModel,
    space_basis,
    coeff_mean: torch.Tensor,
    coeff_std: torch.Tensor,
    t_grid: torch.Tensor,
    t_min_days: float,
    t_max_days: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[dict, dict, dict]:
    """
    Compute per-patient posteriors P(k|x,t) for all sliding windows.

    The patient's absolute-day `t_indices` are globally normalized to [0, 1]
    using the same `t_min_days`/`t_max_days` used during training, then
    clamped so that patients whose recordings extend beyond the training
    range get boundary weights.

    Returns:
        patient_posteriors: dict PtID → ndarray (n_windows, K)
        patient_time_norm:  dict PtID → ndarray (n_windows,)  (globally normalized)
        patient_time_days:  dict PtID → ndarray (n_windows,)
    """
    # Project all curves to L2 coefficients using same basis & normalization
    curves_t = torch.tensor(curves, device=device, dtype=dtype)
    curves_3d = curves_t.unsqueeze(-1)                                # (N, 288, 1)
    all_coeffs = space_basis.project(curves_3d)                       # (N, M)
    all_coeffs = (all_coeffs - coeff_mean.unsqueeze(0)) / coeff_std.unsqueeze(0)

    # Get pi(t) on the model's time grid (already globally normalized to [0,1])
    with torch.no_grad():
        pi_t = model.time_weight_model().cpu().numpy()  # (L_t, K)
    t_grid_np = t_grid.cpu().numpy()

    # Global normalization denominator
    day_range = t_max_days - t_min_days
    if day_range < 1e-9:
        day_range = 1.0

    # Group indices by patient
    patient_indices: dict = {}
    for i, pid in enumerate(patient_ids):
        if pid not in patient_indices:
            patient_indices[pid] = []
        patient_indices[pid].append(i)

    patient_posteriors: dict = {}
    patient_time_norm: dict = {}
    patient_time_days: dict = {}

    for pid, idx_list in patient_indices.items():
        idx = np.array(idx_list)
        x_coeffs = all_coeffs[idx]               # (n_w, M) tensor
        t_vals_abs = t_indices[idx]               # absolute center-days
        days = window_days[idx]

        # Global normalization to [0, 1], clamp for safety
        t_vals_norm = np.clip(
            (t_vals_abs - t_min_days) / day_range, 0.0, 1.0
        )

        # Interpolate pi at this patient's globally-normalized times
        pi_at_t_np = interpolate_pi(pi_t, t_grid_np, t_vals_norm)  # (n_w, K)
        pi_at_t = torch.tensor(pi_at_t_np, device=device, dtype=dtype)

        # Compute posterior P(k|x,t) using the model
        with torch.no_grad():
            posteriors = model.responsibilities(x_coeffs, pi_at_t)  # (n_w, K)

        patient_posteriors[pid] = posteriors.cpu().numpy()
        patient_time_norm[pid] = t_vals_norm
        patient_time_days[pid] = days

    return patient_posteriors, patient_time_norm, patient_time_days


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Glucodensity temporal mixture fitting (Direct vs Two-Stage)."
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
    p.add_argument("--n-components", type=int, default=3, help="Mixture components K")
    p.add_argument("--r-s", type=int, default=12, help="Cosine basis functions for intraday")
    p.add_argument("--n-time-bins", type=int, default=20, help="Number of temporal bins L_t")
    p.add_argument("--r-pi", type=int, default=6, help="Basis functions for pi(t)")
    p.add_argument("--ode-hidden", type=int, default=64, help="NeuralODE hidden dim")

    # Optimization
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--l2-epochs", type=int, default=300, help="Epochs for stage-2 L² distillation")
    p.add_argument("--l2-lr", type=float, default=0.01, help="LR for stage-2 L² distillation")
    p.add_argument("--sigma", type=float, default=0, help="Gaussian kernel bandwidth (0=auto median heuristic)")

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
    print("Glucodensity Temporal Mixture: Direct vs Two-Stage (4 models)")
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
    curves, t_indices, patient_ids, window_days, patient_n_days = compute_sliding_windows(
        patient_data=patient_data,
        window_size=args.window_size,
        stride=args.window_stride,
        verbose=True,
    )

    # ----- Step 3: Bin into temporal slices -----
    print("\n[Step 3] Binning into temporal slices...")
    X_time_raw, t_grid, mask, t_min_days, t_max_days = build_temporal_data(
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
        mask=mask,
    )
    coeff_dim = X_time.shape[-1]
    L_t = X_time.shape[0]
    n_samples = X_time.shape[1]

    print(f"  X_time shape: (L_t={L_t}, n={n_samples}, M_s={coeff_dim})")

    # ----- Step 5: Set up kernel and time basis -----
    # Median heuristic for sigma if set to 0 (auto)
    sigma = args.sigma
    if sigma <= 0:
        # Use only valid (non-padding) coefficients for the heuristic
        if mask is not None:
            all_coeffs = torch.cat(
                [X_time[l][mask[l]] for l in range(L_t)], dim=0
            )
        else:
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

    # ==================================================================
    # Direct MMD experiments (end-to-end)
    # ==================================================================

    # ----- Experiment 1: Direct MMD with Basis temporal weights -----
    print(f"\n[Experiment 1] Direct MMD: Basis temporal weights (K={n_components})...")
    basis_weight_model_direct = BasisLogitsTimeWeights(
        basis_matrix=time_basis.Phi,
        num_components=n_components,
        device=device,
        dtype=dtype,
    )
    model_basis_direct, history_basis_direct, pi_basis_direct = run_experiment(
        name="Direct-Basis",
        time_weight_model=basis_weight_model_direct,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
        mask=mask,
    )

    # ----- Experiment 2: Direct MMD with NeuralODE temporal weights -----
    print(f"\n[Experiment 2] Direct MMD: NeuralODE temporal weights (K={n_components})...")
    ode_weight_model_direct = NeuralODETimeWeights(
        t_grid=t_grid,
        num_components=n_components,
        hidden_dim=args.ode_hidden,
        device=device,
        dtype=dtype,
    )
    model_ode_direct, history_ode_direct, pi_ode_direct = run_experiment(
        name="Direct-NeuralODE",
        time_weight_model=ode_weight_model_direct,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
        mask=mask,
    )

    # ==================================================================
    # Two-stage experiments (discrete pi → L² distillation)
    # ==================================================================

    # ----- Experiment 3a: Stage 1 — free discrete pi(t_i) via MMD -----
    print(f"\n[Experiment 3] Two-stage (stage 1): free discrete pi(t_i) (K={n_components})...")
    discrete_weight_model = DiscreteLogitsTimeWeights(
        num_times=L_t,
        num_components=n_components,
        device=device,
        dtype=dtype,
    )
    stage1_model, history_stage1, pi_stage1 = run_experiment(
        name="TwoStage-Stage1-DiscretePi",
        time_weight_model=discrete_weight_model,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
        mask=mask,
    )

    # ----- Experiment 3b: Stage 2 — fit Basis to stage-1 pi(t) -----
    print(f"\n[Experiment 3 cont.] Two-stage (stage 2): Basis L² fit to stage-1 pi...")
    basis_weight_model_l2 = BasisLogitsTimeWeights(
        basis_matrix=time_basis.Phi,
        num_components=n_components,
        device=device,
        dtype=dtype,
    )
    history_l2_basis, pi_basis_l2 = fit_time_weights_l2(
        name="TwoStage-Basis",
        time_weight_model=basis_weight_model_l2,
        target_pi_t=pi_stage1,
        num_epochs=args.l2_epochs,
        lr=args.l2_lr,
    )

    # ----- Experiment 4: Stage 2 — fit NeuralODE to stage-1 pi(t) -----
    print(f"\n[Experiment 4] Two-stage (stage 2): NeuralODE L² fit to stage-1 pi...")
    ode_weight_model_l2 = NeuralODETimeWeights(
        t_grid=t_grid,
        num_components=n_components,
        hidden_dim=args.ode_hidden,
        device=device,
        dtype=dtype,
    )
    history_l2_ode, pi_ode_l2 = fit_time_weights_l2(
        name="TwoStage-NeuralODE",
        time_weight_model=ode_weight_model_l2,
        target_pi_t=pi_stage1,
        num_epochs=args.l2_epochs,
        lr=args.l2_lr,
    )

    # ==================================================================
    # Evaluate two-stage MMD² (stage-1 Gaussian params + stage-2 weights)
    # ==================================================================
    twostage_basis_mmd = evaluate_mmd_with_fixed_components(
        source_model=stage1_model,
        time_weight_model=basis_weight_model_l2,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        device=device,
        dtype=dtype,
        mask=mask,
    )
    twostage_ode_mmd = evaluate_mmd_with_fixed_components(
        source_model=stage1_model,
        time_weight_model=ode_weight_model_l2,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        device=device,
        dtype=dtype,
        mask=mask,
    )

    # ==================================================================
    # Per-patient posteriors (use best model among all 4)
    # ==================================================================
    all_final_mmd = {
        "Direct-Basis": (history_basis_direct[-1], model_basis_direct),
        "Direct-NeuralODE": (history_ode_direct[-1], model_ode_direct),
        "TwoStage-Basis": (twostage_basis_mmd, stage1_model),
        "TwoStage-NeuralODE": (twostage_ode_mmd, stage1_model),
    }
    best_name = min(all_final_mmd, key=lambda k: all_final_mmd[k][0])
    best_model = all_final_mmd[best_name][1]

    print(f"\n[Step 8] Computing per-patient posteriors (using {best_name})...")
    patient_posteriors, patient_time_norm, patient_time_days = compute_all_patient_posteriors(
        curves=curves,
        t_indices=t_indices,
        patient_ids=patient_ids,
        window_days=window_days,
        model=best_model,
        space_basis=space_basis,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        t_grid=t_grid,
        t_min_days=t_min_days,
        t_max_days=t_max_days,
        device=device,
        dtype=dtype,
    )
    n_ctrl = sum(1 for pid in patient_posteriors if pid in CONTROL_IDS)
    n_treat = sum(1 for pid in patient_posteriors if pid in TREATMENT_IDS)
    print(f"  Patients with posteriors: {len(patient_posteriors)} "
          f"(control: {n_ctrl}, treatment: {n_treat})")

    # ==================================================================
    # Summary
    # ==================================================================
    direct_basis_mmd = history_basis_direct[-1]
    direct_ode_mmd = history_ode_direct[-1]
    stage1_mmd = history_stage1[-1]
    l2_basis_final = float(torch.mean((pi_basis_l2 - pi_stage1) ** 2).item())
    l2_ode_final = float(torch.mean((pi_ode_l2 - pi_stage1) ** 2).item())

    print("\n" + "=" * 72)
    print("Final comparison")
    print("=" * 72)
    print(f"  Direct-Basis      MMD²(avg_t): {direct_basis_mmd:.6f}")
    print(f"  Direct-NeuralODE  MMD²(avg_t): {direct_ode_mmd:.6f}")
    print(f"  Stage1-DiscretePi MMD²(avg_t): {stage1_mmd:.6f}")
    print(f"  TwoStage-Basis    MMD²(avg_t): {twostage_basis_mmd:.6f}")
    print(f"  TwoStage-NeuralODE MMD²(avg_t):{twostage_ode_mmd:.6f}")
    print(f"  TwoStage-Basis L²(pi):         {l2_basis_final:.8f}")
    print(f"  TwoStage-NeuralODE L²(pi):     {l2_ode_final:.8f}")

    # ==================================================================
    # Visualization
    # ==================================================================
    out_dir = os.path.join(project_root, "paper", "images")

    # Prepare numpy arrays for visualization
    with torch.no_grad():
        means_basis = model_basis_direct.mean.cpu()
        means_ode = model_ode_direct.mean.cpu()
        means_stage1 = stage1_model.mean.cpu()
        var_basis = model_basis_direct.variance.cpu().numpy()
        var_ode = model_ode_direct.variance.cpu().numpy()

    # Reconstruct intraday mean curves (un-normalize coefficients first)
    means_basis_orig = means_basis * coeff_std.cpu() + coeff_mean.cpu()
    means_ode_orig = means_ode * coeff_std.cpu() + coeff_mean.cpu()
    means_stage1_orig = means_stage1 * coeff_std.cpu() + coeff_mean.cpu()
    recon_basis = space_basis.reconstruct(means_basis_orig)
    recon_ode = space_basis.reconstruct(means_ode_orig)
    recon_stage1 = space_basis.reconstruct(means_stage1_orig)
    recon_basis_np = recon_basis.detach().squeeze(-1).numpy()
    recon_ode_np = recon_ode.detach().squeeze(-1).numpy()
    recon_stage1_np = recon_stage1.detach().squeeze(-1).numpy()

    t_np = t_grid.detach().cpu().numpy()
    pi_basis_direct_np = pi_basis_direct.cpu().numpy()
    pi_ode_direct_np = pi_ode_direct.cpu().numpy()
    pi_stage1_np = pi_stage1.cpu().numpy()
    pi_basis_l2_np = pi_basis_l2.cpu().numpy()
    pi_ode_l2_np = pi_ode_l2.cpu().numpy()

    # Convert normalized t_grid [0,1] back to actual treatment days
    # t_grid is globally normalized: t_norm = (t_days - t_min_days) / (t_max_days - t_min_days)
    day_range = t_max_days - t_min_days
    t_grid_days_np = t_np * day_range + t_min_days

    plot_glucodensity_temporal_4model_comparison(
        t_grid_np=t_grid_days_np,
        # Training histories
        history_basis_direct=history_basis_direct,
        history_ode_direct=history_ode_direct,
        history_stage1=history_stage1,
        history_l2_basis=history_l2_basis,
        history_l2_ode=history_l2_ode,
        # Mixture weights pi(t)
        pi_basis_direct_np=pi_basis_direct_np,
        pi_ode_direct_np=pi_ode_direct_np,
        pi_stage1_np=pi_stage1_np,
        pi_basis_l2_np=pi_basis_l2_np,
        pi_ode_l2_np=pi_ode_l2_np,
        # Reconstructed mean curves
        recon_basis_np=recon_basis_np,
        recon_ode_np=recon_ode_np,
        recon_stage1_np=recon_stage1_np,
        # Final MMD values
        direct_basis_mmd=direct_basis_mmd,
        direct_ode_mmd=direct_ode_mmd,
        stage1_mmd=stage1_mmd,
        twostage_basis_mmd=twostage_basis_mmd,
        twostage_ode_mmd=twostage_ode_mmd,
        # Misc
        t_min_days=t_min_days,
        t_max_days=t_max_days,
        out_dir=out_dir,
        show=not args.no_show,
    )

    plot_glucodensity_variance(
        var_basis_np=var_basis,
        var_ode_np=var_ode,
        out_dir=out_dir,
        show=not args.no_show,
    )

    plot_cluster_probabilities_by_group(
        patient_posteriors=patient_posteriors,
        patient_time_days=patient_time_days,
        control_ids=CONTROL_IDS,
        treatment_ids=TREATMENT_IDS,
        out_dir=out_dir,
        show=not args.no_show,
    )

    if args.n_components == 3:
        plot_ternary_simplex_evolution(
            patient_posteriors=patient_posteriors,
            patient_time_norm=patient_time_norm,
            control_ids=CONTROL_IDS,
            treatment_ids=TREATMENT_IDS,
            out_dir=out_dir,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
