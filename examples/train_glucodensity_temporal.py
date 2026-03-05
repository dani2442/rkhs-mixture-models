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
from itertools import product
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
    plot_glucodensity_temporal_comparison,
    plot_glucodensity_variance,
    plot_cluster_probabilities_by_group,
    plot_ternary_simplex_evolution,
    plot_ternary_simplex_grid,
    plot_ternary_simplex_interactive,
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
        window_indices: np.ndarray of shape (total_windows,) — center day of
                        each sliding window (actual day number).
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

            # Use actual center day of the sliding window
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
# 5. Training representation helpers
# ---------------------------------------------------------------------------


def estimate_sigma_median_heuristic(
    X_time: torch.Tensor,
    mask: torch.Tensor | None = None,
    max_points: int = 2000,
) -> float:
    """Median heuristic on projected coefficients."""
    if mask is not None:
        all_coeffs = torch.cat([X_time[l][mask[l]] for l in range(X_time.shape[0])], dim=0)
    else:
        all_coeffs = X_time.reshape(-1, X_time.shape[-1])

    if all_coeffs.shape[0] < 2:
        return 1.0

    n_sub = min(max_points, all_coeffs.shape[0])
    idx = torch.randperm(all_coeffs.shape[0], device=all_coeffs.device)[:n_sub]
    sub = all_coeffs[idx]
    dists = torch.cdist(sub, sub)
    positive = dists[dists > 0]
    if positive.numel() == 0:
        return 1.0
    return float(torch.median(positive).item())


def build_training_representation(
    curves: np.ndarray,
    t_indices: np.ndarray,
    n_time_bins: int,
    r_s: int,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = True,
) -> dict:
    """Build temporal slices + projected coefficients for a given config."""
    X_time_raw, t_grid, mask, t_min_days, t_max_days = build_temporal_data(
        curves=curves,
        t_indices=t_indices,
        n_time_bins=n_time_bins,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )

    X_time, space_basis, coeff_mean, coeff_std = project_intraday_to_l2(
        X_time_raw=X_time_raw,
        R_s=r_s,
        device=device,
        dtype=dtype,
        mask=mask,
    )
    sigma_auto = estimate_sigma_median_heuristic(X_time=X_time, mask=mask)

    return {
        "X_time_raw": X_time_raw,
        "X_time": X_time,
        "t_grid": t_grid,
        "mask": mask,
        "t_min_days": t_min_days,
        "t_max_days": t_max_days,
        "space_basis": space_basis,
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "coeff_dim": int(X_time.shape[-1]),
        "L_t": int(X_time.shape[0]),
        "n_samples": int(X_time.shape[1]),
        "sigma_auto": sigma_auto,
    }


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
    verbose: bool = True,
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
        verbose=verbose,
        log_interval=max(1, num_epochs // 8),
        mask=mask,
    )

    with torch.no_grad():
        pi_t = model.time_weight_model()

    if verbose:
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
# 8. Group divergence analysis
# ---------------------------------------------------------------------------


def compute_group_divergence_stats(
    patient_posteriors: dict,
    patient_time_days: dict,
    control_ids: list[int],
    treatment_ids: list[int],
    n_time_bins: int = 12,
    clinical_w_divergence: float = 0.70,
    clinical_w_dynamics: float = 0.30,
) -> dict:
    """
    Compute treatment-vs-control separation from posterior trajectories.

    Separation per time bin is measured by TV distance between group means:
        TV(t) = 0.5 * || mean_treat(t) - mean_ctrl(t) ||_1

    We also quantify group dynamics:
      - control should remain stable (small stepwise TV drift),
      - treatment should change (larger drift / net change).

    Final clinical score (used by search/model selection when requested):
      clinical_score = w_div * increasing_score + w_dyn * dynamics_score
                       - dynamics_penalty

    where dynamics_penalty enforces the clinical preference:
      - treatment should evolve at least as much as control,
      - control should remain relatively stable.
    """
    if not patient_posteriors:
        return {
            "increasing_score": -np.inf,
            "dynamics_score": -np.inf,
            "clinical_score": -np.inf,
            "n_valid_bins": 0,
        }

    control_ids_set = set(control_ids)
    treatment_ids_set = set(treatment_ids)

    K = next(iter(patient_posteriors.values())).shape[1]

    control_data = []
    treatment_data = []
    for pid, posteriors in patient_posteriors.items():
        days = patient_time_days.get(pid)
        if days is None:
            continue
        for i in range(len(days)):
            entry = (float(days[i]), posteriors[i])
            if pid in control_ids_set:
                control_data.append(entry)
            elif pid in treatment_ids_set:
                treatment_data.append(entry)

    if not (control_data and treatment_data):
        return {
            "increasing_score": -np.inf,
            "dynamics_score": -np.inf,
            "clinical_score": -np.inf,
            "n_valid_bins": 0,
        }

    all_days = [d for d, _ in control_data + treatment_data]
    t_min = min(all_days)
    t_max = max(all_days)
    if t_max - t_min < 1e-9:
        t_max = t_min + 1.0

    bin_edges = np.linspace(t_min, t_max + 1e-9, n_time_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    def _bin_and_average(data):
        binned = [[] for _ in range(n_time_bins)]
        for day, post in data:
            b = min(
                int(np.searchsorted(bin_edges, day, side="right")) - 1,
                n_time_bins - 1,
            )
            b = max(0, b)
            binned[b].append(post)
        means = np.full((n_time_bins, K), np.nan)
        counts = np.zeros(n_time_bins, dtype=int)
        for b in range(n_time_bins):
            if binned[b]:
                arr = np.array(binned[b])
                means[b] = arr.mean(axis=0)
                counts[b] = len(binned[b])
        return means, counts

    ctrl_means, ctrl_counts = _bin_and_average(control_data)
    treat_means, treat_counts = _bin_and_average(treatment_data)

    def _trajectory_tv_stats(means: np.ndarray, counts: np.ndarray) -> dict:
        valid = np.where(counts > 0)[0]
        steps = np.full(max(0, n_time_bins - 1), np.nan)
        if len(valid) >= 2:
            traj = means[valid].copy()
            if len(valid) >= 3:
                traj_smooth = traj.copy()
                for i in range(len(valid)):
                    left = max(0, i - 1)
                    right = min(len(valid), i + 2)
                    traj_smooth[i] = traj[left:right].mean(axis=0)
                traj = traj_smooth

            step_vals = []
            for i in range(len(valid) - 1):
                left = valid[i]
                tv_step = 0.5 * np.sum(np.abs(traj[i + 1] - traj[i]))
                step_vals.append(float(tv_step))
                steps[left] = float(tv_step)
            net_tv = float(0.5 * np.sum(np.abs(traj[-1] - traj[0])))
            return {
                "valid_idx": valid,
                "step_tv": steps,
                "mean_step_tv": float(np.mean(step_vals)),
                "total_step_tv": float(np.sum(step_vals)),
                "net_tv": net_tv,
            }
        if len(valid) == 1:
            return {
                "valid_idx": valid,
                "step_tv": steps,
                "mean_step_tv": 0.0,
                "total_step_tv": 0.0,
                "net_tv": 0.0,
            }
        return {
            "valid_idx": valid,
            "step_tv": steps,
            "mean_step_tv": np.nan,
            "total_step_tv": np.nan,
            "net_tv": np.nan,
        }

    ctrl_traj = _trajectory_tv_stats(ctrl_means, ctrl_counts)
    treat_traj = _trajectory_tv_stats(treat_means, treat_counts)

    valid_both = (ctrl_counts > 0) & (treat_counts > 0)
    n_valid_bins = int(valid_both.sum())

    delta = np.full((n_time_bins, K), np.nan)
    sep_tv = np.full(n_time_bins, np.nan)
    if n_valid_bins > 0:
        delta[valid_both] = treat_means[valid_both] - ctrl_means[valid_both]
        sep_tv[valid_both] = 0.5 * np.sum(np.abs(delta[valid_both]), axis=1)

    valid_idx = np.where(valid_both)[0]
    if len(valid_idx) >= 2:
        t_valid = bin_centers[valid_idx]
        sep_valid = sep_tv[valid_idx]
        duration = max(float(t_valid[-1] - t_valid[0]), 1e-9)
        slope_per_day = float(np.polyfit(t_valid, sep_valid, 1)[0])
        slope_total = slope_per_day * duration
        start_sep = float(sep_valid[0])
        final_sep = float(sep_valid[-1])
        delta_sep = final_sep - start_sep
        auc_sep = float(np.trapz(sep_valid, t_valid) / duration)
        monotonic_ratio = float(np.mean(np.diff(sep_valid) >= -1e-10))
        # Main objective used by the hyperparameter search.
        increasing_score = 0.45 * final_sep + 0.35 * delta_sep + 0.20 * slope_total
    elif len(valid_idx) == 1:
        start_sep = float(sep_tv[valid_idx[0]])
        final_sep = start_sep
        delta_sep = 0.0
        slope_per_day = 0.0
        slope_total = 0.0
        auc_sep = start_sep
        monotonic_ratio = 0.0
        increasing_score = 0.25 * final_sep
    else:
        start_sep = np.nan
        final_sep = np.nan
        delta_sep = np.nan
        slope_per_day = np.nan
        slope_total = np.nan
        auc_sep = np.nan
        monotonic_ratio = np.nan
        increasing_score = -np.inf

    ctrl_mean_step_tv = ctrl_traj["mean_step_tv"]
    ctrl_total_step_tv = ctrl_traj["total_step_tv"]
    ctrl_net_tv = ctrl_traj["net_tv"]
    treat_mean_step_tv = treat_traj["mean_step_tv"]
    treat_total_step_tv = treat_traj["total_step_tv"]
    treat_net_tv = treat_traj["net_tv"]

    if np.isfinite(ctrl_mean_step_tv) and np.isfinite(treat_mean_step_tv):
        dynamics_contrast = treat_mean_step_tv - ctrl_mean_step_tv
        dynamics_score = (
            0.60 * dynamics_contrast
            + 0.25 * treat_net_tv
            - 0.15 * ctrl_net_tv
        )
    else:
        dynamics_contrast = np.nan
        dynamics_score = -np.inf

    if np.isfinite(ctrl_mean_step_tv) and np.isfinite(treat_mean_step_tv):
        control_step_target = 0.055
        dynamics_penalty = 0.0
        if treat_mean_step_tv <= ctrl_mean_step_tv:
            dynamics_penalty += 0.08
        dynamics_penalty += 2.0 * max(0.0, ctrl_mean_step_tv - treat_mean_step_tv)
        dynamics_penalty += 1.2 * max(0.0, ctrl_mean_step_tv - control_step_target)
    else:
        dynamics_penalty = 0.0

    w_sum = max(float(clinical_w_divergence + clinical_w_dynamics), 1e-12)
    w_div = float(clinical_w_divergence) / w_sum
    w_dyn = float(clinical_w_dynamics) / w_sum
    if np.isfinite(increasing_score) and np.isfinite(dynamics_score):
        clinical_score = w_div * increasing_score + w_dyn * dynamics_score - dynamics_penalty
    elif np.isfinite(increasing_score):
        clinical_score = increasing_score
    else:
        clinical_score = -np.inf

    if len(valid_idx) > 0:
        final_component_delta = delta[valid_idx[-1]]
    else:
        final_component_delta = np.full(K, np.nan)

    return {
        "bin_centers_days": bin_centers,
        "ctrl_means": ctrl_means,
        "treat_means": treat_means,
        "delta_means": delta,
        "ctrl_counts": ctrl_counts,
        "treat_counts": treat_counts,
        "valid_both": valid_both,
        "sep_tv": sep_tv,
        "n_valid_bins": n_valid_bins,
        "start_sep": start_sep,
        "final_sep": final_sep,
        "delta_sep": delta_sep,
        "slope_per_day": slope_per_day,
        "slope_total": slope_total,
        "auc_sep": auc_sep,
        "monotonic_ratio": monotonic_ratio,
        "increasing_score": increasing_score,
        "ctrl_step_tv": ctrl_traj["step_tv"],
        "treat_step_tv": treat_traj["step_tv"],
        "ctrl_mean_step_tv": ctrl_mean_step_tv,
        "ctrl_total_step_tv": ctrl_total_step_tv,
        "ctrl_net_tv": ctrl_net_tv,
        "treat_mean_step_tv": treat_mean_step_tv,
        "treat_total_step_tv": treat_total_step_tv,
        "treat_net_tv": treat_net_tv,
        "dynamics_contrast": dynamics_contrast,
        "dynamics_score": dynamics_score,
        "dynamics_penalty": dynamics_penalty,
        "clinical_score": clinical_score,
        "final_component_delta": final_component_delta,
    }


def print_group_divergence_summary(name: str, stats: dict) -> None:
    """Human-readable summary of treatment-vs-control separation metrics."""
    n_valid = int(stats.get("n_valid_bins", 0))
    divergence_score = stats.get("increasing_score", -np.inf)
    dynamics_score = stats.get("dynamics_score", -np.inf)
    clinical_score = stats.get("clinical_score", -np.inf)
    final_sep = stats.get("final_sep", np.nan)
    delta_sep = stats.get("delta_sep", np.nan)
    slope_total = stats.get("slope_total", np.nan)
    auc_sep = stats.get("auc_sep", np.nan)
    ctrl_mean_step_tv = stats.get("ctrl_mean_step_tv", np.nan)
    treat_mean_step_tv = stats.get("treat_mean_step_tv", np.nan)
    ctrl_net_tv = stats.get("ctrl_net_tv", np.nan)
    treat_net_tv = stats.get("treat_net_tv", np.nan)
    dynamics_penalty = stats.get("dynamics_penalty", np.nan)
    print(
        f"{name} metrics: "
        f"clinical={clinical_score:.6f}, divergence={divergence_score:.6f}, "
        f"dynamics={dynamics_score:.6f}, final_TV={final_sep:.6f}, "
        f"delta_TV={delta_sep:.6f}, slope_total={slope_total:.6f}, "
        f"AUC={auc_sep:.6f}, penalty={dynamics_penalty:.6f}, "
        f"ctrl_stepTV={ctrl_mean_step_tv:.6f}, "
        f"treat_stepTV={treat_mean_step_tv:.6f}, "
        f"ctrl_netTV={ctrl_net_tv:.6f}, treat_netTV={treat_net_tv:.6f}, "
        f"valid_bins={n_valid}"
    )


def parse_int_csv(value: str, default: list[int]) -> list[int]:
    if not value.strip():
        return list(default)
    parsed = [int(v.strip()) for v in value.split(",") if v.strip()]
    return sorted(set(parsed))


def parse_float_csv(value: str, default: list[float]) -> list[float]:
    if not value.strip():
        return list(default)
    parsed = [float(v.strip()) for v in value.split(",") if v.strip()]
    return sorted(set(parsed))


def parse_str_csv(value: str, default: list[str]) -> list[str]:
    if not value.strip():
        return list(default)
    parsed = [v.strip() for v in value.split(",") if v.strip()]
    return list(dict.fromkeys(parsed))  # preserve order, drop duplicates


def build_search_configs(args: argparse.Namespace) -> list[dict]:
    """Build sampled search configurations from CSV ranges."""
    model_types = [m for m in parse_str_csv(args.search_models, ["basis", "ode"]) if m in {"basis", "ode"}]
    n_components_list = parse_int_csv(args.search_components, [args.n_components])
    r_s_list = parse_int_csv(args.search_r_s, [args.r_s])
    n_time_bins_list = parse_int_csv(args.search_time_bins, [args.n_time_bins])
    r_pi_list = parse_int_csv(args.search_r_pi, [args.r_pi])
    ode_hidden_list = parse_int_csv(args.search_ode_hidden, [args.ode_hidden])
    lr_list = parse_float_csv(args.search_lrs, [args.lr])
    sigma_mult_list = parse_float_csv(args.search_sigma_mults, [1.0])

    full_grid = [
        {
            "model_type": model_type,
            "n_components": n_components,
            "r_s": r_s,
            "n_time_bins": n_time_bins,
            "r_pi": r_pi,
            "ode_hidden": ode_hidden,
            "lr": lr,
            "sigma_mult": sigma_mult,
        }
        for model_type, n_components, r_s, n_time_bins, r_pi, ode_hidden, lr, sigma_mult in product(
            model_types,
            n_components_list,
            r_s_list,
            n_time_bins_list,
            r_pi_list,
            ode_hidden_list,
            lr_list,
            sigma_mult_list,
        )
    ]
    if not full_grid:
        return []

    rng = np.random.RandomState(args.seed + 11)
    order = rng.permutation(len(full_grid))
    n_keep = min(args.search_trials, len(full_grid))
    return [full_grid[i] for i in order[:n_keep]]


def run_hyperparameter_search(
    curves: np.ndarray,
    t_indices: np.ndarray,
    patient_ids: list,
    window_days: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict | None, list[dict]]:
    """Search hyperparameters maximizing clinical or divergence objectives."""
    search_configs = build_search_configs(args)
    if not search_configs:
        return None, []

    objective_key = "clinical_score" if args.search_objective == "clinical" else "increasing_score"
    objective_label = "clinical score" if args.search_objective == "clinical" else "divergence score"

    print(f"\n[Search] Hyperparameter sweep maximizing {objective_label}...")
    print(f"  Trials requested: {len(search_configs)}")
    print("  Divergence score = 0.45*final_TV + 0.35*delta_TV + 0.20*slope_total")
    print("  Dynamics score  = 0.60*(treat_stepTV-ctrl_stepTV) + 0.25*treat_netTV - 0.15*ctrl_netTV")
    print(
        "  Clinical score  = "
        f"{args.clinical_w_divergence:.2f}*divergence + "
        f"{args.clinical_w_dynamics:.2f}*dynamics - penalty"
    )
    print("  Penalty         = 0.08[if treat_stepTV<=ctrl_stepTV]"
          " + 2.0*max(ctrl_stepTV-treat_stepTV,0) + 1.2*max(ctrl_stepTV-0.055,0)")

    repr_cache: dict[tuple[int, int], dict] = {}
    results: list[dict] = []

    for trial_idx, cfg in enumerate(search_configs, start=1):
        key = (cfg["n_time_bins"], cfg["r_s"])
        if key not in repr_cache:
            repr_cache[key] = build_training_representation(
                curves=curves,
                t_indices=t_indices,
                n_time_bins=cfg["n_time_bins"],
                r_s=cfg["r_s"],
                device=device,
                dtype=dtype,
                verbose=False,
            )
        rep = repr_cache[key]

        sigma_base = args.sigma if args.sigma > 0 else rep["sigma_auto"]
        sigma = max(1e-8, sigma_base * cfg["sigma_mult"])
        kernel = GaussianKernel(sigma=sigma)

        if cfg["model_type"] == "basis":
            time_basis = L2CosineBasis(
                T=1.0,
                R=cfg["r_pi"],
                grid_size=rep["L_t"],
                d=1,
                device=device,
                dtype=dtype,
            )
            time_weight_model = BasisLogitsTimeWeights(
                basis_matrix=time_basis.Phi,
                num_components=cfg["n_components"],
                device=device,
                dtype=dtype,
            )
        else:
            time_weight_model = NeuralODETimeWeights(
                t_grid=rep["t_grid"],
                num_components=cfg["n_components"],
                hidden_dim=cfg["ode_hidden"],
                device=device,
                dtype=dtype,
            )

        model, history, _ = run_experiment(
            name=f"Search-{trial_idx:02d}-{cfg['model_type']}",
            time_weight_model=time_weight_model,
            X_time=rep["X_time"],
            kernel=kernel,
            n_components=cfg["n_components"],
            coeff_dim=rep["coeff_dim"],
            num_epochs=args.search_epochs,
            lr=cfg["lr"],
            device=device,
            dtype=dtype,
            mask=rep["mask"],
            verbose=False,
        )

        patient_posteriors, _, patient_time_days = compute_all_patient_posteriors(
            curves=curves,
            t_indices=t_indices,
            patient_ids=patient_ids,
            window_days=window_days,
            model=model,
            space_basis=rep["space_basis"],
            coeff_mean=rep["coeff_mean"],
            coeff_std=rep["coeff_std"],
            t_grid=rep["t_grid"],
            t_min_days=rep["t_min_days"],
            t_max_days=rep["t_max_days"],
            device=device,
            dtype=dtype,
        )
        stats = compute_group_divergence_stats(
            patient_posteriors=patient_posteriors,
            patient_time_days=patient_time_days,
            control_ids=CONTROL_IDS,
            treatment_ids=TREATMENT_IDS,
            n_time_bins=args.analysis_time_bins,
            clinical_w_divergence=args.clinical_w_divergence,
            clinical_w_dynamics=args.clinical_w_dynamics,
        )

        result = dict(cfg)
        result.update(
            trial=trial_idx,
            score=float(stats[objective_key]),
            divergence_score=float(stats["increasing_score"]),
            dynamics_score=float(stats["dynamics_score"]),
            clinical_score=float(stats["clinical_score"]),
            dynamics_penalty=float(stats["dynamics_penalty"]),
            final_sep=float(stats["final_sep"]) if np.isfinite(stats["final_sep"]) else np.nan,
            delta_sep=float(stats["delta_sep"]) if np.isfinite(stats["delta_sep"]) else np.nan,
            ctrl_step_tv=float(stats["ctrl_mean_step_tv"]) if np.isfinite(stats["ctrl_mean_step_tv"]) else np.nan,
            treat_step_tv=float(stats["treat_mean_step_tv"]) if np.isfinite(stats["treat_mean_step_tv"]) else np.nan,
            slope_total=float(stats["slope_total"]) if np.isfinite(stats["slope_total"]) else np.nan,
            mmd=float(history[-1]),
            sigma=float(sigma),
            valid_bins=int(stats["n_valid_bins"]),
        )
        results.append(result)

        print(
            f"  Trial {trial_idx:02d}/{len(search_configs)} "
            f"[{cfg['model_type']}, K={cfg['n_components']}, r_s={cfg['r_s']}, "
            f"L_t={cfg['n_time_bins']}, r_pi={cfg['r_pi']}, lr={cfg['lr']:.4f}, "
            f"sigma={sigma:.4f}] -> obj={result['score']:.6f}, "
            f"div={result['divergence_score']:.6f}, dyn={result['dynamics_score']:.6f}, "
            f"pen={result['dynamics_penalty']:.6f}, "
            f"final_TV={result['final_sep']:.6f}, "
            f"ctrl_stepTV={result['ctrl_step_tv']:.6f}, treat_stepTV={result['treat_step_tv']:.6f}, "
            f"MMD²={result['mmd']:.6f}"
        )

    if not results:
        return None, []

    results_sorted = sorted(
        results,
        key=lambda r: (
            -(r["score"] if np.isfinite(r["score"]) else -np.inf),
            r["mmd"],
        ),
    )

    print(f"\n[Search] Top configurations by {objective_label}:")
    for rank, r in enumerate(results_sorted[: min(5, len(results_sorted))], start=1):
        print(
            f"  {rank}. obj={r['score']:.6f}, clinical={r['clinical_score']:.6f}, "
            f"div={r['divergence_score']:.6f}, dyn={r['dynamics_score']:.6f}, "
            f"pen={r['dynamics_penalty']:.6f}, "
            f"final_TV={r['final_sep']:.6f}, delta_TV={r['delta_sep']:.6f}, "
            f"MMD²={r['mmd']:.6f}, "
            f"type={r['model_type']}, K={r['n_components']}, r_s={r['r_s']}, "
            f"L_t={r['n_time_bins']}, r_pi={r['r_pi']}, "
            f"ode_hidden={r['ode_hidden']}, lr={r['lr']:.4f}, sigma={r['sigma']:.4f}"
        )

    return results_sorted[0], results_sorted


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Glucodensity temporal mixture fitting + divergence-driven tuning."
    )
    # Data / preprocessing
    p.add_argument(
        "--data-path",
        type=str,
        default="data/glucodensities/cgm_all_patients.csv",
        help="Path to CGM CSV relative to project root.",
    )
    p.add_argument("--max-prop-missing", type=float, default=0.20)
    p.add_argument("--block-size", type=int, default=4, help="Min days for patient inclusion")
    p.add_argument("--window-size", type=int, default=4, help="Sliding window size (days)")
    p.add_argument("--window-stride", type=int, default=1, help="Sliding window stride (days)")

    # Model
    p.add_argument("--n-components", type=int, default=3, help="Mixture components K")
    p.add_argument("--r-s", type=int, default=12, help="Cosine basis functions for intraday")
    p.add_argument("--n-time-bins", type=int, default=20, help="Number of temporal bins L_t")
    p.add_argument("--r-pi", type=int, default=6, help="Basis functions for pi(t)")
    p.add_argument("--ode-hidden", type=int, default=64, help="NeuralODE hidden dim")

    # Optimization
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0, help="Gaussian kernel bandwidth (0=auto median heuristic)")

    # Divergence analysis / model selection
    p.add_argument(
        "--analysis-time-bins",
        type=int,
        default=12,
        help="Time bins for control-vs-treatment posterior analysis.",
    )
    p.add_argument(
        "--selection-metric",
        choices=["clinical", "divergence", "mmd"],
        default="clinical",
        help="Criterion for selecting best model for posteriors/plots.",
    )
    p.add_argument(
        "--clinical-w-divergence",
        type=float,
        default=0.70,
        help="Weight on divergence score inside the clinical objective.",
    )
    p.add_argument(
        "--clinical-w-dynamics",
        type=float,
        default=0.30,
        help="Weight on group-dynamics score inside the clinical objective.",
    )

    # Hyperparameter search
    p.add_argument(
        "--search-trials",
        type=int,
        default=0,
        help="If >0, run random sampled sweep and choose config by search objective.",
    )
    p.add_argument(
        "--search-objective",
        choices=["clinical", "divergence"],
        default="clinical",
        help="Objective optimized by hyperparameter search.",
    )
    p.add_argument(
        "--search-epochs",
        type=int,
        default=120,
        help="Epochs per trial for hyperparameter search.",
    )
    p.add_argument(
        "--search-models",
        type=str,
        default="basis,ode",
        help="Comma-separated model families for search: basis,ode.",
    )
    p.add_argument(
        "--search-components",
        type=str,
        default="",
        help="Comma-separated K values for search (empty uses --n-components).",
    )
    p.add_argument(
        "--search-r-s",
        type=str,
        default="",
        help="Comma-separated intraday basis sizes (empty uses --r-s).",
    )
    p.add_argument(
        "--search-time-bins",
        type=str,
        default="",
        help="Comma-separated temporal bin counts (empty uses --n-time-bins).",
    )
    p.add_argument(
        "--search-r-pi",
        type=str,
        default="",
        help="Comma-separated basis sizes for temporal logits (empty uses --r-pi).",
    )
    p.add_argument(
        "--search-ode-hidden",
        type=str,
        default="",
        help="Comma-separated ODE hidden dims (empty uses --ode-hidden).",
    )
    p.add_argument(
        "--search-lrs",
        type=str,
        default="",
        help="Comma-separated learning rates for search (empty uses --lr).",
    )
    p.add_argument(
        "--search-sigma-mults",
        type=str,
        default="0.8,1.0,1.2",
        help="Multipliers on base sigma (base=auto or --sigma).",
    )

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
    print("Glucodensity Temporal Mixture: Divergence-aware analysis")
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

    selected_cfg = {
        "n_components": args.n_components,
        "r_s": args.r_s,
        "n_time_bins": args.n_time_bins,
        "r_pi": args.r_pi,
        "ode_hidden": args.ode_hidden,
        "lr": args.lr,
        "sigma_mult": 1.0,
        "selected_from_search_model_type": "n/a",
    }

    if args.search_trials > 0:
        best_search, _ = run_hyperparameter_search(
            curves=curves,
            t_indices=t_indices,
            patient_ids=patient_ids,
            window_days=window_days,
            args=args,
            device=device,
            dtype=dtype,
        )
        if best_search is not None:
            selected_cfg.update(
                n_components=best_search["n_components"],
                r_s=best_search["r_s"],
                n_time_bins=best_search["n_time_bins"],
                r_pi=best_search["r_pi"],
                ode_hidden=best_search["ode_hidden"],
                lr=best_search["lr"],
                sigma_mult=best_search["sigma_mult"],
                selected_from_search_model_type=best_search["model_type"],
            )
            print("\n[Search] Selected config for final training:")
            print(
                f"  model_type_hint={selected_cfg['selected_from_search_model_type']}, "
                f"K={selected_cfg['n_components']}, r_s={selected_cfg['r_s']}, "
                f"L_t={selected_cfg['n_time_bins']}, r_pi={selected_cfg['r_pi']}, "
                f"ode_hidden={selected_cfg['ode_hidden']}, lr={selected_cfg['lr']:.4f}, "
                f"sigma_mult={selected_cfg['sigma_mult']:.3f}"
            )

    # ----- Step 3/4: Build final representation with selected config -----
    print(
        f"\n[Step 3] Building final representation with "
        f"K={selected_cfg['n_components']}, r_s={selected_cfg['r_s']}, "
        f"L_t={selected_cfg['n_time_bins']}..."
    )
    rep = build_training_representation(
        curves=curves,
        t_indices=t_indices,
        n_time_bins=selected_cfg["n_time_bins"],
        r_s=selected_cfg["r_s"],
        device=device,
        dtype=dtype,
        verbose=True,
    )
    X_time = rep["X_time"]
    t_grid = rep["t_grid"]
    mask = rep["mask"]
    space_basis = rep["space_basis"]
    coeff_mean = rep["coeff_mean"]
    coeff_std = rep["coeff_std"]
    t_min_days = rep["t_min_days"]
    t_max_days = rep["t_max_days"]
    coeff_dim = rep["coeff_dim"]
    L_t = rep["L_t"]
    n_samples = rep["n_samples"]
    print(f"  X_time shape: (L_t={L_t}, n={n_samples}, M_s={coeff_dim})")

    # ----- Step 5: Set up kernel and time basis -----
    sigma_base = args.sigma if args.sigma > 0 else rep["sigma_auto"]
    if args.sigma <= 0:
        print(f"  Auto sigma (median heuristic): {sigma_base:.4f}")
    sigma = max(1e-8, sigma_base * selected_cfg["sigma_mult"])
    print(f"  Final sigma: {sigma:.4f}")
    kernel = GaussianKernel(sigma=sigma)

    time_basis = L2CosineBasis(
        T=1.0,
        R=selected_cfg["r_pi"],
        grid_size=L_t,
        d=1,
        device=device,
        dtype=dtype,
    )

    n_components = selected_cfg["n_components"]
    final_lr = selected_cfg["lr"]

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
        lr=final_lr,
        device=device,
        dtype=dtype,
        mask=mask,
        verbose=True,
    )

    # ----- Step 7: Direct MMD with NeuralODE temporal weights -----
    print(f"\n[Experiment 2] Direct MMD: NeuralODE temporal weights (K={n_components})...")
    ode_weight_model = NeuralODETimeWeights(
        t_grid=t_grid,
        num_components=n_components,
        hidden_dim=selected_cfg["ode_hidden"],
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
        lr=final_lr,
        device=device,
        dtype=dtype,
        mask=mask,
        verbose=True,
    )

    # ----- Step 8: Analyze group divergence for each model -----
    patient_posteriors_basis, patient_time_norm_basis, patient_time_days_basis = compute_all_patient_posteriors(
        curves=curves,
        t_indices=t_indices,
        patient_ids=patient_ids,
        window_days=window_days,
        model=model_basis,
        space_basis=space_basis,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        t_grid=t_grid,
        t_min_days=t_min_days,
        t_max_days=t_max_days,
        device=device,
        dtype=dtype,
    )
    div_basis = compute_group_divergence_stats(
        patient_posteriors=patient_posteriors_basis,
        patient_time_days=patient_time_days_basis,
        control_ids=CONTROL_IDS,
        treatment_ids=TREATMENT_IDS,
        n_time_bins=args.analysis_time_bins,
        clinical_w_divergence=args.clinical_w_divergence,
        clinical_w_dynamics=args.clinical_w_dynamics,
    )
    print_group_divergence_summary("Direct-Basis", div_basis)

    patient_posteriors_ode, patient_time_norm_ode, patient_time_days_ode = compute_all_patient_posteriors(
        curves=curves,
        t_indices=t_indices,
        patient_ids=patient_ids,
        window_days=window_days,
        model=model_ode,
        space_basis=space_basis,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        t_grid=t_grid,
        t_min_days=t_min_days,
        t_max_days=t_max_days,
        device=device,
        dtype=dtype,
    )
    div_ode = compute_group_divergence_stats(
        patient_posteriors=patient_posteriors_ode,
        patient_time_days=patient_time_days_ode,
        control_ids=CONTROL_IDS,
        treatment_ids=TREATMENT_IDS,
        n_time_bins=args.analysis_time_bins,
        clinical_w_divergence=args.clinical_w_divergence,
        clinical_w_dynamics=args.clinical_w_dynamics,
    )
    print_group_divergence_summary("Direct-NeuralODE", div_ode)

    # ----- Step 9: Choose model for final plots -----
    if args.selection_metric == "mmd":
        best_model = model_basis if history_basis[-1] <= history_ode[-1] else model_ode
        best_name = "Basis" if history_basis[-1] <= history_ode[-1] else "NeuralODE"
        patient_posteriors = (
            patient_posteriors_basis if best_name == "Basis" else patient_posteriors_ode
        )
        patient_time_norm = (
            patient_time_norm_basis if best_name == "Basis" else patient_time_norm_ode
        )
        patient_time_days = (
            patient_time_days_basis if best_name == "Basis" else patient_time_days_ode
        )
    else:
        metric_key = "clinical_score" if args.selection_metric == "clinical" else "increasing_score"
        basis_score = div_basis[metric_key]
        ode_score = div_ode[metric_key]
        if basis_score >= ode_score:
            best_model = model_basis
            best_name = "Basis"
            patient_posteriors = patient_posteriors_basis
            patient_time_norm = patient_time_norm_basis
            patient_time_days = patient_time_days_basis
        else:
            best_model = model_ode
            best_name = "NeuralODE"
            patient_posteriors = patient_posteriors_ode
            patient_time_norm = patient_time_norm_ode
            patient_time_days = patient_time_days_ode

    n_ctrl = sum(1 for pid in patient_posteriors if pid in CONTROL_IDS)
    n_treat = sum(1 for pid in patient_posteriors if pid in TREATMENT_IDS)
    print(f"\n[Step 9] Using {best_name} for patient trajectory plots "
          f"(selection metric: {args.selection_metric})")
    print(f"  Patients with posteriors: {len(patient_posteriors)} "
          f"(control: {n_ctrl}, treatment: {n_treat})")

    # ----- Step 10: Summary -----
    print("\n" + "=" * 72)
    print("Final comparison")
    print("=" * 72)
    print(f"  Direct-Basis    MMD²(avg_t): {history_basis[-1]:.6f}")
    print(f"  Direct-NeuralODE MMD²(avg_t): {history_ode[-1]:.6f}")
    print(f"  Improvement (Basis→ODE): {history_basis[-1] - history_ode[-1]:.6f}")
    print(
        f"  Direct-Basis scores: clinical={div_basis['clinical_score']:.6f}, "
        f"divergence={div_basis['increasing_score']:.6f}, dynamics={div_basis['dynamics_score']:.6f}, "
        f"penalty={div_basis['dynamics_penalty']:.6f}"
    )
    print(
        f"  Direct-NeuralODE scores: clinical={div_ode['clinical_score']:.6f}, "
        f"divergence={div_ode['increasing_score']:.6f}, dynamics={div_ode['dynamics_score']:.6f}, "
        f"penalty={div_ode['dynamics_penalty']:.6f}"
    )
    print(
        f"  Mean step TV (control vs treatment): "
        f"Basis=({div_basis['ctrl_mean_step_tv']:.6f}, {div_basis['treat_mean_step_tv']:.6f}), "
        f"ODE=({div_ode['ctrl_mean_step_tv']:.6f}, {div_ode['treat_mean_step_tv']:.6f})"
    )
    print(
        f"  Selected model ({args.selection_metric}): {best_name} | "
        f"Basis final_TV={div_basis['final_sep']:.6f}, ODE final_TV={div_ode['final_sep']:.6f}"
    )

    # ----- Step 11: Visualization -----
    out_dir = os.path.join(project_root, "paper", "images")

    # Prepare numpy arrays for visualization
    with torch.no_grad():
        means_basis = model_basis.mean.cpu()
        means_ode = model_ode.mean.cpu()
        var_basis = model_basis.variance.cpu().numpy()
        var_ode = model_ode.variance.cpu().numpy()

    means_basis_orig = means_basis * coeff_std.cpu() + coeff_mean.cpu()
    means_ode_orig = means_ode * coeff_std.cpu() + coeff_mean.cpu()
    recon_basis = space_basis.reconstruct(means_basis_orig)
    recon_ode = space_basis.reconstruct(means_ode_orig)
    recon_basis_np = recon_basis.detach().squeeze(-1).numpy()
    recon_ode_np = recon_ode.detach().squeeze(-1).numpy()

    t_np = t_grid.detach().cpu().numpy()
    pi_basis_np = pi_basis.cpu().numpy()
    pi_ode_np = pi_ode.cpu().numpy()

    # Convert normalized t_grid [0,1] back to actual treatment days
    day_range = t_max_days - t_min_days
    t_grid_days_np = t_np * day_range + t_min_days

    plot_glucodensity_temporal_comparison(
        t_grid_np=t_grid_days_np,
        history_basis=history_basis,
        history_ode=history_ode,
        pi_basis_np=pi_basis_np,
        pi_ode_np=pi_ode_np,
        recon_basis_np=recon_basis_np,
        recon_ode_np=recon_ode_np,
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
        n_time_bins=args.analysis_time_bins,
        out_dir=out_dir,
        show=not args.no_show,
    )

    if n_components == 3:
        # plot_ternary_simplex_evolution(
        #     patient_posteriors=patient_posteriors,
        #     patient_time_norm=patient_time_norm,
        #     control_ids=CONTROL_IDS,
        #     treatment_ids=TREATMENT_IDS,
        #     out_dir=out_dir,
        #     show=not args.no_show,
        # )

        # plot_ternary_simplex_grid(
        #     patient_posteriors=patient_posteriors,
        #     patient_time_norm=patient_time_norm,
        #     control_ids=CONTROL_IDS,
        #     treatment_ids=TREATMENT_IDS,
        #     n_cols=4,
        #     n_rows=4,
        #     out_dir=out_dir,
        #     show=not args.no_show,
        # )

        plot_ternary_simplex_interactive(
            patient_posteriors=patient_posteriors,
            patient_time_days=patient_time_days,
            control_ids=CONTROL_IDS,
            treatment_ids=TREATMENT_IDS,
            n_time_steps=30,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
