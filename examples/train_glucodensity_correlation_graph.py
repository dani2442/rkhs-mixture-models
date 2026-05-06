"""
Sliding-window hourly graph-signal pipeline for glucodensity data:
helpers used by the use_case_visualization notebook.
"""
from typing import Tuple

import numpy as np
import torch


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

