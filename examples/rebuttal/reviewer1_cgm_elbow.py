#!/usr/bin/env python
"""
W1/Q1: within-cluster MMD^2 vs K (model-selection elbow) for the three CGM
case-study representations:
  (a) intraday H^1 glucose curves        -> expect elbow at K=2
  (b) 24x24 inter-hour correlation mats  -> expect elbow at K=3
  (c) individual-similarity graph signals-> expect elbow at K=3

Reuses the paper's own preprocessing / bases. Static MMD-GMM is fit for each
candidate K (diagonal covariance, Gaussian kernel, median-heuristic bandwidth,
several k-means++ restarts, best retained). This reproduces the "within-cluster
MMD across candidate K + elbow" model-selection rule (Franca et al.).
"""
import os
import sys
import json

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from examples.train_glucodensity_temporal import (
    load_and_preprocess_cgm, compute_sliding_windows,
)
from examples.train_glucodensity_correlation import compute_sliding_window_correlations
from examples.train_glucodensity_correlation_graph import threshold_to_adjacency
from src import (
    GaussianKernel, GaussianMixtureModel, GraphLaplacianBasis, SymmetricMatrixBasis,
)
from src.spaces import H1CosineBasis
from src.mixture import fit_gaussian_mixture_mmd

DEVICE = torch.device("cpu")
DTYPE = torch.float64
CSV = os.path.join(ROOT, "data/glucodensities/cgm_all_patients.csv")
IMG = os.path.join(ROOT, "paper", "images")
KS = [1, 2, 3, 4, 5, 6]
RESTARTS = 5
EPOCHS = 300

# graph representation (mirrors examples/patient_correlation_network_graph.ipynb)
G_WINDOW = 21
G_MIN_PRESENCE = 0.4
G_EDGE_PERCENTILE = 85
G_RANK = 4
G_ALPHA = 1.0
G_R_S = 16
N_SLOTS = 288


def median_sigma(X, max_samples=1500):
    with torch.no_grad():
        Xs = X if X.shape[0] <= max_samples else X[torch.randperm(X.shape[0])[:max_samples]]
        d = torch.cdist(Xs, Xs)
        iu = torch.triu_indices(d.shape[0], d.shape[1], 1)
        pos = d[iu[0], iu[1]]
        pos = pos[pos > 0]
        return float(pos.median()) if pos.numel() else 1.0


def normalize(coeffs):
    mu = coeffs.mean(0, keepdim=True)
    sd = coeffs.std(0, keepdim=True).clamp_min(1e-8)
    return (coeffs - mu) / sd


def mmd_curve(X, name, seed=42):
    torch.manual_seed(seed)
    sigma = median_sigma(X)
    kernel = GaussianKernel(sigma=sigma)
    curve = []
    for K in KS:
        best = np.inf
        for r in range(RESTARTS):
            torch.manual_seed(seed + 100 * r)
            _, hist = fit_gaussian_mixture_mmd(
                X, num_components=K, kernel=kernel, num_epochs=EPOCHS, lr=0.05,
                covariance_type="diagonal", init_method="kmeans++", verbose=False)
            best = min(best, hist[-1])
        curve.append(best)
        print(f"    [{name}] K={K}: within-cluster MMD^2={best:.6e}", flush=True)
    return sigma, curve


def build_intraday(patient_data, W=4, R_s=8, max_n=4000, seed=42):
    curves, t_idx, pids, wdays, _ = compute_sliding_windows(
        patient_data, window_size=W, stride=1, min_valid_frac=0.5, verbose=True)
    curves_t = torch.tensor(curves, device=DEVICE, dtype=DTYPE)  # (N,288)
    basis = H1CosineBasis(T=1.0, R=R_s, grid_size=288, d=1, device=DEVICE, dtype=DTYPE)
    coeffs = basis.project(curves_t.unsqueeze(-1))               # (N, R_s)
    coeffs = normalize(coeffs)
    if coeffs.shape[0] > max_n:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(coeffs.shape[0], generator=g)[:max_n]
        coeffs = coeffs[idx]
    print(f"  intraday features: {tuple(coeffs.shape)}")
    return coeffs


def build_correlation(patient_data, W=14, max_n=2000, seed=42):
    corr, t_idx, pids, wdays, _ = compute_sliding_window_correlations(
        patient_data, window_size=W, stride=1, shrinkage=0.1, verbose=True)
    corr_t = torch.tensor(corr, device=DEVICE, dtype=DTYPE)      # (N,24,24)
    basis = SymmetricMatrixBasis(n=24, device=DEVICE, dtype=DTYPE)  # full vech M=300
    coeffs = basis.project(corr_t)                               # (N, 300)
    coeffs = normalize(coeffs)
    if coeffs.shape[0] > max_n:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(coeffs.shape[0], generator=g)[:max_n]
        coeffs = coeffs[idx]
    print(f"  correlation features: {tuple(coeffs.shape)}")
    return coeffs


def build_graph(patient_data, verbose=True):
    """Individual-similarity graph signals, as in the case study notebook.

    Per sliding window, patients are compared through their H^1 curve
    coefficients; the resulting N_pat x N_pat similarity matrix is treated as a
    vector-valued signal on a fixed patient graph and projected onto the leading
    Laplacian eigenvectors.
    """
    # per-patient window-averaged daily curves
    patient_window_curves = {}
    for pid, days_matrix in patient_data.items():
        pw = {}
        for w in range(0, days_matrix.shape[0] - G_WINDOW + 1):
            avg_curve = np.nanmean(days_matrix[w: w + G_WINDOW], axis=0)
            nan_mask = np.isnan(avg_curve)
            if nan_mask.all():
                continue
            if nan_mask.any():
                valid_idx = np.where(~nan_mask)[0]
                avg_curve = np.interp(np.arange(N_SLOTS), valid_idx, avg_curve[valid_idx])
            pw[w] = avg_curve
        if pw:
            patient_window_curves[pid] = pw

    all_w = sorted({w for pw in patient_window_curves.values() for w in pw})
    w_pids = {w: {pid for pid, pw in patient_window_curves.items() if w in pw} for w in all_w}
    presence = {pid: sum(1 for w in all_w if pid in w_pids[w]) / len(all_w)
                for pid in patient_window_curves}
    common_pids = sorted(pid for pid, f in presence.items() if f >= G_MIN_PRESENCE)
    N_pat = len(common_pids)
    valid_w = [w for w in all_w if all(pid in w_pids[w] for pid in common_pids)]
    if verbose:
        print(f"  common patients: {N_pat}, valid windows: {len(valid_w)}/{len(all_w)}")

    # patient-to-patient similarity per window, from H^1 curve coefficients
    curve_basis = H1CosineBasis(T=1.0, R=G_R_S, grid_size=N_SLOTS, d=1,
                                device=DEVICE, dtype=DTYPE)
    dists = np.zeros((len(valid_w), N_pat, N_pat))
    for idx, w in enumerate(valid_w):
        curves = np.stack([patient_window_curves[pid][w] for pid in common_pids])
        curves_t = torch.tensor(curves, device=DEVICE, dtype=DTYPE).unsqueeze(-1)
        with torch.no_grad():
            c = curve_basis.project(curves_t).cpu().numpy()
        dists[idx] = squareform(pdist(c, metric="euclidean"))
    sim = 1.0 - dists / (dists.max() + 1e-10)

    # fixed patient graph from the time-averaged similarity
    avg_sim = sim.mean(axis=0)
    np.fill_diagonal(avg_sim, 0.0)
    thr = np.percentile(avg_sim[~np.eye(N_pat, dtype=bool)], G_EDGE_PERCENTILE)
    weights = avg_sim * threshold_to_adjacency(avg_sim, threshold=thr, absolute=False)
    graph_basis = GraphLaplacianBasis.from_adjacency(
        adjacency=torch.tensor(weights, device=DEVICE, dtype=DTYPE),
        alpha=G_ALPHA, num_eigenvectors=min(G_RANK, N_pat), normalized=False,
        device=DEVICE, dtype=DTYPE)

    coeffs = graph_basis.project(torch.tensor(sim, device=DEVICE, dtype=DTYPE))
    coeffs = normalize(coeffs)
    print(f"  graph features: {tuple(coeffs.shape)}")
    return coeffs


def main():
    print("[load]")
    patient_data = load_and_preprocess_cgm(CSV, max_prop_missing=0.20, block_size=4, verbose=True)

    results = {}
    print("\n[intraday H^1]")
    Xi = build_intraday(patient_data)
    s_i, c_i = mmd_curve(Xi, "intraday")
    results["intraday"] = {"sigma": s_i, "curve": c_i, "M": Xi.shape[1], "n": Xi.shape[0]}

    print("\n[correlation Sym(24)]")
    Xc = build_correlation(patient_data)
    s_c, c_c = mmd_curve(Xc, "correlation")
    results["correlation"] = {"sigma": s_c, "curve": c_c, "M": Xc.shape[1], "n": Xc.shape[0]}

    print("\n[patient-similarity graph signals]")
    Xg = build_graph(patient_data)
    s_g, c_g = mmd_curve(Xg, "graph")
    results["graph"] = {"sigma": s_g, "curve": c_g, "M": Xg.shape[1], "n": Xg.shape[0]}

    with open(os.path.join(RESULTS, "cgm_elbow.json"), "w") as f:
        json.dump({"KS": KS, **results}, f, indent=2)
    print("\nDONE (intraday + correlation + graph)")


if __name__ == "__main__":
    main()
