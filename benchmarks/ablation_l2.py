#!/usr/bin/env python
"""
Ablation study for L² MMD-based Gaussian mixture fitting.

Generates three publication-quality figures:
1. Consistency contour: Final MMD² vs M (basis dim) and n (sample size)
2. Loss curves: Training MMD² for different K (fixed σ)
3. Chinchilla-style: Final MMD² for different K and σ
"""

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_l2_gaussian import generate_l2_gaussian_data
from src import GaussianKernel, GaussianMixtureModel, L2CosineBasis

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
GRID_SIZE = 100
T = 1.0
D = 2
DEVICE = torch.device("cpu")
DTYPE = torch.float64
TRUE_K = 5
TRUE_WEIGHTS = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10], dtype=DTYPE)
BASE_SEED = 42
N_SEEDS = 3

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "images"
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _median_sigma(X, max_samples=500):
    """Kernel bandwidth via median heuristic."""
    with torch.no_grad():
        X_sub = X
        if X.shape[0] > max_samples:
            X_sub = X[torch.randperm(X.shape[0])[:max_samples]]
        dists = torch.cdist(X_sub, X_sub)
        iu = torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)
        pos = dists[iu[0], iu[1]]
        pos = pos[pos > 0]
        return torch.median(pos).item() if pos.numel() > 0 else 1.0


def _train_mmd(X, basis, K, sigma, num_epochs, lr=0.1, seed=42,
               return_history=False):
    """Train an MMD GMM and return final MMD² (+ optional history)."""
    torch.manual_seed(seed)
    M = X.shape[1]
    kernel = GaussianKernel(sigma=sigma)

    model = GaussianMixtureModel(
        num_components=K,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",
        device=X.device,
        dtype=X.dtype,
    )
    model.initialize_from_data(X, method="kmeans++")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gram = kernel.compute_gram_matrix(X)
    const_term = gram.mean()

    history = []
    for _ in range(num_epochs):
        optimizer.zero_grad()
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()
        history.append((const_term + mmd2.detach()).item())

    if return_history:
        return history[-1], history
    return history[-1]


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Consistency contour  (M vs n)
# ──────────────────────────────────────────────────────────────────────
def _compute_mmd2_no_train(X, basis, K, sigma, seed=42):
    """Compute MMD² at initialization (no gradient steps)."""
    torch.manual_seed(seed)
    M = X.shape[1]
    kernel = GaussianKernel(sigma=sigma)
    model = GaussianMixtureModel(
        num_components=K,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",
        device=X.device,
        dtype=X.dtype,
    )
    model.initialize_from_data(X, method="kmeans++")
    with torch.no_grad():
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=True)
    return mmd2.item()


def plot_consistency():
    """Relative error of the MMD² estimator due to finite M and n."""
    print("\n" + "=" * 60)
    print("Plot 1: Consistency – MMD² approximation error (M vs n)")
    print("=" * 60)

    R_values = [3, 5, 8, 10, 15, 20, 25]
    n_values = [25, 50, 100, 200, 500]
    R_max = max(R_values)
    n_max = max(n_values)
    sigma = 1.2

    errors = np.zeros((len(R_values), len(n_values)))

    for si in range(N_SEEDS):
        seed = BASE_SEED + si
        print(f"\n  Seed {si + 1}/{N_SEEDS}")

        # Generate raw trajectories with R_max
        X_raw, _, _, _ = generate_l2_gaussian_data(
            n_samples=n_max,
            n_components=TRUE_K,
            grid_size=GRID_SIZE,
            R=R_max,
            T=T,
            d=D,
            component_weights=TRUE_WEIGHTS,
            seed=seed,
            device=DEVICE,
            dtype=DTYPE,
        )

        # Reference MMD² at (R_max, n_max) — the "true" value
        basis_ref = L2CosineBasis(
            T=T, R=R_max, grid_size=GRID_SIZE, d=D, device=DEVICE, dtype=DTYPE
        )
        X_ref = basis_ref.project(X_raw)
        mmd2_ref = _compute_mmd2_no_train(X_ref, basis_ref, TRUE_K, sigma, seed)
        print(f"    Reference MMD² (M={R_max * D}, n={n_max}): {mmd2_ref:.6f}")

        for i, R in enumerate(R_values):
            basis = L2CosineBasis(
                T=T, R=R, grid_size=GRID_SIZE, d=D, device=DEVICE, dtype=DTYPE
            )
            X_full = basis.project(X_raw)

            for j, n in enumerate(n_values):
                X_sub = X_full[:n]
                mmd2 = _compute_mmd2_no_train(X_sub, basis, TRUE_K, sigma, seed)
                rel_err = abs(mmd2 - mmd2_ref) / max(abs(mmd2_ref), 1e-15)
                errors[i, j] += rel_err
                M = R * D
                print(
                    f"    R={R:2d} (M={M:3d}), n={n:4d}: "
                    f"MMD²={mmd2:.6f}, rel_err={rel_err:.4f}"
                )

    errors /= N_SEEDS

    # ── Plotting ───────────────────────────────────────────────────
    M_values = np.array([R * D for R in R_values], dtype=float)
    n_arr = np.array(n_values, dtype=float)

    # Interpolate onto a fine grid (linear + nearest fill)
    pts = np.array([[m, n] for m in M_values for n in n_arr])
    vals = errors.flatten()
    M_fine = np.linspace(M_values.min(), M_values.max(), 200)
    n_fine = np.linspace(n_arr.min(), n_arr.max(), 200)
    Mg, ng = np.meshgrid(M_fine, n_fine)
    Zg = griddata(pts, vals, (Mg, ng), method="linear")
    mask = np.isnan(Zg)
    if mask.any():
        Zg[mask] = griddata(pts, vals, (Mg[mask], ng[mask]), method="nearest")
    Zg = np.clip(Zg, vals.min(), vals.max())

    fig, ax = plt.subplots(figsize=(4, 3))

    vmin, vmax = max(vals.min(), 1e-6), vals.max()
    if vmax / vmin > 10:
        levels = np.linspace(0, 3, 25) #np.logspace(np.log10(vmin), np.log10(vmax), 25)
        norm = None #mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        levels = np.linspace(vmin, vmax, 25)
        norm = None

    cf = ax.contourf(Mg, ng, Zg, levels=levels, cmap="viridis_r", norm=norm)
    ax.contour(Mg, ng, Zg, levels=levels[::3], colors="k", alpha=0.2,
               linewidths=0.5)
    cbar = fig.colorbar(cf, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    for m in M_values:
        for n in n_arr:
            ax.plot(m, n, "k.", markersize=3)

    ax.set_xlabel("Projection dimension $M$", fontsize=13)
    ax.set_ylabel("Sample size $n$", fontsize=13)
    ax.set_title(
        r"Relative error of $\widehat{\mathrm{MMD}}^2_{M,n}$",
        fontsize=14,
    )
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    path = os.path.join(SAVE_DIR, "ablation_consistency.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Loss curves for different K
# ──────────────────────────────────────────────────────────────────────
def plot_loss_curves():
    """Training MMD² vs epochs for different numbers of components K."""
    print("\n" + "=" * 60)
    print("Plot 2: Loss Curves (different K)")
    print("=" * 60)

    K_values = [2, 3, 4, 5, 6, 7, 8]
    sigma = 1.2
    R = 15
    n = 500
    epochs = 200

    X_raw, _, _, info = generate_l2_gaussian_data(
        n_samples=n,
        n_components=TRUE_K,
        grid_size=GRID_SIZE,
        R=R,
        T=T,
        d=D,
        component_weights=TRUE_WEIGHTS,
        seed=BASE_SEED,
        device=DEVICE,
        dtype=DTYPE,
    )
    basis = info["basis"]
    X = basis.project(X_raw)

    histories = {}
    for K in K_values:
        _, h = _train_mmd(
            X, basis, K, sigma, epochs, seed=BASE_SEED, return_history=True
        )
        histories[K] = h
        print(f"  K={K}: final MMD²={h[-1]:.6f}")

    # ── Plotting ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(K_values)))

    for idx, K in enumerate(K_values):
        lw = 2.5 if K == TRUE_K else 1.5
        alpha = 1.0 if K == TRUE_K else 0.8
        suffix = " (true)" if K == TRUE_K else ""
        ax.plot(
            range(1, epochs + 1),
            histories[K],
            color=colors[idx],
            linewidth=lw,
            alpha=alpha,
            label=f"$K={K}${suffix}",
        )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("MMD$^2$", fontsize=13)
    ax.set_title(
        f"Training for different $K$", fontsize=14
    )
    ax.set_yscale("log")
    ax.legend(fontsize=10, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    path = os.path.join(SAVE_DIR, "ablation_loss_K.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Chinchilla-style  (K × σ)
# ──────────────────────────────────────────────────────────────────────
def plot_k_sigma():
    """Final MMD² for different K and σ, styled after Chinchilla scaling plots."""
    print("\n" + "=" * 60)
    print("Plot 3: K vs σ (Chinchilla-style)")
    print("=" * 60)

    K_values = [2, 3, 4, 5, 6, 7, 8, 10]
    sigma_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    R = 15
    n = 500
    epochs = 300

    X_raw, _, _, info = generate_l2_gaussian_data(
        n_samples=n,
        n_components=TRUE_K,
        grid_size=GRID_SIZE,
        R=R,
        T=T,
        d=D,
        component_weights=TRUE_WEIGHTS,
        seed=BASE_SEED,
        device=DEVICE,
        dtype=DTYPE,
    )
    basis = info["basis"]
    X = basis.project(X_raw)

    results = {}
    for sigma in sigma_values:
        results[sigma] = {}
        for K in K_values:
            vals = []
            for si in range(N_SEEDS):
                mmd2 = _train_mmd(
                    X, basis, K, sigma, epochs, seed=BASE_SEED + si * 100
                )
                vals.append(mmd2)
            results[sigma][K] = np.mean(vals)
            print(f"  σ={sigma:.1f}, K={K}: MMD²={results[sigma][K]:.6f}")

    # ── Plotting ───────────────────────────────────────────────────
    # Chinchilla-style gradient: light green → teal → blue → dark purple
    chinchilla_cmap = mcolors.LinearSegmentedColormap.from_list(
        "chinchilla",
        ["#a8d8a8", "#78c878", "#48b8a0", "#3898c0",
         "#3070c0", "#4850b8", "#5838a0", "#301860"],
    )
    n_sigma = len(sigma_values)
    colors = [chinchilla_cmap(i / (n_sigma - 1)) for i in range(n_sigma)]

    fig, ax = plt.subplots(figsize=(4, 3))

    for idx, sigma in enumerate(sigma_values):
        K_arr = np.array(K_values, dtype=float)
        mmd_arr = np.array([results[sigma][K] for K in K_values])

        # Data points + solid line
        ax.plot(
            K_arr, mmd_arr, "o-",
            color=colors[idx], markersize=8, linewidth=2,
            markeredgecolor="white", markeredgewidth=0.5,
            label=f"$\\sigma={sigma}$", zorder=3,
        )

        # Dashed quadratic fit in log-space (extrapolated)
        log_K = np.log(K_arr)
        log_mmd = np.log(np.clip(mmd_arr, 1e-15, None))
        if len(log_K) >= 3:
            coeffs = np.polyfit(log_K, log_mmd, 2)
            K_fit = np.linspace(K_arr.min() * 0.85, K_arr.max() * 1.15, 100)
            ax.plot(
                K_fit, np.exp(np.polyval(coeffs, np.log(K_fit))),
                "--", color=colors[idx], alpha=0.4, linewidth=1.5, zorder=2,
            )

    ax.set_xlabel("Number of components $K$", fontsize=13)
    ax.set_ylabel("MMD$^2$", fontsize=13)
    ax.set_title("Final MMD$^2$ for different $K$ and $\\sigma$", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(K_values)
    ax.set_xticklabels([str(k) for k in K_values])
    ax.legend(fontsize=9, ncol=1, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.15, which="minor")
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    path = os.path.join(SAVE_DIR, "ablation_k_sigma.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    plot_consistency()
    plot_loss_curves()
    plot_k_sigma()
    print(f"\nAll ablation plots saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
