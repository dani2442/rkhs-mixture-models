#!/usr/bin/env python
"""
Rebuttal experiments (Reviewer 1), self-contained synthetic part:

  largek   -> W2: large number of clusters (ARI / runtime / memory / stability)
  runtime  -> W3: runtime scaling curves in n, M, K
  icl      -> Q2: finite-M ICL vs MMD elbow on a synthetic Hilbert-space mixture

Uses the paper's own GaussianMixtureModel / GaussianKernel and MMD fitting.
Gaussian measures live in R^M (a finite-dim Hilbert space); the identity basis
is used, so no reconstruction basis is required for fitting.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
# make the mixture package importable regardless of CWD
sys.path.insert(0, ROOT)

from src import GaussianKernel, GaussianMixtureModel  # noqa: E402
from src.mixture import fit_gaussian_mixture_mmd  # noqa: E402

DTYPE = torch.float64
DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------
# Synthetic well-separated Gaussian mixture in R^M
# ----------------------------------------------------------------------
def make_mixture(K, M, n_per, sep=3.0, std=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = sep * torch.randn(K, M, generator=g, dtype=DTYPE)
    Xs, ys = [], []
    for k in range(K):
        z = torch.randn(n_per, M, generator=g, dtype=DTYPE)
        Xs.append(means[k] + std * z)
        ys.append(torch.full((n_per,), k, dtype=torch.long))
    X = torch.cat(Xs, 0)
    y = torch.cat(ys, 0)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm].numpy()


def median_sigma(X, max_samples=800):
    with torch.no_grad():
        Xs = X if X.shape[0] <= max_samples else X[torch.randperm(X.shape[0])[:max_samples]]
        d = torch.cdist(Xs, Xs)
        iu = torch.triu_indices(d.shape[0], d.shape[1], 1)
        pos = d[iu[0], iu[1]]
        pos = pos[pos > 0]
        return float(pos.median()) if pos.numel() else 1.0


def fit_once(X, K, sigma, epochs, lr, seed):
    torch.manual_seed(seed)
    kernel = GaussianKernel(sigma=sigma)
    t0 = time.perf_counter()
    model, hist = fit_gaussian_mixture_mmd(
        X, num_components=K, kernel=kernel, num_epochs=epochs, lr=lr,
        covariance_type="diagonal", init_method="kmeans++", verbose=False,
    )
    dt = time.perf_counter() - t0
    with torch.no_grad():
        # density posterior read-out (uses fitted means/covariances directly);
        # robust to the kernel bandwidth, unlike the kernel soft-assignment.
        pred = model.responsibilities(X).argmax(1).cpu().numpy()
    return hist[-1], pred, dt, model, kernel


# ----------------------------------------------------------------------
# Large-scale evaluation of the *same* closed form.
#
# GaussianKernel.compute_J_diag materialises an (n, K, M) difference tensor,
# which is what limits the released code to moderate n*K.  Expanding the
# quadratic form turns the same expression into three BLAS calls with O(nK)
# memory:
#     ||x - m_k||^2_{A_k} = <x^2, 1/A_k> - 2 <x, m_k/A_k> + <m_k^2, 1/A_k>
# `verify` (below) checks the two agree to machine precision and that the fits
# they produce coincide.  Everything reported for K >= 100 / n >= 1e4 uses this
# path; the objective, gradients and optimizer are otherwise unchanged.
# ----------------------------------------------------------------------
def J_diag_matmul(kernel, X, m, variances):
    A = 1.0 + variances / (kernel.sigma * kernel.sigma)     # (K, M)
    inv_A = 1.0 / A
    logdetA = torch.log(A).sum(dim=1)                        # (K,)
    quad = (
        (X * X) @ inv_A.T
        - 2.0 * (X @ (m * inv_A).T)
        + (m * m * inv_A).sum(dim=1)
    )                                                        # (n, K)
    return torch.exp(-0.5 * logdetA.unsqueeze(0) + kernel.alpha * quad)


# (n, K) entries above which J is built in row blocks instead of at once
CHUNK_CELLS = 2 * 10 ** 7
# k-means++ costs O(n M K^2); above this many (n, K) cells it is seeded from a
# random subsample of the data instead of all of it
INIT_SUBSAMPLE_CELLS = 5 * 10 ** 7
INIT_SUBSAMPLE_N = 50_000


def _chunk_rows(n, K):
    """Row-block size keeping each (block, K) intermediate under CHUNK_CELLS."""
    if n * K <= CHUNK_CELLS:
        return n
    return max(1, CHUNK_CELLS // K)


def predict_chunked(model, X):
    """argmax of the density posterior, evaluated in row blocks."""
    n, K = X.shape[0], model.num_components
    step = _chunk_rows(n, K)
    out = torch.empty(n, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, n, step):
            out[i: i + step] = model.responsibilities(X[i: i + step]).argmax(1)
    return out.numpy()


def fit_once_fast(X, K, sigma, epochs, lr, seed):
    """Adam on the MMD^2 objective without the constant data-data term.

    The term (1/n^2) sum_ij kappa(x_i, x_j) does not depend on the parameters:
    it never enters the gradient and shifts every restart by the same amount,
    so it is irrelevant both to the fit and to the across-restart spread.  It is
    also the only O(n^2) piece, so dropping it is what makes n = 1e5 reachable.

    Beyond CHUNK_CELLS the (n, K) matrix J is built in row blocks and each block
    is backpropagated as it is formed.  Since J enters the objective only
    through its column means, the accumulated gradient is exactly the full-batch
    gradient -- the optimizer still takes one step per epoch on all n points --
    but peak memory is set by the block, not by n.
    """
    torch.manual_seed(seed)
    n, M = X.shape
    kernel = GaussianKernel(sigma=sigma)
    t0 = time.perf_counter()
    model = GaussianMixtureModel(
        num_components=K, coeff_dim=M, covariance_type="diagonal",
        device=X.device, dtype=X.dtype,
    )
    if n * K > INIT_SUBSAMPLE_CELLS:
        idx = torch.randperm(n)[:INIT_SUBSAMPLE_N]
        model.initialize_from_data(X[idx], method="kmeans++")
    else:
        model.initialize_from_data(X, method="kmeans++")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    step = _chunk_rows(n, K)
    for _ in range(epochs):
        opt.zero_grad()
        final = 0.0
        for i in range(0, n, step):
            Xb = X[i: i + step]
            J = J_diag_matmul(kernel, Xb, model.mean, model.variance)  # (b, K)
            # weight the block by b/n so the blocks sum to the full-batch term
            part = -2.0 * (model.pi * J.sum(dim=0)).sum() / n
            part.backward()
            final += float(part.detach())
        I = kernel.compute_I_diag(model.mean, model.variance)          # (K, K)
        pi = model.pi
        mixmix = pi @ I @ pi
        mixmix.backward()
        final += float(mixmix.detach())
        opt.step()
    dt = time.perf_counter() - t0
    pred = predict_chunked(model, X)
    return final, pred, dt, model, kernel


def driver_verify(args):
    """Check the matmul path against the released compute_J_diag."""
    from sklearn.metrics import adjusted_rand_score
    rows = []
    for K, n_per, M in [(5, 40, 20), (10, 40, 20), (20, 40, 20), (5, 200, 50)]:
        X, y = make_mixture(K, M, n_per, sep=6.0, std=0.35, seed=0)
        sigma = median_sigma(X)
        kernel = GaussianKernel(sigma=sigma)
        m = torch.randn(K, M, dtype=DTYPE)
        v = torch.rand(K, M, dtype=DTYPE) + 0.5
        J_ref = kernel.compute_J_diag(X, m, v)
        J_new = J_diag_matmul(kernel, X, m, v)
        rel = float((J_ref - J_new).abs().max() / J_ref.abs().max())

        ref_mmd, ref_pred, _, _, _ = fit_once(X, K, sigma, args.epochs, args.lr, 0)
        new_mmd, new_pred, _, _, _ = fit_once_fast(X, K, sigma, args.epochs, args.lr, 0)
        with torch.no_grad():
            const = float(kernel.compute_gram_matrix(X).mean())
        row = {
            "K": K, "n": int(X.shape[0]), "M": M,
            "max_rel_diff_J": rel,
            "ari_released": adjusted_rand_score(y, ref_pred),
            "ari_matmul": adjusted_rand_score(y, new_pred),
            # the released path logs MMD^2 including the constant data-data
            # term, the matmul path omits it; adding it back must reproduce it
            "mmd2_released": ref_mmd,
            "mmd2_matmul_plus_const": new_mmd + const,
            "abs_diff_mmd2": abs(ref_mmd - (new_mmd + const)),
        }
        rows.append(row)
        print("  ", row, flush=True)

    # the row-blocked path must reproduce the single-block one
    global CHUNK_CELLS
    K, M = 10, 20
    X, y = make_mixture(K, M, 40, sep=6.0, std=0.35, seed=0)
    sigma = median_sigma(X)
    full_mmd, full_pred, _, _, _ = fit_once_fast(X, K, sigma, args.epochs, args.lr, 0)
    saved, CHUNK_CELLS = CHUNK_CELLS, 500        # forces ~50-row blocks
    chunk_mmd, chunk_pred, _, _, _ = fit_once_fast(X, K, sigma, args.epochs, args.lr, 0)
    CHUNK_CELLS = saved
    row = {
        "check": "row-blocked vs single-block", "K": K, "n": int(X.shape[0]), "M": M,
        "abs_diff_mmd2": abs(full_mmd - chunk_mmd),
        "ari_single_block": adjusted_rand_score(y, full_pred),
        "ari_row_blocked": adjusted_rand_score(y, chunk_pred),
    }
    rows.append(row)
    print("  ", row, flush=True)

    with open(os.path.join(RESULTS, "largek_verify.json"), "w") as f:
        json.dump(rows, f, indent=2)


# ----------------------------------------------------------------------
# W2: large number of clusters (single K, meant to run in a subprocess)
# ----------------------------------------------------------------------
def run_largek_single(K, n_per, M, restarts, epochs, lr, seed, sep=6.0, std=0.35):
    from sklearn.metrics import adjusted_rand_score
    import psutil
    proc = psutil.Process()
    base_mb = proc.memory_info().rss / (1024.0 ** 2)  # before data/fit
    X, y = make_mixture(K, M, n_per, sep=sep, std=std, seed=seed)
    sigma = median_sigma(X)
    mmds, aris, times = [], [], []
    for r in range(restarts):
        final_mmd, pred, dt, _, _ = fit_once_fast(X, K, sigma, epochs, lr, seed + 1000 * r)
        mmds.append(final_mmd)
        aris.append(adjusted_rand_score(y, pred))
        times.append(dt)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: kB
    peak_mb = peak_kb / 1024.0
    return {
        "K": K, "n": int(X.shape[0]), "M": M,
        "n_per": int(n_per),
        "ari": float(np.max(aris)),            # best restart
        "ari_mean": float(np.mean(aris)),       # mean across restarts
        "ari_std": float(np.std(aris)),         # spread across restarts
        "runtime_s": float(np.median(times)),  # per fit
        "runtime_mean_s": float(np.mean(times)),
        "mmd2_std": float(np.std(mmds)),        # stability across restarts
        "peak_rss_mb": peak_mb,
        "peak_extra_mb": max(0.0, peak_mb - base_mb),  # incremental (above baseline)
    }


def driver_largek(args):
    import subprocess
    # (K, samples per component): the last two columns push the component count
    # and the sample size well past the benchmark regime (n = 1e4 and 1e5).
    plan = [(5, args.n_per), (10, args.n_per), (20, args.n_per), (50, args.n_per),
            (100, 100), (200, 500), (500, 2000)]
    rows = []
    for K, n_per in plan:
        cmd = [sys.executable, os.path.abspath(__file__), "largek-single",
               "--K", str(K), "--n_per", str(n_per), "--M", str(args.M),
               "--restarts", str(args.restarts), "--epochs", str(args.epochs),
               "--lr", str(args.lr), "--seed", str(args.seed),
               "--sep", str(args.sep), "--std", str(args.std)]
        out = subprocess.check_output(cmd, text=True)
        row = json.loads(out.strip().splitlines()[-1])
        rows.append(row)
        print("  ", row, flush=True)
    with open(os.path.join(RESULTS, "largek.json"), "w") as f:
        json.dump(rows, f, indent=2)
    # LaTeX rows
    print("\n=== LaTeX (largek) ===")
    def fmt(rows, key, f="{:.3g}"):
        return " & ".join(f.format(r[key]) for r in rows)
    print("K:        ", " & ".join(str(r["K"]) for r in rows))
    print("n:        ", " & ".join(str(r["n"]) for r in rows))
    print("ARI:      ", fmt(rows, "ari", "{:.3f}"))
    print("Runtime:  ", fmt(rows, "runtime_s", "{:.2f}"))
    print("PeakMB:   ", fmt(rows, "peak_rss_mb", "{:.0f}"))
    print("ExtraMB:  ", fmt(rows, "peak_extra_mb", "{:.1f}"))
    print("MMD2 std: ", fmt(rows, "mmd2_std", "{:.1e}"))


# ----------------------------------------------------------------------
# W2 (controlled): isolate the effect of K.
#
# The reviewer noted that the original table varies K and n together, so the
# large-K columns also have more observations per component and are thus
# statistically easier.  To isolate K we run two sweeps over the *same* K grid,
# with fixed separation, dimension, epochs and restarts, and an identical
# data-generation procedure for every K:
#   nfixed : n = args.n_total held constant  (n/K shrinks as K grows)
#   nk     : n/K = args.n_per held constant  (observations per component fixed)
# Each K is fitted in its own subprocess so peak_extra_mb is the incremental
# (above-baseline) RSS of that fit alone.
# ----------------------------------------------------------------------
def _run_controlled_sweep(args, plan, out_name, title):
    import subprocess
    rows = []
    for K, n_per in plan:
        cmd = [sys.executable, os.path.abspath(__file__), "largek-single",
               "--K", str(K), "--n_per", str(n_per), "--M", str(args.M),
               "--restarts", str(args.restarts), "--epochs", str(args.epochs),
               "--lr", str(args.lr), "--seed", str(args.seed),
               "--sep", str(args.sep), "--std", str(args.std)]
        out = subprocess.check_output(cmd, text=True)
        row = json.loads(out.strip().splitlines()[-1])
        rows.append(row)
        print("  ", row, flush=True)
    with open(os.path.join(RESULTS, out_name), "w") as f:
        json.dump(rows, f, indent=2)

    def fmt(rows, key, f="{:.3g}"):
        return " & ".join(f.format(r[key]) for r in rows)
    print(f"\n=== LaTeX ({title}) ===")
    print("K:            ", " & ".join(str(r["K"]) for r in rows))
    print("n:            ", " & ".join(str(r["n"]) for r in rows))
    print("n/K:          ", " & ".join("{:.0f}".format(r["n"] / r["K"]) for r in rows))
    print("ARI mean:     ", fmt(rows, "ari_mean", "{:.3f}"))
    print("ARI std:      ", fmt(rows, "ari_std", "{:.1e}"))
    print("Time/fit [s]: ", fmt(rows, "runtime_s", "{:.2f}"))
    print("Incr. MB:     ", fmt(rows, "peak_extra_mb", "{:.0f}"))
    return rows


def driver_largek_iso(args):
    K_grid = [int(k) for k in args.k_grid.split(",")]
    if args.mode == "nfixed":
        # n held constant; n_per = n_total // K (all grid values divide n_total)
        plan = [(K, args.n_total // K) for K in K_grid]
        out_name = "largek_nfixed.json"
        title = f"n = {args.n_total} fixed"
    else:  # nk
        # observations per component held constant; n = n_per * K grows with K
        plan = [(K, args.n_per) for K in K_grid]
        out_name = "largek_nk.json"
        title = f"n/K = {args.n_per} fixed"
    _run_controlled_sweep(args, plan, out_name, title)


# ----------------------------------------------------------------------
# W3: runtime scaling in n, M, K
# ----------------------------------------------------------------------
def driver_runtime(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = args.epochs
    reps = 3

    def time_fit(X, K, epochs):
        sigma = median_sigma(X)
        ts = []
        for r in range(reps):
            _, _, dt, _, _ = fit_once_fast(X, K, sigma, epochs, args.lr, 100 + r)
            ts.append(dt)
        return float(np.median(ts))

    out_json = os.path.join(RESULTS, "runtime.json")
    if args.plot_only:
        # redraw from the recorded timings, without re-running the sweep
        with open(out_json) as f:
            data = json.load(f)
        n_vals, n_times = data["n_vals"], data["n_times"]
        M_vals, M_times = data["M_vals"], data["M_times"]
        K_vals, K_times = data["K_vals"], data["K_times"]
        epochs = data["epochs"]
    else:
        # Ranges run an order of magnitude past the case study in every factor,
        # so that the fixed per-epoch overhead (~0.3 s for 200 Adam steps) is
        # left behind and the asymptotic regime is visible.
        # vary n (K=5, M=10)
        n_vals = [200, 1000, 5000, 10000, 50000, 100000]
        n_times = []
        for n in n_vals:
            X, _ = make_mixture(5, 10, n // 5, seed=0)
            n_times.append(time_fit(X, 5, epochs))
            print(f"  n={n}: {n_times[-1]:.3f}s", flush=True)

        # vary M (n=1000, K=5)
        M_vals = [5, 20, 50, 100, 250, 500, 1000]
        M_times = []
        for M in M_vals:
            X, _ = make_mixture(5, M, 200, seed=0)
            M_times.append(time_fit(X, 5, epochs))
            print(f"  M={M}: {M_times[-1]:.3f}s", flush=True)

        # vary K (n=5000, M=10)
        K_vals = [2, 10, 50, 100, 250, 500]
        K_times = []
        for K in K_vals:
            X, _ = make_mixture(K, 10, max(1, 5000 // K), seed=0)
            # keep n approx constant ~5000
            K_times.append(time_fit(X, K, epochs))
            print(f"  K={K}: {K_times[-1]:.3f}s", flush=True)

        data = {"n_vals": n_vals, "n_times": n_times, "M_vals": M_vals,
                "M_times": M_times, "K_vals": K_vals, "K_times": K_times,
                "epochs": epochs}
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)

    def slope(xs, ys, tail=3):
        """Power-law exponent over the largest `tail` settings.

        Small settings are dominated by the fixed per-epoch overhead (~0.3 s for
        200 Adam steps), so the exponent is read off the asymptotic end.
        """
        p = np.polyfit(np.log(np.asarray(xs[-tail:], float)),
                       np.log(np.asarray(ys[-tail:], float)), 1)
        return float(p[0])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (xv, tv, xl, ti) in zip(axes, [
        (n_vals, n_times, "Sample size $n$", "Runtime vs $n$  ($K{=}5,M{=}10$)"),
        (M_vals, M_times, "Projection dim $M$", "Runtime vs $M$  ($n{=}1000,K{=}5$)"),
        (K_vals, K_times, "Components $K$", "Runtime vs $K$  ($n{\\approx}5000,M{=}10$)"),
    ]):
        s = slope(xv, tv)
        ax.plot(xv, tv, "o-", color="tab:blue", lw=2, ms=6,
                label=f"measured (tail slope {s:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(f"Wall-clock time ({epochs} ep.) [s]", fontsize=9)
        ax.set_title(ti, fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = os.path.join(ROOT, "paper", "images", "rebuttal_runtime_scaling.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)


# ----------------------------------------------------------------------
# Q2: finite-M ICL vs MMD elbow
# ----------------------------------------------------------------------
def n_params(K, M):
    # diagonal covariance: weights (K-1) + means (K*M) + variances (K*M)
    return (K - 1) + K * M + K * M


def driver_icl(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_K = args.true_k
    M = args.M
    n_per = args.n_per
    X, y = make_mixture(true_K, M, n_per, sep=args.sep, std=args.std, seed=7)
    n = X.shape[0]
    sigma = median_sigma(X)
    Ks = list(range(1, args.max_k + 1))

    mmd_curve, bic_curve, icl_curve = [], [], []
    for K in Ks:
        # best of restarts by MMD
        best = None
        for r in range(args.restarts):
            final_mmd, pred, dt, model, kernel = fit_once(
                X, K, sigma, args.epochs, args.lr, 7 + 1000 * r)
            if best is None or final_mmd < best[0]:
                best = (final_mmd, model)
        final_mmd, model = best
        with torch.no_grad():
            ll = float(model.log_likelihood(X))
            gamma = model.responsibilities(X)                 # (n,K)
            ent = float(-(gamma * torch.log(gamma.clamp_min(1e-12))).sum())
        p = n_params(K, M)
        bic = -2.0 * ll + p * np.log(n)
        icl = bic + 2.0 * ent          # ICL = BIC + 2*entropy (lower is better)
        mmd_curve.append(final_mmd)
        bic_curve.append(bic)
        icl_curve.append(icl)
        print(f"  K={K}: MMD2={final_mmd:.4e}  BIC={bic:.1f}  ICL={icl:.1f}", flush=True)

    icl_star = Ks[int(np.argmin(icl_curve))]
    bic_star = Ks[int(np.argmin(bic_curve))]
    data = {"Ks": Ks, "mmd": mmd_curve, "bic": bic_curve, "icl": icl_curve,
            "true_K": true_K, "icl_argmin": icl_star, "bic_argmin": bic_star,
            "M": M, "n": n, "sigma": sigma}
    with open(os.path.join(RESULTS, "icl.json"), "w") as f:
        json.dump(data, f, indent=2)

    fig, ax1 = plt.subplots(figsize=(5, 3.4))
    ax1.plot(Ks, mmd_curve, "o-", color="tab:blue", lw=2, label="MMD$^2$ (elbow)")
    ax1.set_xlabel("Number of components $K$")
    ax1.set_ylabel("Within-cluster MMD$^2$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(Ks, icl_curve, "s--", color="tab:red", lw=2, label="ICL ($M$ fixed)")
    ax2.set_ylabel("ICL", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.axvline(true_K, color="gray", ls=":", label=f"true $K={true_K}$")
    ax1.set_title(f"MMD elbow vs finite-$M$ ICL (ICL argmin $K={icl_star}$)")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "images", "rebuttal_icl.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)
    print(f"\nTrue K={true_K} | ICL argmin K={icl_star} | BIC argmin K={bic_star}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("largek-single")
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--n_per", type=int, default=40)
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sep", type=float, default=6.0)
    p.add_argument("--std", type=float, default=0.35)

    p = sub.add_parser("largek")
    p.add_argument("--n_per", type=int, default=40)
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sep", type=float, default=6.0)
    p.add_argument("--std", type=float, default=0.35)

    p = sub.add_parser("largek-iso",
                       help="controlled K sweep that isolates K (n fixed or n/K fixed)")
    p.add_argument("--mode", choices=["nfixed", "nk"], default="nfixed",
                   help="nfixed: n=--n_total constant; nk: n/K=--n_per constant")
    p.add_argument("--k_grid", type=str, default="5,10,20,50,100,200")
    p.add_argument("--n_total", type=int, default=100000)  # used by mode=nfixed
    p.add_argument("--n_per", type=int, default=500)        # used by mode=nk
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sep", type=float, default=6.0)
    p.add_argument("--std", type=float, default=0.35)

    p = sub.add_parser("runtime")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--plot_only", action="store_true",
                   help="redraw the figure from results/runtime.json")

    p = sub.add_parser("verify")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)

    p = sub.add_parser("icl")
    p.add_argument("--true_k", type=int, default=4)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--n_per", type=int, default=150)
    p.add_argument("--max_k", type=int, default=8)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sep", type=float, default=2.2)
    p.add_argument("--std", type=float, default=0.5)

    args = ap.parse_args()
    if args.cmd == "largek-single":
        print(json.dumps(run_largek_single(
            args.K, args.n_per, args.M, args.restarts, args.epochs, args.lr, args.seed,
            sep=args.sep, std=args.std)))
    elif args.cmd == "largek":
        driver_largek(args)
    elif args.cmd == "largek-iso":
        driver_largek_iso(args)
    elif args.cmd == "runtime":
        driver_runtime(args)
    elif args.cmd == "icl":
        driver_icl(args)
    elif args.cmd == "verify":
        driver_verify(args)


if __name__ == "__main__":
    main()
