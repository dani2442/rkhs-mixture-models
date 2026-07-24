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
        final_mmd, pred, dt, _, _ = fit_once(X, K, sigma, epochs, lr, seed + 1000 * r)
        mmds.append(final_mmd)
        aris.append(adjusted_rand_score(y, pred))
        times.append(dt)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: kB
    peak_mb = peak_kb / 1024.0
    return {
        "K": K, "n": int(X.shape[0]), "M": M,
        "ari": float(np.max(aris)),            # best restart
        "ari_mean": float(np.mean(aris)),
        "runtime_s": float(np.median(times)),  # per fit
        "mmd2_std": float(np.std(mmds)),        # stability across restarts
        "peak_rss_mb": peak_mb,
        "peak_extra_mb": max(0.0, peak_mb - base_mb),  # above interpreter baseline
    }


def driver_largek(args):
    import subprocess
    Ks = [2, 5, 10, 20, 50]
    rows = []
    for K in Ks:
        cmd = [sys.executable, os.path.abspath(__file__), "largek-single",
               "--K", str(K), "--n_per", str(args.n_per), "--M", str(args.M),
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
            _, _, dt, _, _ = fit_once(X, K, sigma, epochs, args.lr, 100 + r)
            ts.append(dt)
        return float(np.median(ts))

    # vary n (K=5, M=10)
    n_vals = [200, 500, 1000, 2000, 4000]
    n_times = []
    for n in n_vals:
        X, _ = make_mixture(5, 10, n // 5, seed=0)
        n_times.append(time_fit(X, 5, epochs))
        print(f"  n={n}: {n_times[-1]:.3f}s", flush=True)

    # vary M (n=1000, K=5)
    M_vals = [5, 10, 20, 40, 80]
    M_times = []
    for M in M_vals:
        X, _ = make_mixture(5, M, 200, seed=0)
        M_times.append(time_fit(X, 5, epochs))
        print(f"  M={M}: {M_times[-1]:.3f}s", flush=True)

    # vary K (n=1000, M=10)
    K_vals = [2, 5, 10, 20, 40]
    K_times = []
    for K in K_vals:
        X, _ = make_mixture(K, 10, 1000 // K, seed=0)
        # keep n approx constant ~1000
        K_times.append(time_fit(X, K, epochs))
        print(f"  K={K}: {K_times[-1]:.3f}s", flush=True)

    data = {"n_vals": n_vals, "n_times": n_times, "M_vals": M_vals,
            "M_times": M_times, "K_vals": K_vals, "K_times": K_times,
            "epochs": epochs}
    with open(os.path.join(RESULTS, "runtime.json"), "w") as f:
        json.dump(data, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (xv, tv, xl, ti) in zip(axes, [
        (n_vals, n_times, "Sample size $n$", "Runtime vs $n$  ($K{=}5,M{=}10$)"),
        (M_vals, M_times, "Projection dim $M$", "Runtime vs $M$  ($n{=}1000,K{=}5$)"),
        (K_vals, K_times, "Components $K$", "Runtime vs $K$  ($n{\\approx}1000,M{=}10$)"),
    ]):
        ax.plot(xv, tv, "o-", color="tab:blue", lw=2, ms=6)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(f"Wall-clock time ({epochs} ep.) [s]", fontsize=9)
        ax.set_title(ti, fontsize=11)
        ax.grid(alpha=0.3)
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

    p = sub.add_parser("runtime")
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
    elif args.cmd == "runtime":
        driver_runtime(args)
    elif args.cmd == "icl":
        driver_icl(args)


if __name__ == "__main__":
    main()
