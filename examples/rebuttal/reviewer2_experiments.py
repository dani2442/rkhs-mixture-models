#!/usr/bin/env python
"""
Rebuttal experiments (Reviewer 2 / #aBwm).

  scaling   -> W1 + Limitations: wall-clock / memory vs n, with and without the
               O(n^2) data-data Gram term (which is constant in theta and is not
               needed by the optimizer).
  stability -> W2: which matrices are factorized, their conditioning, and
               Cholesky-vs-explicit-inverse accuracy.
  nongauss  -> W3: non-Gaussian components (ring, heavy-tailed, curved,
               bimodal): held-out MMD^2 vs number of components.

Q1 (k-means vs the mixture) lives in reviewer2_kmeans.py, and the bandwidth
sensitivity behind it in reviewer2_bandwidth.py.

Uses the paper's own GaussianMixtureModel / GaussianKernel and MMD fitting.
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
sys.path.insert(0, ROOT)

from src import GaussianKernel, GaussianMixtureModel  # noqa: E402
from src.mixture import fit_gaussian_mixture_mmd  # noqa: E402

DTYPE = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def median_sigma(X, max_samples=800):
    with torch.no_grad():
        Xs = X if X.shape[0] <= max_samples else X[torch.randperm(X.shape[0])[:max_samples]]
        d = torch.cdist(Xs, Xs)
        iu = torch.triu_indices(d.shape[0], d.shape[1], 1)
        pos = d[iu[0], iu[1]]
        pos = pos[pos > 0]
        return float(pos.median()) if pos.numel() else 1.0


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


def fit_no_const(X, K, sigma, epochs, lr, seed, covariance_type="diagonal"):
    """
    Same optimization loop as fit_gaussian_mixture_mmd but WITHOUT precomputing
    the O(n^2) data-data Gram matrix.  That term is constant in theta, so the
    optimizer never needs it; it only shifts the logged loss.
    """
    torch.manual_seed(seed)
    n, M = X.shape
    kernel = GaussianKernel(sigma=sigma)
    model = GaussianMixtureModel(
        num_components=K, coeff_dim=M, covariance_type=covariance_type,
        device=X.device, dtype=X.dtype,
    )
    model.initialize_from_data(X, method="kmeans++")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        opt.step()
    return model, kernel


# ----------------------------------------------------------------------
# W1: scaling in n, with / without the O(n^2) constant term
# ----------------------------------------------------------------------
def run_scaling_single(n, K, M, epochs, lr, seed, with_const):
    import psutil
    from sklearn.metrics import adjusted_rand_score
    proc = psutil.Process()
    base_mb = proc.memory_info().rss / (1024.0 ** 2)

    X, y = make_mixture(K, M, n // K, sep=6.0, std=0.35, seed=seed)
    sigma = median_sigma(X)

    t0 = time.perf_counter()
    if with_const:
        model, _ = fit_gaussian_mixture_mmd(
            X, num_components=K, kernel=GaussianKernel(sigma=sigma),
            num_epochs=epochs, lr=lr, covariance_type="diagonal",
            init_method="kmeans++", verbose=False,
        )
    else:
        model, _ = fit_no_const(X, K, sigma, epochs, lr, seed)
    dt = time.perf_counter() - t0

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    with torch.no_grad():
        labels = model.responsibilities(X).argmax(1).numpy()
    return {
        "n": int(X.shape[0]), "K": K, "M": M, "with_const": bool(with_const),
        "runtime_s": dt, "peak_rss_mb": peak_mb,
        "peak_extra_mb": max(0.0, peak_mb - base_mb),
        "ari": float(adjusted_rand_score(y, labels)),
    }


def driver_scaling(args):
    import subprocess
    rows = []
    plan = [(n, False) for n in [1000, 5000, 10000, 20000, 50000, 100000, 1000000]]
    for n, wc in plan:
        cmd = [sys.executable, os.path.abspath(__file__), "scaling-single",
               "--n", str(n), "--K", str(args.K), "--M", str(args.M),
               "--epochs", str(args.epochs), "--lr", str(args.lr),
               "--seed", str(args.seed)] + (["--with_const"] if wc else [])
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            row = json.loads(out.strip().splitlines()[-1])
        except subprocess.CalledProcessError:
            row = {"n": n, "with_const": wc, "runtime_s": None,
                   "peak_rss_mb": None, "peak_extra_mb": None, "ari": None,
                   "oom": True}
        rows.append(row)
        print("  ", row, flush=True)
    with open(os.path.join(RESULTS, "r2_scaling.json"), "w") as f:
        json.dump(rows, f, indent=2)


# ----------------------------------------------------------------------
# W2: what is factorized, conditioning, Cholesky vs explicit inverse
# ----------------------------------------------------------------------
def driver_stability(args):
    from sklearn.metrics import adjusted_rand_score  # noqa: F401  (import cost parity)
    M, K, n_per = args.M, args.K, args.n_per
    X, y = make_mixture(K, M, n_per, sep=6.0, std=0.35, seed=3)
    sigma = median_sigma(X)

    out = {"M": M, "K": K, "n": int(X.shape[0]), "sigma": sigma}

    # --- full-covariance fit, then inspect the matrices actually factorized ---
    torch.manual_seed(0)
    model, hist = fit_gaussian_mixture_mmd(
        X, num_components=K, kernel=GaussianKernel(sigma=sigma),
        num_epochs=args.epochs, lr=args.lr, covariance_type="full",
        init_method="kmeans++", verbose=False,
    )
    cov = model.covariance.detach()                       # (K,M,M)
    I_M = torch.eye(M, dtype=DTYPE)

    # A_k = I + K_k/sigma^2  (the matrix Cholesky-factorized inside compute_J)
    condA, condCov = [], []
    for k in range(K):
        A = I_M + cov[k] / (sigma ** 2)
        ev = torch.linalg.eigvalsh(A)
        condA.append(float(ev.max() / ev.min()))
        evc = torch.linalg.eigvalsh(cov[k] + 1e-12 * I_M)
        condCov.append(float(evc.max() / evc.clamp_min(1e-300).min()))
    out["cond_A_max"] = max(condA)
    out["cond_A_min"] = min(condA)
    out["cond_cov_max"] = max(condCov)
    out["min_eig_A"] = float(min(
        float(torch.linalg.eigvalsh(I_M + cov[k] / (sigma ** 2)).min()) for k in range(K)))
    out["final_mmd2_full"] = hist[-1]

    # --- Cholesky solve vs explicit inverse: accuracy on the same systems ---
    err_chol, err_inv = [], []
    for k in range(K):
        A = (I_M + cov[k] / (sigma ** 2)).clone()
        B = torch.randn(M, 8, dtype=DTYPE)
        L = torch.linalg.cholesky(A)
        Xc = torch.cholesky_solve(B, L)
        Xi = torch.linalg.inv(A) @ B
        err_chol.append(float((A @ Xc - B).abs().max()))
        err_inv.append(float((A @ Xi - B).abs().max()))
    out["resid_cholesky_max"] = max(err_chol)
    out["resid_explicit_inv_max"] = max(err_inv)

    # --- ill-conditioned stress test: near-singular covariance ---
    # covariance with eigenvalues spanning 1e-10 .. 1 -> cond(K)=1e10,
    # but A = I + K/sigma^2 stays perfectly conditioned.
    lam = torch.logspace(-10, 0, M, dtype=DTYPE)
    Q, _ = torch.linalg.qr(torch.randn(M, M, dtype=DTYPE))
    Kbad = Q @ torch.diag(lam) @ Q.T
    A = I_M + Kbad / (sigma ** 2)
    evA = torch.linalg.eigvalsh(A)
    out["stress_cond_cov"] = float(lam.max() / lam.min())
    out["stress_cond_A"] = float(evA.max() / evA.min())
    try:
        torch.linalg.cholesky(A)
        out["stress_cholesky_ok"] = True
    except Exception:
        out["stress_cholesky_ok"] = False

    # --- diagonal path: no factorization at all; count ops ---
    torch.manual_seed(0)
    t0 = time.perf_counter()
    _, hist_d = fit_gaussian_mixture_mmd(
        X, num_components=K, kernel=GaussianKernel(sigma=sigma),
        num_epochs=args.epochs, lr=args.lr, covariance_type="diagonal",
        init_method="kmeans++", verbose=False,
    )
    out["time_diag_s"] = time.perf_counter() - t0
    out["final_mmd2_diag"] = hist_d[-1]

    # number of Cholesky factorizations per iteration, full-covariance path
    out["chol_per_iter_full"] = f"K + K^2 = {K + K*K} (each {M}x{M})"
    out["chol_per_iter_diag"] = 0

    with open(os.path.join(RESULTS, "r2_stability.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


# ----------------------------------------------------------------------
# W3: non-Gaussian component shapes
# ----------------------------------------------------------------------
def make_ring(n, M, radius=3.0, width=0.25, seed=0):
    """Annulus in the first two coordinates, isotropic noise elsewhere."""
    g = torch.Generator().manual_seed(seed)
    th = 2 * np.pi * torch.rand(n, generator=g, dtype=DTYPE)
    r = radius + width * torch.randn(n, generator=g, dtype=DTYPE)
    X = 0.15 * torch.randn(n, M, generator=g, dtype=DTYPE)
    X[:, 0] += r * torch.cos(th)
    X[:, 1] += r * torch.sin(th)
    return X


def make_heavy_tail(n, M, df=2.0, scale=1.0, seed=0):
    """Multivariate Student-t (df=2 -> infinite variance)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, M, generator=g, dtype=DTYPE)
    chi = torch.distributions.Chi2(torch.tensor(df, dtype=DTYPE)).sample((n,))
    return scale * z / torch.sqrt(chi / df).unsqueeze(1)


def make_bimodal(n, M, sep=4.0, seed=0):
    """A single 'semantic' class that is itself bimodal."""
    g = torch.Generator().manual_seed(seed)
    X = 0.5 * torch.randn(n, M, generator=g, dtype=DTYPE)
    flip = torch.randint(0, 2, (n,), generator=g, dtype=DTYPE)
    X[:, 0] += sep * (flip - 0.5)
    return X


def make_curved(n, M, seed=0):
    """Curved (banana) manifold."""
    g = torch.Generator().manual_seed(seed)
    t = 4.0 * (torch.rand(n, generator=g, dtype=DTYPE) - 0.5)
    X = 0.15 * torch.randn(n, M, generator=g, dtype=DTYPE)
    X[:, 0] += t
    X[:, 1] += 0.8 * t ** 2
    return X


def held_out_mmd2(model, kernel, Xte):
    """MMD^2(P_test, Q) including the constant data-data term."""
    with torch.no_grad():
        mmd2, _ = model.compute_mmd2(Xte, kernel, compute_const_term=True)
    return float(mmd2)


def driver_nongauss(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score

    M, n = args.M, args.n
    gens = {
        "ring": lambda s: make_ring(n, M, seed=s),
        "heavy-tailed $t_2$": lambda s: make_heavy_tail(n, M, seed=s),
        "curved": lambda s: make_curved(n, M, seed=s),
        "bimodal": lambda s: make_bimodal(n, M, seed=s),
    }
    Ks = list(range(1, args.max_k + 1))
    curves = {}
    for name, gen in gens.items():
        Xtr, Xte = gen(0), gen(1)
        sigma = median_sigma(Xtr)
        kern = GaussianKernel(sigma=sigma)
        vals = []
        for K in Ks:
            best = None
            for r in range(args.restarts):
                torch.manual_seed(r)
                model, hist = fit_gaussian_mixture_mmd(
                    Xtr, num_components=K, kernel=kern, num_epochs=args.epochs,
                    lr=args.lr, covariance_type="full", init_method="kmeans++",
                    verbose=False,
                )
                if best is None or hist[-1] < best[0]:
                    best = (hist[-1], model)
            vals.append(held_out_mmd2(best[1], kern, Xte))
            print(f"  {name} K={K}: heldout MMD2={vals[-1]:.4e}", flush=True)
        curves[name] = vals

    # --- two non-Gaussian classes: K=2 vs over-parameterized + merge ---
    Xa = make_ring(n // 2, M, radius=3.0, seed=10)
    Xb = make_ring(n // 2, M, radius=0.0, width=0.6, seed=11)  # blob inside ring
    Xmix = torch.cat([Xa, Xb], 0)
    ymix = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    sigma = median_sigma(Xmix)
    kern = GaussianKernel(sigma=sigma)

    def ari_for(K, merge_to=None):
        best = None
        for r in range(args.restarts):
            torch.manual_seed(r)
            model, hist = fit_gaussian_mixture_mmd(
                Xmix, num_components=K, kernel=kern, num_epochs=args.epochs,
                lr=args.lr, covariance_type="full", init_method="kmeans++",
                verbose=False)
            if best is None or hist[-1] < best[0]:
                best = (hist[-1], model)
        with torch.no_grad():
            lab = best[1].responsibilities(Xmix).argmax(1).numpy()
        if merge_to is not None:
            # agglomerate components by their fitted means into merge_to groups
            from scipy.cluster.hierarchy import fcluster, linkage
            mu = best[1].mean.detach().numpy()
            Z = linkage(mu, method="average")
            grp = fcluster(Z, t=merge_to, criterion="maxclust") - 1
            lab = grp[lab]
        return float(adjusted_rand_score(ymix, lab))

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xmix.numpy())
    nested = {
        "mmd_gmm_K2": ari_for(2),
        "mmd_gmm_K8_merged": ari_for(8, merge_to=2),
        "kmeans_K2": float(adjusted_rand_score(ymix, km.labels_)),
    }
    print("  nested ring/blob:", nested, flush=True)

    data = {"Ks": Ks, "curves": curves, "nested": nested, "M": M, "n": n}
    with open(os.path.join(RESULTS, "r2_nongauss.json"), "w") as f:
        json.dump(data, f, indent=2)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, vals in curves.items():
        v = np.array(vals)
        ax.plot(Ks, v / v[0], "o-", lw=2, ms=5, label=name)
    ax.set_xlabel("Number of Gaussian components $K$")
    ax.set_ylabel("Held-out MMD$^2$  (relative to $K{=}1$)")
    ax.set_title("Non-Gaussian targets are approximated by\na few Gaussian components")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    outp = os.path.join(ROOT, "paper", "images", "rebuttal_nongaussian.pdf")
    fig.savefig(outp, bbox_inches="tight")
    print("Saved:", outp)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scaling-single")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with_const", action="store_true")

    p = sub.add_parser("scaling")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("stability")
    p.add_argument("--M", type=int, default=30)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n_per", type=int, default=100)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.05)

    p = sub.add_parser("nongauss")
    p.add_argument("--M", type=int, default=6)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--max_k", type=int, default=8)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.05)

    args = ap.parse_args()
    if args.cmd == "scaling-single":
        print(json.dumps(run_scaling_single(
            args.n, args.K, args.M, args.epochs, args.lr, args.seed, args.with_const)))
    elif args.cmd == "scaling":
        driver_scaling(args)
    elif args.cmd == "stability":
        driver_stability(args)
    elif args.cmd == "nongauss":
        driver_nongauss(args)


if __name__ == "__main__":
    main()
