#!/usr/bin/env python
"""
Is the (c) concentric-shells gap a model limitation or a bandwidth artifact?

Hypothesis: the median heuristic is dominated by the WIDE component, giving a
sigma so large that the Gaussian kernel is nearly flat across the narrow one,
so MMD cannot resolve the fine scale.  Sweep sigma and the covariance type.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src import GaussianKernel  # noqa: E402
from src.mixture import fit_gaussian_mixture_mmd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

DTYPE = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
EPOCHS, LR, RESTARTS = 400, 0.05, 8


def median_sigma(X, max_samples=800):
    with torch.no_grad():
        Xs = X if X.shape[0] <= max_samples else X[torch.randperm(X.shape[0])[:max_samples]]
        d = torch.cdist(Xs, Xs)
        iu = torch.triu_indices(d.shape[0], d.shape[1], 1)
        pos = d[iu[0], iu[1]]
        pos = pos[pos > 0]
        return float(pos.median())


def run(X, y, K, sigma, cov):
    kern = GaussianKernel(sigma=sigma)
    best = None
    for r in range(RESTARTS):
        torch.manual_seed(r)
        model, hist = fit_gaussian_mixture_mmd(
            X, num_components=K, kernel=kern, num_epochs=EPOCHS, lr=LR,
            covariance_type=cov, init_method="kmeans++", verbose=False)
        if best is None or hist[-1] < best[0]:
            best = (hist[-1], model)
    with torch.no_grad():
        lab = best[1].responsibilities(X).argmax(1).numpy()
    return float(adjusted_rand_score(y, lab))


def main():
    M, n_per = 10, 300
    g = torch.Generator().manual_seed(0)
    out = {}

    # ---- scenario (c): concentric shells ----
    r1 = 0.35 * torch.randn(n_per, M, generator=g, dtype=DTYPE)
    r2 = 2.5 * torch.randn(n_per, M, generator=g, dtype=DTYPE)
    Xc = torch.cat([r1, r2], 0)
    yc = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)

    # ---- scenario (b): equal mean, different covariance ----
    s1 = torch.full((M,), 0.4, dtype=DTYPE); s1[:3] = torch.tensor([3.0, .25, .25], dtype=DTYPE)
    s2 = torch.full((M,), 0.4, dtype=DTYPE); s2[:3] = torch.tensor([.25, 3.0, .25], dtype=DTYPE)
    Xb = torch.cat([torch.randn(n_per, M, generator=g, dtype=DTYPE) * s1,
                    torch.randn(n_per, M, generator=g, dtype=DTYPE) * s2], 0)
    yb = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)

    # ---- scenario (a): well-separated isotropic (control: should be insensitive) ----
    means = 6.0 * torch.randn(3, M, generator=g, dtype=DTYPE)
    Xa = torch.cat([means[k] + 0.35 * torch.randn(n_per, M, generator=g, dtype=DTYPE)
                    for k in range(3)], 0)
    ya = np.concatenate([np.full(n_per, k) for k in range(3)])

    for name, X, y, K in [("c_concentric", Xc, yc, 2),
                          ("b_equalmean", Xb, yb, 2),
                          ("a_separated", Xa, ya, 3)]:
        med = median_sigma(X)
        rows = []
        for mult in [0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0]:
            for cov in ["diagonal", "full"]:
                ari = run(X, y, K, med * mult, cov)
                rows.append({"mult": mult, "sigma": med * mult, "cov": cov, "ari": ari})
                print(f"  {name} sigma={mult:>4}x median ({med*mult:6.2f}) {cov:8s} ARI={ari:.3f}",
                      flush=True)
        out[name] = {"median_sigma": med, "rows": rows}

    with open(os.path.join(RESULTS, "r2_bandwidth.json"), "w") as f:
        json.dump(out, f, indent=2)

    for name in out:
        best = max(out[name]["rows"], key=lambda r: r["ari"])
        print(f"\nBEST {name}: ARI={best['ari']:.3f} at sigma={best['mult']}x median, {best['cov']}")


if __name__ == "__main__":
    main()
