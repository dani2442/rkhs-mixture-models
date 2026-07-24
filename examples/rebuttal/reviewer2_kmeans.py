#!/usr/bin/env python
"""
Q1 (Reviewer 2): k-means vs the MMD mixture, with a Bayes-optimal ceiling.

For every scenario we report:
  - k-means ARI              (hard assignment, distance to centroid)
  - MMD-GMM ARI              (posterior argmax)
  - Bayes-optimal ARI        (the best ANY method could do on this instance)
  - uncertainty diagnostic   (accuracy on confident vs unconfident points)
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

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

DTYPE = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
EPOCHS, LR, RESTARTS = 400, 0.05, 6


def median_sigma(X, max_samples=800):
    with torch.no_grad():
        Xs = X if X.shape[0] <= max_samples else X[torch.randperm(X.shape[0])[:max_samples]]
        d = torch.cdist(Xs, Xs)
        iu = torch.triu_indices(d.shape[0], d.shape[1], 1)
        pos = d[iu[0], iu[1]]
        pos = pos[pos > 0]
        return float(pos.median()) if pos.numel() else 1.0


def bayes_labels_diag(X, means, variances, priors):
    """Bayes-optimal assignment for a known diagonal Gaussian mixture."""
    lp = []
    for m, v, p in zip(means, variances, priors):
        d = X - m
        ll = -0.5 * (torch.log(2 * np.pi * v).sum() + (d * d / v).sum(1))
        lp.append(ll + np.log(p))
    return torch.stack(lp, 1).argmax(1).numpy()


def fit_mmd(X, K, cov):
    sigma = median_sigma(X)
    kern = GaussianKernel(sigma=sigma)
    best = None
    for r in range(RESTARTS):
        torch.manual_seed(r)
        model, hist = fit_gaussian_mixture_mmd(
            X, num_components=K, kernel=kern, num_epochs=EPOCHS, lr=LR,
            covariance_type=cov, init_method="kmeans++", verbose=False)
        if best is None or hist[-1] < best[0]:
            best = (hist[-1], model)
    return best[1]


def evaluate(name, X, y, K, cov, bayes=None, res=None):
    model = fit_mmd(X, K, cov)
    with torch.no_grad():
        gamma = model.responsibilities(X)
        lab = gamma.argmax(1).numpy()
        conf = gamma.max(1).values.numpy()
    km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(X.numpy())

    row = {
        "kmeans_ari": float(adjusted_rand_score(y, km.labels_)),
        "mmd_gmm_ari": float(adjusted_rand_score(y, lab)),
    }
    if bayes is not None:
        row["bayes_ari"] = float(adjusted_rand_score(y, bayes))
    # uncertainty diagnostic: is the posterior informative about correctness?
    hi = conf >= np.quantile(conf, 0.5)
    row["ari_confident_half"] = float(adjusted_rand_score(y[hi], lab[hi]))
    row["ari_unconfident_half"] = float(adjusted_rand_score(y[~hi], lab[~hi]))
    row["mean_confidence"] = float(conf.mean())
    res[name] = row
    print(f"  {name}: {json.dumps(row)}", flush=True)


def main():
    res = {}
    M = 10
    n_per = 300
    g = torch.Generator().manual_seed(0)

    # ---- (a) well-separated isotropic: k-means is entirely adequate ----
    means = 6.0 * torch.randn(3, M, generator=g, dtype=DTYPE)
    Xs, ys = [], []
    for k in range(3):
        Xs.append(means[k] + 0.35 * torch.randn(n_per, M, generator=g, dtype=DTYPE))
        ys.append(np.full(n_per, k))
    Xa, ya = torch.cat(Xs, 0), np.concatenate(ys)
    ba = bayes_labels_diag(Xa, [means[k] for k in range(3)],
                           [torch.full((M,), 0.35 ** 2, dtype=DTYPE)] * 3, [1 / 3] * 3)
    evaluate("(a) separated isotropic", Xa, ya, 3, "diagonal", ba, res)

    # ---- (b) equal means, different covariance operators ----
    s1 = torch.full((M,), 0.4, dtype=DTYPE); s1[:3] = torch.tensor([3.0, 0.25, 0.25], dtype=DTYPE)
    s2 = torch.full((M,), 0.4, dtype=DTYPE); s2[:3] = torch.tensor([0.25, 3.0, 0.25], dtype=DTYPE)
    z1 = torch.randn(n_per, M, generator=g, dtype=DTYPE) * s1
    z2 = torch.randn(n_per, M, generator=g, dtype=DTYPE) * s2
    Xb = torch.cat([z1, z2], 0)
    yb = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)
    bb = bayes_labels_diag(Xb, [torch.zeros(M, dtype=DTYPE)] * 2,
                           [s1 ** 2, s2 ** 2], [0.5, 0.5])
    evaluate("(b) equal mean, different covariance", Xb, yb, 2, "diagonal", bb, res)

    # ---- (c) concentric shells (same mean, different scale) ----
    v1 = torch.full((M,), 0.35 ** 2, dtype=DTYPE)
    v2 = torch.full((M,), 2.5 ** 2, dtype=DTYPE)
    r1 = 0.35 * torch.randn(n_per, M, generator=g, dtype=DTYPE)
    r2 = 2.5 * torch.randn(n_per, M, generator=g, dtype=DTYPE)
    Xc = torch.cat([r1, r2], 0)
    yc = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)
    bc = bayes_labels_diag(Xc, [torch.zeros(M, dtype=DTYPE)] * 2, [v1, v2], [0.5, 0.5])
    evaluate("(c) concentric, different scale", Xc, yc, 2, "diagonal", bc, res)

    with open(os.path.join(RESULTS, "r2_kmeans.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
