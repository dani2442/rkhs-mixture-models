#!/usr/bin/env python
"""
Rebuttal experiments (Reviewer 3, #xDmR), on the five real L^2 benchmarks:

  gpmix       -> W3: GPmix as an additional functional-mixture baseline, swept
                 over its own basis/n_proj grid, plus a plain feature-space GMM
                 (EM, full covariance) on the identical cosine coefficients.
  covariance  -> Q1: full vs diagonal vs spherical projected covariance, and the
                 attained objective alongside the ARI -- full covariance reaches
                 a *lower* MMD^2 and a far worse ARI, so the collapse is not an
                 optimization artifact.
  stability   -> Q2: initialization and projection-dimension stability, scored by
                 pairwise ARI, variation of information, and posterior agreement.
  descent     -> W1: monotonicity of the alternating objective, under both the
                 exact scheme Proposition 4 assumes (exact simplex QP + gradient
                 step with backtracking) and the Adam optimizer actually used.

GPmix itself is a permanent competitor (``src/competitors/gpmix.py``) and is
registered in ``benchmarks/config.py``; the sweep here exists only to justify
which single configuration the benchmark should use.

Uses the paper's own GaussianMixtureModel / GaussianKernel and MMD fitting.
"""
import argparse
import json
import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src import GaussianKernel, GaussianMixtureModel, L2CosineBasis  # noqa: E402
from src.competitors import GPmixClustering  # noqa: E402
from benchmarks.bench_l2_realdata import DATASET_SPECS  # noqa: E402

DTYPE = torch.float64
DEVICE = torch.device("cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

NUM_EPOCHS = 400
LR = 0.1
SEED = 42


def _save(name, payload):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[reviewer3] wrote {path}")


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------
def load_dataset(loader_fn, R):
    """Return (X_raw, standardised coefficients, basis, y, K, name, sigma)."""
    X_np, y_true, K, name = loader_fn()
    grid_size = X_np.shape[1]

    basis = L2CosineBasis(
        T=1.0, R=R, grid_size=grid_size, d=1, device=DEVICE, dtype=DTYPE
    )
    X_coeffs = basis.project(
        torch.tensor(X_np, device=DEVICE, dtype=DTYPE).unsqueeze(-1)
    )

    mu = X_coeffs.mean(dim=0, keepdim=True)
    sd = X_coeffs.std(dim=0, keepdim=True).clamp(min=1e-8)
    X = (X_coeffs - mu) / sd

    sigma = max(float(np.median(pdist(X.cpu().numpy(), metric="euclidean"))), 1e-6)
    return X_np, X, basis, y_true, K, name, sigma


def fit_mmd_gmm(
    X, basis, K, kernel,
    covariance_type="diagonal", seed=SEED, epochs=NUM_EPOCHS, lr=LR,
    init="kmeans++", return_model=False, track_loss=False,
):
    """Fit an MMD GMM; return labels (and optionally responsibilities / losses)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GaussianMixtureModel(
        num_components=K, coeff_dim=X.shape[1], basis=basis,
        covariance_type=covariance_type, device=X.device, dtype=X.dtype,
    )
    model.initialize_from_data(X, method=init)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()
        if track_loss:
            losses.append(float(mmd2.detach()))

    with torch.no_grad():
        resp = model.responsibilities(X)
        labels = resp.argmax(dim=1).cpu().numpy()

    out = [labels]
    if return_model:
        out.append((model, resp.cpu().numpy()))
    if track_loss:
        out.append(losses)
    return out[0] if len(out) == 1 else tuple(out)


def variation_of_information(a, b):
    """VI(a,b) = H(a) + H(b) - 2 I(a,b), in nats."""
    def entropy(z):
        _, counts = np.unique(z, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum())
    return entropy(a) + entropy(b) - 2.0 * mutual_info_score(a, b)


def posterior_agreement(p, q):
    """Mean overlap sum_k min(p_k, q_k) after optimal component matching."""
    K = p.shape[1]
    cost = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = -np.minimum(p[:, i], q[:, j]).sum()
    r, c = linear_sum_assignment(cost)
    return float(-cost[r, c].sum() / p.shape[0])


def project_simplex(v):
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, 0)
    ind = torch.arange(1, v.numel() + 1, dtype=v.dtype)
    rho = int((u - (css - 1.0) / ind > 0).nonzero().max()) + 1
    return (v - (css[rho - 1] - 1.0) / rho).clamp(min=0.0)


def solve_pi_exact(I, J, iters=500):
    """argmin_{pi in simplex} pi^T I pi - 2 J^T pi, by projected gradient."""
    pi = torch.full((I.shape[0],), 1.0 / I.shape[0], dtype=I.dtype)
    L = 2.0 * torch.linalg.eigvalsh(I).max().clamp(min=1e-12)
    for _ in range(iters):
        pi = project_simplex(pi - (2.0 * (I @ pi) - 2.0 * J) / L)
    return pi


# ----------------------------------------------------------------------
# W3: GPmix and feature-space GMM baselines
# ----------------------------------------------------------------------
GPMIX_GRID = [
    (b, n)
    for b in ["fpc", "fourier", "ou", "bspline", "rl-fpc"]
    for n in [3, 5, 8, 12, 20]
]


def driver_gpmix(args):
    print("=" * 70)
    print("W3: GPmix and feature-space GMM baselines")
    print("=" * 70)

    out = {"per_dataset": {}, "gpmix_grid": {}}
    order = []

    for loader_fn, R in DATASET_SPECS:
        X_raw, X, basis, y, K, name, sigma = load_dataset(loader_fn, R)
        order.append(name)
        print(f"\n--- {name} (n={X_raw.shape[0]}, grid={X_raw.shape[1]}, K={K}) ---")
        entry = {}

        labels = fit_mmd_gmm(X, basis, K, GaussianKernel(sigma=sigma))
        entry["MMD GMM (Gaussian)"] = adjusted_rand_score(y, labels)

        entry["Feature GMM (EM, full)"] = adjusted_rand_score(
            y,
            GaussianMixture(
                n_components=K, covariance_type="full", n_init=10, random_state=SEED
            ).fit_predict(X.cpu().numpy()),
        )

        # Sweep GPmix's own hyperparameters so the comparison is not a strawman.
        per_cfg = {}
        for basis_type, n_proj in GPMIX_GRID:
            try:
                t0 = time.perf_counter()
                pred = GPmixClustering(
                    n_clusters=K, basis_type=basis_type, n_proj=n_proj,
                    random_state=SEED,
                ).fit_predict(X_raw)
                ari = adjusted_rand_score(y, pred)
                dt = time.perf_counter() - t0
            except Exception as e:  # some bases fail on short grids
                ari, dt = float("nan"), float("nan")
                print(f"    GPmix {basis_type}-{n_proj} failed: {e}")
            per_cfg[f"{basis_type}-{n_proj}"] = {"ari": ari, "time_s": dt}

        out["gpmix_grid"][name] = per_cfg
        valid = {k: v["ari"] for k, v in per_cfg.items() if not np.isnan(v["ari"])}
        best_cfg = max(valid, key=valid.get)
        entry["GPmix (reference cfg, fpc-8)"] = per_cfg["fpc-8"]["ari"]
        entry["GPmix (oracle cfg per dataset)"] = valid[best_cfg]
        entry["GPmix oracle cfg"] = best_cfg

        for k, v in entry.items():
            if isinstance(v, float):
                print(f"  {k:<34} ARI={v:.4f}")
        out["per_dataset"][name] = entry

    # Best *single* configuration held fixed across all datasets: the only
    # like-for-like comparison, since our method also uses one fixed config.
    cfg_means = {}
    for cfg in [f"{b}-{n}" for b, n in GPMIX_GRID]:
        vals = [out["gpmix_grid"][d][cfg]["ari"] for d in order]
        if not any(np.isnan(v) for v in vals):
            cfg_means[cfg] = (float(np.mean(vals)), float(np.std(vals)), vals)
    best_single = max(cfg_means, key=lambda c: cfg_means[c][0])

    summary = {
        "dataset_order": order,
        "gpmix_best_single_cfg": best_single,
        "gpmix_best_single": cfg_means[best_single],
        "gpmix_reference_mean": float(np.mean(
            [out["per_dataset"][d]["GPmix (reference cfg, fpc-8)"] for d in order])),
        "gpmix_oracle_mean": float(np.mean(
            [out["per_dataset"][d]["GPmix (oracle cfg per dataset)"] for d in order])),
        "ours_mean": float(np.mean(
            [out["per_dataset"][d]["MMD GMM (Gaussian)"] for d in order])),
        "feature_gmm_mean": float(np.mean(
            [out["per_dataset"][d]["Feature GMM (EM, full)"] for d in order])),
    }
    out["summary"] = summary

    print("\nMean ARI over the five datasets")
    print(f"  GPmix, reference cfg (fpc-8)        {summary['gpmix_reference_mean']:.3f}")
    print(f"  GPmix, best single cfg ({best_single:<11}) {cfg_means[best_single][0]:.3f}")
    print(f"  GPmix, oracle cfg per dataset       {summary['gpmix_oracle_mean']:.3f}  (upper bound, uses labels)")
    print(f"  MMD GMM (Gaussian), fixed cfg       {summary['ours_mean']:.3f}")
    print(f"  Feature GMM (EM, full covariance)   {summary['feature_gmm_mean']:.3f}")

    _save("r3_gpmix.json", out)


# ----------------------------------------------------------------------
# Q1: covariance structure
# ----------------------------------------------------------------------
def driver_covariance(args):
    print("=" * 70)
    print("Q1: full vs diagonal vs spherical projected covariance")
    print("=" * 70)

    cov_types = ["spherical", "diagonal", "full"]
    rows = {c: [] for c in cov_types}
    times = {c: [] for c in cov_types}
    out = {"per_dataset": {}, "objective_check": {}}

    for loader_fn, R in DATASET_SPECS:
        X_raw, X, basis, y, K, name, sigma = load_dataset(loader_fn, R)
        print(f"\n--- {name} (M={X.shape[1]}, K={K}) ---")
        entry = {}
        for cov in cov_types:
            t0 = time.perf_counter()
            labels = fit_mmd_gmm(
                X, basis, K, GaussianKernel(sigma=sigma), covariance_type=cov
            )
            dt = time.perf_counter() - t0
            ari = adjusted_rand_score(y, labels)
            rows[cov].append(ari)
            times[cov].append(dt)
            entry[cov] = {"ari": ari, "time_s": dt}
            print(f"  {cov:<10} ARI={ari:.4f}  ({dt:.1f}s)")
        out["per_dataset"][name] = entry

    # Is the full-covariance collapse an optimization failure? Compare the
    # attained objective, across learning rates and epoch budgets.
    print("\nObjective check: does full covariance actually fit worse?")
    for loader_fn, R in [DATASET_SPECS[i] for i in args.objective_datasets]:
        X_raw, X, basis, y, K, name, sigma = load_dataset(loader_fn, R)
        print(f"\n--- {name} (M={X.shape[1]}, K={K}) ---")
        checks = {}
        for cov in ["diagonal", "full"]:
            for lr, ep in [(0.1, 400), (0.02, 1500)]:
                labels, _, losses = fit_mmd_gmm(
                    X, basis, K, GaussianKernel(sigma=sigma),
                    covariance_type=cov, lr=lr, epochs=ep,
                    return_model=True, track_loss=True,
                )
                ari = adjusted_rand_score(y, labels)
                checks[f"{cov}-lr{lr}-ep{ep}"] = {
                    "final_mmd2": losses[-1], "ari": ari
                }
                print(f"  {cov:<9} lr={lr:<5} ep={ep:<5} "
                      f"final MMD2={losses[-1]:.6f}  ARI={ari:.4f}")
        out["objective_check"][name] = checks

    summary = {}
    print("\nMean ARI / mean fit time")
    for c in cov_types:
        summary[c] = {
            "ari_scores": rows[c],
            "mean_ari": float(np.mean(rows[c])),
            "std_ari": float(np.std(rows[c])),
            "mean_time_s": float(np.mean(times[c])),
        }
        print(f"  {c:<10} {summary[c]['mean_ari']:.3f} +/- "
              f"{summary[c]['std_ari']:.3f}   {summary[c]['mean_time_s']:.1f}s")
    out["summary"] = summary
    _save("r3_covariance.json", out)


# ----------------------------------------------------------------------
# Q2: initialization and projection-dimension stability
# ----------------------------------------------------------------------
def driver_stability(args):
    print("=" * 70)
    print(f"Q2: stability across {args.restarts} restarts and over M")
    print("=" * 70)

    out = {}
    for loader_fn, R in DATASET_SPECS:
        X_raw, X, basis, y, K, name, sigma = load_dataset(loader_fn, R)
        print(f"\n--- {name} (M={X.shape[1]}, K={K}) ---")
        kernel = GaussianKernel(sigma=sigma)
        entry = {}

        for init in ["kmeans++", "random"]:
            labs, resps, aris = [], [], []
            for s in range(args.restarts):
                lab, (_, resp) = fit_mmd_gmm(
                    X, basis, K, kernel, seed=SEED + s, init=init, return_model=True
                )
                labs.append(lab)
                resps.append(resp)
                aris.append(adjusted_rand_score(y, lab))
            pw_ari = [adjusted_rand_score(a, b) for a, b in combinations(labs, 2)]
            pw_vi = [variation_of_information(a, b) for a, b in combinations(labs, 2)]
            pw_pa = [posterior_agreement(a, b) for a, b in combinations(resps, 2)]
            entry[init] = {
                "ari_vs_truth_mean": float(np.mean(aris)),
                "ari_vs_truth_std": float(np.std(aris)),
                "pairwise_ari_mean": float(np.mean(pw_ari)),
                "pairwise_ari_min": float(np.min(pw_ari)),
                "pairwise_vi_mean": float(np.mean(pw_vi)),
                "posterior_agreement_mean": float(np.mean(pw_pa)),
            }
            e = entry[init]
            print(f"  init={init:<9} ARI(truth)={e['ari_vs_truth_mean']:.3f}"
                  f"+/-{e['ari_vs_truth_std']:.3f}  "
                  f"pairwise ARI={e['pairwise_ari_mean']:.3f}  "
                  f"VI={e['pairwise_vi_mean']:.3f}  "
                  f"post.agree={e['posterior_agreement_mean']:.3f}")

        m_grid = [m for m in args.m_grid if m <= X_raw.shape[1] // 2]
        m_labels, m_aris = {}, {}
        for M in m_grid:
            _, Xm, basis_m, _, _, _, sigma_m = load_dataset(loader_fn, M)
            m_labels[M] = fit_mmd_gmm(Xm, basis_m, K, GaussianKernel(sigma=sigma_m))
            m_aris[M] = adjusted_rand_score(y, m_labels[M])
        pw_m = [adjusted_rand_score(m_labels[a], m_labels[b])
                for a, b in combinations(m_grid, 2)]
        entry["projection_dim"] = {
            "M_grid": m_grid,
            "ari_vs_truth": m_aris,
            "pairwise_ari_across_M_mean": float(np.mean(pw_m)) if pw_m else float("nan"),
        }
        print("  over M: " + ", ".join(f"M={m}:{m_aris[m]:.3f}" for m in m_grid)
              + f"  | pairwise ARI across M="
                f"{entry['projection_dim']['pairwise_ari_across_M_mean']:.3f}")
        out[name] = entry

    _save("r3_stability.json", out)


# ----------------------------------------------------------------------
# W1: monotone descent (Proposition 4)
# ----------------------------------------------------------------------
def run_exact_scheme(X, basis, K, kernel, steps=300, seed=SEED):
    """Algorithm 1 exactly as Proposition 4 assumes it.

    Exact minimization of the convex QP over the simplex for pi, then one
    gradient step on xi with the step size backtracked to satisfy the
    descent-lemma condition (which enforces eta <= 1/L without knowing L).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GaussianMixtureModel(
        num_components=K, coeff_dim=X.shape[1], basis=basis,
        covariance_type="diagonal", device=X.device, dtype=X.dtype,
    )
    model.initialize_from_data(X, method="kmeans++")

    xi_params = [model._mean_coeffs, model._log_var]
    losses, gnorms = [], []
    eta = 1.0

    def objective():
        return model.compute_mmd2(X, kernel, compute_const_term=False)

    for _ in range(steps):
        with torch.no_grad():
            _, stats = objective()
            pi_star = solve_pi_exact(stats["I"].double(), stats["Jbar"].double())
            model._logits.copy_(torch.log(pi_star.clamp(min=1e-300)))

        model.zero_grad()
        f0, _ = objective()
        f0.backward()
        grads = [p.grad.detach().clone() for p in xi_params]
        gnorm2 = sum((g * g).sum() for g in grads)
        gnorms.append(float(gnorm2.sqrt()))

        base = [p.detach().clone() for p in xi_params]
        eta = min(eta * 2.0, 1e3)
        for _ in range(60):
            with torch.no_grad():
                for p, b, g in zip(xi_params, base, grads):
                    p.copy_(b - eta * g)
                f1, _ = objective()
            if float(f1) <= float(f0) - 0.5 * eta * float(gnorm2) + 1e-15:
                break
            eta *= 0.5
        losses.append(float(f0))

    with torch.no_grad():
        labels = model.responsibilities(X).argmax(dim=1).cpu().numpy()
    return np.array(losses), np.array(gnorms), labels


def driver_descent(args):
    print("=" * 70)
    print("W1: monotonicity of the objective (Proposition 4)")
    print("=" * 70)

    out = {}
    for loader_fn, R in DATASET_SPECS:
        X_raw, X, basis, y, K, name, sigma = load_dataset(loader_fn, R)
        kernel = GaussianKernel(sigma=sigma)

        losses, gnorms, labels = run_exact_scheme(X, basis, K, kernel, args.steps)
        d = np.diff(losses)

        # Adam, for contrast: not covered by the proposition.
        _, adam_losses = fit_mmd_gmm(X, basis, K, kernel, track_loss=True)
        da = np.diff(np.asarray(adam_losses))

        out[name] = {
            "exact": {
                "n_steps": len(losses),
                "n_increases": int((d > 1e-14).sum()),
                "max_increase": float(d.max()),
                "rel_decrease": float((losses[0] - losses[-1]) / abs(losses[0])),
                "grad_norm_first": float(gnorms[0]),
                "grad_norm_last": float(gnorms[-1]),
                "ari": float(adjusted_rand_score(y, labels)),
            },
            "adam": {
                "n_steps": len(adam_losses),
                "n_increases": int((da > 0).sum()),
                "max_increase": float(da.max()),
            },
        }
        e, a = out[name]["exact"], out[name]["adam"]
        print(f"  {name:<10} exact: {e['n_increases']}/{len(d)} increases, "
              f"rel.dec {e['rel_decrease']*100:.1f}%, "
              f"||grad|| {e['grad_norm_first']:.2e} -> {e['grad_norm_last']:.2e}"
              f"  | adam: {a['n_increases']}/{len(da)} increases, "
              f"max +{a['max_increase']:.1e}")

    _save("r3_descent.json", out)


# ----------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gpmix")

    p = sub.add_parser("covariance")
    p.add_argument("--objective-datasets", type=int, nargs="+", default=[0, 1, 4],
                   help="indices into DATASET_SPECS for the objective check")

    p = sub.add_parser("stability")
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--m-grid", type=int, nargs="+", default=[5, 10, 15, 20, 30])

    p = sub.add_parser("descent")
    p.add_argument("--steps", type=int, default=300)

    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    {"gpmix": driver_gpmix, "covariance": driver_covariance,
     "stability": driver_stability, "descent": driver_descent}[args.cmd](args)


if __name__ == "__main__":
    main()
