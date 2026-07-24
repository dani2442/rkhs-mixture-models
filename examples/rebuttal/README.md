# Rebuttal experiments

Scripts backing the numbers quoted in `paper/rebuttal_Reviewer 1.tex`,
`paper/rebuttal_Reviewer 2.tex`, and `paper/rebuttal_Reviewer 3.tex`. All are CPU-only, seeded, and reproduce the
reported values exactly. Recorded outputs are in `results/`; figures are written
into `paper/images/`.

Run from anywhere (paths resolve relative to the repo root):

```bash
.venv/bin/python examples/rebuttal/reviewer2_experiments.py stability
```

## Reviewer 1 (#WZyT)

| Script | Command | Backs |
|---|---|---|
| `reviewer1_cgm_elbow.py` | — | Within-cluster MMD² vs K for the CGM representations (W1/Q1). Needs `data/glucodensities/`. → `results/cgm_elbow.json` |
| `reviewer1_plot_elbow.py` | — | Replots the above → `paper/images/rebuttal_cgm_elbow.pdf` |
| `reviewer1_synthetic.py` | `largek` | ARI/runtime/memory/stability for K ∈ {2,…,50} (W2) → `results/largek.json` |
| | `runtime` | Runtime scaling in n, M, K (W3) → `paper/images/rebuttal_runtime_scaling.pdf` |
| | `icl` | Finite-M ICL vs the MMD elbow (Q2) → `paper/images/rebuttal_icl.pdf` |

## Reviewer 2 (#aBwm)

| Script | Command | Backs |
|---|---|---|
| `reviewer2_experiments.py` | `scaling` | Time/memory vs n up to 10⁵, with and without the O(n²) data–data term (W1, Limitations) → `results/r2_scaling.json` |
| | `stability` | Which matrices are factorized, their conditioning, Cholesky vs explicit inverse (W2) → `results/r2_stability.json` |
| | `nongauss` | Held-out MMD² vs K for ring / heavy-tailed / curved / bimodal targets (W3) → `paper/images/rebuttal_nongaussian.pdf` |
| `reviewer2_kmeans.py` | — | k-means vs the mixture with a Bayes-rule reference (Q1) → `results/r2_kmeans.json` |
| `reviewer2_bandwidth.py` | — | Bandwidth sensitivity: why the median heuristic underperforms on multiscale data (Q1) → `results/r2_bandwidth.json` |

## Reviewer 3 (#xDmR)

| Script | Command | Backs |
|---|---|---|
| `reviewer3_experiments.py` | `gpmix` | GPmix swept over its own basis/n_proj grid, plus a feature-space GMM (EM, full cov.) on the same coefficients (W3) → `results/r3_gpmix.json` |
| | `covariance` | Full vs diagonal vs spherical covariance, with the attained objective alongside the ARI (Q1) → `results/r3_covariance.json` |
| | `stability` | Pairwise ARI / VI / posterior agreement over 10 restarts, and over M (Q2) → `results/r3_stability.json` |
| | `descent` | Monotonicity of the objective under the exact scheme of Prop. 4 vs Adam (W1) → `results/r3_descent.json` |

GPmix is *also* a permanent competitor (`src/competitors/gpmix.py`), registered in
`benchmarks/config.py` and run by `benchmarks/bench_l2_realdata.py`, so it appears
in Table 1. The sweep here exists only to establish which single configuration the
benchmark should use, and to bound how well GPmix could do if tuned per dataset.

## Notes on two results

**The O(n²) term (W1).** `scaling` reports both paths because the data–data term
`(1/n²)ΣΣκ(Xᵢ,Xⱼ)` is constant in θ: it never enters the gradient and is identical
across K, so it cancels in the elbow too. `fit_no_const` in `reviewer2_experiments.py`
is the training loop without it — that is the path that reaches n = 10⁵ in ~73 s
and 550 MB. Retaining the term costs 2.3 GB already at n = 10⁴.

**The bandwidth finding (Q1).** The concentric-shells scenario scores ARI 0.422
under the median heuristic and 1.000 at σ ≤ 0.35 × median. The heuristic is
dominated by the widest component, so on multiscale data it returns a σ under which
the kernel is nearly flat across the narrow component. This is a property of the
bandwidth heuristic, not of the mixture model; `reviewer2_bandwidth.py` sweeps σ
across all three scenarios, including the well-separated control that is unaffected.

**Three results that went against the paper (R3).** They are reported in the
rebuttal as found, not filtered. (1) GPmix is not beaten: at its reference config
it scores 0.288 against our 0.412, but its best *single* config scores 0.393 — a
tie — and oracle-tuned per dataset it reaches 0.491. (2) A plain EM Gaussian
mixture on the *same* cosine coefficients scores 0.459, beating us on L². (3) Full
covariances reach a strictly *lower* MMD² than diagonal ones on every dataset
while their ARI collapses from 0.412 to 0.043; `covariance` re-checks this across
learning rates and epoch budgets precisely because the obvious explanation — an
optimizer failure — is the wrong one. The diagonal restriction is acting as an
implicit regularizer, and distributional fit and cluster recovery are in tension.

**One negative result is kept deliberately.** `driver_nongauss` also fits a ring
concentric with a central blob and attempts to recover the two-class partition by
merging components on their fitted means. It does not work (ARI ≈ 0). This is the
semantic-clustering caveat acknowledged in the W3 response — when a class is itself
non-Gaussian it spans several components, and merging by mean is not a sufficient
criterion. It is not reported as a result in the rebuttal.
