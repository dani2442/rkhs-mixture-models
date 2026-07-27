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
| `reviewer1_cgm_elbow.py` | — | Within-cluster MMD² vs K for all three CGM representations — intraday H¹ curves, Sym(24) correlation matrices, and patient-similarity graph signals (W1/Q1). Needs `data/glucodensities/`. → `results/cgm_elbow.json` |
| `reviewer1_plot_elbow.py` | — | Replots the above as three panels → `paper/images/rebuttal_cgm_elbow.pdf` |
| `reviewer1_synthetic.py` | `largek` | ARI/runtime/memory/stability for K ∈ {5,…,500}, up to n = 10⁶ (W2) → `results/largek.json` |
| | `largek-iso --mode nk` | Controlled K sweep isolating K with n/K = 500 fixed (W2) → `results/largek_nk.json` |
| | `largek-iso --mode nfixed` | Controlled K sweep isolating K with n = 10⁵ fixed (W2) → `results/largek_nfixed.json` |
| | `runtime` | Runtime scaling in n (≤10⁵), M (≤10³), K (≤500) (W3) → `paper/images/rebuttal_runtime_scaling.pdf`. `--plot_only` redraws from the recorded JSON |
| | `icl` | Finite-M ICL vs the MMD elbow (Q2) → `paper/images/rebuttal_icl.pdf` |
| | `verify` | Checks the matmul evaluation used by `largek`/`runtime` against the released `compute_J_diag` (see below) → `results/largek_verify.json` |

## Reviewer 2 (#aBwm)

| Script | Command | Backs |
|---|---|---|
| `reviewer2_experiments.py` | `scaling` | Time/memory/ARI/final MMD² vs n up to 10⁶, fitted without the O(n²) data–data term (W1, Limitations) → `results/r2_scaling.json` |
| | `stability` | Which matrices are factorized, their conditioning, Cholesky vs explicit inverse (W2) → `results/r2_stability.json` |
| | `nongauss` | Held-out MMD² vs K for ring / heavy-tailed / curved / bimodal targets (W3) → `paper/images/rebuttal_nongaussian.pdf` |
| `reviewer2_kmeans.py` | — | k-means vs the mixture with a Bayes-rule reference (Q1) → `results/r2_kmeans.json` |
| `reviewer2_bandwidth.py` | — | Bandwidth sensitivity: why the median heuristic underperforms on multiscale data (Q1) → `results/r2_bandwidth.json` |

## Reviewer 3 (#xDmR)

| Script | Command | Backs |
|---|---|---|
| `reviewer3_experiments.py` | `gpmix` | GPmix swept over its own basis/n_proj grid, plus a feature-space GMM (EM, full cov.) on the same coefficients (W3) → `results/r3_gpmix.json` |
| | `covariance` | Full vs diagonal vs spherical covariance, with the attained objective alongside the ARI (Q1) → `results/r3_covariance.json` |
| | `stability` | Empirical identifiability: 5 restarts per (dataset, M) for M ∈ {1,5,10,20} — M=1 is the degenerate control where the projection is maximally non-injective — for our method *and* the Projected GMM-EM baseline, scored by ARI ± std vs. truth, pairwise ARI / VI / posterior agreement between restarts; prints the two rebuttal tabulars (Q2) → `results/r3_stability.json` |
| | `descent` | Monotonicity of the objective under the exact scheme of Prop. 4 vs Adam (W1) → `results/r3_descent.json` |

GPmix is *also* a permanent competitor (`src/competitors/gpmix.py`), registered in
`benchmarks/config.py` and run by `benchmarks/bench_l2_realdata.py`, so it appears
in Table 1. The sweep here exists only to establish which single configuration the
benchmark should use, and to bound how well GPmix could do if tuned per dataset.

## Notes on individual results

**The elbow curves (R1, W1/Q1).** The graph panel replicates
`patient_correlation_network_graph.ipynb` (21-day windows, patients present in ≥40% of
them, H¹ curve metric with 16 cosine modes, edges above the 85th percentile of the
time-averaged similarity, 4 Laplacian eigenvectors → n = 141, M = 4×76 = 304), fitted
statically rather than with the temporal π(t) model. Only the two matrix-valued
representations produce a sharp knee: the first relative drop is −36% (correlation) and
−49% (graph) against −8%/−32% for the next component, while the intraday curve falls by
22–45% at *every* step and singles out no K. The rebuttal says so rather than claiming a
knee at the reported K.

**Isolating K (R1, W2).** The reviewer noted that `largek` varies K and n together, so
large-K columns also have more observations per component and are statistically easier.
`largek-iso` re-runs the same K grid ({5,10,20,50,100,200}, M=20, diagonal cov., 200
epochs, 5 k-means++ restarts, same separation/generation) twice: `--mode nk` holds
n/K = 500 fixed, `--mode nfixed` holds n = 10⁵ fixed. Each K is fitted in its own
subprocess so the reported memory is that fit's incremental (above-baseline) RSS. Reported
per K: ARI mean ± std across restarts, median time per fit, peak incremental memory. With
n/K fixed the mean ARI does not degrade with K (≥ 0.966 throughout); the two sweeps agree
at K = 200, n = 10⁵. Both use the same single-block matmul path below (no chunking or
subsampling at any grid point).

**Reaching K = 200 / n = 10⁵ (R1, W2 and W3).** `GaussianKernel.compute_J_diag`
materialises an `(n, K, M)` difference tensor, which needs 3.2 GB at
n = 10⁵, K = 200, M = 20 and dominates the runtime. `J_diag_matmul` in
`reviewer1_synthetic.py` expands the same quadratic form,
`‖x − m_k‖²_{A_k} = ⟨x², 1/A_k⟩ − 2⟨x, m_k/A_k⟩ + ⟨m_k², 1/A_k⟩`, into three BLAS
calls with O(nK) memory — same closed form, same gradients, ~5× faster. The
`verify` command checks it: J agrees to 2.7e-16 relative, the fits give identical
ARI, and the final MMD² agrees to ≤3.3e-16 absolute once the constant data–data
term (which the fast path drops, as it is constant in θ) is added back. `largek`
and `runtime` report timings for this path; it has *not* been upstreamed into
`src/kernel/gaussian.py`.

Two further concessions are needed for the last column (K = 500, n = 10⁶), both
recorded in the rebuttal. (i) Above 2·10⁷ `(n, K)` cells, J is built in row blocks
and each block is backpropagated as it is formed; J enters the objective only through
its column means, so the accumulated gradient is the exact full-batch gradient (one
Adam step per epoch on all n points) and only peak memory changes — `verify` confirms
bit-identical fits with blocking forced on. (ii) Above 5·10⁷ `(n, K)` cells the
k-means⁺⁺ seeding, which costs O(nMK²) and materialises its own `(n, K)` distance
matrix, is run on a random subsample of 50 000 points. That is an initialization
heuristic, not the objective, but it does mean the largest column is not seeded from
all 10⁶ points.

**The O(n²) term (W1).** The data–data term `(1/n²)ΣΣκ(Xᵢ,Xⱼ)` is constant in θ:
it never enters the gradient and is identical across K, so it cancels in the elbow
too. `fit_no_const` in `reviewer2_experiments.py` is the training loop without it —
that is the path `scaling` measures, and the one that reaches n = 10⁶. The
with-the-term path is still reachable via `scaling-single --with_const`; it costs
2.3 GB already at n = 10⁴ and is infeasible beyond that.

Reporting an *absolute* final MMD² does need the term, so `scaling` adds it back
after the fit (and after the peak-memory reading, so the reported memory stays
the fit's). `data_data_const_exact` accumulates it in row blocks — O(n²) time,
O(n·block) memory — for n ≤ `--const_exact_max` (default 10⁵, ~7 min at 10⁵ and
~11 h at 10⁶). Beyond that, `data_data_const_closed_form` evaluates its
expectation under the known generating mixture at the same σ. Subsampling the
term instead is *not* accurate enough: at 2·10⁴ subsampled points its error is
~5·10⁻⁴, twice the MMD² being reported, whereas the closed form is off by
3.9·10⁻⁵ at n = 10⁴ and shrinks like n^(-1/2) (~4·10⁻⁶ at n = 10⁶, ≈1.6% of the
value). `median_sigma` takes a `seed` here for the same reason — σ came from the
unseeded global RNG, and a 0.2% wobble in σ moves the constant by ~10⁻³.

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
