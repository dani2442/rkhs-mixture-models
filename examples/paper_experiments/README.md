# Paper experiments

Code and saved results for the GPmix comparison, component-count scaling, and
numerical diagnostics. Only records supporting these retained comparisons and
checks are included. The experiment drivers also expose exploratory modes through
`--help`; their outputs are generated on demand.

## Setup

Use Python 3.11. The package versions used for reproduction checks are pinned in
`requirements-reproduction.txt`. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r examples/paper_experiments/requirements-reproduction.txt
```

The commands below assume this working directory. Scripts also resolve data and
paper paths correctly when invoked by absolute path from elsewhere.

## Reproduce the paper results

```bash
# Check both paper versions against the saved results without downloading data.
.venv/bin/python examples/paper_experiments/reproduce_paper.py check

# Print the GPmix cell and component-sweep rows directly from the saved JSON.
.venv/bin/python examples/paper_experiments/reproduce_paper.py tables

# Refit GPmix twice per scalar dataset and compare with the saved ARIs.
.venv/bin/python examples/paper_experiments/reproduce_paper.py gpmix --threads 1 --output /tmp/continuous-gmm-rerun/gpmix.json

# Check bandwidth sampling and independent-process repeatability.
.venv/bin/python -m unittest discover -s examples/paper_experiments -p 'test_reproducibility.py' -v

# Verify standard, matrix-product, and blocked objective evaluations.
.venv/bin/python examples/paper_experiments/synthetic_experiments.py verify --threads 1 --results-dir /tmp/continuous-gmm-rerun

# Rerun the component-count sweep at n = 100,000.
.venv/bin/python examples/paper_experiments/synthetic_experiments.py largek-iso --mode nfixed --threads 1 --results-dir /tmp/continuous-gmm-rerun

# Inspect conditioning and Cholesky solve residuals.
.venv/bin/python examples/paper_experiments/numerical_diagnostics.py stability --results-dir /tmp/continuous-gmm-rerun
```

These commands preserve the saved measurements by writing new outputs to a
separate directory. Runtime and memory vary with the execution environment.

## Scripts and saved records

| Script | Purpose | Saved record |
|---|---|---|
| `reproduce_paper.py` | Validate paper tables and refit the fixed GPmix configuration | Reads `results/gpmix_comparison.json` and `results/largek_nfixed.json` |
| `synthetic_experiments.py` | Component-count sweep and algebraic equivalence checks | `results/largek_nfixed.json`, `results/largek_verify.json` |
| `numerical_diagnostics.py` | Inspect shifted covariance matrices and numerical solve accuracy | `results/numerical_stability.json` |
| `functional_benchmarks.py` | GPmix configuration selection and functional baseline comparisons | `results/gpmix_comparison.json` |
| `test_reproducibility.py` | Check seeded bandwidth sampling and repeatability across processes | No persistent output |

The GPmix paper entry uses the five scalar datasets (Growth, Waveform, Phoneme,
Kneading, ECG200), configuration `rl-fpc-12`, and seed 42. Its mean ARI and standard
deviation are **0.409 ± 0.235**. This configuration was selected by mean ARI on a
six-dataset suite that also includes vector-valued Canadian Weather; the saved
configuration grid preserves that selection history. The full grid can be rerun
with `functional_benchmarks.py gpmix --results-dir /tmp/continuous-gmm-rerun`.
The fixed-configuration reproduction check is sufficient for the paper entry and
requires only the public datasets used by the existing loaders, not the private
CGM cohort. `benchmarks/bench_l2_realdata.py` uses the same five-dataset suite and
configuration by default; `--include-weather` saves the extension separately.

The component sweep uses one generated dataset per K and five initializations,
with n = 100,000 and M = 20. The matrix-product evaluation avoids the n × K × M
difference tensor; the paper sweep uses neither row blocking nor subsampled
initialization. **Memory (MiB)** is peak process memory above the baseline before
data generation, measured over all fits and their predictions.

New sweep runs seed bandwidth subsampling independently of the global random
state and record the bandwidth, data and initialization seeds, per-restart scores,
thread settings, and software versions. PyTorch and BLAS default to one thread,
which is propagated to subprocesses. Historical timing runs did not record every
bandwidth subsample or thread setting, so exact refitting of those measurements
is not guaranteed. Saved measurements are preserved unchanged.

The conditioning diagnostic concerns I + K/σ² and I + (K₁ + K₂)/σ² in the kernel
expectations. Their identity shift prevents singularity but does not guarantee
good conditioning at small bandwidths; it also says nothing about conditioning of
the unshifted covariances used for posterior densities.
