# Paper Build & Reproducibility

This repository contains the code and LaTeX sources for *Gaussian Mixture Models in Hilbert Spaces via Kernel Methods*. The paper lives in [`paper/`](paper/), and the compiled artifact is [`paper/main.pdf`](paper/main.pdf).

The repository ships only the executables needed to reproduce the paper: the benchmarks behind Table 1 and the scripts that generate every figure in the main text and appendix. The PEDAP CGM file is supplied under its data-use agreement; everything else is generated or downloaded on first run.

## Setup

Requires Python >= 3.11. All reproducibility scripts use fixed seeds, CPU execution, and `torch.float64` unless noted. The full pipeline usually takes 2-4 hours on a modern laptop.

Install the project environment with `uv`:

```bash
uv sync
source .venv/bin/activate
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```


## End-to-End Recipe

```bash
# Table 1 benchmarks
python -m benchmarks.bench_rd_sklearn
python -m benchmarks.bench_l2_realdata
python -m benchmarks.bench_l2_glucodensity
python -m benchmarks.bench_so3
python -m benchmarks.bench_graph
python -m benchmarks.table_generator

# Main-text figures:
# data overview panels, ablations, model_summary_v3.pdf, corr_ode_training.pdf,
# and patient_graph.pdf
python examples/paper_data_figure.py
python -m benchmarks.ablation_l2
python examples/use_case_visualization.py

# Supplemental figures:
# rd_gmm_summary.pdf, l2_k5_summary.pdf, l2_2d_k3_summary.pdf, so3_summary.pdf,
# graph_summary.pdf, lti_summary.pdf, atomic_qm9_representatives.pdf,
# and ntu_representatives.pdf
# (Optional) This might take a while
python examples/paper_synthetic_experiments.py
```


## Data

| Data | Access |
|---|---|
| Synthetic, sklearn toy datasets | Generated or bundled |
| Growth, Phoneme, Kneading, ECG200 | Downloaded on first run |
| MUTAG, QM9, TMQM, NTU RGB+D skeletons | Downloaded and cached under [`data/`](data/) |
| PEDAP CGM | Static `cgm_all_patients.csv` at `data/glucodensities/` |

## Reproduce the Paper

### Table 1 (clustering benchmark)

The five benchmarks below populate the columns of Table 1 (`tab:unified_benchmark`). Each writes a JSON file under [`benchmarks/results/`](benchmarks/results/); [`benchmarks/table_generator.py`](benchmarks/table_generator.py) assembles them into a LaTeX table.

| Column | Command |
|---|---|
| $\mathbb{R}^d$ | `python -m benchmarks.bench_rd_sklearn` |
| $L^2$ | `python -m benchmarks.bench_l2_realdata` |
| CGM (Gluco.) | `python -m benchmarks.bench_l2_glucodensity` |
| $\mathrm{SO}(3)$ | `python -m benchmarks.bench_so3` |
| Molecular | `python -m benchmarks.bench_graph` |

### Figures

| Figure(s) | Command |
|---|---|
| Ablation figures (`ablation_consistency.pdf`, `ablation_loss_K.pdf`, `ablation_k_sigma.pdf`) | `python -m benchmarks.ablation_l2` |
| Intro data overview (`images/data/data_overview_cgm_hourly_family.pdf`, `images/data/data_overview_fixed_graph_signal.pdf`, `images/data/data_overview_sym10_matrix.pdf`) | `python examples/paper_data_figure.py` |
| Synthetic appendix figures (`images/synthetic/rd_gmm_summary.pdf`, `images/synthetic/l2_k5_summary.pdf`, `images/synthetic/l2_2d_k3_summary.pdf`, `images/synthetic/so3_summary.pdf`, `images/synthetic/graph_summary.pdf`, `images/synthetic/lti_summary.pdf`, `images/synthetic/atomic_qm9_representatives.pdf`, `images/synthetic/ntu_representatives.pdf`) | `python examples/paper_synthetic_experiments.py` |
| Glucodensity case-study figures (`model_summary_v3.pdf`, `corr_ode_training.pdf`, `patient_graph.pdf`) | `python examples/use_case_visualization.py` |


