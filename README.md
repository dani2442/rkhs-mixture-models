# Paper Build & Reproducibility

This repository contains the code and LaTeX sources for *Gaussian Mixture Models in Hilbert Spaces via Kernel Methods*. The paper lives in [`paper/`](paper/), and the compiled artifact is [`paper/main.pdf`](paper/main.pdf).

The repository ships only the executables needed to reproduce the paper: the benchmarks behind Table 1 and the scripts that generate every figure in the main text and appendix. The PEDAP CGM file must be supplied under its data-use agreement; everything else is generated or downloaded on first run.

## Setup

Requires Python >= 3.11.

```bash
uv sync
source .venv/bin/activate
```

All reproducibility scripts use fixed seeds, CPU execution, and `torch.float64` unless noted. The full pipeline usually takes 2-4 hours on a modern laptop.


## End-to-End Recipe

```bash
uv sync
source .venv/bin/activate

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
python examples/patient_correlation_network_graph.py

# Supplemental figures:
# rd_gmm_summary.pdf, l2_k5_summary.pdf, l2_2d_k3_summary.pdf, so3_summary.pdf,
# graph_summary.pdf, lti_summary.pdf, atomic_qm9_representatives.pdf,
# and ntu_representatives.pdf
python examples/paper_synthetic_experiments.py

# Build the paper
cd paper
latexmk -r latexmkrc main.tex
cd ..
```


## Data

| Data | Access |
|---|---|
| Synthetic, sklearn toy datasets | Generated or bundled |
| Growth, Phoneme, Kneading, ECG200 | Downloaded on first run |
| MUTAG, QM9, TMQM, NTU RGB+D skeletons | Downloaded and cached under [`data/`](data/) |
| PEDAP CGM | Manual: place `cgm_all_patients.csv` at `data/glucodensities/` |

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
| Glucodensity case-study figures (`model_summary_v3.pdf`, `corr_ode_training.pdf`) | `python examples/use_case_visualization.py` |
| Patient similarity graph (`patient_graph.pdf`) | `python examples/patient_correlation_network_graph.py` |

The figure scripts are converted from Jupyter notebooks; they import shared training/preprocessing helpers (`train_glucodensity_*.py`, `train_atomic.py`, `train_ntu_skeleton.py`, `lti.py`) from [`examples/`](examples/).


