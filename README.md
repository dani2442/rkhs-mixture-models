# Paper Build & Reproducibility

This repository contains the code and LaTeX sources for *Random Objects in Hilbert Spaces via Kernel Mixture Gaussian Model*. The paper lives in [`paper/`](paper/), and the compiled artifact is [`paper/main.pdf`](paper/main.pdf).

The paper is reproducible from a clean checkout: scripts regenerate the reported tables, figures, and cached datasets, except for the PEDAP CGM file, which must be supplied under its data-use agreement.

## Setup

Requires Python >= 3.11.

```bash
uv sync
source .venv/bin/activate
```

All reproducibility scripts use fixed seeds, CPU execution, and `torch.float64` unless noted. The full pipeline usually takes 2-4 hours on a modern laptop.

## Build the PDF

```bash
cd paper
latexmk -r latexmkrc main.tex
latexmk -C   # optional cleanup
```

The NeurIPS style file, bibliography, and latexmk config are committed in [`paper/`](paper/).

## Data

| Data | Access |
|---|---|
| Synthetic, sklearn toy datasets | Generated or bundled |
| Growth, Phoneme, Kneading, ECG200 | Downloaded on first run |
| MUTAG, QM9, TMQM, NTU RGB+D skeletons | Downloaded and cached under [`data/`](data/) |
| PEDAP CGM | Manual: place `cgm_all_patients.csv` at `data/glucodensities/` |

## Reproduce the Paper

| Output | Command |
|---|---|
| Clustering benchmark table | Run the benchmark commands below |
| Ablation figures | `python -m benchmarks.ablation_l2` |
| Glucodensity case-study figures | `python examples/train_glucodensity_temporal.py`; `python examples/train_glucodensity_correlation.py`; `python examples/train_glucodensity_correlation_graph.py` |
| Synthetic appendix figures | `jupyter nbconvert --to notebook --execute examples/paper_synthetic_experiments.ipynb --inplace` |
| Intro data overview | `jupyter nbconvert --to notebook --execute examples/paper_data_figure.ipynb --inplace` |
| Polished glucodensity figures | Execute [`examples/use_case_visualization.ipynb`](examples/use_case_visualization.ipynb) and [`examples/patient_correlation_network_graph.ipynb`](examples/patient_correlation_network_graph.ipynb) |

The clustering table is assembled from JSON files in [`benchmarks/results/`](benchmarks/results/) and written to [`paper/sections/table_benchmark.tex`](paper/sections/table_benchmark.tex):


## End-to-End Recipe

```bash
uv sync
source .venv/bin/activate

for b in bench_rd_sklearn bench_l2_synthetic bench_l2_realdata \
         bench_l2_glucodensity bench_l2_skeleton bench_so3 bench_graph; do
    python -m benchmarks.$b
done

python -m benchmarks.ablation_l2
python examples/train_glucodensity_temporal.py
python examples/train_glucodensity_correlation.py
python examples/train_glucodensity_correlation_graph.py

jupyter nbconvert --to notebook --execute examples/paper_synthetic_experiments.ipynb --inplace
jupyter nbconvert --to notebook --execute examples/paper_data_figure.ipynb --inplace
jupyter nbconvert --to notebook --execute examples/use_case_visualization.ipynb --inplace
jupyter nbconvert --to notebook --execute examples/patient_correlation_network_graph.ipynb --inplace

```

