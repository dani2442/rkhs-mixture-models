# Kernel Mixture Gaussian Model in Hilbert Spaces

> **Fitting Gaussian mixtures to infinite-dimensional data via Maximum Mean Discrepancy (MMD)**

This repository implements the framework described in *"Gaussian Mixture Models in Hilbert Spaces via Kernel Methods"*. It provides a modular, PyTorch-based toolkit for fitting Gaussian mixture models to data living in separable Hilbert spaces—function spaces, rotation groups, graph signal spaces, and more—using closed-form MMD optimization instead of likelihood-based methods.

---

## Key Idea

Classical Gaussian mixture models (GMMs) rely on evaluating probability densities. In infinite-dimensional Hilbert spaces (e.g., $L^2$ function spaces), no Lebesgue measure exists, making density-based fitting ill-defined.

This package sidesteps the problem entirely by using the **Maximum Mean Discrepancy (MMD)**, a kernel-based distance between probability measures that requires no densities. Given:

- An empirical distribution $P_n = \frac{1}{n}\sum_{i=1}^{n} \delta_{X_i}$ from data
- A Gaussian mixture $Q_\theta = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(m_k, \mathcal{K}_k)$

We fit the mixture by minimizing $\text{MMD}^2(P_n, Q_\theta)$, which admits **exact closed-form evaluation** for Gaussian and polynomial kernels—no sampling from $Q_\theta$ is needed.

### How It Works

1. **Project** data from the Hilbert space onto a finite-dimensional orthonormal basis (cosine, Fourier, graph Laplacian eigenvectors, Wigner D-matrices, etc.)
2. **Compute** the MMD² objective using closed-form expressions.
3. **Optimize** mixture parameters (weights, means, covariances) via gradient descent on the differentiable MMD² loss

All quantities converge to their infinite-dimensional counterparts as the truncation dimension $M \to \infty$.


## Features

- **Closed-form MMD**: Exact evaluation of MMD² between empirical distributions and Gaussian mixtures (no sampling required)
- **Multiple kernels**: Gaussian (RBF) and polynomial kernels with full closed-form expressions
- **Diverse Hilbert spaces**:
  - $L^2([0,T]; \mathbb{R}^d)$ — functional/time-series data (cosine and Fourier bases)
  - $H^1([0,T]; \mathbb{R}^d)$ — Sobolev space (cosine basis)
  - $L^2([0,T]^2; \mathbb{R}^d)$ — 2D spatial data (tensor product bases)
  - $L^2(\text{SO}(3))$ — rotation data (Wigner D-matrix basis via Peter-Weyl theorem)
  - $\text{Sym}(n)$ — symmetric matrices with the Frobenius inner product (scaled vech embedding, optional PCA)
  - Graph signals — signals on graph vertices (Laplacian eigenbasis)
  - $\mathbb{R}^n$ — finite-dimensional vectors (canonical and DCT bases)
- **Flexible covariance**: Diagonal, spherical, and full (Cholesky) covariance parameterizations
- **Convex weight optimization**: Closed-form solution for mixture weights given fixed components
- **Time-varying mixtures**: Support for time-dependent mixture weights via basis expansions or Neural ODEs
- **Molecular fingerprints**: WL hash embeddings for molecular graph data (integrates with PyTorch Geometric)
- **Initialization**: Random and k-means++ initialization strategies
- **Fully differentiable**: End-to-end gradient-based optimization via PyTorch autograd


## Installation

Requires Python ≥ 3.11.

```bash
git clone https://github.com/dani2442/rkhs-mixture-models.git mixture
cd mixture
uv sync
```

## Quick Start

### Minimal Example

```python
import torch
from src import (
    L2CosineBasis,
    GaussianKernel,
    GaussianMixtureModel,
    fit_gaussian_mixture_mmd,
)

# 1. Setup
T, d, grid_size, R = 1.0, 2, 200, 20
basis = L2CosineBasis(T=T, R=R, grid_size=grid_size, d=d)
kernel = GaussianKernel(sigma=1.0)

# 2. Project your functional data onto the basis
# X_raw: (n_samples, grid_size, d) — your trajectories on a time grid
X = basis.project(X_raw)  # (n_samples, R*d)

# 3. Fit a Gaussian mixture by minimizing MMD²
model, history = fit_gaussian_mixture_mmd(
    X=X,
    num_components=3,
    kernel=kernel,
    num_epochs=200,
    lr=0.1,
    basis=basis,
)

# 4. Inspect results
print(f"Mixture weights: {model.pi.detach().numpy()}")
print(f"Final MMD²: {history[-1]:.6f}")

# 5. Sample new functions from the fitted model
functions, coeffs, assignments = model.sample_functions(num_samples=10)
# functions: (10, grid_size, d) — reconstructed trajectories
```

### Manual MMD Computation

```python
from src import mmd2_empirical_vs_gaussian_mixture, GaussianKernel

kernel = GaussianKernel(sigma=2.0)

# X: (n, M) projected data coefficients
# pi: (K,) mixture weights
# m: (K, M) component means
# Kcov: (K, M, M) component covariances
mmd2, stats = mmd2_empirical_vs_gaussian_mixture(
    X=X, pi=pi, m=m, Kcov=Kcov, kernel=kernel
)
```


## Examples

The `examples/` directory contains complete, self-contained scripts for each supported domain. All scripts generate synthetic data, fit the model, and produce visualizations.

| Script | Domain | Description |
|--------|--------|-------------|
| [`train_l2.py`](examples/train_l2.py) | $L^2([0,1]; \mathbb{R}^2)$ | Functional data with sine/cosine mixtures |
| [`train_l2_gaussian.py`](examples/train_l2_gaussian.py) | $L^2([0,1]; \mathbb{R}^2)$ | Gaussian process data with covariance recovery |
| [`train_l2_2d_gaussian.py`](examples/train_l2_2d_gaussian.py) | $L^2([0,1]^2; \mathbb{R})$ | 2D spatial data with tensor product basis |
| [`train_l2_2d_temporal_pi.py`](examples/train_l2_2d_temporal_pi.py) | $L^2([0,1]^2; \mathbb{R})$ | Time-varying mixture weights |
| [`train_so3.py`](examples/train_so3.py) | $L^2(\text{SO}(3))$ | Rotation data with Wigner D-matrix basis |
| [`train_graph.py`](examples/train_graph.py) | Graph signals | Signals on Erdős–Rényi graphs with Laplacian basis |
| [`train_molecules.py`](examples/train_molecules.py) | Molecular graphs | ESOL molecules via WL hash → DCT basis |
| [`train_ntu_skeleton.py`](examples/train_ntu_skeleton.py) | $L^2([0,1]; \mathbb{R}^{75})$ | Human action skeleton sequences (NTU RGB+D) |
| [`train_atomic.py`](examples/train_atomic.py) | Molecular data | QM9/TMQM atomic datasets |
| [`compare_sklearn_clustering_mmd.py`](examples/compare_sklearn_clustering_mmd.py) | $\mathbb{R}^2$ toy datasets | scikit-learn clustering benchmark + MMD with finite-dimensional quadratic kernel |

Run any example from the repository root:

```bash
python examples/train_l2.py
python examples/train_so3.py
python examples/train_graph.py
python examples/compare_sklearn_clustering_mmd.py
```


## Architecture

```
src/
├── __init__.py              # Public API and MMD² computation
├── mixture.py               # GaussianMixtureModel (nn.Module) and fit_gaussian_mixture_mmd()
├── temporal_mixture.py      # Time-varying weights (basis expansion + Neural ODE)
├── data.py                  # Synthetic data generators
├── visualization.py         # Plotting utilities
├── kernel/
│   ├── base.py              # Abstract Kernel class (J, I expectations)
│   ├── gaussian.py          # Gaussian (RBF) kernel with closed-form J, I
│   └── polynomial.py        # Polynomial kernel with moment-based J, I
├── spaces/
│   ├── base.py              # Abstract HilbertBasis class
│   ├── L2.py                # L² and H¹ bases: cosine, Fourier, 2D tensor product
│   ├── Rn.py                # R^n bases: canonical, discrete cosine transform
│   ├── so3.py               # SO(3) bases via Wigner D-matrices
│   ├── symmetric.py         # Sym(n) with Frobenius inner product (vech + optional PCA)
│   ├── graph.py             # Graph Laplacian eigenbasis
│   └── graph_embedding.py   # WL hash fingerprint for molecular graphs
└── competitors/             # Baseline clustering methods used in benchmarks
                             # (functional k-means, funHDDC, kernel k-groups,
                             #  projected GMM-EM, scikit-fda agglomerative, ...)
```
## Mathematical Background

The MMD² between the empirical measure $P$ and the Gaussian mixture $Q$ decomposes as:

$$
\text{MMD}^2(P_n, Q_\theta) = \underbrace{\frac{1}{n^2} \sum_{i,j} \kappa(X_i, X_j)}_{\text{data–data (constant)}} - \frac{2}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} \pi_k J_{i,k} + \sum_{k,s=1}^{K} \pi_k \pi_s I_{k,s}
$$

For the **Gaussian kernel** $\kappa(x, y) = \exp\!\left(-\|x-y\|^2 / (2\sigma^2)\right)$, writing $\alpha = -1/(2\sigma^2)$:

$$
J_{i,k} = \text{det}(I - 2\alpha K_k)^{-1/2} \exp\left(\alpha (X_i - m_k)^\top (I - 2\alpha K_k)^{-1} (X_i - m_k)\right)
$$

$$
I_{k,s} = \text{det}(I - 2\alpha (K_k + K_s))^{-1/2} \exp\!\left(\alpha (m_k - m_s)^\top (I - 2\alpha (K_k + K_s))^{-1} (m_k - m_s)\right)
$$

Both expressions are differentiable and computed exactly via Cholesky decompositions—no Monte Carlo estimation needed.



## Core Components

### Kernels (`src/kernel/`)

Each kernel implements three operations needed for the MMD computation:

| Method | Formula | Description |
|--------|---------|-------------|
| `evaluate(x, y)` | $\kappa(x, y)$ | Pointwise kernel evaluation |
| `compute_J(X, m, Kcov)` | $J_{i,k} = \mathbb{E}_{y \sim \mathcal{N}(m_k, K_k)}[\kappa(X_i, y)]$ | Data-component cross-expectation |
| `compute_I(m, Kcov)` | $I_{k,s} = \mathbb{E}_{y \sim \mathcal{N}(m_k,K_k),\, y' \sim \mathcal{N}(m_s,K_s)}[\kappa(y, y')]$ | Component-component cross-expectation |

**Gaussian kernel** (`GaussianKernel`): Uses Fredholm determinants and Cholesky-based solves for $J$ and $I$.
**Polynomial kernel** (`PolynomialKernel`, plus the `LinearKernel` and `QuadraticKernel` shortcuts): Uses moment formulas for Gaussian random variables.

### Hilbert Bases (`src/spaces/`)

Each basis provides `project()` (data → coefficients) and optionally `reconstruct()` (coefficients → data):

| Basis | Space | Projection |
|-------|-------|------------|
| `L2CosineBasis` | $L^2([0,T]; \mathbb{R}^d)$ | $c_r = \int_0^T f(t) \phi_r(t) \, dt$ via quadrature |
| `L2FourierBasis` | $L^2([0,T]; \mathbb{R}^d)$ | Fourier sin/cos basis |
| `H1CosineBasis` | $H^1([0,T]; \mathbb{R}^d)$ | Cosine basis orthonormal w.r.t. the $H^1$ inner product |
| `L2TensorBasis2D` | $L^2([0,T]^2; \mathbb{R}^d)$ | Tensor product of 1D bases |
| `SO3Basis` / `SO3FourierBasis` | $L^2(\text{SO}(3))$ | Wigner D-matrix evaluations |
| `SymmetricMatrixBasis` | $\text{Sym}(n)$ with Frobenius inner product | Scaled vech embedding (optional PCA) |
| `GraphLaplacianBasis` | $\mathbb{R}^{\|V\|}$ with Laplacian geometry | Graph Fourier transform |
| `CanonicalBasis` | $\mathbb{R}^n$ | Identity / truncation |
| `DiscreteCosineBasis` | $\mathbb{R}^n$ | DCT-II transform |

### Gaussian Mixture Model (`src/mixture.py`)

`GaussianMixtureModel` is a `torch.nn.Module` with:
- Mixture weights via softmax-parameterized logits (simplex constraint)
- Means as learnable coefficient vectors
- Covariances parameterized as diagonal exp-variances or Cholesky factors (positive definiteness)
- Methods: `compute_mmd2()`, `sample()`, `sample_functions()`, `log_likelihood()`, `responsibilities()`





## Configuration Reference

### `fit_gaussian_mixture_mmd()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `Tensor` | — | Projected data coefficients, shape `(n, M)` |
| `num_components` | `int` | — | Number of mixture components $K$ |
| `kernel` | `Kernel` | — | Kernel for MMD computation |
| `num_epochs` | `int` | `100` | Number of Adam optimization steps |
| `lr` | `float` | `0.01` | Learning rate |
| `covariance_type` | `str` | `"diagonal"` | `"diagonal"`, `"spherical"`, or `"full"` |
| `init_method` | `str` | `"kmeans++"` | `"random"` or `"kmeans++"` |
| `basis` | `HilbertBasis` | `None` | Basis for function reconstruction from samples |
| `verbose` | `bool` | `True` | Print progress |

### `GaussianKernel`

| Parameter | Type | Description |
|-----------|------|-------------|
| `sigma` | `float` | Bandwidth parameter $\sigma > 0$ |

### `PolynomialKernel`

| Parameter | Type | Description |
|-----------|------|-------------|
| `degree` | `int` | Polynomial degree $p \geq 1$ |
| `c` | `float` | Constant term $c \geq 0$ |

---

## Benchmarks

The `benchmarks/` directory compares the MMD Gaussian mixture against classical clustering baselines (k-means, hierarchical, fuzzy c-means, funHDDC, kernel k-groups, projected GMM-EM, ...). Entry points:

| Script | What it benchmarks |
|--------|--------------------|
| [`benchmarks/bench_l2_synthetic.py`](benchmarks/bench_l2_synthetic.py) | Synthetic $L^2([0,1]; \mathbb{R}^d)$ datasets |
| [`benchmarks/bench_l2_realdata.py`](benchmarks/bench_l2_realdata.py) | Real scalar- and vector-valued functional datasets (including Canadian Weather) |
| [`benchmarks/bench_l2_glucodensity.py`](benchmarks/bench_l2_glucodensity.py) | CGM glucodensity curves |
| [`benchmarks/bench_l2_skeleton.py`](benchmarks/bench_l2_skeleton.py) | NTU RGB+D skeleton sequences |
| [`benchmarks/bench_so3.py`](benchmarks/bench_so3.py) | SO(3) rotation data |
| [`benchmarks/bench_graph.py`](benchmarks/bench_graph.py) | Graph signals on Erdős–Rényi graphs |
| [`benchmarks/bench_rd_sklearn.py`](benchmarks/bench_rd_sklearn.py) | $\mathbb{R}^d$ sklearn-style toy datasets |
| [`benchmarks/ablation_l2.py`](benchmarks/ablation_l2.py) | Ablations: dimension $M$, sample size $n$, components $K$, bandwidth |

Run individual benchmarks directly, or use [`benchmarks/runner.py`](benchmarks/runner.py) and [`benchmarks/table_generator.py`](benchmarks/table_generator.py) to aggregate results into the paper tables.


## Reproducibility

All experiments from the paper can be reproduced by running the corresponding scripts in `examples/`. The default settings match those in the paper:

- Gaussian radial kernel, $\sigma = 2.0$
- Adam optimizer, learning rate $0.1$
- k-means++ initialization
- Diagonal covariances
- Random seed `42`

All runs are CPU-only and complete in minutes on a standard laptop.

| Experiment | $n$ | $K$ | $M$ | Final MMD² | Script |
|-----------|-----|-----|-----|------------|--------|
| $L^2([0,1]; \mathbb{R}^2)$ | 200 | 3 | 30 | ~$10^{-5}$ | `train_l2.py` |
| $L^2([0,1]; \mathbb{R}^2)$ with cov | 500 | 5 | 30 | ~$10^{-3}$ | `train_l2_gaussian.py` |
| $L^2([0,1]^2; \mathbb{R})$ | 400 | 4 | 64 | ~$10^{-3}$ | `train_l2_2d_gaussian.py` |
| $L^2(\text{SO}(3))$ | 200 | 3 | 84 | ~$10^{-2}$ | `train_so3.py` |
| Graph signals | 150 | 3 | 15 | ~$10^{-4}$ | `train_graph.py` |


## Citation

```bibtex
@misc{mixture2026,
  title         = {Gaussian Mixture Models in Hilbert Spaces via Kernel Methods},
  author        = {L{\'o}pez-Montero, Daniel and {\'A}lvarez-L{\'o}pez, Antonio and Matabuena, Marcos},
  year          = {2026},
  month         = may,
  eprint        = {2605.05996},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  doi           = {10.48550/arXiv.2605.05996},
  url           = {https://arxiv.org/abs/2605.05996}
}
```


## License

See the repository for license details.
