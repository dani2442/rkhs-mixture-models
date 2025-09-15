# Continuous Temporal Learning of Probability Distributions via Neural ODEs with Applications in Digital Health and Clinical Trials

Code implementation for NeurIPS 2025 submission.

## Overview

We present a novel framework for modeling time-evolving probability distributions by combining:

1. **MMD-based GMM Fitting**: A robust method to fit Gaussian Mixture Models using Maximum Mean Discrepancy (MMD) as the objective function, with global parameters (means, covariances) shared across time and local time-varying weights.

2. **Neural ODE Dynamics**: Continuous modeling of weight evolution through Neural Ordinary Differential Equations, enabling smooth interpolation and extrapolation of probability distributions over time.

## Key Implementation Details

### Architectural Choices

Our implementation uses **Neural ODE dynamics with normalization** by projecting the output onto the probability simplex at each time step. This approach provided the best performance in our experiments. We explored several alternatives that are not included in this simplified version:

MMD Fitting:
- **Parameter updates**: Weighted moment matching for updating global parameters (gradient descent outperformed this approach)

Neural ODE:
- **Alternative dynamics**: Time-dependent, piecewise-constant parameters in the Neural ODE with different MLP architectures
- **Normalization schemes**: Softmax normalization, built-in simplex constraints, and other projection methods
- **Loss functions**: L1 loss instead of MSE for Neural ODE training
- **Regularization**: Both L1 and L2 regularization terms in the neural ODE with different weights

### Hyperparameter Tuning

We performed extensive grid search to tune hyperparameters while maintaining reasonable runtime:
- **MMD fitting**: Learning rates, gradient steps, ridge regularization, convergence tolerances
- **Neural ODE**: Hidden dimensions, number of layers, learning rates, regularization weights
- **Integration**: Step sizes, numerical methods (Euler, RK4, RK45)

### L2 Error Computation

For synthetic experiments, we implemented four methods to compute L2 errors between predicted and ground truth distributions:

1. **Standard Monte Carlo**: Uses global integration bounds
2. **Adaptive Monte Carlo**: Time-specific integration bounds for reduced computational cost
3. **Importance Sampling**: Samples from a mixture of model and ground truth (used in paper)
4. **Analytical**: Closed-form computation for Gaussian mixtures

For the paper results, we used **importance sampling** as it provides the best trade-off between accuracy and computational efficiency, especially in higher dimensions.

## Installation

```bash

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
├── main.py                   # Main entry point
├── input.py                  # Parameter configuration
├── Experiment.py             # Experiment orchestration
├── data.py                   # Synthetic data generation
├── load.py                   # Medical data loading with derivatives
│
├── Core/
│   ├── train_mmd.py          # MMD-GMM fitting algorithm
│   ├── node.py               # Neural ODE model architecture
│   ├── train_node.py         # Neural ODE training loop
│   └── test_mmd.py           # L2 error evaluation (4 methods)
│
├── Execution Pipeline/
│   ├── Run/
│   │   ├── run_f.py          # MMD-GMM fitting execution
│   │   ├── run_n.py          # Neural ODE training execution
│   │   ├── run_t.py          # Error evaluation execution
│   │   └── run_b.py          # Confidence bands (for bagging)
│   │
│   └── Utils/
│       ├── MMD_utils.py      # MMD computation and optimization
│       ├── NODE_utils.py     # Simplex projection, initialization
│       ├── IO_utils.py       # Smart caching and folder management
│       └── test_utils.py     # Integration bounds, density computation
│
├── Visualization/
│   ├── plots.py              # Static plots and visualizations
│   ├── gifs.py               # Animated evolution GIFs
│   └── log.py                # Progress logging
│
├── Analysis/
│   ├── Outputs.ipynb              # Generate all paper figures
│   └── Medical Data Analysis.py   # Medical data specific plots
│
└── Results_TimeSeries/ and  Results_MedicalData/    # Auto-generated output directories
```

## Usage

### Basic Execution

```bash
# Run synthetic experiments (default: 100 seeds, 11 time steps)
python main.py

# For medical data: modify input.py
# Set options["load_data"] = True
python main.py
```

### Configuration

All parameters are configured in `input.py`:

```python
# Key hyperparameters (tuned via grid search)
params_optim = {
    "K": [10],                 # GMM components
    "max_iter": 20,            # MMD-GMM iterations
    "lr": [0.01],              # Learning rate
    "grad_steps": [10],        # Gradient steps per iteration
    "lambda_ridge": [[0.1]],   # Ridge regularization
}

params_node = {
    "hidden_dim": 200,         # Neural network width
    "N_layers": 2,             # Network depth
    "max_epochs": 2000,        # Training epochs
    "method": "rk4",           # ODE solver (rk4 performed best)
    "lambda_node": 0.001,      # L2 regularization
}
```

### Output Structure

Results are organized hierarchically based on experiment type and parameters:

#### Synthetic Data (`Results_TimeSeries/`)

```
Results_TimeSeries/
├── run1/                          # Each run has different fixed parameters
│   ├── fixed_params.json          # All parameters for this run
│   ├── data.csv                   # Generated synthetic data summary
│   └── {varying_params}/          # Folders for each parameter combination
│       ├── result_fit.json        # MMD-GMM fitting results
│       ├── bag_1/                 # Plots from fitting (if d=1)
│       │   └── fit_ds{i}.png     # Fit for each time step
│       └── node_1/                # Neural ODE results
│           ├── result_node.json   # Weight trajectories
│           ├── result_error.json  # Error metrics
│           ├── error_comparison.json  # Comparison of L2 methods
│           ├── trajectories.png   # Weight evolution plot
│           ├── L2_errors*.png     # Error plots (4 methods)
│           ├── gmm_evolution.gif  # Animated evolution (if gif=true)
│           └── model_vs_ground_truth_1d.png  # Comparison plots
```

Example with multiple seeds and n_points:
- `run1/n_points=100/` - Fixed seed, varying n_points
- `run2/n_points=20_seed=5/` - Both n_points and seed vary

#### Medical Data (`Results_MedicalData/`)

```
Results_MedicalData/
├── run1/
│   ├── fixed_params.json
│   └── Id_{patient_ids}_blocksize_{size}/  # When no parameters vary
│       └── ids={patient_id}/               # One folder per patient
│           ├── result_fit.json
│           ├── bag_1/                      # Empty for medical data
│           └── node_1/
│               ├── result_node.json
│               ├── result_error.json
│               ├── L2_MMD_empirical.png   # Empirical error plots
│               ├── trajectories.png
│               └── sum_weights.png
```

#### Key Output Files

1. **result_fit.json**: Contains
   - `avg_weights`: Fitted weights at each time step (shape: [t_steps, K])
   - `avg_means`: Global GMM means (shape: [K, d])
   - `avg_covariances`: Global GMM covariances (shape: [K, d, d])
   - `convergence_iter`: Iterations until convergence
   - Individual bag results (if n_bags > 1)

2. **result_node.json**: Contains
   - `traj_weights`: Continuous trajectories (shape: [T/dt+1, K])
   - All training hyperparameters
   - Training time

3. **result_error.json**: Contains
   - For synthetic: L2 errors using 4 methods (standard, adaptive, importance, analytical)
   - For medical: Empirical L2 and MMD errors
   - Comparison metrics between methods

4. **error_comparison.json**: Table comparing L2 computation methods (synthetic only)

## Reproducing Paper Results

### 1. Generate Synthetic Experiments
```bash
# Ensure options["load_data"] = False in input.py
python main.py
```

### 2. Generate Medical Data Results
```bash
# Set options["load_data"] = True in input.py
# Configure patient IDs in params_data["ids"]
python main.py
```

### 3. Generate All Figures
```bash
# Run the Jupyter notebook
jupyter notebook Outputs.ipynb

```

The notebook generates:
- L2 error curves comparing methods
- Weight trajectory plots for medical data
- GMM heatmaps showing distribution evolution
- Confidence bands from bagging experiments

### 4. Medical Data Analysis
```python
# For treatment/control group analysis
python "Medical Data Analysis.py"
```

## Data Format

### Synthetic Data
- Generated using `data.py` with evolving GMM parameters
- Time evolution defined in `get_distrparams()`:
  - t ≤ 0.5: Means evolve linearly
  - t > 0.5: Means converge to same value
  - Variance increases linearly with time

### Medical Data (CSV Format)
- One CSV file per patient: `{patient_id}.csv`
- Each row = time point, each column = measurement
- Preprocessing:
  - Block aggregation (default: 10 rows/block)
  - Optional derivative computation (1st and 2nd order)
  - Robust handling of missing values

## Key Implementation Features

1. **Smart Caching**: Automatically reuses results when parameters match
2. **Flexible Output**: Adapts folder structure to show only varying parameters
3. **Robust Integration**: Multiple L2 computation methods for different scenarios
4. **Uncertainty Quantification**: Bootstrap aggregation for confidence bands
5. **Memory Efficiency**: Adjoint method for Neural ODE backpropagation
6. **Early Stopping**: Monitors validation loss with patience=50 epochs

## Citation

```bibtex
@inproceedings{anonymous2025continuous,
  title={Continuous Temporal Learning of Probability Distributions via Neural ODEs with Applications in Digital Health and Clinical Trials},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.