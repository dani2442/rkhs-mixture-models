#!/usr/bin/env python
"""
Example: L²([0,1]^2) mixture fitting with time-varying weights pi_k(t).

We use the same synthetic 2D Gaussian data as the static L²([0,1]^2) example,
but optimize a temporal mixture:
  Q_t = sum_k pi_k(t) N(m_k, K_k)

At each fixed t, data lives in L²([0,1]) (space variable s), and MMD is
computed in that L²([0,1]) coefficient space.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import GaussianKernel, generate_l2_2d_gaussian_data
from src.spaces import L2CosineBasis
from src.temporal_mixture import (
    BasisLogitsTimeWeights,
    NeuralODETimeWeights,
    TemporalGaussianMixtureModel,
    fit_temporal_gaussian_mixture_mmd,
    project_l2_2d_to_space_slices,
)


def plot_pi_curves(
    ax: plt.Axes,
    t_grid: torch.Tensor,
    pi_t: torch.Tensor,
    title: str,
    true_weights: torch.Tensor,
) -> None:
    t_np = t_grid.detach().cpu().numpy()
    pi_np = pi_t.detach().cpu().numpy()
    true_np = true_weights.detach().cpu().numpy()

    for k in range(pi_np.shape[1]):
        ax.plot(t_np, pi_np[:, k], lw=2.0, label=f"$\\pi_{k+1}(t)$")
        ax.hlines(
            y=float(true_np[k]),
            xmin=float(t_np[0]),
            xmax=float(t_np[-1]),
            linestyles="--",
            linewidth=1.0,
            color=ax.lines[-1].get_color(),
            alpha=0.65,
        )

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)


def run_experiment(
    name: str,
    time_weight_model: torch.nn.Module,
    X_time: torch.Tensor,
    kernel: GaussianKernel,
    n_components: int,
    coeff_dim: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
):
    model = TemporalGaussianMixtureModel(
        num_components=n_components,
        coeff_dim=coeff_dim,
        time_weight_model=time_weight_model,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )

    history = fit_temporal_gaussian_mixture_mmd(
        model=model,
        X_time=X_time,
        kernel=kernel,
        num_epochs=num_epochs,
        lr=lr,
        init_method="kmeans++",
        verbose=True,
        log_interval=max(1, num_epochs // 8),
    )

    with torch.no_grad():
        pi_t = model.time_weight_model()

    print(f"{name} final MMD²(avg_t): {history[-1]:.6f}")
    print(f"{name} mean pi over t: {pi_t.mean(dim=0).detach().cpu().numpy()}")

    return model, history, pi_t.detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal pi_k(t) fitting on L²([0,1]^2) data."
    )
    parser.add_argument("--n-samples", type=int, default=220)
    parser.add_argument("--grid-size-s", type=int, default=28)
    parser.add_argument("--grid-size-t", type=int, default=28)
    parser.add_argument("--r-s", type=int, default=8)
    parser.add_argument("--r-t", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--sigma", type=float, default=1.6)
    parser.add_argument("--r-pi", type=int, default=6)
    parser.add_argument("--ode-hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Data parameters (same synthetic setup as train_l2_2d_gaussian.py)
    n_samples = args.n_samples
    n_components = 4
    grid_size_s = args.grid_size_s
    grid_size_t = args.grid_size_t
    T = 1.0
    S = 1.0
    d = 1
    R_s = args.r_s
    R_t = args.r_t

    # Temporal weight parameterization settings
    R_pi = args.r_pi
    ode_hidden_dim = args.ode_hidden

    # Optimization
    num_epochs = args.epochs
    lr = args.lr

    # Kernel on L²([0,1]) coefficient space (space-only slices)
    sigma_kernel = args.sigma

    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 72)
    print("L²([0,1]^2) Temporal Mixture Weights: Basis vs Neural ODE")
    print("=" * 72)

    true_weights = torch.tensor([0.35, 0.30, 0.20, 0.15], device=device, dtype=dtype)

    X_raw, _, _, info = generate_l2_2d_gaussian_data(
        n_samples=n_samples,
        n_components=n_components,
        grid_size_s=grid_size_s,
        grid_size_t=grid_size_t,
        R_s=R_s,
        R_t=R_t,
        T=T,
        S=S,
        d=d,
        component_weights=true_weights,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    # Project each time slice X(., t) into L²([0,1]) basis over s
    space_basis = L2CosineBasis(
        T=S,
        R=R_s,
        grid_size=grid_size_s,
        d=d,
        device=device,
        dtype=dtype,
    )
    X_time = project_l2_2d_to_space_slices(X_raw, space_basis)
    coeff_dim = X_time.shape[-1]
    t_grid = info["basis"].t

    print(f"Data shape (n, L_s, L_t, d): {tuple(X_raw.shape)}")
    print(f"Time-slice coefficient shape (L_t, n, M_s): {tuple(X_time.shape)}")
    print(f"M_s = {coeff_dim}, K = {n_components}, L_t = {len(t_grid)}")

    kernel = GaussianKernel(sigma=sigma_kernel)

    # Approach 1: basis coefficients for pi_k(t)
    time_basis = L2CosineBasis(
        T=T,
        R=R_pi,
        grid_size=grid_size_t,
        d=1,
        device=device,
        dtype=dtype,
    )
    basis_weight_model = BasisLogitsTimeWeights(
        basis_matrix=time_basis.Phi,
        num_components=n_components,
        device=device,
        dtype=dtype,
    )

    print("\n[1] Training basis-coefficient temporal weights...")
    _, history_basis, pi_basis = run_experiment(
        name="Basis",
        time_weight_model=basis_weight_model,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        dtype=dtype,
    )

    # Approach 2: Neural ODE for pi(t)
    ode_weight_model = NeuralODETimeWeights(
        t_grid=t_grid,
        num_components=n_components,
        hidden_dim=ode_hidden_dim,
        device=device,
        dtype=dtype,
    )

    print("\n[2] Training Neural ODE temporal weights...")
    _, history_ode, pi_ode = run_experiment(
        name="NeuralODE",
        time_weight_model=ode_weight_model,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        dtype=dtype,
    )

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    axes[0].plot(history_basis, label="Basis", lw=2.0)
    axes[0].plot(history_ode, label="Neural ODE", lw=2.0)
    axes[0].set_title("Training: mean_t MMD²")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MMD²")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    plot_pi_curves(
        axes[1],
        t_grid=t_grid,
        pi_t=pi_basis,
        title="Basis logits: $\\pi_k(t)$",
        true_weights=true_weights,
    )

    plot_pi_curves(
        axes[2],
        t_grid=t_grid,
        pi_t=pi_ode,
        title="Neural ODE logits: $\\pi_k(t)$",
        true_weights=true_weights,
    )
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    paper_images_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "images"
    )
    os.makedirs(paper_images_dir, exist_ok=True)
    out_path = os.path.join(paper_images_dir, "l2_2d_temporal_pi_comparison.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")

    print(f"\nSaved figure: {out_path}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
