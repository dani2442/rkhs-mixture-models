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
import itertools
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


class DiscreteLogitsTimeWeights(torch.nn.Module):
    """Trainable per-time logits: pi(t_i) = softmax(z_i)."""

    def __init__(
        self,
        num_times: int,
        num_components: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.zeros(num_times, num_components, device=device, dtype=dtype)
        )

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)


def best_component_permutation(
    true_means: torch.Tensor, pred_means: torch.Tensor
) -> tuple[int, ...]:
    """Find permutation p minimizing sum_k ||pred[p[k]] - true[k]||²."""
    K = true_means.shape[0]
    best_perm = tuple(range(K))
    best_cost = float("inf")
    for perm in itertools.permutations(range(K)):
        perm_tensor = torch.tensor(perm, device=pred_means.device, dtype=torch.long)
        cost = torch.sum((pred_means[perm_tensor] - true_means) ** 2).item()
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return best_perm


def reorder_components(
    X: torch.Tensor, perm: tuple[int, ...], component_dim: int
) -> torch.Tensor:
    idx = torch.tensor(perm, device=X.device, dtype=torch.long)
    return X.index_select(component_dim, idx)


def plot_pi_true_vs_pred(
    ax: plt.Axes,
    t_grid: torch.Tensor,
    pi_true: torch.Tensor,
    pi_pred: torch.Tensor,
    title: str,
) -> None:
    t_np = t_grid.detach().cpu().numpy()
    true_np = pi_true.detach().cpu().numpy()
    pred_np = pi_pred.detach().cpu().numpy()

    for k in range(true_np.shape[1]):
        (line,) = ax.plot(
            t_np, true_np[:, k], linestyle="--", lw=1.2, alpha=0.9, label=f"true $\\pi_{k+1}$"
        )
        ax.plot(
            t_np,
            pred_np[:, k],
            linestyle="-",
            lw=2.0,
            color=line.get_color(),
            label=f"pred $\\pi_{k+1}$",
        )

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)


def plot_pi_fit_vs_target(
    ax: plt.Axes,
    t_grid: torch.Tensor,
    pi_target: torch.Tensor,
    pi_pred: torch.Tensor,
    title: str,
) -> None:
    t_np = t_grid.detach().cpu().numpy()
    tgt_np = pi_target.detach().cpu().numpy()
    pred_np = pi_pred.detach().cpu().numpy()

    for k in range(tgt_np.shape[1]):
        (line,) = ax.plot(
            t_np, tgt_np[:, k], linestyle="--", lw=1.4, alpha=0.9, label=f"target $\\pi_{k+1}$"
        )
        ax.plot(
            t_np,
            pred_np[:, k],
            linestyle="-",
            lw=2.0,
            color=line.get_color(),
            label=f"pred $\\pi_{k+1}$",
        )

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)


def plot_means_true_vs_pred(
    ax: plt.Axes,
    true_means: torch.Tensor,
    pred_means: torch.Tensor,
    title: str,
) -> None:
    true_np = true_means.detach().cpu().numpy()
    pred_np = pred_means.detach().cpu().numpy()
    coeff_idx = range(true_np.shape[1])

    for k in range(true_np.shape[0]):
        (line,) = ax.plot(
            coeff_idx, true_np[k], linestyle="--", lw=1.2, alpha=0.9, label=f"true $m_{k+1}$"
        )
        ax.plot(
            coeff_idx,
            pred_np[k],
            linestyle="-",
            lw=2.0,
            color=line.get_color(),
            label=f"pred $m_{k+1}$",
        )

    ax.set_title(title)
    ax.set_xlabel("coeff index")
    ax.set_ylabel("mean coeff")
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


def fit_time_weights_l2(
    name: str,
    time_weight_model: torch.nn.Module,
    target_pi_t: torch.Tensor,
    num_epochs: int,
    lr: float,
) -> tuple[list[float], torch.Tensor]:
    optimizer = torch.optim.Adam(time_weight_model.parameters(), lr=lr)
    history: list[float] = []
    log_interval = max(1, num_epochs // 8)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pi_pred = time_weight_model()
        loss = torch.mean((pi_pred - target_pi_t) ** 2)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch + 1:4d}/{num_epochs}: {name} L² = {loss.item():.8f}")

    with torch.no_grad():
        pi_fit = time_weight_model()
    print(f"{name} final L²: {history[-1]:.8f}")
    return history, pi_fit.detach()


def copy_component_params(
    dst_model: TemporalGaussianMixtureModel, src_model: TemporalGaussianMixtureModel
) -> None:
    with torch.no_grad():
        dst_model.components._mean_coeffs.copy_(src_model.components._mean_coeffs)
        if hasattr(dst_model.components, "_log_var") and hasattr(src_model.components, "_log_var"):
            dst_model.components._log_var.copy_(src_model.components._log_var)
        if hasattr(dst_model.components, "_chol_factor") and hasattr(
            src_model.components, "_chol_factor"
        ):
            dst_model.components._chol_factor.copy_(src_model.components._chol_factor)


def evaluate_mmd_with_fixed_components(
    source_model: TemporalGaussianMixtureModel,
    time_weight_model: torch.nn.Module,
    X_time: torch.Tensor,
    kernel: GaussianKernel,
    n_components: int,
    coeff_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    eval_model = TemporalGaussianMixtureModel(
        num_components=n_components,
        coeff_dim=coeff_dim,
        time_weight_model=time_weight_model,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )
    copy_component_params(eval_model, source_model)
    with torch.no_grad():
        mmd2, _ = eval_model.compute_mmd2(X_time, kernel, compute_const_term=True)
    return float(mmd2.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal pi_k(t) fitting on L²([0,1]^2) data."
    )
    parser.add_argument("--n-samples", type=int, default=220)
    parser.add_argument("--grid-size-s", type=int, default=28)
    parser.add_argument("--grid-size-t", type=int, default=28)
    parser.add_argument("--r-s", type=int, default=8)
    parser.add_argument("--r-t", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2-epochs", type=int, default=220)
    parser.add_argument("--l2-lr", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=1.6)
    parser.add_argument("--r-pi", type=int, default=6)
    parser.add_argument("--ode-hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Data parameters (same synthetic setup as train_l2_2d_gaussian.py)
    n_samples = args.n_samples
    n_components = 10
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
    print("L²([0,1]^2) Temporal Weights: Direct MMD vs Two-Stage L² Distillation")
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

    # Shared basis used by basis-parameterized temporal weights
    time_basis = L2CosineBasis(
        T=T,
        R=R_pi,
        grid_size=grid_size_t,
        d=1,
        device=device,
        dtype=dtype,
    )

    # ------------------------------------------------------------------
    # Previous baseline (already done): direct MMD with parametric pi(t)
    # ------------------------------------------------------------------
    basis_weight_model_direct = BasisLogitsTimeWeights(
        basis_matrix=time_basis.Phi,
        num_components=n_components,
        device=device,
        dtype=dtype,
    )

    print("\n[1] Direct MMD: Basis temporal weights...")
    model_basis_direct, history_basis_direct, pi_basis_direct = run_experiment(
        name="Direct-Basis",
        time_weight_model=basis_weight_model_direct,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        dtype=dtype,
    )

    ode_weight_model_direct = NeuralODETimeWeights(
        t_grid=t_grid,
        num_components=n_components,
        hidden_dim=ode_hidden_dim,
        device=device,
        dtype=dtype,
    )

    print("\n[2] Direct MMD: Neural ODE temporal weights...")
    model_ode_direct, history_ode_direct, pi_ode_direct = run_experiment(
        name="Direct-NeuralODE",
        time_weight_model=ode_weight_model_direct,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        dtype=dtype,
    )

    # ------------------------------------------------------------------
    # New approach: Stage 1 (free pi(t_i)) + Stage 2 (fit Basis/ODE in L²)
    # ------------------------------------------------------------------
    discrete_weight_model = DiscreteLogitsTimeWeights(
        num_times=len(t_grid),
        num_components=n_components,
        device=device,
        dtype=dtype,
    )
    print("\n[3] Two-stage (stage 1): free discrete pi(t_i) + Gaussian parameters...")
    stage1_model, history_stage1, pi_stage1 = run_experiment(
        name="TwoStage-Stage1-DiscretePi",
        time_weight_model=discrete_weight_model,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        dtype=dtype,
    )

    print("\n[4] Two-stage (stage 2): fit Basis and Neural ODE to stage-1 pi(t_i) with L²...")
    basis_weight_model_l2 = BasisLogitsTimeWeights(
        basis_matrix=time_basis.Phi,
        num_components=n_components,
        device=device,
        dtype=dtype,
    )
    history_l2_basis, pi_basis_l2 = fit_time_weights_l2(
        name="TwoStage-Basis",
        time_weight_model=basis_weight_model_l2,
        target_pi_t=pi_stage1,
        num_epochs=args.l2_epochs,
        lr=args.l2_lr,
    )

    ode_weight_model_l2 = NeuralODETimeWeights(
        t_grid=t_grid,
        num_components=n_components,
        hidden_dim=ode_hidden_dim,
        device=device,
        dtype=dtype,
    )
    history_l2_ode, pi_ode_l2 = fit_time_weights_l2(
        name="TwoStage-NeuralODE",
        time_weight_model=ode_weight_model_l2,
        target_pi_t=pi_stage1,
        num_epochs=args.l2_epochs,
        lr=args.l2_lr,
    )


    # Final comparisons
    direct_basis_mmd = history_basis_direct[-1]
    direct_ode_mmd = history_ode_direct[-1]
    stage1_mmd = history_stage1[-1]
    twostage_basis_mmd = evaluate_mmd_with_fixed_components(
        source_model=stage1_model,
        time_weight_model=basis_weight_model_l2,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        device=device,
        dtype=dtype,
    )
    twostage_ode_mmd = evaluate_mmd_with_fixed_components(
        source_model=stage1_model,
        time_weight_model=ode_weight_model_l2,
        X_time=X_time,
        kernel=kernel,
        n_components=n_components,
        coeff_dim=coeff_dim,
        device=device,
        dtype=dtype,
    )

    l2_basis_final = float(torch.mean((pi_basis_l2 - pi_stage1) ** 2).item())
    l2_ode_final = float(torch.mean((pi_ode_l2 - pi_stage1) ** 2).item())

    print("\n" + "=" * 72)
    print("Final comparison")
    print("=" * 72)
    print(f"Direct-Basis MMD²(avg_t):      {direct_basis_mmd:.6f}")
    print(f"Direct-NeuralODE MMD²(avg_t):  {direct_ode_mmd:.6f}")
    print(f"Stage1-DiscretePi MMD²(avg_t): {stage1_mmd:.6f}")
    print(f"TwoStage-Basis MMD²(avg_t):    {twostage_basis_mmd:.6f}")
    print(f"TwoStage-NeuralODE MMD²(avg_t):{twostage_ode_mmd:.6f}")
    print(f"TwoStage-Basis L²(pi):         {l2_basis_final:.8f}")
    print(f"TwoStage-NeuralODE L²(pi):     {l2_ode_final:.8f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.0))

    axes[0, 0].plot(history_basis_direct, label="Direct Basis", lw=2.0)
    axes[0, 0].plot(history_ode_direct, label="Direct NeuralODE", lw=2.0)
    axes[0, 0].set_title("Direct MMD Training")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("MMD²")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(history_stage1, color="black", lw=2.0, label="Stage1 Discrete pi(t_i)")
    axes[0, 1].set_title("Two-Stage: Stage 1 MMD Training")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("MMD²")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(history_l2_basis, lw=2.0, label="TwoStage Basis")
    axes[0, 2].plot(history_l2_ode, lw=2.0, label="TwoStage NeuralODE")
    axes[0, 2].set_title("Two-Stage: Stage 2 L²(pi) Fit")
    axes[0, 2].set_xlabel("epoch")
    axes[0, 2].set_ylabel("L²(pi)")
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].legend(fontsize=8)

    methods = [
        "Direct\nBasis",
        "Direct\nNeuralODE",
        "Stage1\nDiscrete",
        "TwoStage\nBasis",
        "TwoStage\nNeuralODE",
    ]
    final_mmd_values = [
        direct_basis_mmd,
        direct_ode_mmd,
        stage1_mmd,
        twostage_basis_mmd,
        twostage_ode_mmd,
    ]
    axes[1, 0].bar(methods, final_mmd_values, color=["C0", "C1", "black", "C0", "C1"], alpha=0.85)
    axes[1, 0].set_title("Final MMD²(avg_t): Direct vs Two-Stage")
    axes[1, 0].set_ylabel("MMD²")
    axes[1, 0].grid(alpha=0.3, axis="y")

    plot_pi_fit_vs_target(
        axes[1, 1],
        t_grid=t_grid,
        pi_target=pi_stage1,
        pi_pred=pi_basis_l2,
        title=f"TwoStage Basis vs Stage1 target (L²={l2_basis_final:.2e})",
    )
    plot_pi_fit_vs_target(
        axes[1, 2],
        t_grid=t_grid,
        pi_target=pi_stage1,
        pi_pred=pi_ode_l2,
        title=f"TwoStage NeuralODE vs Stage1 target (L²={l2_ode_final:.2e})",
    )
    axes[1, 2].legend(loc="upper right", fontsize=7)

    plt.tight_layout()

    paper_images_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "images"
    )
    os.makedirs(paper_images_dir, exist_ok=True)
    out_path = os.path.join(
        paper_images_dir, "l2_2d_temporal_pi_direct_vs_twostage_comparison.pdf"
    )
    fig.savefig(out_path, format="pdf", bbox_inches="tight")

    # Additional figure: true-vs-predicted pi(t) for 4 predictors + side mean comparisons
    true_means_raw = info["true_means"]
    true_means_time = project_l2_2d_to_space_slices(true_means_raw, space_basis)
    true_means_space = true_means_time.mean(dim=0)
    pi_true_t = true_weights.unsqueeze(0).expand(len(t_grid), -1)

    perm_basis_direct = best_component_permutation(true_means_space, model_basis_direct.mean.detach())
    perm_ode_direct = best_component_permutation(true_means_space, model_ode_direct.mean.detach())
    perm_stage1 = best_component_permutation(true_means_space, stage1_model.mean.detach())

    pi_basis_direct_aligned = reorder_components(
        pi_basis_direct, perm_basis_direct, component_dim=1
    )
    pi_ode_direct_aligned = reorder_components(
        pi_ode_direct, perm_ode_direct, component_dim=1
    )
    pi_basis_l2_aligned = reorder_components(pi_basis_l2, perm_stage1, component_dim=1)
    pi_ode_l2_aligned = reorder_components(pi_ode_l2, perm_stage1, component_dim=1)

    means_basis_direct_aligned = reorder_components(
        model_basis_direct.mean.detach(), perm_basis_direct, component_dim=0
    )
    means_stage1_aligned = reorder_components(
        stage1_model.mean.detach(), perm_stage1, component_dim=0
    )

    fig2 = plt.figure(figsize=(24, 8.5), constrained_layout=True)
    gs = fig2.add_gridspec(2, 4, width_ratios=[1.25, 1.0, 1.0, 1.25], wspace=0.25, hspace=0.28)

    ax_mean_left = fig2.add_subplot(gs[:, 0])
    ax_pi_1 = fig2.add_subplot(gs[0, 1])
    ax_pi_2 = fig2.add_subplot(gs[0, 2])
    ax_pi_3 = fig2.add_subplot(gs[1, 1])
    ax_pi_4 = fig2.add_subplot(gs[1, 2])
    ax_mean_right = fig2.add_subplot(gs[:, 3])

    plot_means_true_vs_pred(
        ax_mean_left,
        true_means=true_means_space,
        pred_means=means_basis_direct_aligned,
        title="Means: True vs Direct-Basis",
    )
    plot_pi_true_vs_pred(
        ax_pi_1,
        t_grid=t_grid,
        pi_true=pi_true_t,
        pi_pred=pi_basis_direct_aligned,
        title="Direct-Basis: true vs pred $\\pi(t)$",
    )
    plot_pi_true_vs_pred(
        ax_pi_2,
        t_grid=t_grid,
        pi_true=pi_true_t,
        pi_pred=pi_ode_direct_aligned,
        title="Direct-NeuralODE: true vs pred $\\pi(t)$",
    )
    plot_pi_true_vs_pred(
        ax_pi_3,
        t_grid=t_grid,
        pi_true=pi_true_t,
        pi_pred=pi_basis_l2_aligned,
        title="TwoStage-Basis: true vs pred $\\pi(t)$",
    )
    plot_pi_true_vs_pred(
        ax_pi_4,
        t_grid=t_grid,
        pi_true=pi_true_t,
        pi_pred=pi_ode_l2_aligned,
        title="TwoStage-NeuralODE: true vs pred $\\pi(t)$",
    )
    plot_means_true_vs_pred(
        ax_mean_right,
        true_means=true_means_space,
        pred_means=means_stage1_aligned,
        title="Means: True vs Two-Stage (Stage1)",
    )

    ax_pi_4.legend(loc="upper right", fontsize=7)
    ax_mean_right.legend(loc="upper right", fontsize=7)
    out_path_2 = os.path.join(
        paper_images_dir, "l2_2d_temporal_pi_true_vs_pred_with_means.pdf"
    )
    fig2.savefig(out_path_2, format="pdf", bbox_inches="tight")

    print(f"\nSaved figure: {out_path}")
    print(f"Saved figure: {out_path_2}")
    if args.no_show:
        plt.close(fig)
        plt.close(fig2)
    else:
        plt.show()


if __name__ == "__main__":
    main()
