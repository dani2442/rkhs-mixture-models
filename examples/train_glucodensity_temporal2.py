#!/usr/bin/env python
"""
Run and plot the best glucodensity temporal model (no hyperparameter search).

This script is intended to be the execution/plotting companion to
`train_glucodensity_temporal.py`, which is now the search/tuning entry point.

Best config defaults (from search):
  - model_type: ode
  - n_components: 3
  - r_s: 8
  - n_time_bins: 16
  - r_pi: 6
  - ode_hidden: 64
  - lr: 0.005
  - sigma: 2.9707
  - sigma_mult: 1.0
  - block_size/window_size: 4
  - epochs: 90
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_glucodensity_temporal import (
    CONTROL_IDS,
    TREATMENT_IDS,
    build_training_representation,
    compute_all_patient_posteriors,
    compute_group_divergence_stats,
    compute_sliding_windows,
    load_and_preprocess_cgm,
    print_group_divergence_summary,
    run_experiment,
)
from src import GaussianKernel
from src.spaces import L2CosineBasis
from src.temporal_mixture import BasisLogitsTimeWeights, NeuralODETimeWeights
from src.visualization import plot_cluster_probabilities_by_group


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run best glucodensity temporal model and generate plots."
    )

    # Data / preprocessing
    p.add_argument(
        "--data-path",
        type=str,
        default="data/glucodensities/cgm_all_patients.csv",
        help="Path to CGM CSV relative to project root.",
    )
    p.add_argument("--max-prop-missing", type=float, default=0.20)
    p.add_argument("--block-size", type=int, default=4, help="Min days for patient inclusion")
    p.add_argument("--window-size", type=int, default=4, help="Sliding window size (days)")
    p.add_argument("--window-stride", type=int, default=1, help="Sliding window stride (days)")

    # Fixed best-config defaults (can be overridden if needed)
    p.add_argument("--model-type", choices=["basis", "ode"], default="ode")
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--r-s", type=int, default=8, help="Cosine basis functions for intraday")
    p.add_argument("--n-time-bins", type=int, default=16, help="Number of temporal bins")
    p.add_argument(
        "--space-metric",
        choices=["l2", "h1"],
        default="h1",
        help="Geometry for intraday curves before MMD fitting.",
    )
    p.add_argument("--r-pi", type=int, default=6, help="Basis functions for pi(t) when model-type=basis")
    p.add_argument("--ode-hidden", type=int, default=64, help="NeuralODE hidden dim when model-type=ode")

    # Optimization
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=1., help="Base sigma (0=auto median heuristic)")
    p.add_argument("--sigma-mult", type=float, default=1.0, help="Multiplier on base sigma")

    # Analysis / output
    p.add_argument("--analysis-time-bins", type=int, default=12)
    p.add_argument(
        "--out-dir",
        type=str,
        default="paper/images",
        help="Output directory relative to project root.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-show", action="store_true")

    return p.parse_args()


def plot_best_model_summary(
    history: list[float],
    pi_t_np: np.ndarray,
    recon_means_np: np.ndarray,
    t_grid_days_np: np.ndarray,
    out_dir: str,
    model_name: str,
    space_label: str,
    filename_suffix: str,
    show: bool,
) -> str:
    """Single-model summary: training curve, pi(t), and mean glucose curves."""
    os.makedirs(out_dir, exist_ok=True)

    K = pi_t_np.shape[1]
    use_weeks = float(np.max(t_grid_days_np)) > 30.0
    if use_weeks:
        t_axis = t_grid_days_np / 7.0
        t_label = "Treatment week"
    else:
        t_axis = t_grid_days_np
        t_label = "Treatment day"

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # (1) MMD curve
    axes[0].plot(history, lw=2, color="tab:blue")
    axes[0].set_title(f"{model_name}: MMD² training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MMD²(avg_t)")
    axes[0].grid(alpha=0.3)

    # (2) pi(t)
    for k in range(K):
        axes[1].plot(t_axis, pi_t_np[:, k], lw=2, label=f"Cluster {k + 1}")
    axes[1].set_title(f"{model_name}: cluster weights over time")
    axes[1].set_xlabel(t_label)
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # (3) Mean intraday curves
    slot_x = np.linspace(0.0, 24.0, recon_means_np.shape[1])
    for k in range(K):
        axes[2].plot(slot_x, recon_means_np[k], lw=1.8, label=f"Cluster {k + 1}")
    axes[2].set_title(f"{model_name}: mean intraday glucodensity")
    axes[2].set_xlabel("Hour of day")
    axes[2].set_ylabel("Glucose")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Best temporal glucodensity model ({space_label})", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(
        out_dir,
        f"glucodensity_best_model_summary{filename_suffix}.pdf",
    )
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    dtype = torch.float64

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, args.data_path)
    out_dir = os.path.join(project_root, args.out_dir)

    print("=" * 72)
    print("Glucodensity Temporal Mixture: Best-Model Execution")
    print("=" * 72)
    print(
        f"Config: model={args.model_type}, K={args.n_components}, r_s={args.r_s}, "
        f"L_t={args.n_time_bins}, metric={args.space_metric}, "
        f"r_pi={args.r_pi}, ode_hidden={args.ode_hidden}, "
        f"lr={args.lr}, sigma={args.sigma}, sigma_mult={args.sigma_mult}"
    )

    # 1) Data
    print("\n[Step 1] Loading and preprocessing data...")
    patient_data = load_and_preprocess_cgm(
        csv_path=csv_path,
        max_prop_missing=args.max_prop_missing,
        block_size=args.block_size,
        verbose=True,
    )

    # 2) Sliding windows
    print("\n[Step 2] Building sliding-window curves...")
    curves, t_indices, patient_ids, window_days, _ = compute_sliding_windows(
        patient_data=patient_data,
        window_size=args.window_size,
        stride=args.window_stride,
        verbose=True,
    )

    # 3) Representation
    print("\n[Step 3] Building temporal representation...")
    rep = build_training_representation(
        curves=curves,
        t_indices=t_indices,
        n_time_bins=args.n_time_bins,
        r_s=args.r_s,
        device=device,
        dtype=dtype,
        space_metric=args.space_metric,
        verbose=True,
    )

    X_time = rep["X_time"]
    t_grid = rep["t_grid"]
    mask = rep["mask"]

    # 4) Kernel and time-weight model
    sigma_base = args.sigma if args.sigma > 0 else rep["sigma_auto"]
    sigma = max(1e-8, sigma_base * args.sigma_mult)
    print(f"  Sigma base: {sigma_base:.4f} | final sigma: {sigma:.4f}")
    kernel = GaussianKernel(sigma=sigma)

    if args.model_type == "basis":
        time_basis = L2CosineBasis(
            T=1.0,
            R=args.r_pi,
            grid_size=rep["L_t"],
            d=1,
            device=device,
            dtype=dtype,
        )
        time_weight_model = BasisLogitsTimeWeights(
            basis_matrix=time_basis.Phi,
            num_components=args.n_components,
            device=device,
            dtype=dtype,
        )
        model_name = "Best-Basis"
    else:
        time_weight_model = NeuralODETimeWeights(
            t_grid=t_grid,
            num_components=args.n_components,
            hidden_dim=args.ode_hidden,
            device=device,
            dtype=dtype,
        )
        model_name = "Best-NeuralODE"

    # 5) Train
    print(f"\n[Step 4] Training {model_name}...")
    model, history, pi_t = run_experiment(
        name=model_name,
        time_weight_model=time_weight_model,
        X_time=X_time,
        kernel=kernel,
        n_components=args.n_components,
        coeff_dim=rep["coeff_dim"],
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
        mask=mask,
        verbose=True,
    )

    # 6) Patient posteriors + divergence analysis
    print("\n[Step 5] Computing patient posteriors and group divergence...")
    patient_posteriors, patient_time_norm, patient_time_days = compute_all_patient_posteriors(
        curves=curves,
        t_indices=t_indices,
        patient_ids=patient_ids,
        window_days=window_days,
        model=model,
        space_basis=rep["space_basis"],
        coeff_mean=rep["coeff_mean"],
        coeff_std=rep["coeff_std"],
        t_grid=t_grid,
        t_min_days=rep["t_min_days"],
        t_max_days=rep["t_max_days"],
        device=device,
        dtype=dtype,
    )

    div_stats = compute_group_divergence_stats(
        patient_posteriors=patient_posteriors,
        patient_time_days=patient_time_days,
        control_ids=CONTROL_IDS,
        treatment_ids=TREATMENT_IDS,
        n_time_bins=args.analysis_time_bins,
    )
    print_group_divergence_summary(model_name, div_stats)

    # 7) Summary plot for the selected model
    with torch.no_grad():
        means = model.mean.cpu()
    means_orig = means * rep["coeff_std"].cpu() + rep["coeff_mean"].cpu()
    recon_means = rep["space_basis"].reconstruct(means_orig)
    recon_means_np = recon_means.detach().squeeze(-1).numpy()

    t_np = t_grid.detach().cpu().numpy()
    day_range = rep["t_max_days"] - rep["t_min_days"]
    t_grid_days_np = t_np * day_range + rep["t_min_days"]

    plot_best_model_summary(
        history=history,
        pi_t_np=pi_t.cpu().numpy(),
        recon_means_np=recon_means_np,
        t_grid_days_np=t_grid_days_np,
        out_dir=out_dir,
        model_name=model_name,
        space_label="L²" if args.space_metric == "l2" else "H¹",
        filename_suffix="" if args.space_metric == "l2" else "_h1",
        show=not args.no_show,
    )

    # 8) Required clinical plot: cluster membership probability over time
    plot_cluster_probabilities_by_group(
        patient_posteriors=patient_posteriors,
        patient_time_days=patient_time_days,
        control_ids=CONTROL_IDS,
        treatment_ids=TREATMENT_IDS,
        n_time_bins=args.analysis_time_bins,
        include_difference_panel=False,
        out_dir=out_dir,
        filename=(
            "glucodensity_cluster_probs_by_group.pdf"
            if args.space_metric == "l2"
            else "glucodensity_cluster_probs_by_group_h1.pdf"
        ),
        show=not args.no_show,
    )

    n_ctrl = sum(1 for pid in patient_posteriors if pid in CONTROL_IDS)
    n_treat = sum(1 for pid in patient_posteriors if pid in TREATMENT_IDS)
    print("\n" + "=" * 72)
    print("Done")
    print("=" * 72)
    print(f"  Model: {model_name}")
    print(f"  Final MMD²(avg_t): {history[-1]:.6f}")
    print(
        f"  Divergence score: {div_stats['increasing_score']:.6f} | "
        f"final_TV={div_stats['final_sep']:.6f} | delta_TV={div_stats['delta_sep']:.6f}"
    )
    print(f"  Patients with posteriors: {len(patient_posteriors)} (control: {n_ctrl}, treatment: {n_treat})")


if __name__ == "__main__":
    main()
