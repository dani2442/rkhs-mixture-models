#!/usr/bin/env python
"""
Train a Gaussian Mixture Model on NTU RGB+D skeleton sequences.

This script demonstrates:
1. Loading & resampling skeleton sequences to a fixed grid
2. Centering the data (subtracting the mean trajectory)
3. Projecting onto an L² cosine basis
4. Training a Gaussian mixture using MMD minimization
5. Visualizing: training history, weights, t-SNE, and most-similar
   skeleton trajectories per cluster

Usage:
    .venv/bin/python examples/train_ntu_skeleton.py
"""
from __future__ import annotations

import glob
import random
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.visualize_ntu_skeleton import (
    parse_skeleton_file,
    get_action_label,
    plot_skeleton_frame,
    BONES,
    BONE_GROUPS,
    GROUP_COLORS,
)
from src import (
    L2CosineBasis,
    GaussianKernel,
    GaussianMixtureModel,
    plot_training_history,
    plot_mixture_weights,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def skeleton_to_trajectory(frames: list[np.ndarray]) -> np.ndarray:
    """
    Convert parsed skeleton frames to a trajectory array.

    Takes the first body from each frame and flattens the 25×3 joints
    into a 75-dim vector per frame.

    Returns:
        trajectory: shape (num_frames, 75)
    """
    traj = []
    for frame in frames:
        body = frame[0]  # first person, shape (25, 3)
        traj.append(body.reshape(-1))  # (75,)
    return np.stack(traj, axis=0)  # (num_frames, 75)


def resample_trajectory(traj: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a variable-length trajectory to a fixed number of frames
    via linear interpolation.

    Args:
        traj: shape (T_orig, D)
        target_len: desired number of frames

    Returns:
        resampled: shape (target_len, D)
    """
    T_orig, D = traj.shape
    if T_orig == target_len:
        return traj

    t_orig = np.linspace(0, 1, T_orig)
    t_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, D))
    for d in range(D):
        resampled[:, d] = np.interp(t_new, t_orig, traj[:, d])
    return resampled


def trajectory_to_joints(traj: np.ndarray) -> np.ndarray:
    """Reshape a (grid_size, 75) trajectory back to (grid_size, 25, 3)."""
    return traj.reshape(traj.shape[0], 25, 3)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ==========================================
    # Configuration
    # ==========================================
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Data parameters
    n_samples = 300           # number of skeleton files to load
    grid_size = 64            # resample all sequences to this length
    d = 75                    # 25 joints × 3 coords

    # Basis parameters
    T = 1.0
    R = 10                    # basis functions per spatial dim

    # GMM parameters
    n_components = 5

    # Kernel parameters
    sigma_kernel = 2.0

    # Training parameters
    num_epochs = 300
    lr = 0.05

    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 60)
    print("NTU RGB+D Skeleton: MMD-based Gaussian Mixture Fitting")
    print("=" * 60)

    # ==========================================
    # Load skeleton data
    # ==========================================
    print("\n[1] Loading skeleton files...")

    DATA_ROOT = Path("data/ntu_rgbd_skeleton")
    skeleton_files = sorted(glob.glob(str(DATA_ROOT / "**/*.skeleton"), recursive=True))

    if not skeleton_files:
        print(f"No .skeleton files found under {DATA_ROOT}.")
        print("Run  examples/download_ntu_skeleton.py  first.")
        raise SystemExit(1)

    print(f"   Found {len(skeleton_files)} skeleton files total.")

    # Random subsample
    chosen_files = random.sample(skeleton_files, min(n_samples, len(skeleton_files)))
    print(f"   Using {len(chosen_files)} samples.")

    # ==========================================
    # Parse and resample
    # ==========================================
    print("\n[2] Parsing and resampling to fixed grid...")

    trajectories = []
    action_labels = []
    valid_files = []

    for fpath in chosen_files:
        try:
            frames = parse_skeleton_file(fpath)
            if len(frames) < 2:
                continue
            traj = skeleton_to_trajectory(frames)
            traj = resample_trajectory(traj, grid_size)
            trajectories.append(traj)
            action_labels.append(get_action_label(fpath))
            valid_files.append(fpath)
        except Exception as e:
            continue  # skip corrupt files

    X_raw_np = np.stack(trajectories, axis=0)  # (n, grid_size, 75)
    n_actual = X_raw_np.shape[0]
    print(f"   Successfully loaded {n_actual} trajectories, shape {X_raw_np.shape}")

    # ==========================================
    # Center the data
    # ==========================================
    print("\n[3] Centering data (subtracting mean trajectory)...")

    mean_traj = X_raw_np.mean(axis=0)  # (grid_size, 75)
    X_centered_np = X_raw_np - mean_traj[None, :, :]

    print(f"   Mean trajectory norm: {np.linalg.norm(mean_traj):.2f}")
    print(f"   Centered data norm (avg): {np.linalg.norm(X_centered_np, axis=(1,2)).mean():.2f}")

    X_raw = torch.tensor(X_centered_np, device=device, dtype=dtype)  # (n, grid_size, d)

    # ==========================================
    # Project onto L² cosine basis
    # ==========================================
    print("\n[4] Projecting onto L² cosine basis...")

    basis = L2CosineBasis(T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype)
    X = basis.project(X_raw)  # (n, M)  where M = R * d
    M = X.shape[1]

    # Check reconstruction quality
    X_recon = basis.reconstruct(X)
    recon_error = (X_raw - X_recon).norm() / X_raw.norm()

    print(f"   Basis: L² cosine, R={R}, d={d} → M={M}")
    print(f"   Reconstruction relative error: {recon_error:.4f}")

    # ==========================================
    # Create kernel and model
    # ==========================================
    print("\n[5] Setting up model...")

    kernel = GaussianKernel(sigma=sigma_kernel)

    model = GaussianMixtureModel(
        num_components=n_components,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )

    model.initialize_from_data(X, method="kmeans++")

    print(f"   Kernel: Gaussian (σ={sigma_kernel})")
    print(f"   Model: {n_components} components, diagonal covariance, M={M}")
    print(f"   Initial weights: {model.pi.detach().numpy()}")

    # ==========================================
    # Train the model
    # ==========================================
    print("\n[6] Training model by minimizing MMD²...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    gram = kernel.compute_gram_matrix(X)
    const_term = gram.mean()

    history = []
    print_interval = num_epochs // 10

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        mmd2, stats = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()

        full_mmd2 = const_term + mmd2.detach()
        history.append(full_mmd2.item())

        if (epoch + 1) % print_interval == 0:
            print(
                f"   Epoch {epoch+1:4d}/{num_epochs}: "
                f"MMD² = {full_mmd2.item():.6f}, "
                f"π = {model.pi.detach().numpy()}"
            )

    print(f"\n   Final MMD² = {history[-1]:.6f}")
    print(f"   Final weights: {model.pi.detach().numpy()}")

    # ==========================================
    # Assign skeletons to components
    # ==========================================
    print("\n[7] Assigning skeletons to components...")

    with torch.no_grad():
        gamma = model.responsibilities(X)  # (n, K)
        assignments = gamma.argmax(dim=1)   # (n,)

    for k in range(n_components):
        mask = assignments == k
        count = mask.sum().item()
        if count > 0:
            # Collect action labels for this cluster
            labels_k = [action_labels[i] for i in range(n_actual) if assignments[i] == k]
            label_counts = {}
            for lb in labels_k:
                label_counts[lb] = label_counts.get(lb, 0) + 1
            top_actions = sorted(label_counts.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{a}({c})" for a, c in top_actions)
            print(f"   Component {k+1}: {count:4d} skeletons — top actions: {top_str}")

    # ==========================================
    # Visualize results
    # ==========================================
    print("\n[8] Creating visualizations...")

    # ── Figure 1: Training overview (2×2) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Training history
    plot_training_history(history, ax=axes[0, 0], title="MMD² Training History")

    # (b) Mixture weights
    plot_mixture_weights(model.pi, ax=axes[0, 1], title="Learned Mixture Weights")

    # (c) Action label distribution per component
    ax = axes[1, 0]
    comp_colors = plt.cm.Set2(np.linspace(0, 1, n_components))
    bottom = np.zeros(n_components)
    all_actions = sorted(set(action_labels))
    for action in all_actions:
        counts = []
        for k in range(n_components):
            mask = assignments == k
            cnt = sum(1 for i in range(n_actual) if mask[i] and action_labels[i] == action)
            counts.append(cnt)
        counts = np.array(counts)
        if counts.sum() > 0:
            ax.bar(range(n_components), counts, bottom=bottom, label=action, alpha=0.8)
            bottom += counts
    ax.set_xlabel("Component")
    ax.set_ylabel("Count")
    ax.set_title("Action Distribution by Component")
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f"k={k+1}" for k in range(n_components)])
    ax.grid(True, alpha=0.3, axis="y")

    # (d) t-SNE of coefficients
    ax = axes[1, 1]
    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_actual - 1))
        X_2d = tsne.fit_transform(X.detach().numpy())
        for k in range(n_components):
            mask = (assignments == k).numpy()
            if mask.any():
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=[comp_colors[k]], s=12, alpha=0.6,
                    label=f"Component {k+1}",
                )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE of L² Coefficients")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    except ImportError:
        ax.text(
            0.5, 0.5, "Install scikit-learn\nfor t-SNE visualization",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )

    fig.suptitle("NTU Skeleton Mixture Model", fontsize=15)
    plt.tight_layout()

    # ── Figure 2: Representative skeletons per component ──
    print("\n   Creating representative skeleton plots...")

    n_representative = 3
    n_snapshots = 8  # frames to overlay per skeleton

    # Distinct component colors
    COMPONENT_COLORS = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    ]

    fig_rep = plt.figure(
        figsize=(5 * n_representative, 4.5 * n_components),
        facecolor="#0f0f23",
    )
    fig_rep.suptitle(
        "Most Representative Skeleton Sequences per Component",
        color="#e0e0e0", fontsize=16, fontweight="bold", y=0.995,
    )

    with torch.no_grad():
        means = model.mean  # (K, M)

    for k in range(n_components):
        comp_color = COMPONENT_COLORS[k % len(COMPONENT_COLORS)]

        # Indices belonging to component k
        mask_k = (assignments == k).nonzero(as_tuple=True)[0]
        if len(mask_k) == 0:
            continue
        X_k = X[mask_k]

        # Distance to component mean in coefficient space
        dists = (X_k - means[k].unsqueeze(0)).norm(dim=1)
        n_rep = min(n_representative, len(mask_k))
        top_idx = dists.argsort()[:n_rep]
        representative_idx = mask_k[top_idx]

        for j, mol_idx in enumerate(representative_idx):
            idx = mol_idx.item()
            ax = fig_rep.add_subplot(
                n_components, n_representative,
                k * n_representative + j + 1,
                projection="3d", facecolor="#1a1a2e",
            )

            # Un-center the trajectory for display
            traj_centered = X_centered_np[idx]  # (grid_size, 75)
            traj_display = traj_centered + mean_traj  # add mean back
            joints_seq = trajectory_to_joints(traj_display)  # (grid_size, 25, 3)

            # Plot overlaid snapshots
            snap_indices = np.linspace(0, grid_size - 1, n_snapshots, dtype=int)
            for si, frame_idx in enumerate(snap_indices):
                t_frac = si / (len(snap_indices) - 1) if len(snap_indices) > 1 else 1.0
                alpha = 0.15 + 0.85 * t_frac
                lw = 1.0 + 2.0 * t_frac
                plot_skeleton_frame(
                    ax, joints_seq[frame_idx],
                    alpha=alpha, linewidth=lw,
                    joint_size=8 + 14 * t_frac,
                )

            # Styling
            resp_val = gamma[idx, k].item()
            label = action_labels[idx]
            ax.set_title(
                f"{label} · γ={resp_val:.2f}",
                color="#b0b0d0", fontsize=9, fontweight="bold", pad=8,
            )
            ax.tick_params(colors="#555", labelsize=6)
            ax.set_xlabel("X", color="#666", fontsize=7, labelpad=2)
            ax.set_ylabel("Z", color="#666", fontsize=7, labelpad=2)
            ax.set_zlabel("Y", color="#666", fontsize=7, labelpad=2)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("#222")
            ax.yaxis.pane.set_edgecolor("#222")
            ax.zaxis.pane.set_edgecolor("#222")
            ax.grid(True, alpha=0.1, color="#888")
            ax.view_init(elev=15, azim=-60)

            # Row label on first column
            if j == 0:
                ax.text2D(
                    -0.12, 0.5,
                    f"Component {k+1}\n(π={model.pi[k].item():.3f})",
                    transform=ax.transAxes,
                    fontsize=10, fontweight="bold",
                    ha="right", va="center", rotation=90,
                    color=comp_color,
                )

    plt.tight_layout(rect=[0.02, 0, 1, 0.97])

    # ==========================================
    # Save figures
    # ==========================================
    print("\n[9] Saving figures as PDF...")
    paper_images_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "paper", "images",
    )
    os.makedirs(paper_images_dir, exist_ok=True)

    fig.savefig(
        os.path.join(paper_images_dir, "ntu_skeleton_mixture.pdf"),
        format="pdf", bbox_inches="tight",
    )
    fig_rep.savefig(
        os.path.join(paper_images_dir, "ntu_skeleton_representatives.pdf"),
        format="pdf", bbox_inches="tight",
    )
    print(f"   Saved to {paper_images_dir}/ntu_skeleton_mixture.pdf")
    print(f"   Saved to {paper_images_dir}/ntu_skeleton_representatives.pdf")

    print("\n[10] Displaying plots...")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
