"""
Visualize NTU RGB+D skeleton sequences as 3D line plots.

Parses .skeleton files and plots a few sample frames showing the 25-joint
human skeleton with bone connections in 3D.

Usage:
    .venv/bin/python examples/visualize_ntu_skeleton.py
"""
from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection
import matplotlib.cm as cm

# ─── NTU RGB+D joint names (0-indexed) ────────────────────────────────────────
JOINT_NAMES = [
    "Base of spine",       # 0
    "Middle of spine",     # 1
    "Neck",                # 2
    "Head",                # 3
    "Left shoulder",       # 4
    "Left elbow",          # 5
    "Left wrist",          # 6
    "Left hand",           # 7
    "Right shoulder",      # 8
    "Right elbow",         # 9
    "Right wrist",         # 10
    "Right hand",          # 11
    "Left hip",            # 12
    "Left knee",           # 13
    "Left ankle",          # 14
    "Left foot",           # 15
    "Right hip",           # 16
    "Right knee",          # 17
    "Right ankle",         # 18
    "Right foot",          # 19
    "Spine (shoulder)",    # 20
    "Tip of left hand",    # 21
    "Left thumb",          # 22
    "Tip of right hand",   # 23
    "Right thumb",         # 24
]

# Bone connections (0-indexed).  Each tuple is (parent, child).
BONES = [
    # Torso
    (0, 1), (1, 20), (20, 2), (2, 3),
    # Left arm
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    # Right arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    # Left leg
    (0, 12), (12, 13), (13, 14), (14, 15),
    # Right leg
    (0, 16), (16, 17), (17, 18), (18, 19),
]

# Body-part coloring for bones
BONE_GROUPS = {
    "torso":     [0, 1, 2, 3],
    "left_arm":  [4, 5, 6, 7, 8, 9],
    "right_arm": [10, 11, 12, 13, 14, 15],
    "left_leg":  [16, 17, 18, 19],
    "right_leg": [20, 21, 22, 23],
}

GROUP_COLORS = {
    "torso":     "#4FC3F7",
    "left_arm":  "#FFB74D",
    "right_arm": "#AED581",
    "left_leg":  "#CE93D8",
    "right_leg": "#F48FB1",
}


# ─── Parsing ──────────────────────────────────────────────────────────────────
def parse_skeleton_file(path: str | Path) -> list[np.ndarray]:
    """
    Parse a .skeleton file and return a list of frames.

    Each frame is an ndarray of shape (num_bodies, 25, 3) containing the
    (x, y, z) world coordinates of each joint.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    idx = 0
    num_frames = int(lines[idx].strip()); idx += 1

    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        num_bodies = int(lines[idx].strip()); idx += 1
        bodies = []
        for _ in range(num_bodies):
            # Body info line (bodyID, clipedEdges, ...)
            idx += 1
            num_joints = int(lines[idx].strip()); idx += 1
            joints = np.zeros((num_joints, 3))
            for j in range(num_joints):
                parts = lines[idx].strip().split()
                joints[j, 0] = float(parts[0])  # x
                joints[j, 1] = float(parts[1])  # y
                joints[j, 2] = float(parts[2])  # z
                idx += 1
            bodies.append(joints)
        if bodies:
            frames.append(np.stack(bodies))
        else:
            frames.append(np.zeros((1, 25, 3)))
    return frames


def get_action_label(filename: str) -> str:
    """Extract action class number from filename like S001C001P001R001A042."""
    name = Path(filename).stem
    # Action class is after the last 'A' and is 3 digits
    a_idx = name.rfind("A")
    if a_idx >= 0:
        return f"Action {name[a_idx+1:a_idx+4]}"
    return "Unknown"


# ─── Plotting ─────────────────────────────────────────────────────────────────
def bone_color(bone_idx: int) -> str:
    for group, indices in BONE_GROUPS.items():
        if bone_idx in indices:
            return GROUP_COLORS[group]
    return "#FFFFFF"


def plot_skeleton_frame(
    ax: plt.Axes,
    joints: np.ndarray,
    alpha: float = 1.0,
    linewidth: float = 2.5,
    joint_size: float = 18,
) -> None:
    """
    Plot a single skeleton on a 3D axis.

    Parameters
    ----------
    joints : ndarray of shape (25, 3)
    """
    # Draw bones
    for bi, (p, c) in enumerate(BONES):
        color = bone_color(bi)
        ax.plot(
            [joints[p, 0], joints[c, 0]],
            [joints[p, 2], joints[c, 2]],  # swap y/z for nicer view
            [joints[p, 1], joints[c, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )
    # Draw joints
    ax.scatter(
        joints[:, 0],
        joints[:, 2],
        joints[:, 1],
        c="#FFFFFF",
        edgecolors="#333333",
        s=joint_size,
        alpha=alpha,
        zorder=5,
        linewidths=0.5,
    )


def plot_skeleton_sequence(
    frames: list[np.ndarray],
    title: str = "",
    num_snapshots: int = 8,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot several evenly-spaced snapshots of a skeleton sequence overlaid
    in a single 3D plot with a time-based color fade.
    """
    fig = plt.figure(figsize=figsize, facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")

    total = len(frames)
    indices = np.linspace(0, total - 1, num_snapshots, dtype=int)

    for i, frame_idx in enumerate(indices):
        alpha = 0.2 + 0.8 * (i / (len(indices) - 1)) if len(indices) > 1 else 1.0
        lw = 1.0 + 2.0 * (i / (len(indices) - 1)) if len(indices) > 1 else 2.5
        body = frames[frame_idx][0]  # first person
        plot_skeleton_frame(ax, body, alpha=alpha, linewidth=lw, joint_size=10 + 12 * alpha)

    # Style the axes
    ax.set_xlabel("X", color="#888", fontsize=9, labelpad=5)
    ax.set_ylabel("Z", color="#888", fontsize=9, labelpad=5)
    ax.set_zlabel("Y", color="#888", fontsize=9, labelpad=5)
    ax.tick_params(colors="#555", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333")
    ax.yaxis.pane.set_edgecolor("#333")
    ax.zaxis.pane.set_edgecolor("#333")
    ax.grid(True, alpha=0.15, color="#888")

    if title:
        ax.set_title(title, color="#e0e0e0", fontsize=12, fontweight="bold", pad=15)

    ax.view_init(elev=15, azim=-60)

    return fig


def plot_multi_sample_grid(
    skeleton_files: list[str | Path],
    num_samples: int = 6,
    num_snapshots: int = 8,
) -> plt.Figure:
    """
    Plot a grid of skeleton sequences, each showing time-lapsed snapshots.
    """
    num_samples = min(num_samples, len(skeleton_files))
    chosen = random.sample(skeleton_files, num_samples)

    cols = 3
    rows = (num_samples + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows), facecolor="#0f0f23")
    fig.suptitle(
        "NTU RGB+D  —  Skeleton Sequence Samples",
        color="#e0e0e0",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    for i, skel_path in enumerate(chosen):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d", facecolor="#1a1a2e")
        try:
            frames = parse_skeleton_file(skel_path)
            label = get_action_label(str(skel_path))
            fname = Path(skel_path).stem

            total = len(frames)
            indices = np.linspace(0, total - 1, num_snapshots, dtype=int)

            for j, frame_idx in enumerate(indices):
                t = j / (len(indices) - 1) if len(indices) > 1 else 1.0
                alpha = 0.15 + 0.85 * t
                lw = 1.0 + 2.0 * t
                body = frames[frame_idx][0]
                plot_skeleton_frame(ax, body, alpha=alpha, linewidth=lw, joint_size=8 + 14 * t)

            ax.set_title(f"{label}\n{fname}", color="#b0b0d0", fontsize=9, fontweight="bold", pad=8)
        except Exception as e:
            ax.set_title(f"Error: {e}", color="#ff6666", fontsize=8)

        # Style
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

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_ROOT = Path("data/ntu_rgbd_skeleton")

    # Collect all .skeleton files
    skeleton_files = sorted(glob.glob(str(DATA_ROOT / "**/*.skeleton"), recursive=True))

    if not skeleton_files:
        print(f"No .skeleton files found under {DATA_ROOT}.")
        print("Run  examples/download_ntu_skeleton.py  first to download the data.")
        raise SystemExit(1)

    print(f"Found {len(skeleton_files)} skeleton files.")

    # ── Grid of samples ───────────────────────────────────────────────────
    random.seed(42)
    fig_grid = plot_multi_sample_grid(skeleton_files, num_samples=6, num_snapshots=10)
    out_grid = "examples/ntu_skeleton_samples.png"
    fig_grid.savefig(out_grid, dpi=180, bbox_inches="tight")
    print(f"Saved grid → {out_grid}")

    # ── Single detailed sample ────────────────────────────────────────────
    sample_path = skeleton_files[0]
    frames = parse_skeleton_file(sample_path)
    label = get_action_label(sample_path)
    fig_single = plot_skeleton_sequence(
        frames, title=f"{label}  ({Path(sample_path).stem})", num_snapshots=12
    )
    out_single = "examples/ntu_skeleton_single.png"
    fig_single.savefig(out_single, dpi=180, bbox_inches="tight")
    print(f"Saved single → {out_single}")

    plt.show()
    print("Done.")
