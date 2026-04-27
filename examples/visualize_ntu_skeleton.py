"""
Parsing and frame-level plotting helpers for NTU RGB+D skeleton sequences.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

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


def parse_skeleton_file(path: str | Path) -> list[np.ndarray]:
    """Parse a .skeleton file into a list of (num_bodies, 25, 3) frame arrays."""
    with open(path, "r") as f:
        lines = f.readlines()

    idx = 0
    num_frames = int(lines[idx].strip()); idx += 1

    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        num_bodies = int(lines[idx].strip()); idx += 1
        bodies = []
        for _ in range(num_bodies):
            idx += 1  # body info line
            num_joints = int(lines[idx].strip()); idx += 1
            joints = np.zeros((num_joints, 3))
            for j in range(num_joints):
                parts = lines[idx].strip().split()
                joints[j, 0] = float(parts[0])
                joints[j, 1] = float(parts[1])
                joints[j, 2] = float(parts[2])
                idx += 1
            bodies.append(joints)
        if bodies:
            frames.append(np.stack(bodies))
        else:
            frames.append(np.zeros((1, 25, 3)))
    return frames


def get_action_label(filename: str) -> str:
    name = Path(filename).stem
    a_idx = name.rfind("A")
    if a_idx >= 0:
        return f"Action {name[a_idx+1:a_idx+4]}"
    return "Unknown"


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
    for bi, (p, c) in enumerate(BONES):
        color = bone_color(bi)
        ax.plot(
            [joints[p, 0], joints[c, 0]],
            [joints[p, 2], joints[c, 2]],
            [joints[p, 1], joints[c, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )
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
