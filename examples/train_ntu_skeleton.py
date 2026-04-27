"""
NTU RGB+D skeleton trajectory helpers used by paper notebooks and benchmarks.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np


DEFAULT_NTU_ROOT = Path("data/ntu_rgbd_skeleton")


def ensure_ntu_skeleton_data(root: str | Path = DEFAULT_NTU_ROOT) -> list[str]:
    """Return the list of `.skeleton` files under *root*, downloading on cache miss.

    The download step delegates to :func:`src.data.download_ntu_skeleton`, which
    skips the network call when the zip archives are already present and skips
    extraction when the unpacked folders already exist. Safe to call repeatedly.
    """
    root = Path(root)
    files = sorted(glob.glob(str(root / "**/*.skeleton"), recursive=True))
    if files:
        return files

    from src.data import download_ntu_skeleton

    download_ntu_skeleton(root=str(root))
    return sorted(glob.glob(str(root / "**/*.skeleton"), recursive=True))


def skeleton_to_trajectory(frames: list[np.ndarray]) -> np.ndarray:
    """First-body trajectory: flatten each frame's 25 joints x 3 coords into 75 dims."""
    traj = []
    for frame in frames:
        body = frame[0]
        traj.append(body.reshape(-1))
    return np.stack(traj, axis=0)


def resample_trajectory(traj: np.ndarray, target_len: int) -> np.ndarray:
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
    return traj.reshape(traj.shape[0], 25, 3)
