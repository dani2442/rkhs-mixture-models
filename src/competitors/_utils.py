"""
Shared utilities for competitor clustering baselines.
"""

from __future__ import annotations

import numpy as np
import torch


def to_numpy(X) -> np.ndarray:
    """Convert torch tensors or array-like inputs to NumPy arrays."""
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def flatten_features(X) -> np.ndarray:
    """Flatten all but the sample axis into a 2D feature matrix."""
    X_np = to_numpy(X)
    return X_np.reshape(X_np.shape[0], -1)


def reconstruct_functional_data(X, basis=None) -> np.ndarray:
    """
    Recover sampled trajectories from coefficient vectors when possible.

    If ``basis`` exposes ``reconstruct``, we use it. Otherwise the input is
    interpreted as already sampled on a grid.
    """
    if basis is not None and hasattr(basis, "reconstruct"):
        X_tensor = X if torch.is_tensor(X) else torch.as_tensor(X)
        with torch.no_grad():
            curves = basis.reconstruct(X_tensor)
        curves_np = to_numpy(curves)
    else:
        curves_np = to_numpy(X)

    if curves_np.ndim == 2:
        curves_np = curves_np[:, :, None]
    elif curves_np.ndim > 3:
        curves_np = curves_np.reshape(curves_np.shape[0], curves_np.shape[1], -1)
    return curves_np


def functional_features(X, basis=None, metric: str = "l2") -> np.ndarray:
    """
    Build Euclidean feature vectors corresponding to functional metrics.

    ``metric="l2"`` uses sampled curves.
    ``metric="derivative"`` uses first-order finite differences.
    """
    curves = reconstruct_functional_data(X, basis=basis)
    spacing = 1.0
    if basis is not None and hasattr(basis, "dt"):
        spacing = float(to_numpy(basis.dt))
    elif curves.shape[1] > 1:
        spacing = 1.0 / float(curves.shape[1] - 1)

    if metric == "l2":
        feats = curves * np.sqrt(spacing)
    elif metric == "derivative":
        deriv = np.gradient(curves, spacing, axis=1, edge_order=1)
        feats = deriv * np.sqrt(spacing)
    else:
        raise ValueError(f"Unknown functional metric: {metric}")
    return feats.reshape(feats.shape[0], -1)


def weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute a weighted mean over the first axis."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    total = max(weights.sum(), 1e-12)
    return (weights[:, None] * X).sum(axis=0) / total


def weighted_covariance(
    X: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray | None = None,
    reg_covar: float = 1e-6,
) -> np.ndarray:
    """Compute a regularized weighted covariance matrix."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if mean is None:
        mean = weighted_mean(X, weights)
    centered = X - mean
    total = max(weights.sum(), 1e-12)
    cov = (centered * weights[:, None]).T @ centered / total
    cov.flat[:: cov.shape[0] + 1] += reg_covar
    return cov


def select_subspace_dimension(
    eigenvalues: np.ndarray,
    fixed_dimension: int | None = None,
    variance_explained: float = 0.9,
    max_dimension: int | None = None,
) -> int:
    """Choose an intrinsic dimension from eigenvalues."""
    dim = eigenvalues.shape[0]
    if dim == 0:
        return 0

    if fixed_dimension is not None:
        return int(np.clip(fixed_dimension, 0, dim))

    if max_dimension is None:
        max_dimension = dim
    max_dimension = int(np.clip(max_dimension, 1, dim))

    positive = np.clip(eigenvalues[:max_dimension], a_min=0.0, a_max=None)
    total = positive.sum()
    if total <= 0:
        return 1

    cumulative = np.cumsum(positive) / total
    chosen = int(np.searchsorted(cumulative, variance_explained) + 1)
    return int(np.clip(chosen, 1, max_dimension))


def build_parsimonious_covariance(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    dimension: int,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Build a covariance with rich variation in a low-dimensional subspace and
    isotropic variation in the orthogonal complement.
    """
    full_dim = eigenvalues.shape[0]
    dimension = int(np.clip(dimension, 0, full_dim))
    if dimension < full_dim:
        tail = eigenvalues[dimension:]
        noise = float(max(tail.mean(), reg_covar))
    else:
        noise = float(reg_covar)

    signal = np.clip(eigenvalues[:dimension], a_min=noise + reg_covar, a_max=None)
    basis = eigenvectors[:, :dimension]

    cov = noise * np.eye(full_dim)
    if dimension > 0:
        cov += basis @ np.diag(signal - noise) @ basis.T
    cov.flat[:: full_dim + 1] += reg_covar
    return cov, signal, noise, basis


def stable_log_gaussian_density(
    X: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    reg_covar: float = 1e-6,
) -> np.ndarray:
    """Evaluate a multivariate Gaussian log-density in a numerically stable way."""
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    cov.flat[:: cov.shape[0] + 1] += reg_covar

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + reg_covar * np.eye(cov.shape[0])
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")

    diff = X - mean
    precision_diff = np.linalg.solve(cov, diff.T).T
    quad = np.einsum("ij,ij->i", diff, precision_diff)
    dim = cov.shape[0]
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad)


def haar_wavelet_features(X, basis=None, max_levels: int | None = None) -> np.ndarray:
    """
    Compute a lightweight Haar-wavelet representation without extra dependencies.
    """
    curves = reconstruct_functional_data(X, basis=basis)
    outputs = []

    for sample in curves:
        pieces = []
        for dim in range(sample.shape[1]):
            values = np.asarray(sample[:, dim], dtype=float)
            current = values.copy()
            details = []
            level = 0

            while current.shape[0] > 1 and (max_levels is None or level < max_levels):
                if current.shape[0] % 2 == 1:
                    current = np.pad(current, (0, 1), mode="edge")
                approx = (current[0::2] + current[1::2]) / np.sqrt(2.0)
                detail = (current[0::2] - current[1::2]) / np.sqrt(2.0)
                details.append(detail)
                current = approx
                level += 1

            pieces.append(np.concatenate([current] + details[::-1]))
        outputs.append(np.concatenate(pieces))

    return np.vstack(outputs)
