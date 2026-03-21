"""
Shared helpers for scikit-fda competitor wrappers.
"""

from __future__ import annotations

import numpy as np
import torch
import skfda

from ._utils import to_numpy


def prepare_fda_input(X: torch.Tensor, basis=None):
    """Convert coefficients or reconstructed curves to an FDataGrid."""
    X_np = to_numpy(X)

    if basis is not None and hasattr(basis, "reconstruct"):
        with torch.no_grad():
            functions = basis.reconstruct(X)
        func_np = to_numpy(functions)
        n_samples, grid_size, _ = func_np.shape
        grid_points = np.linspace(0.0, 1.0, grid_size)
        return skfda.FDataGrid(data_matrix=func_np, grid_points=grid_points)

    return skfda.FDataGrid(data_matrix=X_np)
