"""
DBSCAN clustering for metric spaces.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from ._utils import to_numpy


class DBSCANClustering:
    """DBSCAN on a precomputed distance matrix."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")

    def fit_predict(self, X: torch.Tensor, dist_matrix: np.ndarray = None) -> np.ndarray:
        if dist_matrix is None:
            X_np = to_numpy(X)
            from scipy.spatial.distance import pdist, squareform

            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        self.labels_ = self.model.fit_predict(dist_matrix)
        return self.labels_
