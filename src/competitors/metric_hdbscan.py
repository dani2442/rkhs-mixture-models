"""
HDBSCAN clustering for metric spaces.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import HDBSCAN

from ._utils import to_numpy


class HDBSCANClustering:
    """HDBSCAN on a precomputed distance matrix."""

    def __init__(self, min_cluster_size: int = 5):
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        self.model = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")

    def fit_predict(self, X: torch.Tensor, dist_matrix: np.ndarray = None) -> np.ndarray:
        if dist_matrix is None:
            X_np = to_numpy(X)
            from scipy.spatial.distance import pdist, squareform

            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        self.labels_ = self.model.fit_predict(dist_matrix)
        return self.labels_
