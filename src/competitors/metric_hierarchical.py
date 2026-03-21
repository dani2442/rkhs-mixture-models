"""
Agglomerative clustering for metric spaces.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from ._utils import to_numpy


class HierarchicalClustering:
    """Agglomerative clustering on a precomputed distance matrix."""

    def __init__(self, n_clusters: int, linkage: str = "average"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage,
        )

    def fit_predict(self, X: torch.Tensor, dist_matrix: np.ndarray = None) -> np.ndarray:
        if dist_matrix is None:
            X_np = to_numpy(X)
            from scipy.spatial.distance import pdist, squareform

            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        self.labels_ = self.model.fit_predict(dist_matrix)
        return self.labels_
