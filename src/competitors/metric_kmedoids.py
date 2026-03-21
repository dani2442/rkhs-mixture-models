"""
K-medoids clustering for metric spaces.
"""

from __future__ import annotations

import numpy as np
import torch
import kmedoids

from ._utils import to_numpy


class KMedoidsClustering:
    """
    K-Medoids using FasterPAM from the ``kmedoids`` package.
    """

    def __init__(self, n_clusters: int, method: str = "fasterpam", max_iter: int = 100):
        self.n_clusters = n_clusters
        self.method = method
        self.max_iter = max_iter
        self.labels_ = None
        self.medoids_ = None

    def fit_predict(self, X: torch.Tensor, dist_matrix: np.ndarray = None) -> np.ndarray:
        if dist_matrix is None:
            X_np = to_numpy(X)
            from scipy.spatial.distance import pdist, squareform

            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        if self.method == "fasterpam":
            result = kmedoids.fasterpam(dist_matrix, self.n_clusters, max_iter=self.max_iter)
        else:
            result = kmedoids.pam(dist_matrix, self.n_clusters, max_iter=self.max_iter)

        self.labels_ = result.labels
        self.medoids_ = result.medoids
        return self.labels_
