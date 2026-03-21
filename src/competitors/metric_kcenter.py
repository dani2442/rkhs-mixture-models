"""
Greedy K-center clustering for Euclidean features.
"""

from __future__ import annotations

import numpy as np
import torch

from ._utils import to_numpy


class KCenterClustering:
    """
    Classical farthest-first 2-approximation for K-center clustering.
    """

    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.centers_ = None

    def fit_predict(self, X: torch.Tensor) -> np.ndarray:
        X_np = to_numpy(X)
        n_samples = X_np.shape[0]

        if self.n_clusters >= n_samples:
            self.labels_ = np.arange(n_samples)
            self.centers_ = np.arange(n_samples)
            return self.labels_

        rng = np.random.RandomState(self.random_state)
        first_center = rng.randint(n_samples)
        centers = [first_center]

        from scipy.spatial.distance import cdist

        min_dists = cdist(X_np, X_np[centers]).min(axis=1)
        for _ in range(1, self.n_clusters):
            next_center = int(np.argmax(min_dists))
            centers.append(next_center)
            new_dists = cdist(X_np, X_np[[next_center]]).ravel()
            min_dists = np.minimum(min_dists, new_dists)

        self.centers_ = np.asarray(centers)
        self.labels_ = np.argmin(cdist(X_np, X_np[self.centers_]), axis=1)
        return self.labels_
