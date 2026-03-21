"""
Two-stage K-means clustering on finite-dimensional curve features.

Citation:
    McQueen, J. B. (1967). Some Methods of Classification and Analysis of
    Multivariate Observations. Proceedings of the Fifth Berkeley Symposium on
    Mathematical Statistics and Probability, 281-297.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from ._utils import flatten_features


class FeatureKMeansClustering:
    """K-means on raw discretizations, basis coefficients, or FPCA scores."""

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: torch.Tensor):
        X_np = flatten_features(X)
        self.labels_ = self.model.fit_predict(X_np)
        self.cluster_centers_ = self.model.cluster_centers_
        return self

    def predict(self, X: torch.Tensor) -> np.ndarray:
        return self.model.predict(flatten_features(X))

    def fit_predict(self, X: torch.Tensor) -> np.ndarray:
        return self.fit(X).labels_
