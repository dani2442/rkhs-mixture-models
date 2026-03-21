"""
Functional K-means using either L2 distances on reconstructed curves or
first-derivative distances.

Citation:
    Jacques, J., and Preda, C. (2014). Functional Data Clustering: A Survey.
    Advances in Data Analysis and Classification, 8(3), 231-255.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from ._utils import functional_features


class FunctionalKMeansClustering:
    """
    K-means in the discretized geometry induced by either the L2 norm or a
    first-derivative norm.
    """

    def __init__(
        self,
        n_clusters: int,
        metric: str = "l2",
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
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

    def fit(self, X: torch.Tensor, basis=None):
        features = functional_features(X, basis=basis, metric=self.metric)
        self.labels_ = self.model.fit_predict(features)
        self.cluster_centers_ = self.model.cluster_centers_
        return self

    def predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        features = functional_features(X, basis=basis, metric=self.metric)
        return self.model.predict(features)

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        return self.fit(X, basis=basis).labels_
