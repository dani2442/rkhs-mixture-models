"""
High-dimensional data clustering on finite-dimensional curve features.

Citation:
    Bouveyron, C., Girard, S., and Schmid, C. (2007). High-Dimensional Data
    Clustering. Computational Statistics and Data Analysis, 52(1), 502-519.
"""

from __future__ import annotations

import numpy as np
import torch

from ._low_rank_mixture import LowRankGaussianMixture
from ._utils import flatten_features


class FeatureHDDCClustering:
    """Parsimonious Gaussian mixture with component-specific signal subspaces."""

    def __init__(
        self,
        n_clusters: int,
        subspace_dimension: int | None = None,
        variance_explained: float = 0.9,
        max_subspace_dimension: int | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 3,
        random_state: int | None = None,
        reg_covar: float = 1e-6,
    ):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.responsibilities_ = None
        self.model = LowRankGaussianMixture(
            n_clusters=n_clusters,
            subspace_dimension=subspace_dimension,
            variance_explained=variance_explained,
            max_subspace_dimension=max_subspace_dimension,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            random_state=random_state,
            reg_covar=reg_covar,
        )

    def fit(self, X: torch.Tensor):
        X_np = flatten_features(X)
        self.model.fit(X_np)
        self.labels_ = self.model.labels_
        self.responsibilities_ = self.model.responsibilities_
        return self

    def predict(self, X: torch.Tensor) -> np.ndarray:
        return self.model.predict(flatten_features(X))

    def fit_predict(self, X: torch.Tensor) -> np.ndarray:
        return self.fit(X).labels_
