"""
Parsimonious functional clustering with adaptive cluster-specific subspaces.

Citation:
    Bouveyron, C., and Jacques, J. (2011). Model-Based Clustering of Time
    Series in Group-Specific Functional Subspaces. Advances in Data Analysis
    and Classification, 5(4), 281-300. doi:10.1007/s11634-011-0095-6
"""

from __future__ import annotations

import numpy as np
import torch

from ._low_rank_mixture import LowRankGaussianMixture
from ._utils import flatten_features


class FunHDDCClustering:
    """
    Functional analogue of HDDC operating on coefficient vectors or other
    finite-dimensional function encodings.
    """

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

    def fit(self, X: torch.Tensor, basis=None):
        X_np = flatten_features(X)
        self.model.fit(X_np)
        self.labels_ = self.model.labels_
        self.responsibilities_ = self.model.responsibilities_
        return self

    def predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        return self.model.predict(flatten_features(X))

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        return self.fit(X, basis=basis).labels_
