"""
Cluster-specific FPCA mixture in a finite-dimensional coefficient space.

Citation:
    Jacques, J., and Preda, C. (2013). Funclust: A Curves Clustering Method
    Using Functional Random Variable Density Approximation. Neurocomputing,
    112, 164-171. doi:10.1016/j.neucom.2012.11.042
"""

from __future__ import annotations

import numpy as np
import torch

from ._low_rank_mixture import LowRankGaussianMixture
from ._utils import flatten_features


class FunclustClustering:
    """
    Finite-dimensional Funclust analogue using cluster-specific PCA subspaces on
    basis coefficients or other precomputed features.
    """

    def __init__(
        self,
        n_clusters: int,
        subspace_dimension: int = 3,
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
