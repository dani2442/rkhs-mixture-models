"""
Gaussian mixture on basis coefficients with a shared covariance structure.

Citation:
    James, G. M., and Sugar, C. A. (2003). Clustering for Sparsely Sampled
    Functional Data. Journal of the American Statistical Association, 98(462),
    397-408. doi:10.1198/016214503000189
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from ._utils import flatten_features


class FclustClustering:
    """
    Finite-dimensional analogue of fclust using basis coefficients and a tied
    covariance Gaussian mixture.
    """

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 3,
        max_iter: int = 200,
        random_state: int | None = None,
        reg_covar: float = 1e-6,
    ):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.responsibilities_ = None
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            reg_covar=reg_covar,
        )

    def fit(self, X: torch.Tensor, basis=None):
        X_np = flatten_features(X)
        self.model.fit(X_np)
        self.responsibilities_ = self.model.predict_proba(X_np)
        self.labels_ = self.responsibilities_.argmax(axis=1)
        return self

    def predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        return self.model.predict(flatten_features(X))

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        return self.fit(X, basis=basis).labels_
