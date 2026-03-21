"""
Gaussian mixture clustering on finite-dimensional curve features.

Citation:
    Fraley, C., and Raftery, A. E. (2002). Model-Based Clustering,
    Discriminant Analysis, and Density Estimation. Journal of the American
    Statistical Association, 97(458), 611-631.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from ._utils import flatten_features


class FeatureGaussianMixtureClustering:
    """EM-fitted Gaussian mixture on feature vectors."""

    def __init__(
        self,
        n_clusters: int,
        covariance_type: str = "full",
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
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            reg_covar=reg_covar,
        )

    def fit(self, X: torch.Tensor):
        X_np = flatten_features(X)
        self.model.fit(X_np)
        self.responsibilities_ = self.model.predict_proba(X_np)
        self.labels_ = self.responsibilities_.argmax(axis=1)
        return self

    def predict(self, X: torch.Tensor) -> np.ndarray:
        return self.model.predict(flatten_features(X))

    def fit_predict(self, X: torch.Tensor) -> np.ndarray:
        return self.fit(X).labels_
