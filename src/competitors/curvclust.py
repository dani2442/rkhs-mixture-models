"""
Wavelet-domain Gaussian mixture clustering for functional data.

Citation:
    Giacofci, M., Lambert-Lacroix, S., Marot, G., and Picard, F. (2013).
    Wavelet-Based Clustering for Mixed-Effects Functional Models in High
    Dimension. Biometrics, 69(1), 31-40.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from ._utils import haar_wavelet_features


class CurvclustClustering:
    """
    Curvclust-style baseline using Haar-wavelet coefficients followed by a
    Gaussian mixture in the transformed domain.
    """

    def __init__(
        self,
        n_clusters: int,
        covariance_type: str = "full",
        max_levels: int | None = None,
        n_init: int = 3,
        max_iter: int = 200,
        random_state: int | None = None,
        reg_covar: float = 1e-6,
    ):
        self.n_clusters = n_clusters
        self.max_levels = max_levels
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

    def fit(self, X: torch.Tensor, basis=None):
        features = haar_wavelet_features(X, basis=basis, max_levels=self.max_levels)
        self.model.fit(features)
        self.responsibilities_ = self.model.predict_proba(features)
        self.labels_ = self.responsibilities_.argmax(axis=1)
        return self

    def predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        features = haar_wavelet_features(X, basis=basis, max_levels=self.max_levels)
        return self.model.predict(features)

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        return self.fit(X, basis=basis).labels_
