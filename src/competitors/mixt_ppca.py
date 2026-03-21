"""
Mixture of probabilistic PCA clusterers on finite-dimensional curve features.

Citation:
    Tipping, M. E., and Bishop, C. M. (1999). Mixtures of Probabilistic
    Principal Component Analysers. Neural Computation, 11(2), 443-482.
"""

from __future__ import annotations

import numpy as np
import torch

from ._low_rank_mixture import LowRankGaussianMixture
from ._utils import flatten_features


class MixturePPCAClustering:
    """Low-rank Gaussian mixture with a fixed latent dimension per cluster."""

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
        self.subspace_dimension = subspace_dimension
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
