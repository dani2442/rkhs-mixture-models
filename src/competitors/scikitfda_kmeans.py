"""
Scikit-FDA K-means wrapper.
"""

from __future__ import annotations

import numpy as np
import torch
from skfda.ml.clustering import KMeans as FDAKMeans

from .scikitfda_utils import prepare_fda_input


class ScikitFDAKMeans:
    """Scikit-FDA KMeans clustering for functional data."""

    def __init__(self, n_clusters: int, **kwargs):
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.labels_ = None
        self.model = FDAKMeans(n_clusters=n_clusters, **kwargs)

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        fd = prepare_fda_input(X, basis=basis)
        self.labels_ = self.model.fit_predict(fd)
        return self.labels_
