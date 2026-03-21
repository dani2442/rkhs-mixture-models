"""
Scikit-FDA fuzzy c-means wrapper.
"""

from __future__ import annotations

import numpy as np
import torch
from skfda.ml.clustering import FuzzyCMeans as FDAFuzzyCMeans

from .scikitfda_utils import prepare_fda_input


class ScikitFDAFuzzyCMeans:
    """Scikit-FDA fuzzy C-means clustering for functional data."""

    def __init__(self, n_clusters: int, **kwargs):
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.labels_ = None
        self.model = FDAFuzzyCMeans(n_clusters=n_clusters, **kwargs)

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        fd = prepare_fda_input(X, basis=basis)
        self.model.fit(fd)
        memberships = self.model.membership_degree_
        self.labels_ = np.argmax(memberships, axis=1)
        return self.labels_
