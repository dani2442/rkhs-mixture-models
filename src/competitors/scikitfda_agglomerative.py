"""
Scikit-FDA agglomerative clustering wrapper.
"""

from __future__ import annotations

import numpy as np
import torch
from skfda.ml.clustering import AgglomerativeClustering as FDAAgglomerative

from .scikitfda_utils import prepare_fda_input


class ScikitFDAAgglomerative:
    """Scikit-FDA agglomerative clustering for functional data."""

    def __init__(self, n_clusters: int, linkage: str = "average", **kwargs):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.kwargs = kwargs
        self.labels_ = None
        self.model = FDAAgglomerative(n_clusters=n_clusters, linkage=linkage, **kwargs)

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        fd = prepare_fda_input(X, basis=basis)
        self.labels_ = self.model.fit_predict(fd)
        return self.labels_
