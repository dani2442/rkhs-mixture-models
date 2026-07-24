"""
GPmix wrapper: model-based clustering of functional data via Gaussian processes.

GPmix smooths each curve, projects it onto a set of univariate directions, fits
an independent one-dimensional Gaussian mixture per projection, and combines the
resulting base partitions by consensus (spectral clustering of a co-association
matrix). It therefore consumes raw curves on a grid rather than the projected
coefficients used by the other Hilbert-space methods.

Citation:
    Akeweje, E., and Zhang, M. (2024). GPmix: Model-based Clustering of
    Functional Data via Gaussian Processes. ICML 2024, PMLR 235:720-740.
"""

from __future__ import annotations

import numpy as np
import torch


class GPmixClustering:
    """Consensus clustering of functional data via an ensemble of univariate GMMs.

    Unlike the other competitors, this one is fit on the *raw* curves: pass the
    ``(n_samples, grid_size)`` array as ``X_raw``, since GPmix performs its own
    smoothing and projection.
    """

    def __init__(
        self,
        n_clusters: int,
        basis_type: str = "fpc",
        n_proj: int = 8,
        smoother_basis: str = "bspline",
        init_method: str = "kmeans",
        n_init: int = 10,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.basis_type = basis_type
        self.n_proj = n_proj
        self.smoother_basis = smoother_basis
        self.init_method = init_method
        self.n_init = n_init
        self.random_state = random_state
        self.labels_ = None

    def fit_predict(self, X_raw, basis=None) -> np.ndarray:
        from GPmix import Projector, Smoother, UniGaussianMixtureEnsemble

        if isinstance(X_raw, torch.Tensor):
            X_raw = X_raw.detach().cpu().numpy()
        X_raw = np.asarray(X_raw, dtype=float)
        if X_raw.ndim == 3 and X_raw.shape[-1] == 1:
            X_raw = X_raw[..., 0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        fd = Smoother(basis=self.smoother_basis).fit(X_raw)
        coeffs = Projector(basis_type=self.basis_type, n_proj=self.n_proj).fit(fd)

        ensemble = UniGaussianMixtureEnsemble(
            n_clusters=self.n_clusters,
            init_method=self.init_method,
            n_init=self.n_init,
        )
        ensemble.fit_gmms(coeffs)
        self.labels_ = np.asarray(ensemble.get_clustering())
        return self.labels_
