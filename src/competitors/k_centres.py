"""
Functional k-centres clustering through alternating cluster-specific subspace
estimation and reconstruction-error assignment.

Citation:
    Chiou, J.-M., and Li, P.-L. (2007). Functional Clustering and Identifying
    Substructures of Longitudinal Data. Journal of the Royal Statistical
    Society: Series B, 69(4), 679-699.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from ._utils import flatten_features, select_subspace_dimension


class KCentresClustering:
    """
    K-subspaces style functional clustering, matching the survey's
    reconstruction-based k-centres description.
    """

    def __init__(
        self,
        n_clusters: int,
        subspace_dimension: int | None = None,
        variance_explained: float = 0.9,
        max_subspace_dimension: int | None = None,
        max_iter: int = 100,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.subspace_dimension = subspace_dimension
        self.variance_explained = variance_explained
        self.max_subspace_dimension = max_subspace_dimension
        self.max_iter = max_iter
        self.random_state = random_state

        self.labels_ = None
        self.centers_ = None
        self.subspaces_ = None
        self.subspace_dims_ = None

    def fit(self, X: torch.Tensor, basis=None):
        X_np = flatten_features(X)
        n_samples, n_features = X_np.shape
        max_dim = self.max_subspace_dimension
        if max_dim is None:
            max_dim = max(1, min(n_features - 1, n_features))

        labels = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        ).fit_predict(X_np)

        rng = np.random.RandomState(self.random_state)
        centers = None
        subspaces = None
        subspace_dims = None

        for _ in range(self.max_iter):
            new_centers = []
            new_subspaces = []
            new_dims = []

            for k in range(self.n_clusters):
                members = X_np[labels == k]
                if members.shape[0] == 0:
                    mean = X_np[rng.randint(0, n_samples)]
                    basis_k = np.zeros((n_features, 0))
                    dim_k = 0
                else:
                    mean = members.mean(axis=0)
                    centered = members - mean
                    if members.shape[0] == 1:
                        basis_k = np.zeros((n_features, 0))
                        dim_k = 0
                    else:
                        cov = centered.T @ centered / max(members.shape[0] - 1, 1)
                        evals, evecs = np.linalg.eigh(cov)
                        order = np.argsort(evals)[::-1]
                        evals = evals[order]
                        evecs = evecs[:, order]
                        dim_k = select_subspace_dimension(
                            evals,
                            fixed_dimension=self.subspace_dimension,
                            variance_explained=self.variance_explained,
                            max_dimension=max_dim,
                        )
                        basis_k = evecs[:, :dim_k]

                new_centers.append(mean)
                new_subspaces.append(basis_k)
                new_dims.append(dim_k)

            errors = []
            for mean, basis_k in zip(new_centers, new_subspaces):
                centered = X_np - mean
                if basis_k.shape[1] > 0:
                    projected = centered @ basis_k @ basis_k.T
                    residual = centered - projected
                else:
                    residual = centered
                errors.append(np.sum(residual * residual, axis=1))

            errors = np.column_stack(errors)
            new_labels = errors.argmin(axis=1)

            centers = new_centers
            subspaces = new_subspaces
            subspace_dims = np.asarray(new_dims, dtype=int)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels

        self.labels_ = labels
        self.centers_ = np.asarray(centers)
        self.subspaces_ = subspaces
        self.subspace_dims_ = subspace_dims
        return self

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        return self.fit(X, basis=basis).labels_
