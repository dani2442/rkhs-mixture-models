"""
Kernel k-Groups via Hartigan's Method.

Implements Algorithm 2 from:
    Kernel k-Groups via Hartigan's Method (2020)

The objective maximized is Q = sum_j Q_j / s_j where Q_j is the within-cluster
weighted kernel sum and s_j the cluster weight sum.  The Hartigan-style
reassignment moves one point at a time to the cluster that most increases Q,
iterating until convergence.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from ._utils import to_numpy


class KernelKGroupsClustering:
    """
    Kernel k-Groups clustering via Hartigan's reassignment method.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    kernel : {"gaussian", "energy"} or callable, optional
        Kernel function.  ``"gaussian"`` uses exp(-||x-y||^2 / (2 sigma^2));
        ``"energy"`` converts a Euclidean distance matrix via the
        energy-distance kernel K(x,y) = (||x|| + ||y|| - ||x-y||) / 2
        (equivalent to fixing x0=0 in the paper's semimetric construction).
        A callable should accept two 1-D arrays and return a scalar.
        Default is ``"gaussian"``.
    sigma : float or None
        Bandwidth for the Gaussian kernel.  If *None*, the median heuristic
        is used.  Ignored when *kernel* is not ``"gaussian"``.
    max_iter : int
        Maximum number of full passes over the data.
    random_state : int
        Seed for the k-means++ initializer.
    """

    def __init__(
        self,
        n_clusters: int,
        kernel: str = "gaussian",
        sigma: float | None = None,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.sigma = sigma
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None

    # ── public API ────────────────────────────────────────────────────

    def fit_predict(
        self,
        X: torch.Tensor,
        dist_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Data points (used for building the kernel Gram matrix).
        dist_matrix : np.ndarray, shape (n, n), optional
            Precomputed pairwise distance matrix.  When supplied, the
            ``"energy"`` kernel uses this directly; the ``"gaussian"``
            kernel derives squared distances from it.

        Returns
        -------
        labels : np.ndarray, shape (n,)
        """
        X_np = to_numpy(X)
        n = X_np.shape[0]
        k = self.n_clusters

        # ── Build Gram matrix A (unweighted: w_i = 1) ────────────────
        A = self._gram_matrix(X_np, dist_matrix)
        a_diag = np.diag(A).copy()  # A_{ii}

        # ── Initialize with k-means++ ─────────────────────────────────
        labels = self._kmpp_init(X_np, k, dist_matrix)

        # ── Build one-hot Z and derived quantities ────────────────────
        Z = np.zeros((n, k), dtype=np.float64)
        Z[np.arange(n), labels] = 1.0

        s = Z.sum(axis=0)          # cluster sizes (n_ell)
        U = A @ Z                  # U[i, ell] = sum_{r in C_ell} A[i,r]
        q = np.sum(Z * U, axis=0)  # q[ell] = Q_ell

        # ── Hartigan passes ───────────────────────────────────────────
        for _ in range(self.max_iter):
            moved = False
            for i in range(n):
                j = labels[i]  # current cluster

                # Never empty a cluster
                if s[j] <= 1.0 + 1e-12:
                    continue

                # Compute gain for each candidate cluster ell != j
                # delta = gain_remove(j) - cost_add(ell)
                #   gain_remove = (q[j]/s[j] - 2*U[i,j] + a_diag[i]) / (s[j] - 1)
                #   cost_add    = (q[ell]/s[ell] - 2*U[i,ell] - a_diag[i]) / (s[ell] + 1)
                gain_remove = (q[j] / s[j] - 2.0 * U[i, j] + a_diag[i]) / (s[j] - 1.0)

                cost_add = (q / s - 2.0 * U[i] - a_diag[i]) / (s + 1.0)
                delta = gain_remove - cost_add  # shape (k,)
                delta[j] = -np.inf  # exclude current cluster

                ell_star = int(np.argmax(delta))
                if delta[ell_star] <= 0.0:
                    continue

                # Accept move j -> ell_star
                moved = True

                # Update Z
                Z[i, j] = 0.0
                Z[i, ell_star] = 1.0

                # Update cluster sizes
                s[j] -= 1.0
                s[ell_star] += 1.0

                # Update Q_j, Q_ell*
                q[j] -= 2.0 * U[i, j] - a_diag[i]
                q[ell_star] += 2.0 * U[i, ell_star] + a_diag[i]

                # Update U = A @ Z  (rank-1 update)
                U[:, j] -= A[:, i]
                U[:, ell_star] += A[:, i]

                labels[i] = ell_star

            if not moved:
                break

        self.labels_ = labels
        return labels

    # ── private helpers ───────────────────────────────────────────────

    def _gram_matrix(
        self, X_np: np.ndarray, dist_matrix: np.ndarray | None
    ) -> np.ndarray:
        """Build the kernel Gram matrix."""
        if callable(self.kernel) and not isinstance(self.kernel, str):
            n = X_np.shape[0]
            G = np.empty((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(i, n):
                    v = self.kernel(X_np[i], X_np[j])
                    G[i, j] = v
                    G[j, i] = v
            return G

        if dist_matrix is None:
            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        if self.kernel == "energy":
            # K(x,y) = (||x|| + ||y|| - ||x-y||) / 2  with x0 = 0
            norms = np.linalg.norm(X_np, axis=1)
            G = 0.5 * (norms[:, None] + norms[None, :] - dist_matrix)
            return G

        # Default: Gaussian kernel
        sigma = self.sigma
        if sigma is None:
            sigma = float(np.median(dist_matrix[dist_matrix > 0]))
        return np.exp(-dist_matrix ** 2 / (2.0 * sigma ** 2))

    def _kmpp_init(
        self, X_np: np.ndarray, k: int, dist_matrix: np.ndarray | None
    ) -> np.ndarray:
        """K-means++ initialization returning integer labels."""
        rng = np.random.RandomState(self.random_state)
        n = X_np.shape[0]

        if dist_matrix is None:
            dist_sq = squareform(pdist(X_np, metric="sqeuclidean"))
        else:
            dist_sq = dist_matrix ** 2

        centers = [rng.randint(n)]
        for _ in range(1, k):
            d_min = dist_sq[centers].min(axis=0)
            probs = d_min / d_min.sum()
            centers.append(rng.choice(n, p=probs))

        # Assign each point to closest center
        labels = np.argmin(dist_sq[np.array(centers)], axis=0)
        return labels
