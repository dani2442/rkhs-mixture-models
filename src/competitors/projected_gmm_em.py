"""
Projected Gaussian-mixture EM (Algorithm 1 from Appendix, Section likelihood).

Implements the projected EM algorithm for the Gaussian-mixture log-likelihood
as described in the paper.  Updates per-component weights, means, and
covariances with ridge regularisation (epsilon * I_M).  Setting
``fixed_covariance=True`` activates the benchmark variant in which only
weights and means are updated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from ._utils import flatten_features, to_numpy, weighted_mean

if TYPE_CHECKING:
    from ..spaces.base import HilbertBasis


class ProjectedGaussianMixtureEM:
    """Projected EM for a Gaussian mixture (Algorithm 1).

    Parameters
    ----------
    n_clusters : int
        Number of mixture components *K*.
    fixed_covariance : bool
        If ``True``, covariances are not updated in the M-step (benchmark
        variant described in the paper).  A shared empirical covariance is
        used for all components.
    max_iter : int
        Maximum number of EM iterations *T_EM*.
    tol : float
        Relative convergence tolerance on the log-likelihood.
    n_init : int
        Number of random restarts; the run with the highest log-likelihood
        is kept.
    random_state : int or None
        Seed for reproducibility.
    reg_covar : float
        Ridge parameter *epsilon* added to the diagonal of every covariance
        after each M-step (``epsilon * I_M``).
    """

    def __init__(
        self,
        n_clusters: int,
        fixed_covariance: bool = False,
        max_iter: int = 200,
        tol: float = 1e-4,
        n_init: int = 3,
        random_state: int | None = None,
        reg_covar: float = 1e-4,
    ):
        self.n_clusters = n_clusters
        self.fixed_covariance = fixed_covariance
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.reg_covar = reg_covar

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.responsibilities_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.lower_bound_: float | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Projected EM expects a 2D feature matrix.")

        n, M = X.shape
        rng = np.random.RandomState(self.random_state)

        best_state = None
        best_lower_bound = -np.inf

        for _ in range(self.n_init):
            seed = (
                None
                if self.random_state is None
                else int(rng.randint(0, 10_000_000))
            )
            weights, means, covariances = self._initialize(X, M, seed)
            chols, logdets = self._cholesky_params(covariances)
            previous_lower_bound = -np.inf

            for iteration in range(1, self.max_iter + 1):
                # E-step: r_{ik} = pi_k phi_M(x_i; m_k, K_k) / sum_s ...
                log_resp = self._e_step(X, weights, means, chols, logdets)
                log_norm = logsumexp(log_resp, axis=1)
                lower_bound = float(log_norm.sum())
                resp = np.exp(log_resp - log_norm[:, None])

                # M-step
                weights, means, covariances = self._m_step(
                    X, resp, covariances, seed
                )
                chols, logdets = self._cholesky_params(covariances)

                if abs(lower_bound - previous_lower_bound) <= self.tol * (
                    1.0 + abs(lower_bound)
                ):
                    break
                previous_lower_bound = lower_bound

            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                best_state = (
                    weights.copy(),
                    means.copy(),
                    covariances.copy(),
                    resp.copy(),
                    chols.copy(),
                    logdets.copy(),
                    iteration,
                )

        if best_state is None:
            raise RuntimeError("Projected EM failed to initialize.")

        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.responsibilities_,
            self._chols,
            self._logdets,
            self.n_iter_,
        ) = best_state
        self.labels_ = self.responsibilities_.argmax(axis=1)
        self.lower_bound_ = best_lower_bound
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_resp = self._e_step(
            X, self.weights_, self.means_, self._chols, self._logdets
        )
        return np.exp(log_resp - logsumexp(log_resp, axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(self, X, weights, means, chols, logdets):
        """log(pi_k * phi_M(x_i; m_k, K_k)) for all i, k."""
        n, M = X.shape
        K = len(weights)
        log_prob = np.empty((n, K))
        const = -0.5 * M * np.log(2.0 * np.pi)

        for k in range(K):
            diff = X - means[k]
            whitened = solve_triangular(chols[k], diff.T, lower=True).T
            quad = np.einsum("ij,ij->i", whitened, whitened)
            log_prob[:, k] = (
                np.log(weights[k] + 1e-300)
                + const
                - 0.5 * logdets[k]
                - 0.5 * quad
            )
        return log_prob

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _m_step(self, X, resp, covariances, seed):
        """Update pi_k, m_k, and (unless fixed_covariance) K_k.

        K_k <- (1/N_k) sum_i r_{ik} (x_i - m_k)(x_i - m_k)^T + eps * I_M
        """
        n, M = X.shape
        K = self.n_clusters
        rng = np.random.RandomState(seed)

        masses = resp.sum(axis=0)
        weights = np.maximum(masses, 1e-12)
        weights /= weights.sum()

        means = np.empty((K, M))
        new_covariances = np.empty((K, M, M)) if not self.fixed_covariance else None

        for k in range(K):
            if masses[k] <= 1e-8:
                means[k] = X[rng.randint(0, n)]
                if new_covariances is not None:
                    new_covariances[k] = covariances[k]
            else:
                means[k] = weighted_mean(X, resp[:, k])
                if new_covariances is not None:
                    centered = X - means[k]
                    new_covariances[k] = (
                        (centered * resp[:, k : k + 1]).T @ centered / masses[k]
                    )
                    new_covariances[k] = 0.5 * (
                        new_covariances[k] + new_covariances[k].T
                    )
                    new_covariances[k].flat[:: M + 1] += self.reg_covar

        if self.fixed_covariance:
            return weights, means, covariances
        return weights, means, new_covariances

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(self, X, M, seed):
        K = self.n_clusters
        n = X.shape[0]

        try:
            kmeans = KMeans(n_clusters=K, n_init=1, random_state=seed)
            labels = kmeans.fit_predict(X)
            means = np.asarray(kmeans.cluster_centers_, dtype=float)
        except Exception:
            rng = np.random.RandomState(seed)
            labels = rng.randint(K, size=n)
            indices = rng.choice(n, size=K, replace=False)
            means = X[indices].copy()

        counts = np.bincount(labels, minlength=K).astype(float)
        counts = np.maximum(counts, 1.0)
        weights = counts / counts.sum()

        if self.fixed_covariance:
            shared = self._empirical_covariance(X, M)
            covariances = np.tile(shared, (K, 1, 1))
        else:
            covariances = np.empty((K, M, M))
            for k in range(K):
                mask = labels == k
                if mask.sum() < 2:
                    covariances[k] = self._empirical_covariance(X, M)
                else:
                    centered = X[mask] - means[k]
                    covariances[k] = centered.T @ centered / centered.shape[0]
                    covariances[k] = 0.5 * (covariances[k] + covariances[k].T)
                    covariances[k].flat[:: M + 1] += self.reg_covar

        return weights, means, covariances

    def _empirical_covariance(self, X, M):
        centered = X - X.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(X.shape[0] - 1, 1)
        cov = 0.5 * (cov + cov.T)
        cov.flat[:: M + 1] += self.reg_covar
        return cov

    @staticmethod
    def _cholesky_params(covariances):
        K = covariances.shape[0]
        chols = np.empty_like(covariances)
        logdets = np.empty(K)
        for k in range(K):
            chols[k] = np.linalg.cholesky(covariances[k])
            logdets[k] = 2.0 * np.log(np.diag(chols[k])).sum()
        return chols, logdets


class ProjectedGMMEMFixedCovarianceClustering:
    """Clustering wrapper that projects onto a basis and runs EM."""

    def __init__(
        self,
        n_clusters: int,
        basis: HilbertBasis | None = None,
        n_init: int = 3,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int | None = None,
        reg_covar: float = 1e-4,
    ):
        self.n_clusters = n_clusters
        self.basis = basis
        self.labels_ = None
        self.responsibilities_ = None
        self.model = ProjectedGaussianMixtureEM(
            n_clusters=n_clusters,
            fixed_covariance=True,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            reg_covar=reg_covar,
        )

    def _project_data(self, X) -> np.ndarray:
        if self.basis is not None:
            X_tensor = X if torch.is_tensor(X) else torch.as_tensor(X)
            with torch.no_grad():
                coeffs = self.basis.project(X_tensor)
            return to_numpy(coeffs)
        return flatten_features(X)

    def fit(self, X):
        self.model.fit(self._project_data(X))
        self.labels_ = self.model.labels_
        self.responsibilities_ = self.model.responsibilities_
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self._project_data(X))

    def fit_predict(self, X) -> np.ndarray:
        return self.fit(X).labels_
