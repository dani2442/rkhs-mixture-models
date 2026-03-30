"""
Projected Gaussian-mixture EM (Algorithm 1 from Appendix, Section likelihood).

Implements the projected EM algorithm for the Gaussian-mixture log-likelihood
as described in the paper.  The primary mode updates per-component weights,
means, and covariances with optional ridge regularisation (epsilon * I_M).
A ``fixed_covariance`` flag activates the benchmark variant in which only
weights and means are updated while the covariance is held fixed.
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
        variant described in the paper).  The shared covariance is set
        according to *covariance_mode* / *covariance*.
    covariance_mode : str
        How to build the initial (or fixed) covariance.  One of
        ``"empirical"``, ``"diagonal_empirical"``, ``"spherical_empirical"``,
        ``"identity"``.  Ignored when *covariance* is provided explicitly.
    covariance : np.ndarray or None
        An explicit shared covariance matrix.  Overrides *covariance_mode*.
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
        covariance_mode: str = "empirical",
        covariance: np.ndarray | None = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        n_init: int = 3,
        random_state: int | None = None,
        reg_covar: float = 1e-4,
    ):
        self.n_clusters = n_clusters
        self.fixed_covariance = fixed_covariance
        self.covariance_mode = covariance_mode
        self.covariance = covariance
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.reg_covar = reg_covar

        # Fitted attributes
        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None  # (K, M, M)
        self.responsibilities_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.lower_bound_: float | None = None
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray):
        """Run the projected EM algorithm on data ``X`` of shape (n, M)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Projected EM expects a 2D feature matrix.")

        n, M = X.shape
        K = self.n_clusters
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
                # --- E-step: compute responsibilities r_{ik} ---
                log_resp = self._e_step(X, weights, means, chols, logdets)
                log_norm = logsumexp(log_resp, axis=1)
                lower_bound = float(log_norm.sum())
                resp = np.exp(log_resp - log_norm[:, None])

                # --- M-step ---
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

    def _e_step(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        chols: np.ndarray,
        logdets: np.ndarray,
    ) -> np.ndarray:
        """Compute log(pi_k * phi_M(x_i; m_k, K_k)) for all i, k.

        Uses Cholesky factors for numerical stability:
            phi_M(x; m, K) = (2pi)^{-M/2} det(K)^{-1/2}
                             exp(-0.5 (x-m)^T K^{-1} (x-m))
        """
        n, M = X.shape
        K = len(weights)
        log_prob = np.empty((n, K))
        const = -0.5 * M * np.log(2.0 * np.pi)

        for k in range(K):
            diff = X - means[k]  # (n, M)
            # solve L_k z = diff^T  =>  z = L_k^{-1} diff^T
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

    def _m_step(
        self,
        X: np.ndarray,
        resp: np.ndarray,
        covariances: np.ndarray,
        seed: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M-step: update pi_k, m_k, and (optionally) K_k.

        Full-covariance update (Algorithm 1, line 13):
            K_k <- (1/N_k) sum_i r_{ik} (x_i - m_k)(x_i - m_k)^T + eps * I_M

        Fixed-covariance variant: only weights and means are updated.
        """
        n, M = X.shape
        K = self.n_clusters
        rng = np.random.RandomState(seed)

        masses = resp.sum(axis=0)  # N_k
        weights = np.maximum(masses, 1e-12)
        weights = weights / weights.sum()

        means = np.empty((K, M))
        new_covariances = covariances  # keep unchanged for fixed-cov mode

        if not self.fixed_covariance:
            new_covariances = np.empty((K, M, M))

        for k in range(K):
            if masses[k] <= 1e-8:
                means[k] = X[rng.randint(0, n)]
                if not self.fixed_covariance:
                    new_covariances[k] = covariances[k]
            else:
                means[k] = weighted_mean(X, resp[:, k])
                if not self.fixed_covariance:
                    centered = X - means[k]  # (n, M)
                    # K_k = (1/N_k) sum_i r_{ik} (x_i - m_k)(x_i - m_k)^T
                    new_covariances[k] = (
                        (centered * resp[:, k : k + 1]).T @ centered
                        / masses[k]
                    )
                    # + epsilon * I_M
                    new_covariances[k].flat[:: M + 1] += self.reg_covar

        if self.fixed_covariance:
            return weights, means, covariances

        # Symmetrise for numerical safety
        for k in range(K):
            new_covariances[k] = 0.5 * (
                new_covariances[k] + new_covariances[k].T
            )
        return weights, means, new_covariances

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(
        self,
        X: np.ndarray,
        M: int,
        seed: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialise weights, means, and per-component covariances."""
        K = self.n_clusters
        n = X.shape[0]

        # --- Weights and means via K-means ---
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

        # --- Covariances ---
        if self.fixed_covariance:
            shared = self._build_fixed_covariance(X, M)
            covariances = np.tile(shared, (K, 1, 1))
        else:
            # Initialise each K_k from within-cluster scatter + ridge
            covariances = np.empty((K, M, M))
            for k in range(K):
                mask = labels == k
                if mask.sum() < 2:
                    # Fall back to global empirical covariance
                    covariances[k] = self._build_fixed_covariance(X, M)
                else:
                    cluster_data = X[mask]
                    centered = cluster_data - means[k]
                    covariances[k] = centered.T @ centered / centered.shape[0]
                    covariances[k] = 0.5 * (
                        covariances[k] + covariances[k].T
                    )
                    covariances[k].flat[:: M + 1] += self.reg_covar

        return weights, means, covariances

    def _build_fixed_covariance(self, X: np.ndarray, M: int) -> np.ndarray:
        """Build a shared covariance matrix for the fixed-covariance variant."""
        if self.covariance is not None:
            cov = np.asarray(self.covariance, dtype=float)
        elif self.covariance_mode == "empirical":
            centered = X - X.mean(axis=0, keepdims=True)
            denom = max(X.shape[0] - 1, 1)
            cov = centered.T @ centered / denom
        elif self.covariance_mode == "diagonal_empirical":
            cov = np.diag(X.var(axis=0) + self.reg_covar)
        elif self.covariance_mode == "spherical_empirical":
            scale = float(max(X.var(axis=0).mean(), self.reg_covar))
            cov = scale * np.eye(M)
        elif self.covariance_mode == "identity":
            cov = np.eye(M)
        else:
            raise ValueError(f"Unknown covariance_mode: {self.covariance_mode}")

        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        cov.flat[:: M + 1] += self.reg_covar
        return cov

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cholesky_params(
        covariances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Cholesky factors and log-determinants for all components."""
        K = covariances.shape[0]
        chols = np.empty_like(covariances)
        logdets = np.empty(K)
        for k in range(K):
            chols[k] = np.linalg.cholesky(covariances[k])
            logdets[k] = 2.0 * np.log(np.diag(chols[k])).sum()
        return chols, logdets


# ---- Backward-compatible alias for the fixed-covariance-only class ----
FixedCovarianceGaussianMixture = ProjectedGaussianMixtureEM


class ProjectedGMMEMFixedCovarianceClustering:
    """Clustering wrapper for the projected EM baseline.

    When a *basis* is supplied the wrapper projects raw observations onto
    the basis to obtain coefficient vectors in R^M, and the EM algorithm
    operates in that coordinate space.  When no basis is given the input
    is simply flattened.
    """

    def __init__(
        self,
        n_clusters: int,
        basis: HilbertBasis | None = None,
        covariance_mode: str = "empirical",
        covariance: np.ndarray | None = None,
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
            covariance_mode=covariance_mode,
            covariance=covariance,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            reg_covar=reg_covar,
        )

    def _project_data(self, X) -> np.ndarray:
        """Project input data to coefficient space."""
        if self.basis is not None:
            X_tensor = X if torch.is_tensor(X) else torch.as_tensor(X)
            with torch.no_grad():
                coeffs = self.basis.project(X_tensor)
            return to_numpy(coeffs)
        return flatten_features(X)

    def fit(self, X):
        X_np = self._project_data(X)
        self.model.fit(X_np)
        self.labels_ = self.model.labels_
        self.responsibilities_ = self.model.responsibilities_
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self._project_data(X))

    def fit_predict(self, X) -> np.ndarray:
        return self.fit(X).labels_
