"""
Generic EM routine for low-rank Gaussian mixture competitors.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from ._utils import (
    build_parsimonious_covariance,
    select_subspace_dimension,
    stable_log_gaussian_density,
    weighted_covariance,
    weighted_mean,
)


class LowRankGaussianMixture:
    """
    Gaussian mixture with component-specific signal subspaces and isotropic noise.

    This covers the low-rank covariance structure used by Mixt-PPCA, HDDC,
    Funclust, and FunHDDC in a finite-dimensional coefficient representation.
    """

    def __init__(
        self,
        n_clusters: int,
        subspace_dimension: int | None = None,
        variance_explained: float = 0.9,
        max_subspace_dimension: int | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 3,
        random_state: int | None = None,
        reg_covar: float = 1e-6,
    ):
        self.n_clusters = n_clusters
        self.subspace_dimension = subspace_dimension
        self.variance_explained = variance_explained
        self.max_subspace_dimension = max_subspace_dimension
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.reg_covar = reg_covar

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.subspace_dims_ = None
        self.subspace_bases_ = None
        self.signal_variances_ = None
        self.noise_variances_ = None
        self.responsibilities_ = None
        self.labels_ = None
        self.lower_bound_ = None
        self.n_iter_ = 0

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        max_dim = self.max_subspace_dimension
        if max_dim is None:
            max_dim = max(1, min(n_features - 1, n_features))

        best_state = None
        best_lower_bound = -np.inf
        rng = np.random.RandomState(self.random_state)
        global_cov = np.cov(X, rowvar=False)
        if global_cov.ndim == 0:
            global_cov = np.array([[float(global_cov)]])
        global_cov = global_cov + self.reg_covar * np.eye(n_features)

        for _ in range(self.n_init):
            seed = None if self.random_state is None else int(rng.randint(0, 10_000_000))
            resp = self._initialize_responsibilities(X, seed)
            previous_lower_bound = -np.inf

            for iteration in range(1, self.max_iter + 1):
                params = self._m_step(X, resp, max_dim=max_dim, global_cov=global_cov)
                log_prob = self._estimate_log_component_densities(X, params)
                lower_bound = float(logsumexp(log_prob, axis=1).sum())
                resp = np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))

                if abs(lower_bound - previous_lower_bound) <= self.tol * (1.0 + abs(lower_bound)):
                    break
                previous_lower_bound = lower_bound

            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                best_state = (params, resp, iteration)

        if best_state is None:
            raise RuntimeError("Low-rank Gaussian mixture failed to initialize.")

        params, resp, n_iter = best_state
        self.weights_ = params["weights"]
        self.means_ = params["means"]
        self.covariances_ = params["covariances"]
        self.subspace_dims_ = params["subspace_dims"]
        self.subspace_bases_ = params["subspace_bases"]
        self.signal_variances_ = params["signal_variances"]
        self.noise_variances_ = params["noise_variances"]
        self.responsibilities_ = resp
        self.labels_ = resp.argmax(axis=1)
        self.lower_bound_ = best_lower_bound
        self.n_iter_ = n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        params = {
            "weights": self.weights_,
            "means": self.means_,
            "covariances": self.covariances_,
        }
        log_prob = self._estimate_log_component_densities(X, params)
        return np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def _initialize_responsibilities(self, X: np.ndarray, seed: int | None) -> np.ndarray:
        try:
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init=1,
                random_state=seed,
            )
            labels = kmeans.fit_predict(X)
        except Exception:
            rng = np.random.RandomState(seed)
            labels = rng.randint(self.n_clusters, size=X.shape[0])

        resp = np.zeros((X.shape[0], self.n_clusters), dtype=float)
        resp[np.arange(X.shape[0]), labels] = 1.0
        return resp

    def _m_step(
        self,
        X: np.ndarray,
        resp: np.ndarray,
        max_dim: int,
        global_cov: np.ndarray,
    ) -> dict:
        n_samples, n_features = X.shape
        weights = resp.sum(axis=0)
        weights = np.maximum(weights, 1e-12)
        mixing = weights / weights.sum()

        means = []
        covariances = []
        subspace_dims = []
        subspace_bases = []
        signal_variances = []
        noise_variances = []

        for k in range(self.n_clusters):
            component_weights = resp[:, k]
            component_mass = component_weights.sum()

            if component_mass <= 1e-8:
                mean = X[np.random.randint(0, n_samples)]
                cov = global_cov.copy()
                evals, evecs = np.linalg.eigh(cov)
                order = np.argsort(evals)[::-1]
                evals = evals[order]
                evecs = evecs[:, order]
                dim = min(max_dim, n_features)
                basis = evecs[:, :dim]
                signal = evals[:dim]
                noise = float(max(evals[dim:].mean() if dim < n_features else self.reg_covar, self.reg_covar))
            else:
                mean = weighted_mean(X, component_weights)
                cov_emp = weighted_covariance(X, component_weights, mean=mean, reg_covar=self.reg_covar)
                evals, evecs = np.linalg.eigh(cov_emp)
                order = np.argsort(evals)[::-1]
                evals = evals[order]
                evecs = evecs[:, order]
                dim = select_subspace_dimension(
                    evals,
                    fixed_dimension=self.subspace_dimension,
                    variance_explained=self.variance_explained,
                    max_dimension=max_dim,
                )
                cov, signal, noise, basis = build_parsimonious_covariance(
                    eigenvectors=evecs,
                    eigenvalues=evals,
                    dimension=dim,
                    reg_covar=self.reg_covar,
                )

            means.append(mean)
            covariances.append(cov)
            subspace_dims.append(dim)
            subspace_bases.append(basis)
            signal_variances.append(signal)
            noise_variances.append(noise)

        return {
            "weights": mixing,
            "means": np.asarray(means),
            "covariances": np.asarray(covariances),
            "subspace_dims": np.asarray(subspace_dims, dtype=int),
            "subspace_bases": subspace_bases,
            "signal_variances": signal_variances,
            "noise_variances": np.asarray(noise_variances, dtype=float),
        }

    def _estimate_log_component_densities(self, X: np.ndarray, params: dict) -> np.ndarray:
        log_prob = []
        for weight, mean, cov in zip(
            params["weights"],
            params["means"],
            params["covariances"],
        ):
            density = stable_log_gaussian_density(X, mean, cov, reg_covar=self.reg_covar)
            log_prob.append(np.log(weight + 1e-12) + density)
        return np.column_stack(log_prob)
