"""
GPmix: model-based clustering of functional data via Gaussian processes.

This is a self-contained re-implementation of the GPmix algorithm of Akeweje and
Zhang (ICML 2024) that, unlike the authors' released package, also handles
**vector-valued** (multivariate) functional data ``x_i : [0, T] -> R^d``.

Algorithm (unchanged from the paper):

  1. Smooth each observed curve (optional, B-spline smoothing spline per channel).
  2. Project every curve onto ``n_proj`` projection functions, obtaining a scalar
     coefficient per (curve, projection).  A projection function is an element
     ``beta`` of the observation space and the coefficient is the ambient inner
     product ``<x_i, beta>``.
  3. Fit an independent univariate Gaussian mixture (spherical, EM) to each
     projection's coefficients -> ``n_proj`` base clusterings.
  4. Combine the base clusterings by consensus: a weighted co-association affinity
     matrix (each base clustering weighted by the inverse of its estimated total
     misclassification probability), followed by spectral clustering.

Multivariate extension.  Everything above is stated for a general separable
Hilbert space in the paper; only the released *code* is restricted to scalar
curves.  For observations in ``H = L^2([0, T]; R^d)`` the inner product is
``<f, g> = sum_{c=1}^d \\int f_c(t) g_c(t) dt`` and a projection function is a
genuine vector field ``beta = (beta_1, ..., beta_d) in H``:

  * ``fpc`` / ``rl-fpc``: multivariate FPCA -- the eigenfunctions are naturally
    vector-valued and capture cross-channel structure; ``rl-fpc`` uses random
    linear combinations of the leading eigenfunctions, exactly as in the paper.
  * ``fourier`` / ``bspline`` / ``ou``: a scalar dictionary ``g_v(t)`` is lifted
    to a vector field ``beta_v(t) = g_v(t) u_v`` with a random unit direction
    ``u_v in R^d`` (a random projection across channels).  For ``d = 1`` this is
    the identity ``u_v = 1`` and the whole pipeline reduces exactly to the
    univariate GPmix of the paper.

Citation:
    Akeweje, E., and Zhang, M. (2024). GPmix: Model-based Clustering of
    Functional Data via Gaussian Processes. ICML 2024, PMLR 235:720-740.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from scipy.stats import norm
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture


class GPmixClustering:
    """Consensus clustering of (possibly multivariate) functional data.

    Unlike the other competitors, this one is fit on the *raw* curves: pass an
    ``(n_samples, grid_size)`` array for scalar curves or ``(n_samples,
    grid_size, d)`` for vector-valued curves, since GPmix performs its own
    smoothing and projection.

    Parameters
    ----------
    n_clusters : int
        Number of mixture components (and consensus clusters), ``K``.
    basis_type : {'fpc', 'rl-fpc', 'fourier', 'bspline', 'ou'}
        Family of projection functions.
    n_proj : int
        Number of projection functions / base clusterings.
    smoother_basis : {'bspline', None}
        Presmoothing of each channel.  ``'bspline'`` fits a GCV smoothing spline
        per channel (matches the paper's default); ``None`` disables smoothing.
    init_method : {'kmeans', 'k-means++', 'random', 'random_from_data'}
        Initialisation passed to the univariate ``GaussianMixture`` fits.
    n_init : int
        Number of EM restarts per univariate GMM.
    random_state : int or None
        Seed controlling the random projection directions, OU / RL-FPC draws,
        the GMM restarts and the final spectral clustering.
    """

    def __init__(
        self,
        n_clusters: int,
        basis_type: str = "fpc",
        n_proj: int = 8,
        smoother_basis: str | None = "bspline",
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

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit_predict(self, X_raw, basis=None) -> np.ndarray:
        X = self._as_array(X_raw)  # (n, grid, d), float64
        n, grid, d = X.shape

        rng = np.random.RandomState(self.random_state)

        # domain [0, 1] with trapezoidal quadrature weights for the L^2 inner product.
        t = np.linspace(0.0, 1.0, grid)
        w = np.full(grid, 1.0 / (grid - 1)) if grid > 1 else np.ones(1)
        if grid > 1:
            w[0] *= 0.5
            w[-1] *= 0.5

        if self.smoother_basis == "bspline":
            X = self._smooth(X, t)

        # centre the sample (the paper projects the centred process).
        X = X - X.mean(axis=0, keepdims=True)

        # (n_proj, n) matrix of scalar projection coefficients.
        coeffs = self._project(X, t, w, d, rng)

        # one univariate GMM per projection.
        gmms = self._fit_gmms(coeffs, rng)

        # weighted co-association consensus + spectral clustering.
        self.labels_ = self._consensus(coeffs, gmms, rng)
        return self.labels_

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_array(X_raw) -> np.ndarray:
        if isinstance(X_raw, torch.Tensor):
            X_raw = X_raw.detach().cpu().numpy()
        X = np.asarray(X_raw, dtype=float)
        if X.ndim == 2:
            X = X[..., None]
        assert X.ndim == 3, "X_raw must be (n, grid) or (n, grid, d)"
        return X

    def _smooth(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """GCV smoothing spline per (curve, channel); robust to short/degenerate grids."""
        from scipy.interpolate import make_smoothing_spline

        n, grid, d = X.shape
        if grid < 5:
            return X
        Xs = np.empty_like(X)
        for i in range(n):
            for c in range(d):
                y = X[i, :, c]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        spl = make_smoothing_spline(t, y)
                        Xs[i, :, c] = spl(t)
                except Exception:
                    Xs[i, :, c] = y
        return Xs

    # ---- projection functions -------------------------------------------------
    def _project(self, X, t, w, d, rng) -> np.ndarray:
        """Return projection coefficients of shape (n_proj, n)."""
        bt = self.basis_type
        if bt in ("fpc", "rl-fpc"):
            return self._project_fpc(X, w, bt, rng)
        if bt in ("fourier", "bspline", "ou"):
            G = self._scalar_basis(bt, t, w, rng)  # (grid, n_proj)
            return self._project_scalar_basis(X, w, G, d, rng)
        raise ValueError(
            f"Unknown basis_type {bt!r}; choose from "
            "'fpc', 'rl-fpc', 'fourier', 'bspline', 'ou'."
        )

    def _project_fpc(self, X, w, bt, rng) -> np.ndarray:
        """Multivariate FPCA scores (fpc) or random eigenfunction combinations (rl-fpc)."""
        n, grid, d = X.shape
        sw = np.sqrt(w)  # quadrature half-weights
        # Weighted, flattened data matrix so that the Euclidean inner product in
        # this space equals the L^2([0,T]; R^d) inner product of the curves.
        Z = (X * sw[None, :, None]).reshape(n, grid * d)
        Z = Z - Z.mean(axis=0, keepdims=True)
        # scores_v = <x, e_v>, with e_v the v-th (weighted) right singular vector.
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        scores = U * S[None, :]  # (n, r), columns ordered by explained variance

        if bt == "fpc":
            k = min(self.n_proj, scores.shape[1])
            coeffs = scores[:, :k].T  # (k, n)
            if k < self.n_proj:  # pad if fewer components than requested
                coeffs = np.vstack([coeffs, np.zeros((self.n_proj - k, n))])
            return coeffs

        # rl-fpc: random linear combinations of the leading eigenfunctions that
        # jointly explain >= 95% of the variance.
        lam = S ** 2
        if lam.sum() <= 0:
            jn = 1
        else:
            jn = int(np.argmax(np.cumsum(lam / lam.sum()) >= 0.95)) + 1
        jn = max(1, min(jn, scores.shape[1]))
        s2 = scores[:, :jn].var(axis=0)  # variance of each score
        gammas = rng.normal(
            0.0, np.sqrt(np.clip(s2, 1e-30, None))[:, None], size=(jn, self.n_proj)
        )
        return (scores[:, :jn] @ gammas).T  # (n_proj, n)

    def _scalar_basis(self, bt, t, w, rng) -> np.ndarray:
        """Scalar projection functions on the grid, shape (grid, n_proj)."""
        grid = t.shape[0]
        p = self.n_proj
        if bt == "fourier":
            cols = [np.ones(grid)]
            k = 1
            while len(cols) < p:
                cols.append(np.cos(2.0 * math.pi * k * t))
                if len(cols) < p:
                    cols.append(np.sin(2.0 * math.pi * k * t))
                k += 1
            G = np.stack(cols[:p], axis=1)
            return self._gram_schmidt(G, w)

        if bt == "bspline":
            from scipy.interpolate import BSpline

            order = 3
            deg = order - 1
            n_basis = max(p, deg + 1)
            n_interior = n_basis - deg - 1
            interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1] if n_interior > 0 else []
            knots = np.concatenate(
                [np.zeros(deg + 1), np.asarray(interior, dtype=float), np.ones(deg + 1)]
            )
            x = np.clip(t, knots[deg], knots[-deg - 1] - 1e-12)
            G = np.asarray(BSpline.design_matrix(x, knots, deg).todense())[:, :p]
            return self._gram_schmidt(G, w)

        if bt == "ou":
            # Ornstein-Uhlenbeck sample paths: mean 0, cov(s,t) = exp(-|s - t|).
            diff = np.abs(t[:, None] - t[None, :])
            cov = np.exp(-diff) + 1e-8 * np.eye(grid)
            L = np.linalg.cholesky(cov)
            return L @ rng.standard_normal(size=(grid, p))

        raise ValueError(bt)

    @staticmethod
    def _gram_schmidt(G: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Orthonormalise the columns of G w.r.t. the weighted inner product diag(w)."""
        grid, p = G.shape
        Q = np.zeros_like(G)
        for j in range(p):
            v = G[:, j].copy()
            for i in range(j):
                proj = (Q[:, i] * w) @ v
                v -= proj * Q[:, i]
            nrm = math.sqrt(max((v * w) @ v, 0.0))
            Q[:, j] = v / nrm if nrm > 1e-12 else v
        return Q

    def _project_scalar_basis(self, X, w, G, d, rng) -> np.ndarray:
        """Lift scalar basis functions to R^d via random directions and project."""
        n, grid, _ = X.shape
        p = G.shape[1]
        # Per-channel scalar projections: P[c] = <x^c, g_v>, shape (n, p).
        Xw = X * w[None, :, None]  # (n, grid, d)
        P = np.einsum("ngc,gp->cnp", Xw, G)  # (d, n, p)
        if d == 1:
            return P[0].T  # (p, n) -- exactly the univariate pipeline
        # random unit direction per projection function
        dirs = rng.standard_normal(size=(p, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-12)
        coeffs = np.einsum("cnp,pc->np", P, dirs)  # (n, p)
        return coeffs.T  # (p, n)

    # ---- univariate GMMs ------------------------------------------------------
    def _fit_gmms(self, coeffs: np.ndarray, rng) -> list[GaussianMixture]:
        gmms = []
        for v in range(coeffs.shape[0]):
            gmm = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type="spherical",
                n_init=self.n_init,
                init_params=self.init_method,
                random_state=rng.randint(0, 2**31 - 1),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm.fit(coeffs[v].reshape(-1, 1))
            gmms.append(gmm)
        return gmms

    # ---- consensus ------------------------------------------------------------
    def _consensus(self, coeffs, gmms, rng) -> np.ndarray:
        n = coeffs.shape[1]
        K = self.n_clusters

        # binary membership indicator matrices, one per projection
        bms = []
        for v, gmm in enumerate(gmms):
            proba = gmm.predict_proba(coeffs[v].reshape(-1, 1))  # (n, K)
            hard = np.zeros_like(proba)
            hard[np.arange(n), proba.argmax(axis=1)] = 1.0
            bms.append(hard)

        weights = self._clustering_weights(gmms)

        affinity = np.zeros((n, n))
        for v in range(len(gmms)):
            affinity += weights[v] * (bms[v] @ bms[v].T)

        clustering = SpectralClustering(
            n_clusters=K,
            affinity="precomputed",
            assign_labels="discretize",
            random_state=rng.randint(0, 2**31 - 1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clustering.fit(affinity)
        return np.asarray(clustering.labels_)

    def _clustering_weights(self, gmms) -> np.ndarray:
        """Weight each base clustering by 1 / (total misclassification probability)."""
        if len(gmms) == 1:
            return np.array([1.0])
        total = np.array([self._total_omega(g) for g in gmms])
        total[total < 1e-30] = 1e-30
        inv = 1.0 / total
        return inv / inv.sum()

    def _total_omega(self, gmm) -> float:
        weights = gmm.weights_
        means = gmm.means_.ravel()
        varis = gmm.covariances_.ravel()  # spherical -> per-component variance
        K = self.n_clusters
        omega = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                omega[i, j] = self._omega_prob(
                    (weights[i], means[i], varis[i]),
                    (weights[j], means[j], varis[j]),
                )
        row = omega.sum(axis=1)
        return float(weights @ row)

    @staticmethod
    def _omega_prob(dist_a, dist_b) -> float:
        """P(sample from component a is assigned to component b) for 1-D Gaussians."""
        pa, ma, va = dist_a
        pb, mb, vb = dist_b
        va = max(va, 1e-30)
        vb = max(vb, 1e-30)
        pa = max(pa, 1e-30)
        pb = max(pb, 1e-30)

        coeff_a = 1.0 / vb - 1.0 / va
        coeff_b = 2.0 * (ma / va - mb / vb)
        coeff_c = mb ** 2 / vb - ma ** 2 / va - math.log((pb / pa) ** 2 * va / vb)

        if coeff_a != 0.0:
            disc = coeff_b ** 2 - 4.0 * coeff_a * coeff_c
            if disc >= 0.0:
                roots = [
                    (-coeff_b + math.sqrt(disc)) / (2.0 * coeff_a),
                    (-coeff_b - math.sqrt(disc)) / (2.0 * coeff_a),
                ]
                z = (np.array(roots) - ma) / math.sqrt(va)
                left, right = float(np.min(z)), float(np.max(z))
                if 2.0 * coeff_a > 0.0:
                    return norm.cdf(right) - norm.cdf(left)
                return 1.0 + norm.cdf(left) - norm.cdf(right)
            return 0.0 if 2.0 * coeff_a > 0.0 else 1.0

        if coeff_b != 0.0:
            zero = -coeff_c / coeff_b
            z = (zero - ma) / math.sqrt(va)
            return norm.cdf(z) if coeff_b > 0.0 else 1.0 - norm.cdf(z)

        return 1.0 if coeff_c <= 0.0 else 0.0
