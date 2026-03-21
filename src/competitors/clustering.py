"""
Competitor clustering methods for metric and Hilbert spaces.

These wrappers provide a scikit-learn compatible interface for clustering
the coefficient representation X (shape n x M) of the data. 
Since the coefficients are in an orthonormal Hilbert basis (by default), 
the Euclidean distance in R^M corresponds to the distance in the Hilbert space.
"""
import numpy as np
import torch
import warnings

# kmedoids
import kmedoids
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN

# scikit-fda
import skfda
from skfda.ml.clustering import KMeans as FDAKMeans
from skfda.ml.clustering import FuzzyCMeans as FDAFuzzyCMeans
from skfda.ml.clustering import AgglomerativeClustering as FDAAgglomerative


def to_numpy(X):
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X)


class KMedoidsClustering:
    """
    K-Medoids using FasterPAM algorithm from the `kmedoids` package.
    Robust for arbitrary metric spaces when distance matrix is precomputed.
    """
    def __init__(self, n_clusters: int, method: str = "fasterpam", max_iter: int = 100):
        self.n_clusters = n_clusters
        self.method = method
        self.max_iter = max_iter
        self.labels_ = None
        self.medoids_ = None

    def fit_predict(self, X: torch.Tensor, dist_matrix: np.ndarray = None) -> np.ndarray:
        if kmedoids is None:
            raise ImportError("kmedoids package is not installed.")

        if dist_matrix is None:
            X_np = to_numpy(X)
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        if self.method == "fasterpam":
            res = kmedoids.fasterpam(dist_matrix, self.n_clusters, max_iter=self.max_iter)
        else:
            # fallback or you could support others like pam, fastpam1
            res = kmedoids.pam(dist_matrix, self.n_clusters, max_iter=self.max_iter)

        self.labels_ = res.labels
        self.medoids_ = res.medoids
        return self.labels_


class HierarchicalClustering:
    """
    Hierarchical / Agglomerative Clustering using scikit-learn.
    Supports single, complete, average linkages via standard metric space.
    """
    def __init__(self, n_clusters: int, linkage: str = "average"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        # Using precomputed distances to ensure pure metric space grouping
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters, 
            metric="precomputed", 
            linkage=self.linkage
        )

    def fit_predict(self, X: torch.Tensor, dist_matrix: np.ndarray = None) -> np.ndarray:
        if AgglomerativeClustering is None:
            raise ImportError("scikit-learn is not installed.")

        if dist_matrix is None:
            X_np = to_numpy(X)
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        self.labels_ = self.model.fit_predict(dist_matrix)
        return self.labels_


class DBSCANClustering:
    """
    DBSCAN clustering using scikit-learn with precomputed metric space distances.
    """
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")

    def fit_predict(self, X: torch.Tensor, dist_matrix: np.ndarray = None) -> np.ndarray:
        if DBSCAN is None:
            raise ImportError("scikit-learn is not installed.")

        if dist_matrix is None:
            X_np = to_numpy(X)
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        self.labels_ = self.model.fit_predict(dist_matrix)
        return self.labels_


class HDBSCANClustering:
    """
    HDBSCAN clustering using scikit-learn with precomputed metric space distances.
    """
    def __init__(self, min_cluster_size: int = 5):
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        self.model = HDBSCAN(min_cluster_size=self.min_cluster_size, metric="precomputed")

    def fit_predict(self, X: torch.Tensor, dist_matrix: np.ndarray = None) -> np.ndarray:
        if HDBSCAN is None:
            raise ImportError("scikit-learn >= 1.3 is required for HDBSCAN.")

        if dist_matrix is None:
            X_np = to_numpy(X)
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(X_np, metric="euclidean"))

        self.labels_ = self.model.fit_predict(dist_matrix)
        return self.labels_


class KCenterClustering:
    """
    Classical greedy 2-approximation for K-Center Clustering 
    (Farthest-First Traversal / Gonzalez's algorithm).
    """
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.centers_ = None

    def fit_predict(self, X: torch.Tensor) -> np.ndarray:
        X_np = to_numpy(X)
        n_samples = X_np.shape[0]

        if self.n_clusters >= n_samples:
            self.labels_ = np.arange(n_samples)
            self.centers_ = np.arange(n_samples)
            return self.labels_

        rng = np.random.RandomState(self.random_state)
        first_center = rng.randint(n_samples)
        centers = [first_center]

        from scipy.spatial.distance import cdist
        # Distances to the nearest center for each point
        min_dists = cdist(X_np, X_np[centers]).min(axis=1)

        for _ in range(1, self.n_clusters):
            # Pick the point farthest from its nearest center
            next_center = np.argmax(min_dists)
            centers.append(next_center)
            
            # Update minimum distances
            new_dists = cdist(X_np, X_np[[next_center]]).flatten()
            min_dists = np.minimum(min_dists, new_dists)

        self.centers_ = np.array(centers)
        
        # Finally, assign each point to its nearest center
        all_dists = cdist(X_np, X_np[self.centers_])
        self.labels_ = np.argmin(all_dists, axis=1)
        
        return self.labels_


# ----------------------------------------------------------------------------
# Scikit-FDA wrappers
# ----------------------------------------------------------------------------

def _prepare_fda_input(X: torch.Tensor, basis=None):
    """
    Convert coefficient tensor X to an FDataGrid so it can be used 
    by scikit-fda clustering estimators.
    """
    if skfda is None:
        raise ImportError("scikit-fda is not installed.")

    X_np = to_numpy(X)

    if basis is not None and hasattr(basis, "reconstruct"):
        # We can reconstruct on whatever grid the basis provides.
        # But reconstruct(X) usually just returns the tensor evaluated points.
        with torch.no_grad():
            functions = basis.reconstruct(X)
            func_np = to_numpy(functions)
        
        # func_np has shape (n_samples, grid_size, d)
        n_samples, grid_size, d = func_np.shape
        grid_points = np.linspace(0, 1, grid_size)
        return skfda.FDataGrid(data_matrix=func_np, grid_points=grid_points)
    else:
        # If no basis is provided or reconstructed isn't possible,
        # we treat the coefficients themselves as discrete functional points.
        return skfda.FDataGrid(data_matrix=X_np)


class ScikitFDAKMeans:
    """
    Scikit-FDA KMeans clustering for functional data.
    """
    def __init__(self, n_clusters: int, **kwargs):
        if skfda is None:
            raise ImportError("scikit-fda is not installed.")
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.labels_ = None
        self.model = FDAKMeans(n_clusters=self.n_clusters, **self.kwargs)

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        fd = _prepare_fda_input(X, basis)
        self.labels_ = self.model.fit_predict(fd)
        return self.labels_


class ScikitFDAFuzzyCMeans:
    """
    Scikit-FDA Fuzzy C-Means clustering for functional data.
    """
    def __init__(self, n_clusters: int, **kwargs):
        if skfda is None:
            raise ImportError("scikit-fda is not installed.")
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.labels_ = None
        self.model = FDAFuzzyCMeans(n_clusters=self.n_clusters, **self.kwargs)

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        fd = _prepare_fda_input(X, basis)
        # fit_predict returns crisp labels for fuzzy means typically, or we take argmax
        # FDAFuzzyCMeans doesn't have fit_predict that returns crisp, we must fit & assign
        self.model.fit(fd)
        memberships = self.model.membership_degree_
        self.labels_ = np.argmax(memberships, axis=1)
        return self.labels_


class ScikitFDAAgglomerative:
    """
    Scikit-FDA Agglomerative clustering for functional data.
    """
    def __init__(self, n_clusters: int, linkage: str = "average", **kwargs):
        if skfda is None:
            raise ImportError("scikit-fda is not installed.")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.kwargs = kwargs
        self.labels_ = None
        self.model = FDAAgglomerative(n_clusters=self.n_clusters, linkage=self.linkage, **self.kwargs)

    def fit_predict(self, X: torch.Tensor, basis=None) -> np.ndarray:
        fd = _prepare_fda_input(X, basis)
        self.labels_ = self.model.fit_predict(fd)
        return self.labels_
