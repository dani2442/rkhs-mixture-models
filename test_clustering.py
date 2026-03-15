import torch
import numpy as np
from src.competitors.clustering import (
    KMedoidsClustering,
    HierarchicalClustering,
    DBSCANClustering,
    HDBSCANClustering,
    KCenterClustering,
    ScikitFDAKMeans,
    ScikitFDAFuzzyCMeans,
    ScikitFDAAgglomerative,
)

def test_all():
    # 20 samples, 5 base coefficients
    X = torch.randn(20, 5)

    clustering_methods = [
        ("K-Medoids", KMedoidsClustering(n_clusters=3)),
        ("Hierarchical", HierarchicalClustering(n_clusters=3)),
        ("DBSCAN", DBSCANClustering(eps=2.0, min_samples=3)),
        ("HDBSCAN", HDBSCANClustering(min_cluster_size=3)),
        ("K-Center", KCenterClustering(n_clusters=3)),
        ("FDA KMeans", ScikitFDAKMeans(n_clusters=3)),
        ("FDA Fuzzy C-Means", ScikitFDAFuzzyCMeans(n_clusters=3)),
        ("FDA Agglomerative", ScikitFDAAgglomerative(n_clusters=3)),
    ]

    for name, method in clustering_methods:
        try:
            labels = method.fit_predict(X)
            print(f"{name}: successfully clustered into labels: {np.unique(labels)}")
        except Exception as e:
            print(f"{name}: FAILED with error: {e}")

if __name__ == "__main__":
    test_all()
