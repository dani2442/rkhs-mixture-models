"""
Competitor clustering methods for benchmarking against GaussianMixtureModel.
"""

from .curvclust import CurvclustClustering
from .fclust import FclustClustering
from .feature_gaussian_mixture import FeatureGaussianMixtureClustering
from .feature_hddc import FeatureHDDCClustering
from .feature_kmeans import FeatureKMeansClustering
from .funclust import FunclustClustering
from .funhddc import FunHDDCClustering
from .k_centres import KCentresClustering
from .kernel_kgroups import KernelKGroupsClustering
from .metric_dbscan import DBSCANClustering
from .metric_hdbscan import HDBSCANClustering
from .metric_hierarchical import HierarchicalClustering
from .metric_kcenter import KCenterClustering
from .metric_kmedoids import KMedoidsClustering
from .mixt_ppca import MixturePPCAClustering
from .projected_gmm_em import ProjectedGMMEMFixedCovarianceClustering
from .scikitfda_agglomerative import ScikitFDAAgglomerative
from .scikitfda_fuzzy_cmeans import ScikitFDAFuzzyCMeans
from .scikitfda_kmeans import ScikitFDAKMeans

__all__ = [
    "KMedoidsClustering",
    "HierarchicalClustering",
    "DBSCANClustering",
    "HDBSCANClustering",
    "KCenterClustering",
    "KernelKGroupsClustering",
    "ProjectedGMMEMFixedCovarianceClustering",
    "ScikitFDAKMeans",
    "ScikitFDAFuzzyCMeans",
    "ScikitFDAAgglomerative",
    "FeatureKMeansClustering",
    "FeatureGaussianMixtureClustering",
    "FeatureHDDCClustering",
    "MixturePPCAClustering",
    "FunclustClustering",
    "FunHDDCClustering",
    "FclustClustering",
    "KCentresClustering",
    "CurvclustClustering",
]
