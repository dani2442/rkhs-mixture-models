"""
Static metadata for benchmark methods and columns.

METHOD_REGISTRY defines each clustering method's applicable spaces,
computational complexity, benchmark participation, and citation metadata.

BENCHMARK_COLUMNS defines the dataset columns in the unified table.
"""

from collections import OrderedDict

# ──────────────────────────────────────────────────────────────────────
# Method registry: order here determines row order in the LaTeX table
# ──────────────────────────────────────────────────────────────────────

METHOD_REGISTRY = OrderedDict(
    {
        # --- Our methods ---
        "MMD GMM (Gaussian)": {
            "space": "Hilbert",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
            "citation_title": "Random Objects in Hilbert Spaces via Kernel Mixture Gaussian Model",
            "citation_doi": None,
        },
        "MMD GMM (Polynomial)": {
            "space": "Hilbert",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
            "citation_title": "Random Objects in Hilbert Spaces via Kernel Mixture Gaussian Model",
            "citation_doi": None,
        },
        # --- Metric-space competitors ---
        "K-Medoids": {
            "space": "Metric",
            "train": r"$O(n^2 K)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
            "citation_title": "Fast and Eager k-Medoids Clustering: O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms",
            "citation_doi": "10.1016/j.is.2021.101804",
        },
        "Hierarchical (Avg)": {
            "space": "Metric",
            "train": r"$O(n^2 \log n)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
            "citation_title": "Algorithms for Hierarchical Clustering: An Overview",
            "citation_doi": "10.1002/widm.53",
        },
        "DBSCAN": {
            "space": "Metric",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
            "citation_title": "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise",
            "citation_doi": None,
        },
        "HDBSCAN": {
            "space": "Metric",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
            "citation_title": "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection",
            "citation_doi": "10.1145/2733381",
        },
        "K-Center": {
            "space": "Metric",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(n)$",
            "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
            "citation_title": "Clustering to Minimize the Maximum Intercluster Distance",
            "citation_doi": "10.1016/0304-3975(85)90224-5",
        },
        # --- FDA methods (functional data only) ---
        "FDA K-Means": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "l2_real", "glucodensity", "skeleton"],
            "citation_title": "Some Methods of Classification and Analysis of Multivariate Observations",
            "citation_doi": None,
        },
        "FDA Fuzzy C-Means": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "l2_real", "glucodensity", "skeleton"],
            "citation_title": "Pattern Recognition with Fuzzy Objective Function Algorithms",
            "citation_doi": "10.1007/978-1-4757-0450-1",
        },
        "FDA Agglomerative": {
            "space": r"$L^2$",
            "train": r"$O(n^2 \log n)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["l2", "l2_real", "glucodensity", "skeleton"],
            "citation_title": "Algorithms for Hierarchical Clustering: An Overview",
            "citation_doi": "10.1002/widm.53",
        },
        "Funclust": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "l2_real", "glucodensity", "skeleton"],
            "citation_title": "Funclust: A Curves Clustering Method Using Functional Random Variable Density Approximation",
            "citation_doi": "10.1016/j.neucom.2012.11.042",
        },
        "FunHDDC": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "l2_real", "glucodensity", "skeleton"],
            "citation_title": "Model-Based Clustering of Time Series in Group-Specific Functional Subspaces",
            "citation_doi": "10.1007/s11634-011-0095-6",
        },
        "fclust": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "l2_real", "glucodensity", "skeleton"],
            "citation_title": "Clustering for Sparsely Sampled Functional Data",
            "citation_doi": "10.1198/016214503000189",
        },
        "K-Centres": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "l2_real", "glucodensity", "skeleton"],
            "citation_title": "Functional Clustering and Identifying Substructures of Longitudinal Data",
            "citation_doi": "10.1111/j.1467-9868.2007.00605.x",
        },
        "Curvclust": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "l2_real", "glucodensity", "skeleton"],
            "citation_title": "Wavelet-Based Clustering for Mixed-Effects Functional Models in High Dimension",
            "citation_doi": "10.1111/j.1541-0420.2012.01828.x",
        },
        # --- Feature-based competitors on projected coefficients/embeddings ---
        # "Feature K-Means": {
        #     "space": "Feature",
        #     "train": r"$O(nK)$",
        #     "infer": r"$O(nK)$",
        #     "memory": r"$O(nK)$",
        #     "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
        #     "citation_title": "Some Methods of Classification and Analysis of Multivariate Observations",
        #     "citation_doi": None,
        # },
        # "Feature GMM": {
        #     "space": "Feature",
        #     "train": r"$O(nK)$",
        #     "infer": r"$O(nK)$",
        #     "memory": r"$O(nK)$",
        #     "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
        #     "citation_title": "Model-Based Clustering, Discriminant Analysis, and Density Estimation",
        #     "citation_doi": "10.1198/016214502760047131",
        # },
        # "Feature HDDC": {
        #     "space": "Feature",
        #     "train": r"$O(nK)$",
        #     "infer": r"$O(nK)$",
        #     "memory": r"$O(nK)$",
        #     "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
        #     "citation_title": "High-Dimensional Data Clustering",
        #     "citation_doi": "10.1016/j.csda.2007.02.009",
        # },
        # "Mixt-PPCA": {
        #     "space": "Feature",
        #     "train": r"$O(nK)$",
        #     "infer": r"$O(nK)$",
        #     "memory": r"$O(nK)$",
        #     "benchmarks": ["rd", "l2", "l2_real", "glucodensity", "skeleton", "so3", "graph"],
        #     "citation_title": "Mixtures of Probabilistic Principal Component Analysers",
        #     "citation_doi": "10.1162/089976699300016728",
        # },
        # --- sklearn R^d only ---
        "MiniBatch KMeans": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(n)$",
            "benchmarks": ["rd"],
            "citation_title": "Web-Scale K-Means Clustering",
            "citation_doi": "10.1145/1772690.1772862",
        },
        "Spectral": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^3)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd"],
            "citation_title": "A Tutorial on Spectral Clustering",
            "citation_doi": "10.1007/s11222-007-9033-z",
        },
        "Ward": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd"],
            "citation_title": "Hierarchical Grouping to Optimize an Objective Function",
            "citation_doi": "10.1080/01621459.1963.10500845",
        },
        "Agglomerative (Avg)": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd"],
            "citation_title": "Algorithms for Hierarchical Clustering: An Overview",
            "citation_doi": "10.1002/widm.53",
        },
        "BIRCH": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(n)$",
            "benchmarks": ["rd"],
            "citation_title": "BIRCH: An Efficient Data Clustering Method for Very Large Databases",
            "citation_doi": "10.1145/233269.233324",
        },
        "Gaussian Mixture (EM)": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["rd"],
            "citation_title": "Maximum Likelihood from Incomplete Data via the EM Algorithm",
            "citation_doi": "10.1111/j.2517-6161.1977.tb01600.x",
        },
        "Affinity Propagation": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd"],
            "citation_title": "Clustering by Passing Messages Between Data Points",
            "citation_doi": "10.1126/science.1136800",
        },
        "MeanShift": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": r"$O(n)$",
            "memory": r"$O(n)$",
            "benchmarks": ["rd"],
            "citation_title": "Mean Shift, Mode Seeking, and Clustering",
            "citation_doi": "10.1109/34.400568",
        },
        "OPTICS": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n)$",
            "benchmarks": ["rd"],
            "citation_title": "OPTICS: Ordering Points To Identify the Clustering Structure",
            "citation_doi": "10.1145/304181.304187",
        },
    }
)

# ──────────────────────────────────────────────────────────────────────
# Benchmark columns: order here determines column order in the table
# ──────────────────────────────────────────────────────────────────────

BENCHMARK_COLUMNS = OrderedDict(
    {
        "rd": {
            "header": r"$\mathbb{R}^d$",
        },
        # "l2": {
        #     "header": r"$L^2$ Synth.",
        # },
        "l2_real": {
            "header": r"$L^2$",
        },
        "glucodensity": {
            "header": "Gluco.",
        },
        # "skeleton": {
        #     "header": "Skeleton",
        # },
        "so3": {
            "header": r"$\mathrm{SO}(3)$",
        },
        "graph": {
            "header": "Molecular",
        },
    }
)
