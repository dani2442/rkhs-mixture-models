"""
Static metadata for benchmark methods and columns.

METHOD_REGISTRY defines each clustering method's applicable spaces,
computational complexity, and which benchmarks it participates in.

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
            "benchmarks": ["rd", "l2", "glucodensity", "skeleton", "so3", "graph"],
        },
        "MMD GMM (Polynomial)": {
            "space": "Hilbert",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["rd", "l2", "glucodensity", "skeleton", "so3", "graph"],
        },
        # --- Metric-space competitors ---
        "K-Medoids": {
            "space": "Metric",
            "train": r"$O(n^2 K)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd", "l2", "glucodensity", "skeleton", "so3", "graph"],
        },
        "Hierarchical (Avg)": {
            "space": "Metric",
            "train": r"$O(n^2 \log n)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd", "l2", "glucodensity", "skeleton", "so3", "graph"],
        },
        "DBSCAN": {
            "space": "Metric",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd", "l2", "glucodensity", "skeleton", "so3", "graph"],
        },
        "HDBSCAN": {
            "space": "Metric",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd", "l2", "glucodensity", "skeleton", "so3", "graph"],
        },
        "K-Center": {
            "space": "Metric",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(n)$",
            "benchmarks": ["rd", "l2", "glucodensity", "skeleton", "so3", "graph"],
        },
        # --- FDA methods (functional data only) ---
        "FDA K-Means": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "glucodensity", "skeleton"],
        },
        "FDA Fuzzy C-Means": {
            "space": r"$L^2$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["l2", "glucodensity", "skeleton"],
        },
        "FDA Agglomerative": {
            "space": r"$L^2$",
            "train": r"$O(n^2 \log n)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["l2", "glucodensity", "skeleton"],
        },
        # --- sklearn R^d only ---
        "MiniBatch KMeans": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(n)$",
            "benchmarks": ["rd"],
        },
        "Spectral": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^3)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd"],
        },
        "Ward": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd"],
        },
        "Agglomerative (Avg)": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd"],
        },
        "BIRCH": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(n)$",
            "benchmarks": ["rd"],
        },
        "Gaussian Mixture (EM)": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(nK)$",
            "infer": r"$O(nK)$",
            "memory": r"$O(nK)$",
            "benchmarks": ["rd"],
        },
        "Affinity Propagation": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n^2)$",
            "benchmarks": ["rd"],
        },
        "MeanShift": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": r"$O(n)$",
            "memory": r"$O(n)$",
            "benchmarks": ["rd"],
        },
        "OPTICS": {
            "space": r"$\mathbb{R}^n$",
            "train": r"$O(n^2)$",
            "infer": "---",
            "memory": r"$O(n)$",
            "benchmarks": ["rd"],
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
        "l2": {
            "header": r"$L^2$ Synth.",
        },
        "glucodensity": {
            "header": "Gluco.",
        },
        "skeleton": {
            "header": "Skeleton",
        },
        "so3": {
            "header": r"$\mathrm{SO}(3)$",
        },
        "graph": {
            "header": "Molecular",
        },
    }
)
