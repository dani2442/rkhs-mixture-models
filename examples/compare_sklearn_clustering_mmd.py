#!/usr/bin/env python
"""
Compare scikit-learn clustering algorithms with MMD-based clustering.

This example mirrors scikit-learn's toy benchmark and adds one extra column:
an MMD-trained Gaussian mixture using a finite-dimensional quadratic kernel.
"""
import argparse
import os
import sys
import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import cluster, datasets, mixture
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuadraticKernel, fit_gaussian_mixture_mmd


def make_toy_datasets(n_samples: int, seed: int):
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
    )
    noisy_moons = datasets.make_moons(
        n_samples=n_samples, noise=0.05, random_state=seed
    )
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    rng = np.random.RandomState(seed)
    no_structure = rng.rand(n_samples, 2), None

    random_state = 170
    X_aniso, y_aniso = datasets.make_blobs(
        n_samples=n_samples, random_state=random_state
    )
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    aniso = (X_aniso, y_aniso)

    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 3,
        "n_clusters": 3,
        "min_samples": 7,
        "xi": 0.05,
        "min_cluster_size": 0.1,
        "allow_single_cluster": True,
        "hdbscan_min_cluster_size": 15,
        "hdbscan_min_samples": 3,
        "random_state": 42,
    }

    dataset_specs = [
        (
            "Noisy circles",
            noisy_circles,
            {
                "damping": 0.77,
                "preference": -240,
                "quantile": 0.2,
                "n_clusters": 2,
                "min_samples": 7,
                "xi": 0.08,
            },
        ),
        (
            "Noisy moons",
            noisy_moons,
            {
                "damping": 0.75,
                "preference": -220,
                "n_clusters": 2,
                "min_samples": 7,
                "xi": 0.1,
            },
        ),
        (
            "Varied",
            varied,
            {
                "eps": 0.18,
                "n_neighbors": 2,
                "min_samples": 7,
                "xi": 0.01,
                "min_cluster_size": 0.2,
            },
        ),
        (
            "Anisotropic",
            aniso,
            {
                "eps": 0.15,
                "n_neighbors": 2,
                "min_samples": 7,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        (
            "Blobs",
            blobs,
            {
                "min_samples": 7,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        ("No structure", no_structure, {}),
    ]

    merged_specs = []
    for name, dataset_tuple, algo_params in dataset_specs:
        params = default_base.copy()
        params.update(algo_params)
        merged_specs.append((name, dataset_tuple, params))

    return merged_specs


def run_mmd_clustering(
    X: np.ndarray,
    n_clusters: int,
    kernel_c: float,
    mmd_epochs: int,
    mmd_lr: float,
    mmd_seed: int,
) -> np.ndarray:
    torch.manual_seed(mmd_seed)
    X_torch = torch.as_tensor(X, dtype=torch.float64)
    kernel = QuadraticKernel(c=kernel_c)

    model, _ = fit_gaussian_mixture_mmd(
        X=X_torch,
        num_components=n_clusters,
        kernel=kernel,
        num_epochs=mmd_epochs,
        lr=mmd_lr,
        covariance_type="spherical",
        init_method="kmeans++",
        verbose=False,
    )

    with torch.no_grad():
        y_pred = model.responsibilities(X_torch).argmax(dim=1)

    return y_pred.cpu().numpy().astype(int)


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    palette = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]
    labels = labels.astype(int)
    unique_labels = [label for label in np.unique(labels) if label != -1]
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    color_map[-1] = "#000000"
    return np.array([color_map.get(label, "#000000") for label in labels], dtype=object)


def benchmark(args: argparse.Namespace):
    dataset_specs = make_toy_datasets(n_samples=args.n_samples, seed=args.seed)
    has_hdbscan = hasattr(cluster, "HDBSCAN")

    n_cols = 12 if has_hdbscan else 11
    fig = plt.figure(figsize=(n_cols * 2.1 + 1.0, len(dataset_specs) * 2.1 + 1.0))
    plt.subplots_adjust(
        left=0.03,
        right=0.985,
        bottom=0.02,
        top=0.95,
        wspace=0.05,
        hspace=0.05,
    )

    print("=" * 80)
    print("Scikit-learn toy clustering benchmark + MMD (Quadratic finite-dimensional kernel)")
    print("=" * 80)
    print(f"n_samples={args.n_samples}, mmd_epochs={args.mmd_epochs}, mmd_lr={args.mmd_lr}")
    print(f"kernel=QuadraticKernel(c={args.kernel_c})")
    if not has_hdbscan:
        print("HDBSCAN is unavailable in this scikit-learn version; skipping that column.")

    results = []
    plot_num = 1

    for i_dataset, (dataset_name, dataset_tuple, params) in enumerate(dataset_specs):
        X, y_true = dataset_tuple
        X = StandardScaler().fit_transform(X)

        bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])
        connectivity = kneighbors_graph(
            X, n_neighbors=params["n_neighbors"], include_self=False
        )
        connectivity = 0.5 * (connectivity + connectivity.T)

        two_means = cluster.MiniBatchKMeans(
            n_clusters=params["n_clusters"],
            n_init="auto",
            random_state=params["random_state"],
        )
        affinity_propagation = cluster.AffinityPropagation(
            damping=params["damping"],
            preference=params["preference"],
            random_state=params["random_state"],
        )
        mean_shift = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        spectral = cluster.SpectralClustering(
            n_clusters=params["n_clusters"],
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            random_state=params["random_state"],
        )
        ward = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
        )
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average",
            metric="cityblock",
            n_clusters=params["n_clusters"],
            connectivity=connectivity,
        )
        dbscan = cluster.DBSCAN(eps=params["eps"])
        optics = cluster.OPTICS(
            min_samples=params["min_samples"],
            xi=params["xi"],
            min_cluster_size=params["min_cluster_size"],
        )
        birch = cluster.Birch(n_clusters=params["n_clusters"])
        gmm = mixture.GaussianMixture(
            n_components=params["n_clusters"],
            covariance_type="full",
            random_state=params["random_state"],
        )

        algorithm_list = [
            ("MiniBatch\nKMeans", two_means),
            ("Affinity\nPropagation", affinity_propagation),
            ("MeanShift", mean_shift),
            ("Spectral\nClustering", spectral),
            ("Ward", ward),
            ("Agglomerative\nClustering", average_linkage),
            ("DBSCAN", dbscan),
        ]

        if has_hdbscan:
            hdbscan = cluster.HDBSCAN(
                min_samples=params["hdbscan_min_samples"],
                min_cluster_size=params["hdbscan_min_cluster_size"],
                allow_single_cluster=params["allow_single_cluster"],
            )
            algorithm_list.append(("HDBSCAN", hdbscan))

        algorithm_list.extend(
            [
                ("OPTICS", optics),
                ("BIRCH", birch),
                ("Gaussian\nMixture", gmm),
                ("MMD\nQuadratic", None),
            ]
        )

        for name, algorithm in algorithm_list:
            t0 = time.perf_counter()

            if name == "MMD\nQuadratic":
                y_pred = run_mmd_clustering(
                    X=X,
                    n_clusters=params["n_clusters"],
                    kernel_c=args.kernel_c,
                    mmd_epochs=args.mmd_epochs,
                    mmd_lr=args.mmd_lr,
                    mmd_seed=args.mmd_seed,
                )
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the "
                        + "connectivity matrix is [0-9]{1,2}"
                        + " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding"
                        + " may not work as expected.",
                        category=UserWarning,
                    )
                    algorithm.fit(X)

                if hasattr(algorithm, "labels_"):
                    y_pred = algorithm.labels_.astype(int)
                else:
                    y_pred = algorithm.predict(X)

            elapsed = time.perf_counter() - t0
            ari: Optional[float] = None
            if y_true is not None:
                ari = float(adjusted_rand_score(y_true, y_pred))

            ax = plt.subplot(len(dataset_specs), len(algorithm_list), plot_num)
            ax.scatter(X[:, 0], X[:, 1], s=10, color=labels_to_colors(np.asarray(y_pred)))
            ax.set_facecolor("#f0f0f0")
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_xticks(())
            ax.set_yticks(())

            if i_dataset == 0:
                ax.set_title(name, size=12)
            if name.startswith("MiniBatch"):
                ax.set_ylabel(dataset_name, size=10)

            ax.text(
                0.99,
                0.01,
                f"{elapsed:.2f}s".lstrip("0"),
                transform=ax.transAxes,
                size=9,
                horizontalalignment="right",
            )
            if ari is not None:
                ax.text(
                    0.01,
                    0.01,
                    f"ARI {ari:.2f}",
                    transform=ax.transAxes,
                    size=8,
                    horizontalalignment="left",
                )

            results.append(
                {
                    "dataset": dataset_name,
                    "algorithm": name.replace("\n", " "),
                    "time_s": elapsed,
                    "ari": ari,
                }
            )
            plot_num += 1

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        fig.savefig(args.save_path, format="pdf", bbox_inches="tight")
        print(f"\nSaved comparison figure to {args.save_path}")

    valid_results = [r for r in results if r["ari"] is not None]
    algorithms = []
    for r in results:
        if r["algorithm"] not in algorithms:
            algorithms.append(r["algorithm"])

    print("\nSummary (mean ARI over labeled datasets, mean runtime over all datasets):")
    for algo in algorithms:
        ari_values = [r["ari"] for r in valid_results if r["algorithm"] == algo]
        times = [r["time_s"] for r in results if r["algorithm"] == algo]
        mean_ari = float(np.mean(ari_values)) if ari_values else float("nan")
        mean_time = float(np.mean(times))
        print(f"  {algo:<24} ARI={mean_ari:6.3f}   time={mean_time:6.3f}s")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MMD clustering with scikit-learn clustering baselines."
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--mmd-seed", type=int, default=42)
    parser.add_argument("--mmd-epochs", type=int, default=200)
    parser.add_argument("--mmd-lr", type=float, default=0.05)
    parser.add_argument("--kernel-c", type=float, default=1.0)
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper",
            "images",
            "sklearn_clustering_mmd_comparison.pdf",
        ),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
