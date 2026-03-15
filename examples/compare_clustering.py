#!/usr/bin/env python
"""
Example: Comparing clustering methods on L2 functional data.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import adjusted_rand_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    L2CosineBasis,
    GaussianKernel,
    GaussianMixtureModel,
)
from examples.train_l2_gaussian import generate_l2_gaussian_data
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

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ==========================================
    # Configuration
    # ==========================================
    # Data parameters
    n_samples = 500
    n_components = 5
    grid_size = 100
    T = 1.0
    d = 2  # 2D functions
    
    # Basis parameters
    R = 15  # Number of basis functions per dimension

    # Kernel parameters
    sigma_kernel = 1.2

    # Training parameters
    num_epochs = 200
    lr = 0.1
    
    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 60)
    print("Clustering Comparison on L2 Functional Data")
    print("=" * 60)

    # ==========================================
    # Generate synthetic data
    # ==========================================
    print("\n[1] Generating synthetic data...")
    true_weights = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10], device=device, dtype=dtype)
    X_raw, X_coeffs, true_assignments, info = generate_l2_gaussian_data(
        n_samples=n_samples,
        n_components=n_components,
        grid_size=grid_size,
        R=R,
        T=T,
        d=d,
        component_weights=true_weights,
        seed=42,
        device=device,
        dtype=dtype,
    )
    true_assignments = true_assignments.cpu().numpy()

    # ==========================================
    # Create basis and project data
    # ==========================================
    print("\n[2] Projecting data onto L2 cosine basis...")
    basis = L2CosineBasis(
        T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype
    )
    
    X = basis.project(X_raw)  # (n_samples, M)
    M = X.shape[1]
    
    # ==========================================
    # Evaluate Ours (MMD Gaussian Mixture Model)
    # ==========================================
    print("\n[3] Training our MMD Gaussian Mixture Model...")
    kernel = GaussianKernel(sigma=sigma_kernel)
    model = GaussianMixtureModel(
        num_components=n_components,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )
    model.initialize_from_data(X, method="kmeans++")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        mmd2, _ = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()
        
    with torch.no_grad():
        resp = model.responsibilities(X) # (n, K)
        our_labels = resp.argmax(dim=1).cpu().numpy()
        
    our_ari = adjusted_rand_score(true_assignments, our_labels)
    print(f"Ours (MMD GMM) ARI: {our_ari:.4f}")

    # ==========================================
    # Evaluate Competitors
    # ==========================================
    print("\n[4] Evaluating competitor models...")
    competitors = [
        ("K-Medoids", KMedoidsClustering(n_clusters=n_components)),
        ("Hierarchical", HierarchicalClustering(n_clusters=n_components, linkage="average")),
        ("DBSCAN", DBSCANClustering(eps=1.5, min_samples=5)),
        ("HDBSCAN", HDBSCANClustering(min_cluster_size=5)),
        ("K-Center", KCenterClustering(n_clusters=n_components)),
    ]

    # FDA methods if available
    try:
        competitors.extend([
            ("FDA K-Means", ScikitFDAKMeans(n_clusters=n_components)),
            ("FDA Fuzzy C-Means", ScikitFDAFuzzyCMeans(n_clusters=n_components)),
            ("FDA Agglomerative", ScikitFDAAgglomerative(n_clusters=n_components, linkage="average")),
        ])
    except Exception as e:
        print("Scikit-FDA methods omitted")

    results = []
    results.append({"Method": "Ours (MMD GMM)", "ARI": our_ari})

    for name, method in competitors:
        try:
            # FDA methods take X and basis
            if "FDA" in name:
                labels = method.fit_predict(X, basis=basis)
            else:
                labels = method.fit_predict(X)
            
            ari = adjusted_rand_score(true_assignments, labels)
            print(f"{name} ARI: {ari:.4f}")
            results.append({"Method": name, "ARI": ari})
        except Exception as e:
            print(f"{name} failed: {e}")
            results.append({"Method": name, "ARI": np.nan})

    # ==========================================
    # Generate Outputs
    # ==========================================
    print("\n[5] Generating plot and table...")
    # Sort results by ARI descending, filter out nan
    valid_results = [r for r in results if not np.isnan(r["ARI"])]
    valid_results.sort(key=lambda x: x["ARI"], reverse=True)
    
    methods = [r["Method"] for r in valid_results]
    aris = [r["ARI"] for r in valid_results]
    
    paper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
    images_dir = os.path.join(paper_dir, "images")
    sections_dir = os.path.join(paper_dir, "sections")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sections_dir, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, aris, color='skyblue', edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("Clustering Performance Comparison")
    ax.set_ylim(-0.1, 1.1)
    plt.xticks(rotation=45, ha='right')

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, max(height, 0)),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(images_dir, "clustering_comparison.pdf")
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    # LaTeX Table
    table_str = "\\begin{table}[htpb]\n\\centering\n"
    table_str += "\\caption{Clustering performance comparison based on Adjusted Rand Index (ARI).}\n"
    table_str += "\\label{tab:clustering_comparison}\n"
    table_str += "\\begin{tabular}{lc}\n\\toprule\n"
    table_str += "Method & ARI \\\\\n\\midrule\n"
    for r in valid_results:
        table_str += f"{r['Method']} & {r['ARI']:.3f} \\\\\n"
    table_str += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    table_path = os.path.join(sections_dir, "clustering_comparison_table.tex")
    with open(table_path, "w") as f:
        f.write(table_str)
    
    print(f"Saved LaTeX table to {table_path}")

if __name__ == "__main__":
    main()
