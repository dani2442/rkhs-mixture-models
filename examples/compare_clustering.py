#!/usr/bin/env python
"""
Example: Comparing clustering methods on L2 functional data over multiple datasets.
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
from examples.train_l2_gaussian import generate_n_datasets
from src.competitors import (
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
    n_datasets = 10
    n_samples = 500
    n_components = 5
    grid_size = 100
    T = 1.0
    d = 2  # 2D functions
    frequencies = [1.0, 2.0, 3.0, 4.0, 5.0]
    noise_std = 0.15
    
    # Basis parameters
    R = 15  # Number of basis functions per dimension

    # Kernel parameters
    sigma_kernel = 1.2

    # Training parameters
    num_epochs = 500
    lr = 0.1
    
    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 60)
    print(f"Clustering Comparison on L2 Functional Data ({n_datasets} datasets)")
    print("=" * 60)

    # ==========================================
    # Generate synthetic datasets
    # ==========================================
    print(f"\n[1] Generating {n_datasets} synthetic datasets...")
    true_weights = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10], device=device, dtype=dtype)
    datasets = generate_n_datasets(
        n_datasets=n_datasets,
        n_samples=n_samples,
        n_components=n_components,
        grid_size=grid_size,
        R=R,
        T=T,
        d=d,
        component_weights=true_weights,
        base_seed=42,
        device=device,
        dtype=dtype,
    )

    # We will accumulate ARIs for each method across datasets
    ari_results = {
        "Ours (MMD GMM)": [],
        "K-Medoids": [],
        "Hierarchical": [],
        "DBSCAN": [],
        "HDBSCAN": [],
        "K-Center": [],
    }
    
    # Add FDA methods
    try:
        # Just instantiate to see if scikit-fda is available
        _ = ScikitFDAKMeans(n_clusters=n_components)
        ari_results.update({
            "FDA K-Means": [],
            "FDA Fuzzy C-Means": [],
            "FDA Agglomerative": [],
        })
        use_fda = True
    except Exception:
        print("Scikit-FDA methods omitted")
        use_fda = False

    for i, (X_raw, X_coeffs, true_assignments, info) in enumerate(datasets):
        print(f"\n--- Processing Dataset {i+1}/{n_datasets} ---")
        true_assignments = true_assignments.cpu().numpy()

        # ==========================================
        # Create basis and project data
        # ==========================================
        basis = L2CosineBasis(
            T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype
        )
        
        X = basis.project(X_raw)  # (n_samples, M)
        M = X.shape[1]
        
        # ==========================================
        # Evaluate Ours (MMD Gaussian Mixture Model)
        # ==========================================
        print("Training our MMD Gaussian Mixture Model...")
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
        ari_results["Ours (MMD GMM)"].append(our_ari)

        # ==========================================
        # Evaluate Competitors
        # ==========================================
        print("Evaluating competitor models...")
        competitors = [
            ("K-Medoids", KMedoidsClustering(n_clusters=n_components)),
            ("Hierarchical", HierarchicalClustering(n_clusters=n_components, linkage="average")),
            ("DBSCAN", DBSCANClustering(eps=1.5, min_samples=5)),
            ("HDBSCAN", HDBSCANClustering(min_cluster_size=5)),
            ("K-Center", KCenterClustering(n_clusters=n_components)),
        ]

        if use_fda:
            competitors.extend([
                ("FDA K-Means", ScikitFDAKMeans(n_clusters=n_components)),
                ("FDA Fuzzy C-Means", ScikitFDAFuzzyCMeans(n_clusters=n_components)),
                ("FDA Agglomerative", ScikitFDAAgglomerative(n_clusters=n_components, linkage="average")),
            ])

        for name, method in competitors:
            try:
                # FDA methods take X and basis
                if "FDA" in name:
                    labels = method.fit_predict(X, basis=basis)
                else:
                    labels = method.fit_predict(X)
                
                ari = adjusted_rand_score(true_assignments, labels)
                print(f"{name} ARI: {ari:.4f}")
                ari_results[name].append(ari)
            except Exception as e:
                print(f"{name} failed: {e}")
                ari_results[name].append(np.nan)

    # ==========================================
    # Generate Outputs
    # ==========================================
    print("\n[5] Generating plot and table...")
    
    # Calculate statistics and filter valid results
    method_stats = []
    for name, aris in ari_results.items():
        valid_aris = [a for a in aris if not np.isnan(a)]
        if len(valid_aris) > 0:
            mean_ari = np.mean(valid_aris)
            std_ari = np.std(valid_aris)
            method_stats.append({
                "Method": name, 
                "Mean": mean_ari, 
                "Std": std_ari, 
                "ARIs": valid_aris
            })
            
    # Sort results by Mean ARI descending
    method_stats.sort(key=lambda x: x["Mean"], reverse=True)
    
    methods = [r["Method"] for r in method_stats]
    plot_data = [r["ARIs"] for r in method_stats]
    
    paper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
    images_dir = os.path.join(paper_dir, "images")
    sections_dir = os.path.join(paper_dir, "sections")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sections_dir, exist_ok=True)

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box = ax.boxplot(plot_data, tick_labels=methods, patch_artist=True)
    
    # Customize colors
    for patch in box['boxes']:
        patch.set_facecolor('skyblue')
    for median in box['medians']:
        median.set(color='red', linewidth=2)
        
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title(f"Clustering Performance Comparison ({n_datasets} Datasets)")
    ax.set_ylim(-0.1, 1.1)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plot_path = os.path.join(images_dir, "clustering_comparison_boxplot.pdf")
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    # LaTeX Table
    table_str = "\\begin{table}[htpb]\n\\centering\n"
    table_str += f"\\caption{{Clustering performance comparison over {n_datasets} datasets based on Adjusted Rand Index (ARI).}}\n"
    table_str += "\\label{tab:clustering_comparison}\n"
    table_str += "\\begin{tabular}{lc}\n\\toprule\n"
    table_str += "Method & ARI (Mean $\\pm$ Std) \\\\\n\\midrule\n"
    for r in method_stats:
        table_str += f"{r['Method']} & {r['Mean']:.3f} $\\pm$ {r['Std']:.3f} \\\\\n"
    table_str += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    table_path = os.path.join(sections_dir, "clustering_comparison_table.tex")
    with open(table_path, "w") as f:
        f.write(table_str)
    
    print(f"Saved LaTeX table to {table_path}")

if __name__ == "__main__":
    main()
