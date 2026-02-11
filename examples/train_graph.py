#!/usr/bin/env python
"""
Example: Training a Gaussian Mixture Model on graph signal data.

This script demonstrates:
1. Generating synthetic graph signals from a mixture model
2. Projecting the data onto a graph Laplacian eigenbasis
3. Training a Gaussian mixture using MMD minimization
4. Visualizing the results: graph signals, predicted means, and training history
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    GraphLaplacianBasis,
    GaussianKernel,
    GaussianMixtureModel,
    fit_gaussian_mixture_mmd,
    generate_graph_mixture_data,
    plot_graph_signals,
    plot_graph_means_comparison,
    plot_training_history,
    plot_mixture_weights,
)

# Check for networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Some visualizations will be limited.")
    print("Install with: pip install networkx")


def main():
    # ==========================================
    # Configuration
    # ==========================================
    torch.manual_seed(42)
    
    # Graph parameters
    num_nodes = 30
    edge_probability = 0.25
    
    # Data parameters
    n_samples = 150
    n_components = 3
    signal_std = 1.5
    noise_std = 0.2
    
    # Basis parameters
    num_eigenvectors = 15  # Use first 15 Laplacian eigenvectors
    alpha = 0.1  # Regularization for inner product
    
    # Kernel parameters
    sigma_kernel = 2.0
    
    # Training parameters
    num_epochs = 200
    lr = 0.1
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print("=" * 60)
    print("Graph Signals: MMD-based Gaussian Mixture Fitting")
    print("=" * 60)
    
    # ==========================================
    # Generate synthetic data
    # ==========================================
    print("\n[1] Generating synthetic graph and signals...")
    
    X_raw, true_assignments, adjacency, info = generate_graph_mixture_data(
        n_samples=n_samples,
        n_components=n_components,
        num_nodes=num_nodes,
        edge_probability=edge_probability,
        signal_std=signal_std,
        noise_std=noise_std,
        seed=42,
        device=device,
        dtype=dtype,
    )
    
    true_means = info["base_signals"]
    true_weights = info["component_weights"]
    
    # Count edges
    num_edges = int(adjacency.sum().item() / 2)
    
    print(f"   Graph: {num_nodes} nodes, {num_edges} edges")
    print(f"   Generated {n_samples} signals with {n_components} components")
    print(f"   True weights: {true_weights.numpy()}")
    
    # ==========================================
    # Create basis and project data
    # ==========================================
    print("\n[2] Creating Graph Laplacian basis...")
    
    basis = GraphLaplacianBasis.from_adjacency(
        adjacency=adjacency,
        alpha=alpha,
        num_eigenvectors=num_eigenvectors,
        device=device,
        dtype=dtype,
    )
    
    X = basis.project(X_raw)  # (n_samples, M)
    M = X.shape[1]
    
    print(f"   Using {num_eigenvectors} Laplacian eigenvectors")
    print(f"   Coefficient dimension M = {M}")
    print(f"   Eigenvalue range: [{basis.eigenvalues.min():.3f}, {basis.eigenvalues.max():.3f}]")
    
    # ==========================================
    # Create kernel and model
    # ==========================================
    print("\n[3] Setting up model and kernel...")
    
    kernel = GaussianKernel(sigma=sigma_kernel)
    
    model = GaussianMixtureModel(
        num_components=n_components,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )
    
    # Initialize from data
    model.initialize_from_data(X, method="kmeans++")
    
    print(f"   Kernel: Gaussian (σ={sigma_kernel})")
    print(f"   Model: {model}")
    print(f"   Initial weights: {model.pi.detach().numpy()}")
    
    # ==========================================
    # Train the model
    # ==========================================
    print("\n[4] Training model by minimizing MMD²...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Precompute constant term
    gram = kernel.compute_gram_matrix(X)
    const_term = gram.mean()
    
    history = []
    print_interval = num_epochs // 10
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        mmd2, stats = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()
        
        full_mmd2 = const_term + mmd2.detach()
        history.append(full_mmd2.item())
        
        if (epoch + 1) % print_interval == 0:
            print(
                f"   Epoch {epoch+1:4d}/{num_epochs}: "
                f"MMD² = {full_mmd2.item():.6f}, "
                f"π = {model.pi.detach().numpy()}"
            )
    
    print(f"\n   Final MMD² = {history[-1]:.6f}")
    print(f"   Final weights: {model.pi.detach().numpy()}")
    
    # ==========================================
    # Visualize results
    # ==========================================
    print("\n[5] Creating visualizations...")
    
    if HAS_NETWORKX:
        # Compute graph layout once
        adj_np = adjacency.detach().cpu().numpy()
        G = nx.from_numpy_array(adj_np)
        node_positions = nx.spring_layout(G, seed=42)
        
        # Figure 1: Sample signals from each component
        fig1, axes1 = plt.subplots(1, n_components, figsize=(5 * n_components, 5))
        
        for k in range(n_components):
            # Find a sample from component k
            mask = true_assignments == k
            if mask.any():
                idx = mask.nonzero()[0][0].item()
                plot_graph_signals(
                    X_raw, adjacency,
                    assignments=true_assignments,
                    node_positions=node_positions,
                    ax=axes1[k],
                    sample_idx=idx,
                    title=f"Component {k+1} Sample",
                )
        
        fig1.suptitle("Sample Signals from Each Component", fontsize=14)
        plt.tight_layout()
        
        # Figure 2: True vs Predicted means
        fig2 = plot_graph_means_comparison(
            X=X_raw,
            predicted_means=model.mean,
            adjacency=adjacency,
            basis=basis,
            true_means=true_means,
            predicted_weights=model.pi,
            node_positions=node_positions,
        )
    
    # Figure 3: Training history and weights (works without networkx)
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_training_history(history, ax=axes3[0], title="MMD² Training History")
    plot_mixture_weights(
        model.pi, true_weights=true_weights, ax=axes3[1],
        title="Mixture Weights: Predicted vs True"
    )
    
    fig3.suptitle("Training Summary", fontsize=14)
    plt.tight_layout()
    
    # Figure 4: Signal distribution per component
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot signal magnitudes per component
    ax = axes4[0]
    for k in range(n_components):
        mask = true_assignments == k
        if mask.any():
            magnitudes = X_raw[mask].norm(dim=1).detach().cpu().numpy()
            ax.hist(magnitudes, bins=20, alpha=0.5, label=f"Component {k+1}")
    ax.set_xlabel("Signal Magnitude ||x||")
    ax.set_ylabel("Count")
    ax.set_title("Signal Magnitude Distribution by Component")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot coefficient magnitudes
    ax = axes4[1]
    for k in range(n_components):
        mask = true_assignments == k
        if mask.any():
            coeff_mags = X[mask].norm(dim=1).detach().cpu().numpy()
            ax.hist(coeff_mags, bins=20, alpha=0.5, label=f"Component {k+1}")
    ax.set_xlabel("Coefficient Magnitude ||c||")
    ax.set_ylabel("Count")
    ax.set_title("Coefficient Magnitude Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig4.suptitle("Data Distribution Analysis", fontsize=14)
    plt.tight_layout()
    
    # Figure 5: Eigenvalue spectrum
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    eigenvalues = basis.eigenvalues.detach().cpu().numpy()
    ax5.bar(range(len(eigenvalues)), eigenvalues, color="steelblue", alpha=0.7)
    ax5.set_xlabel("Eigenvalue Index")
    ax5.set_ylabel("λ")
    ax5.set_title("Graph Laplacian Eigenvalue Spectrum (First {} Eigenvalues)".format(num_eigenvectors))
    ax5.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    # Save figures as PDF for paper
    print("\n[6] Saving figures as PDF...")
    paper_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "images")
    os.makedirs(paper_images_dir, exist_ok=True)
    
    if HAS_NETWORKX:
        fig1.savefig(os.path.join(paper_images_dir, "graph_samples.pdf"), format="pdf", bbox_inches="tight")
        fig2.savefig(os.path.join(paper_images_dir, "graph_means_comparison.pdf"), format="pdf", bbox_inches="tight")
    fig3.savefig(os.path.join(paper_images_dir, "graph_training_summary.pdf"), format="pdf", bbox_inches="tight")
    print(f"   Saved figures to {paper_images_dir}")

    print("\n[7] Displaying plots...")
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
