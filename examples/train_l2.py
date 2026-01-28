#!/usr/bin/env python
"""
Example: Training a Gaussian Mixture Model on L² functional data.

This script demonstrates:
1. Generating synthetic functional data from a mixture model
2. Projecting the data onto an L² basis
3. Training a Gaussian mixture using MMD minimization
4. Visualizing the results: data, predicted means, and training history
"""
import torch
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmd import (
    L2CosineBasis,
    GaussianKernel,
    GaussianMixtureModel,
    fit_gaussian_mixture_mmd,
    generate_l2_sine_cosine_data,
    plot_l2_trajectories,
    plot_l2_means_comparison,
    plot_training_history,
    plot_mixture_weights,
)


def main():
    # ==========================================
    # Configuration
    # ==========================================
    torch.manual_seed(42)
    
    # Data parameters
    n_samples = 200
    n_components = 3
    grid_size = 100
    T = 1.0
    d = 2  # 2D functions
    noise_std = 0.15
    frequencies = [1.0, 2.0, 3.0]  # Different frequency for each component
    
    # Basis parameters
    R = 15  # Number of basis functions per dimension
    
    # Kernel parameters
    sigma_kernel = 2.0
    
    # Training parameters
    num_epochs = 200
    lr = 0.1
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print("=" * 60)
    print("L² Functional Data: MMD-based Gaussian Mixture Fitting")
    print("=" * 60)
    
    # ==========================================
    # Generate synthetic data
    # ==========================================
    print("\n[1] Generating synthetic data...")
    
    X_raw, true_assignments, info = generate_l2_sine_cosine_data(
        n_samples=n_samples,
        grid_size=grid_size,
        T=T,
        d=d,
        frequencies=frequencies,
        noise_std=noise_std,
        seed=42,
        device=device,
        dtype=dtype,
    )
    
    true_means = info["base_functions"]
    true_weights = info["component_weights"]
    t = info["t"]
    
    print(f"   Generated {n_samples} trajectories with {n_components} components")
    print(f"   Grid size: {grid_size}, Dimensions: {d}")
    print(f"   True weights: {true_weights.numpy()}")
    print(f"   Frequencies: {frequencies}")
    
    # ==========================================
    # Create basis and project data
    # ==========================================
    print("\n[2] Projecting data onto L² cosine basis...")
    
    basis = L2CosineBasis(
        T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype
    )
    
    X = basis.project(X_raw)  # (n_samples, M)
    M = X.shape[1]
    
    print(f"   Coefficient dimension M = {M} (R={R} × d={d})")
    
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
    
    # Figure 1: Data trajectories by component
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_l2_trajectories(
        X_raw, t=t, assignments=true_assignments,
        ax=axes1[0], dim=0, title="Dimension 1: x₀(t)"
    )
    plot_l2_trajectories(
        X_raw, t=t, assignments=true_assignments,
        ax=axes1[1], dim=1, title="Dimension 2: x₁(t)"
    )
    
    fig1.suptitle("Data Trajectories (colored by true component)", fontsize=14)
    plt.tight_layout()
    
    # Figure 2: Data vs Predicted means (dimension 0)
    fig2 = plot_l2_means_comparison(
        X=X_raw,
        predicted_means=model.mean,
        basis=basis,
        t=t,
        true_means=true_means,
        assignments=true_assignments,
        predicted_weights=model.pi,
        dim=0,
        title="Dimension 1: Data vs Means",
    )
    
    # Figure 3: Data vs Predicted means (dimension 1)
    fig3 = plot_l2_means_comparison(
        X=X_raw,
        predicted_means=model.mean,
        basis=basis,
        t=t,
        true_means=true_means,
        assignments=true_assignments,
        predicted_weights=model.pi,
        dim=1,
        title="Dimension 2: Data vs Means",
    )
    
    # Figure 4: Training history and weights
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_training_history(history, ax=axes4[0], title="MMD² Training History")
    plot_mixture_weights(
        model.pi, true_weights=true_weights, ax=axes4[1],
        title="Mixture Weights: Predicted vs True"
    )
    
    fig4.suptitle("Training Summary", fontsize=14)
    plt.tight_layout()
    
    # Figure 5: Sample from fitted model
    print("\n[6] Sampling from fitted model...")
    
    samples_functions, samples_coeffs, samples_assignments = model.sample_functions(20)
    
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_l2_trajectories(
        samples_functions, t=t, assignments=samples_assignments,
        ax=axes5[0], dim=0, title="Sampled: Dimension 1"
    )
    plot_l2_trajectories(
        samples_functions, t=t, assignments=samples_assignments,
        ax=axes5[1], dim=1, title="Sampled: Dimension 2"
    )
    
    fig5.suptitle("Samples from Fitted Gaussian Mixture", fontsize=14)
    plt.tight_layout()
    
    # Save figures as PDF for paper
    print("\n[7] Saving figures as PDF...")
    paper_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "images")
    os.makedirs(paper_images_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(paper_images_dir, "l2_trajectories.pdf"), format="pdf", bbox_inches="tight")
    fig2.savefig(os.path.join(paper_images_dir, "l2_means_dim0.pdf"), format="pdf", bbox_inches="tight")
    fig4.savefig(os.path.join(paper_images_dir, "l2_training_summary.pdf"), format="pdf", bbox_inches="tight")
    fig5.savefig(os.path.join(paper_images_dir, "l2_samples.pdf"), format="pdf", bbox_inches="tight")
    print(f"   Saved figures to {paper_images_dir}")

    print("\n[8] Displaying plots...")


if __name__ == "__main__":
    main()
