#!/usr/bin/env python
"""
Example: Training a Gaussian Mixture Model on L² functional data generated
from true Gaussian distributions in coefficient space.

This script demonstrates:
1. Generating synthetic functional data from Gaussian distributions with
   complex/convoluted mean functions and varying covariances
2. Projecting the data onto an L² basis
3. Training a Gaussian mixture model (means AND covariances) using MMD minimization
4. Visualizing the results: data, predicted means, covariances, and training history
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    L2CosineBasis,
    GaussianKernel,
    GaussianMixtureModel,
    plot_l2_trajectories,
    plot_l2_means_comparison,
    plot_training_history,
    plot_mixture_weights,
)


def generate_l2_gaussian_data(
    n_samples: int,
    n_components: int,
    grid_size: int,
    R: int,
    T: float = 1.0,
    d: int = 2,
    component_weights: torch.Tensor = None,
    seed: int = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
):
    """
    Generate L² functional data from true Gaussian distributions in coefficient space.
    
    Each component k has:
    - A complex mean function (combination of harmonics)
    - A distinct covariance structure (diagonal with varying scales)
    
    Args:
        n_samples: Number of samples to generate
        n_components: Number of mixture components
        grid_size: Number of time discretization points
        R: Number of basis functions per dimension
        T: End of time interval [0, T]
        d: Spatial dimension (R^d valued functions)
        component_weights: Mixture weights, shape (n_components,)
        seed: Random seed for reproducibility
        device: Torch device
        dtype: Torch dtype
    
    Returns:
        X_raw: Generated trajectories, shape (n_samples, grid_size, d)
        X_coeffs: Coefficient representation, shape (n_samples, M)
        assignments: True component assignments, shape (n_samples,)
        info: Dictionary with ground truth parameters
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    M = R * d  # Total coefficient dimension
    t = torch.linspace(0, T, grid_size, device=device, dtype=dtype)
    
    # Create basis for reconstruction
    basis = L2CosineBasis(T=T, R=R, grid_size=grid_size, d=d, device=device, dtype=dtype)
    
    # Default uniform weights
    if component_weights is None:
        component_weights = torch.ones(n_components, device=device, dtype=dtype) / n_components
    
    # Sample component assignments
    assignments = torch.multinomial(
        component_weights.expand(n_samples, -1), num_samples=1
    ).squeeze(-1)
    
    # =========================================
    # Define complex mean coefficients for each component
    # Each mean is a combination of multiple harmonics with different phases
    # =========================================
    true_mean_coeffs = torch.zeros(n_components, M, device=device, dtype=dtype)
    
    for k in range(n_components):
        for dim in range(d):
            base_idx = dim * R
            # Create distinct and abnormal patterns for each component
            
            for r in range(R):
                freq_factor = (r + 1) / R
                
                # Highly varied amplitude profiles per component
                if k == 0:
                    # Sharp spike at low frequency
                    amplitude = 4.0 * np.exp(-5.0 * freq_factor) + 0.1
                elif k == 1:
                    # Bimodal: peaks at low and high frequencies
                    amplitude = 2.0 * (np.exp(-10.0 * freq_factor) + np.exp(-10.0 * (1 - freq_factor)))
                elif k == 2:
                    # Oscillating with decay
                    amplitude = 2.5 * np.abs(np.sin(4 * np.pi * freq_factor)) * np.exp(-1.5 * freq_factor)
                elif k == 3:
                    # Sawtooth-like pattern
                    amplitude = 3.0 * (freq_factor % 0.3) / 0.3 * np.exp(-2.0 * freq_factor)
                elif k == 4:
                    # Narrow band-pass (very peaked)
                    amplitude = 3.5 * np.exp(-25.0 * (freq_factor - 0.3) ** 2)
                elif k == 5:
                    # Inverted: high frequencies emphasized
                    amplitude = 2.0 * freq_factor ** 1.5 + 0.3
                elif k == 6:
                    # Step function approximation
                    amplitude = 2.5 if freq_factor < 0.5 else 0.5
                else:  # k == 7
                    # Chaotic/irregular pattern
                    amplitude = 2.0 * np.abs(np.sin(7 * np.pi * freq_factor + k)) * (1 + 0.5 * np.cos(3 * np.pi * freq_factor))
                
                # Different phase patterns per component to create abnormal shapes
                if k % 2 == 0:
                    phase_shift = np.pi * dim / (d + 1) + np.pi * k / 4
                else:
                    phase_shift = -np.pi * dim / 2 + np.pi * (k + r) / 6
                
                # Add component-specific offsets and sign flips
                sign = 1 if (k + dim) % 3 != 0 else -1
                true_mean_coeffs[k, base_idx + r] = sign * amplitude * np.cos(phase_shift + r * np.pi / (k + 2))
    
    # =========================================
    # Define distinct covariance structures for each component
    # Diagonal covariances with different variance profiles
    # =========================================
    true_variances = torch.zeros(n_components, M, device=device, dtype=dtype)
    
    for k in range(n_components):
        for dim in range(d):
            base_idx = dim * R
            for r in range(R):
                freq_factor = (r + 1) / R
                
                # Highly varied variance profiles per component
                if k == 0:
                    # Very tight/concentrated (small variance)
                    var = 0.02 + 0.01 * freq_factor
                elif k == 1:
                    # Large variance at low frequencies, tight at high
                    var = 0.5 * np.exp(-4.0 * freq_factor) + 0.01
                elif k == 2:
                    # Oscillating variance pattern
                    var = 0.15 * (1 + 0.8 * np.abs(np.sin(3 * np.pi * freq_factor)))
                elif k == 3:
                    # Very spread out (high variance overall)
                    var = 0.4 + 0.1 * freq_factor
                elif k == 4:
                    # Peaked variance in middle frequencies
                    var = 0.05 + 0.35 * np.exp(-15.0 * (freq_factor - 0.5) ** 2)
                elif k == 5:
                    # Inverse pattern: tight at low, spread at high
                    var = 0.03 + 0.3 * freq_factor ** 2
                elif k == 6:
                    # Step pattern variance
                    var = 0.1 if freq_factor < 0.4 else 0.35
                else:  # k == 7
                    # Irregular variance pattern
                    var = 0.1 + 0.2 * np.abs(np.sin(5 * np.pi * freq_factor)) * (1 - 0.5 * freq_factor)
                
                # Add dimension-specific scaling (more varied)
                dim_scale = 1.0 + 0.5 * dim + 0.2 * np.sin(np.pi * k * dim / d)
                true_variances[k, base_idx + r] = var * dim_scale
    
    # =========================================
    # Sample from the Gaussian mixture
    # =========================================
    X_coeffs = torch.zeros(n_samples, M, device=device, dtype=dtype)
    
    for i in range(n_samples):
        k = assignments[i].item()
        # Sample from N(mean_k, diag(var_k))
        std = torch.sqrt(true_variances[k])
        X_coeffs[i] = true_mean_coeffs[k] + std * torch.randn(M, device=device, dtype=dtype)
    
    # Reconstruct functional data from coefficients
    X_raw = basis.reconstruct(X_coeffs)  # (n_samples, grid_size, d)
    
    # Reconstruct true mean functions
    true_means = basis.reconstruct(true_mean_coeffs)  # (n_components, grid_size, d)
    
    info = {
        "component_weights": component_weights,
        "true_mean_coeffs": true_mean_coeffs,
        "true_variances": true_variances,
        "true_means": true_means,
        "t": t,
        "T": T,
        "basis": basis,
    }
    
    return X_raw, X_coeffs, assignments, info


def plot_variance_comparison(
    predicted_var: torch.Tensor,
    true_var: torch.Tensor,
    R: int,
    d: int,
    ax: plt.Axes = None,
    title: str = "Variance Comparison",
):
    """
    Plot comparison of predicted vs true variances for each component.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    pred_np = predicted_var.detach().cpu().numpy()
    true_np = true_var.detach().cpu().numpy()
    K = pred_np.shape[0]
    M = pred_np.shape[1]
    
    x = np.arange(M)
    width = 0.35
    
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for k in range(K):
        offset = (k - K / 2 + 0.5) * width / K * 3
        ax.bar(x + offset - width/2, true_np[k], width/K, 
               alpha=0.6, color=colors[k], label=f"True k={k+1}" if k == 0 else None,
               edgecolor="black", linewidth=0.5, hatch="//")
        ax.bar(x + offset + width/2, pred_np[k], width/K, 
               alpha=0.8, color=colors[k], label=f"Pred k={k+1}" if k == 0 else None,
               edgecolor="black", linewidth=0.5)
    
    # Add dimension separators
    for dim in range(1, d):
        ax.axvline(x=dim * R - 0.5, color="gray", linestyle="--", alpha=0.5)
    
    ax.set_xlabel("Coefficient Index")
    ax.set_ylabel("Variance")
    ax.set_title(title)
    ax.legend(["True (hatched)", "Predicted (solid)"], loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    
    return ax


def plot_variance_by_component(
    predicted_var: torch.Tensor,
    true_var: torch.Tensor,
    figsize=(14, 4),
):
    """
    Plot variance comparison for each component separately.
    """
    pred_np = predicted_var.detach().cpu().numpy()
    true_np = true_var.detach().cpu().numpy()
    K = pred_np.shape[0]
    
    fig, axes = plt.subplots(1, K, figsize=figsize, sharey=True)
    if K == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for k in range(K):
        ax = axes[k]
        x = np.arange(len(pred_np[k]))
        ax.fill_between(x, 0, true_np[k], alpha=0.4, color=colors[k], label="True")
        ax.plot(x, pred_np[k], color=colors[k], linewidth=2, label="Predicted")
        ax.set_xlabel("Coefficient Index")
        if k == 0:
            ax.set_ylabel("Variance")
        ax.set_title(f"Component {k+1}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Variance Profiles: True vs Predicted", fontsize=14)
    plt.tight_layout()
    return fig


def main():
    # ==========================================
    # Configuration
    # ==========================================
    torch.manual_seed(42)
    
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
    num_epochs = 400
    lr = 0.1
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print("=" * 60)
    print("L² Functional Data from Gaussian Distributions")
    print("MMD-based Gaussian Mixture Fitting (Means + Covariances)")
    print("=" * 60)
    
    # ==========================================
    # Generate synthetic data from Gaussians
    # ==========================================
    print("\n[1] Generating synthetic data from Gaussian mixture...")
    
    # Non-uniform weights to make it more interesting
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
    
    true_means = info["true_means"]
    true_mean_coeffs = info["true_mean_coeffs"]
    true_variances = info["true_variances"]
    basis = info["basis"]
    t = info["t"]
    M = X_coeffs.shape[1]
    
    print(f"   Generated {n_samples} trajectories with {n_components} components")
    print(f"   Grid size: {grid_size}, Dimensions: {d}")
    print(f"   Coefficient dimension M = {M} (R={R} × d={d})")
    print(f"   True weights: {true_weights.numpy()}")
    print(f"   Mean coefficient norms: {torch.norm(true_mean_coeffs, dim=1).numpy()}")
    print(f"   Mean variances per component: {true_variances.mean(dim=1).numpy()}")
    
    # ==========================================
    # Create kernel and model
    # ==========================================
    print("\n[2] Setting up model and kernel...")
    
    kernel = GaussianKernel(sigma=sigma_kernel)
    
    model = GaussianMixtureModel(
        num_components=n_components,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",  # Learning diagonal covariances
        device=device,
        dtype=dtype,
    )
    
    # Initialize from data
    model.initialize_from_data(X_coeffs, method="kmeans++")
    
    print(f"   Kernel: Gaussian (σ={sigma_kernel})")
    print(f"   Model: {n_components} components, diagonal covariance")
    print(f"   Initial weights: {model.pi.detach().numpy()}")
    print(f"   Initial mean variances: {model.variance.mean(dim=1).detach().numpy()}")
    
    # ==========================================
    # Train the model
    # ==========================================
    print("\n[3] Training model by minimizing MMD²...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Precompute constant term (data-data kernel expectation)
    gram = kernel.compute_gram_matrix(X_coeffs)
    const_term = gram.mean()
    
    history = []
    print_interval = num_epochs // 10
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        mmd2, stats = model.compute_mmd2(X_coeffs, kernel, compute_const_term=False)
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
    print(f"   Final mean variances: {model.variance.mean(dim=1).detach().numpy()}")
    
    # ==========================================
    # Visualize results
    # ==========================================
    print("\n[4] Creating visualizations...")
    
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
    
    fig1.suptitle("Data Trajectories from Gaussian Mixture (colored by true component)", fontsize=14)
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
    
    # Figure 5: Variance comparison
    fig5 = plot_variance_by_component(
        predicted_var=model.variance,
        true_var=true_variances,
    )
    
    # Figure 6: Sample from fitted model
    print("\n[5] Sampling from fitted model...")
    
    samples_functions, samples_coeffs, samples_assignments = model.sample_functions(50)
    
    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_l2_trajectories(
        samples_functions, t=t, assignments=samples_assignments,
        ax=axes6[0], dim=0, title="Sampled: Dimension 1"
    )
    plot_l2_trajectories(
        samples_functions, t=t, assignments=samples_assignments,
        ax=axes6[1], dim=1, title="Sampled: Dimension 2"
    )
    
    fig6.suptitle("Samples from Fitted Gaussian Mixture", fontsize=14)
    plt.tight_layout()
    
    # Figure 7: Coefficient space visualization
    print("\n[6] Visualizing coefficient space...")
    
    fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot first two coefficients
    X_np = X_coeffs.detach().cpu().numpy()
    assignments_np = true_assignments.detach().cpu().numpy()
    pred_means_np = model.mean.detach().cpu().numpy()
    true_means_coeff_np = true_mean_coeffs.detach().cpu().numpy()
    colors = plt.cm.tab20(np.linspace(0, 1, n_components))  # Use tab20 for more colors
    
    for k in range(n_components):
        mask = assignments_np == k
        axes7[0].scatter(X_np[mask, 0], X_np[mask, 1], 
                        c=[colors[k]], alpha=0.4, s=15, label=f"k={k+1}")
    for k in range(n_components):
        axes7[0].scatter(pred_means_np[k, 0], pred_means_np[k, 1], 
                        c=[colors[k]], marker="*", s=250, edgecolors="black", linewidths=1.5)
        axes7[0].scatter(true_means_coeff_np[k, 0], true_means_coeff_np[k, 1], 
                        c=[colors[k]], marker="^", s=150, edgecolors="black", linewidths=1.5)
    axes7[0].set_xlabel("Coefficient 1")
    axes7[0].set_ylabel("Coefficient 2")
    axes7[0].set_title("Coefficient Space (dim 1-2)\n★=Predicted, ▲=True")
    axes7[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes7[0].grid(True, alpha=0.3)
    
    # Plot coefficients 3 and 4
    for k in range(n_components):
        mask = assignments_np == k
        axes7[1].scatter(X_np[mask, 2], X_np[mask, 3], 
                        c=[colors[k]], alpha=0.4, s=15, label=f"k={k+1}")
    for k in range(n_components):
        axes7[1].scatter(pred_means_np[k, 2], pred_means_np[k, 3], 
                        c=[colors[k]], marker="*", s=250, edgecolors="black", linewidths=1.5)
        axes7[1].scatter(true_means_coeff_np[k, 2], true_means_coeff_np[k, 3], 
                        c=[colors[k]], marker="^", s=150, edgecolors="black", linewidths=1.5)
    axes7[1].set_xlabel("Coefficient 3")
    axes7[1].set_ylabel("Coefficient 4")
    axes7[1].set_title("Coefficient Space (dim 3-4)\n★=Predicted, ▲=True")
    axes7[1].legend(loc="upper right", ncol=2, fontsize=8)
    axes7[1].grid(True, alpha=0.3)
    
    fig7.suptitle("Data and Means in Coefficient Space", fontsize=14)
    plt.tight_layout()
    
    # Save figures as PDF for paper
    print("\n[7] Saving figures as PDF...")
    paper_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "images")
    os.makedirs(paper_images_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(paper_images_dir, "l2_gaussian_trajectories.pdf"), format="pdf", bbox_inches="tight")
    fig2.savefig(os.path.join(paper_images_dir, "l2_gaussian_means_dim0.pdf"), format="pdf", bbox_inches="tight")
    fig4.savefig(os.path.join(paper_images_dir, "l2_gaussian_training.pdf"), format="pdf", bbox_inches="tight")
    fig5.savefig(os.path.join(paper_images_dir, "l2_gaussian_variance.pdf"), format="pdf", bbox_inches="tight")
    fig7.savefig(os.path.join(paper_images_dir, "l2_gaussian_coefficients.pdf"), format="pdf", bbox_inches="tight")
    print(f"   Saved figures to {paper_images_dir}")

    print("\n[8] Displaying plots...")


if __name__ == "__main__":
    main()
