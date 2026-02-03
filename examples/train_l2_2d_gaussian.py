#!/usr/bin/env python
"""
Example: Training a Gaussian Mixture Model on L²([0,T]×[0,S]; R^d) functional data
using tensor product basis.

This script demonstrates:
1. Generating synthetic 2D functional data from Gaussian distributions
   with complex mean surfaces and varying covariances
2. Projecting the data onto a tensor product cosine basis
3. Training a Gaussian mixture model (means AND covariances) using MMD minimization
4. Visualizing the results with 3D mesh plots

The space L²([0,T]×[0,S]; R^d) uses tensor products of 1D orthonormal bases:
    Ψ_{j,m,n}(s,t) = φ_m(s) · φ_n(t) · e_j

where φ_r is the 1D cosine basis and e_j is the j-th standard basis vector in R^d.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmd import (
    GaussianKernel,
    GaussianMixtureModel,
    plot_training_history,
    plot_mixture_weights,
)
from mmd.spaces.L2 import L2TensorBasis2D
from mmd.visualization import (
    plot_l2_2d_surfaces_grid,
    plot_l2_2d_means_comparison,
    plot_l2_2d_samples_by_component,
)


def generate_l2_2d_gaussian_data(
    n_samples: int,
    n_components: int,
    grid_size_s: int,
    grid_size_t: int,
    R_s: int,
    R_t: int,
    T: float = 1.0,
    S: float = 1.0,
    d: int = 1,
    component_weights: torch.Tensor = None,
    seed: int = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
):
    """
    Generate L²([0,T]×[0,S]; R^d) functional data from Gaussian distributions.
    
    Each component k has:
    - A complex mean surface (combination of 2D harmonics)
    - A distinct covariance structure (diagonal with varying scales)
    
    Args:
        n_samples: Number of samples to generate
        n_components: Number of mixture components
        grid_size_s: Number of discretization points in s direction
        grid_size_t: Number of discretization points in t direction
        R_s: Number of basis functions in s direction
        R_t: Number of basis functions in t direction
        T: End of interval [0, T] in t direction
        S: End of interval [0, S] in s direction
        d: Spatial dimension (R^d valued functions)
        component_weights: Mixture weights, shape (n_components,)
        seed: Random seed for reproducibility
        device: Torch device
        dtype: Torch dtype
    
    Returns:
        X_raw: Generated 2D functions, shape (n_samples, grid_size_s, grid_size_t, d)
        X_coeffs: Coefficient representation, shape (n_samples, M)
        assignments: True component assignments, shape (n_samples,)
        info: Dictionary with ground truth parameters
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    M = R_s * R_t * d  # Total coefficient dimension
    
    # Create basis for reconstruction
    basis = L2TensorBasis2D(
        T=T, S=S, R_t=R_t, R_s=R_s,
        grid_size_t=grid_size_t, grid_size_s=grid_size_s,
        d=d, device=device, dtype=dtype
    )
    
    # Get meshgrid for visualization
    s_grid, t_grid = basis.get_meshgrid()
    
    # Default uniform weights
    if component_weights is None:
        component_weights = torch.ones(n_components, device=device, dtype=dtype) / n_components
    
    # Sample component assignments
    assignments = torch.multinomial(
        component_weights.expand(n_samples, -1), num_samples=1
    ).squeeze(-1)
    
    # =========================================
    # Define complex mean coefficients for each component
    # Each mean is a 2D surface with distinct patterns
    # =========================================
    true_mean_coeffs = torch.zeros(n_components, M, device=device, dtype=dtype)
    
    for k in range(n_components):
        for dim_idx in range(d):
            for m in range(R_s):
                for n in range(R_t):
                    # Linear index in coefficient vector
                    coeff_idx = dim_idx * R_s * R_t + m * R_t + n
                    
                    freq_s = (m + 1) / R_s
                    freq_t = (n + 1) / R_t
                    
                    # Create distinct 2D patterns for each component
                    if k == 0:
                        # Low-frequency dominant (smooth hill)
                        amplitude = 3.0 * np.exp(-3.0 * (freq_s + freq_t))
                    elif k == 1:
                        # Diagonal pattern (ridges)
                        amplitude = 2.5 * np.exp(-4.0 * np.abs(freq_s - freq_t)) * np.exp(-2.0 * (freq_s + freq_t))
                    elif k == 2:
                        # Checkerboard-like pattern
                        amplitude = 2.0 * np.abs(np.sin(3 * np.pi * freq_s) * np.sin(3 * np.pi * freq_t)) * np.exp(-1.5 * (freq_s + freq_t))
                    elif k == 3:
                        # Radial pattern (from corner)
                        r = np.sqrt(freq_s**2 + freq_t**2)
                        amplitude = 2.8 * np.exp(-5.0 * r) * (1 + 0.5 * np.cos(4 * np.pi * r))
                    elif k == 4:
                        # Band pattern in s direction
                        amplitude = 2.2 * np.exp(-8.0 * (freq_s - 0.3)**2) * np.exp(-2.0 * freq_t)
                    elif k == 5:
                        # Oscillating decay
                        amplitude = 2.0 * np.abs(np.sin(2 * np.pi * freq_s)) * np.exp(-3.0 * freq_t)
                    elif k == 6:
                        # Corner-peaked
                        amplitude = 3.5 * np.exp(-10.0 * (freq_s**2 + freq_t**2))
                    else:  # k >= 7
                        # Complex mixed pattern
                        amplitude = 1.8 * (np.sin(4 * np.pi * freq_s) + np.cos(3 * np.pi * freq_t)) * np.exp(-2.0 * (freq_s + freq_t))
                    
                    # Add phase variation based on component and dimension
                    phase = np.pi * k / 4 + np.pi * dim_idx / (d + 1)
                    sign = 1 if (k + dim_idx + m) % 3 != 0 else -1
                    
                    true_mean_coeffs[k, coeff_idx] = sign * amplitude * np.cos(phase + (m + n) * np.pi / (k + 2))
    
    # =========================================
    # Define distinct covariance structures for each component
    # Diagonal covariances with different 2D variance profiles
    # =========================================
    true_variances = torch.zeros(n_components, M, device=device, dtype=dtype)
    
    for k in range(n_components):
        for dim_idx in range(d):
            for m in range(R_s):
                for n in range(R_t):
                    coeff_idx = dim_idx * R_s * R_t + m * R_t + n
                    
                    freq_s = (m + 1) / R_s
                    freq_t = (n + 1) / R_t
                    
                    # Varied variance profiles per component
                    if k == 0:
                        # Very tight (small variance)
                        var = 0.02 + 0.01 * (freq_s + freq_t)
                    elif k == 1:
                        # Large at low frequencies
                        var = 0.4 * np.exp(-3.0 * (freq_s + freq_t)) + 0.02
                    elif k == 2:
                        # Oscillating pattern
                        var = 0.1 * (1 + 0.6 * np.abs(np.sin(2 * np.pi * freq_s) * np.sin(2 * np.pi * freq_t)))
                    elif k == 3:
                        # Spread out (high variance)
                        var = 0.3 + 0.1 * (freq_s + freq_t)
                    elif k == 4:
                        # Peaked in middle frequencies
                        var = 0.05 + 0.25 * np.exp(-10.0 * ((freq_s - 0.4)**2 + (freq_t - 0.4)**2))
                    elif k == 5:
                        # Inverse: high at high frequencies
                        var = 0.03 + 0.2 * (freq_s * freq_t)
                    elif k == 6:
                        # Step pattern
                        var = 0.08 if (freq_s < 0.5 and freq_t < 0.5) else 0.25
                    else:
                        # Irregular
                        var = 0.08 + 0.15 * np.abs(np.sin(4 * np.pi * freq_s)) * (1 - 0.5 * freq_t)
                    
                    # Add dimension-specific scaling
                    dim_scale = 1.0 + 0.3 * dim_idx
                    true_variances[k, coeff_idx] = var * dim_scale
    
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
    X_raw = basis.reconstruct(X_coeffs)  # (n_samples, grid_size_s, grid_size_t, d)
    
    # Reconstruct true mean functions
    true_means = basis.reconstruct(true_mean_coeffs)  # (n_components, grid_size_s, grid_size_t, d)
    
    info = {
        "component_weights": component_weights,
        "true_mean_coeffs": true_mean_coeffs,
        "true_variances": true_variances,
        "true_means": true_means,
        "s_grid": s_grid,
        "t_grid": t_grid,
        "T": T,
        "S": S,
        "basis": basis,
    }
    
    return X_raw, X_coeffs, assignments, info


def plot_variance_comparison_2d(
    predicted_var: torch.Tensor,
    true_var: torch.Tensor,
    R_s: int,
    R_t: int,
    d: int,
    figsize: tuple = (14, 4),
):
    """
    Plot comparison of predicted vs true variances for 2D basis.
    Shows variance as a function of coefficient index.
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
        ax.plot(x, pred_np[k], color=colors[k], linewidth=1.5, label="Predicted")
        ax.set_xlabel("Coefficient Index")
        if k == 0:
            ax.set_ylabel("Variance")
        ax.set_title(f"Component {k+1}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Variance Profiles: True vs Predicted (2D Tensor Basis)", fontsize=14)
    plt.tight_layout()
    return fig


def main():
    # ==========================================
    # Configuration
    # ==========================================
    torch.manual_seed(42)
    
    # Data parameters
    n_samples = 400
    n_components = 4
    grid_size_s = 30
    grid_size_t = 30
    T = 1.0
    S = 1.0
    d = 1  # Scalar-valued 2D functions for simplicity
    
    # Basis parameters
    R_s = 8  # Number of basis functions in s direction
    R_t = 8  # Number of basis functions in t direction
    M = R_s * R_t * d
    
    # Kernel parameters
    sigma_kernel = 2.0
    
    # Training parameters
    num_epochs = 400
    lr = 0.1
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print("=" * 70)
    print("L²([0,T]×[0,S]; R^d) Functional Data from Gaussian Distributions")
    print("MMD-based Gaussian Mixture Fitting with Tensor Product Basis")
    print("=" * 70)
    
    # ==========================================
    # Generate synthetic data from Gaussians
    # ==========================================
    print("\n[1] Generating synthetic 2D functional data from Gaussian mixture...")
    
    # Non-uniform weights
    true_weights = torch.tensor([0.35, 0.30, 0.20, 0.15], device=device, dtype=dtype)
    
    X_raw, X_coeffs, true_assignments, info = generate_l2_2d_gaussian_data(
        n_samples=n_samples,
        n_components=n_components,
        grid_size_s=grid_size_s,
        grid_size_t=grid_size_t,
        R_s=R_s,
        R_t=R_t,
        T=T,
        S=S,
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
    s_grid = info["s_grid"]
    t_grid = info["t_grid"]
    
    print(f"   Generated {n_samples} 2D functions with {n_components} components")
    print(f"   Grid size: {grid_size_s} × {grid_size_t}, Dimensions: {d}")
    print(f"   Coefficient dimension M = {M} (R_s={R_s} × R_t={R_t} × d={d})")
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
        covariance_type="diagonal",
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
    # Visualize results with 3D mesh plots
    # ==========================================
    print("\n[4] Creating 3D mesh visualizations...")
    
    # Figure 1: Sample 2D functions by component (3D surfaces)
    fig1 = plot_l2_2d_samples_by_component(
        X=X_raw,
        assignments=true_assignments,
        s_grid=s_grid,
        t_grid=t_grid,
        dim=0,
        samples_per_component=3,
        figsize=(15, 12),
        title="L²([0,T]×[0,S]) Sample Functions by Component",
    )
    
    # Figure 2: Data samples grid
    fig2 = plot_l2_2d_surfaces_grid(
        X=X_raw,
        s_grid=s_grid,
        t_grid=t_grid,
        assignments=true_assignments,
        n_samples=9,
        dim=0,
        figsize=(15, 12),
        title="Sample 2D Functions (colored by component)",
    )
    
    # Figure 3: Predicted vs True means comparison
    fig3 = plot_l2_2d_means_comparison(
        predicted_means=model.mean,
        basis=basis,
        s_grid=s_grid,
        t_grid=t_grid,
        true_means=true_means,
        dim=0,
        figsize=(16, 8),
        title="Mean Surfaces: Predicted (top) vs True (bottom)",
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
    fig5 = plot_variance_comparison_2d(
        predicted_var=model.variance,
        true_var=true_variances,
        R_s=R_s,
        R_t=R_t,
        d=d,
    )
    
    # Figure 6: Samples from fitted model
    print("\n[5] Sampling from fitted model...")
    
    samples_functions, samples_coeffs, samples_assignments = model.sample_functions(12)
    
    fig6 = plot_l2_2d_surfaces_grid(
        X=samples_functions,
        s_grid=s_grid,
        t_grid=t_grid,
        assignments=samples_assignments,
        n_samples=9,
        dim=0,
        figsize=(15, 12),
        title="Samples from Fitted Gaussian Mixture (3D surfaces)",
    )
    
    # Save figures as PDF for paper
    print("\n[6] Saving figures as PDF...")
    paper_images_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "paper", "images"
    )
    os.makedirs(paper_images_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(paper_images_dir, "l2_2d_samples_by_component.pdf"), 
                 format="pdf", bbox_inches="tight")
    fig2.savefig(os.path.join(paper_images_dir, "l2_2d_trajectories.pdf"), 
                 format="pdf", bbox_inches="tight")
    fig3.savefig(os.path.join(paper_images_dir, "l2_2d_means_comparison.pdf"), 
                 format="pdf", bbox_inches="tight")
    fig4.savefig(os.path.join(paper_images_dir, "l2_2d_training.pdf"), 
                 format="pdf", bbox_inches="tight")
    fig5.savefig(os.path.join(paper_images_dir, "l2_2d_variance.pdf"), 
                 format="pdf", bbox_inches="tight")
    fig6.savefig(os.path.join(paper_images_dir, "l2_2d_samples_fitted.pdf"), 
                 format="pdf", bbox_inches="tight")
    
    print(f"   Saved figures to {paper_images_dir}")
    
    print("\n[7] Displaying plots...")
    plt.show()


if __name__ == "__main__":
    main()
