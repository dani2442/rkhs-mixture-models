#!/usr/bin/env python
"""
Example: Training a Gaussian Mixture Model on SO(3) rotation data.

This script demonstrates:
1. Generating synthetic rotation data from a mixture model on SO(3)
2. Projecting the data onto the Wigner D-matrix basis (L²(SO(3)) basis)
3. Training a Gaussian mixture using MMD minimization
4. Visualizing the results: rotation distributions and training history

The L²(SO(3)) space uses the Peter-Weyl basis:
    φ^ℓ_mn(R) = √(2ℓ+1) · D^ℓ_mn(R)

where D^ℓ_mn are Wigner D-matrix elements.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    SO3Basis,
    GaussianKernel,
    GaussianMixtureModel,
    generate_so3_mixture_data,
    plot_training_history,
    plot_mixture_weights,
)


def euler_to_rotation_matrix(alpha, beta, gamma):
    """Convert Euler angles to rotation matrix (z-y-z convention)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    R = np.array([
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg, sb * sg, cb]
    ])
    return R


def plot_so3_samples(X, assignments, ax, title="SO(3) Samples"):
    """
    Visualize SO(3) samples by showing where they map a reference point.
    
    Each rotation is visualized by where it maps the vector [1, 0, 0].
    """
    n = X.shape[0]
    colors = plt.cm.tab10(assignments.numpy() % 10)
    
    # Reference point on the sphere
    ref = np.array([1.0, 0.0, 0.0])
    
    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    # Plot rotated points
    for i in range(n):
        euler = X[i].numpy()
        R = euler_to_rotation_matrix(euler[0], euler[1], euler[2])
        point = R @ ref
        ax.scatter(point[0], point[1], point[2], c=[colors[i]], s=20, alpha=0.7)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])


def plot_euler_angles(X, assignments, axes, title_prefix=""):
    """Plot Euler angle distributions."""
    labels = [r'$\alpha$ (rad)', r'$\beta$ (rad)', r'$\gamma$ (rad)']
    n_components = assignments.max().item() + 1
    colors = plt.cm.tab10(np.arange(n_components) % 10)
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        for k in range(n_components):
            mask = assignments == k
            ax.hist(X[mask, i].numpy(), bins=30, alpha=0.5, 
                   label=f'Component {k+1}', color=colors[k])
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        if i == 0:
            ax.legend()
        ax.set_title(f'{title_prefix}{label} distribution')


def plot_coefficient_distribution(X_proj, assignments, ax, max_coeffs=20):
    """Plot distribution of first few coefficients."""
    n_components = assignments.max().item() + 1
    n_coeffs = min(X_proj.shape[1], max_coeffs)
    
    for k in range(n_components):
        mask = assignments == k
        means = X_proj[mask, :n_coeffs].mean(dim=0).numpy()
        stds = X_proj[mask, :n_coeffs].std(dim=0).numpy()
        
        x = np.arange(n_coeffs)
        ax.errorbar(x + 0.1 * k, means, yerr=stds, fmt='o-', 
                   label=f'Component {k+1}', alpha=0.7, capsize=3)
    
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('Value')
    ax.set_title('Wigner D-matrix coefficient distribution')
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)


def main():
    # ==========================================
    # Configuration
    # ==========================================
    torch.manual_seed(42)
    
    # Data parameters
    n_samples = 200
    n_components = 3
    noise_concentration = 30.0  # Higher = more concentrated around mean
    
    # Basis parameters (L_max = maximum degree of Wigner D-matrices)
    L_max = 3  # Uses ℓ = 0, 1, 2, 3
    
    # Kernel parameters
    sigma_kernel = 2.0
    
    # Training parameters
    num_epochs = 400
    lr = 0.1
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print("=" * 60)
    print("SO(3) Rotation Data: MMD-based Gaussian Mixture Fitting")
    print("=" * 60)
    
    # ==========================================
    # Generate synthetic data
    # ==========================================
    print("\n[1] Generating synthetic SO(3) data...")
    
    X_euler, true_assignments, info = generate_so3_mixture_data(
        n_samples=n_samples,
        n_components=n_components,
        noise_concentration=noise_concentration,
        seed=42,
        device=device,
        dtype=dtype,
    )
    
    true_means = info["mean_rotations"]
    true_weights = info["component_weights"]
    
    print(f"   Generated {n_samples} rotations with {n_components} components")
    print(f"   Noise concentration: {noise_concentration}")
    print(f"   True weights: {true_weights.numpy()}")
    print(f"   Mean Euler angles (rad):")
    for k in range(n_components):
        print(f"      Component {k+1}: α={true_means[k,0]:.3f}, β={true_means[k,1]:.3f}, γ={true_means[k,2]:.3f}")
    
    # ==========================================
    # Create basis and project data
    # ==========================================
    print("\n[2] Projecting data onto Wigner D-matrix basis...")
    
    basis = SO3Basis(
        L_max=L_max,
        use_real_basis=True,
        device=device,
        dtype=dtype,
    )
    
    X = basis.project(X_euler)  # (n_samples, M)
    M = X.shape[1]
    
    print(f"   Maximum degree L_max = {L_max}")
    print(f"   Number of basis functions: {basis.num_complex_coeffs} complex")
    print(f"   Coefficient dimension M = {M} (real + imaginary parts)")
    
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
    
    # Figure 1: 3D visualization of rotations on the sphere
    fig1 = plt.figure(figsize=(12, 5))
    
    ax1 = fig1.add_subplot(121, projection='3d')
    plot_so3_samples(X_euler, true_assignments, ax1, "Data (colored by true component)")
    
    # Plot mean rotations
    ax2 = fig1.add_subplot(122, projection='3d')
    ref = np.array([1.0, 0.0, 0.0])
    
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    colors = plt.cm.tab10(np.arange(n_components) % 10)
    for k in range(n_components):
        euler = true_means[k].numpy()
        R = euler_to_rotation_matrix(euler[0], euler[1], euler[2])
        point = R @ ref
        ax2.scatter(point[0], point[1], point[2], c=[colors[k]], s=200, 
                   marker='*', edgecolors='black', linewidth=1,
                   label=f'Mean {k+1}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('True Mean Rotations')
    ax2.legend()
    ax2.set_box_aspect([1, 1, 1])
    
    fig1.suptitle("SO(3) Rotation Data Visualization", fontsize=14)
    plt.tight_layout()
    
    # Figure 2: Euler angle distributions
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    plot_euler_angles(X_euler, true_assignments, axes2, "")
    fig2.suptitle("Euler Angle Distributions by Component", fontsize=14)
    plt.tight_layout()
    
    # Figure 3: Coefficient distributions
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    plot_coefficient_distribution(X, true_assignments, ax3)
    fig3.suptitle("Wigner D-matrix Coefficients", fontsize=14)
    plt.tight_layout()
    
    # Figure 4: Training history
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_training_history(history, ax=axes4[0])
    
    # Weight comparison
    x_pos = np.arange(n_components)
    width = 0.35
    axes4[1].bar(x_pos - width/2, true_weights.numpy(), width, label='True', alpha=0.7)
    axes4[1].bar(x_pos + width/2, model.pi.detach().numpy(), width, label='Learned', alpha=0.7)
    axes4[1].set_xlabel('Component')
    axes4[1].set_ylabel('Weight')
    axes4[1].set_title('Mixture Weights: True vs Learned')
    axes4[1].set_xticks(x_pos)
    axes4[1].set_xticklabels([f'{k+1}' for k in range(n_components)])
    axes4[1].legend()
    
    fig4.suptitle("Training Results", fontsize=14)
    plt.tight_layout()
    
    # Save figures as PDF for paper
    print("\n[6] Saving figures as PDF...")
    paper_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "images")
    os.makedirs(paper_images_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(paper_images_dir, "so3_sphere_visualization.pdf"), format="pdf", bbox_inches="tight")
    fig2.savefig(os.path.join(paper_images_dir, "so3_euler_angles.pdf"), format="pdf", bbox_inches="tight")
    fig4.savefig(os.path.join(paper_images_dir, "so3_training_summary.pdf"), format="pdf", bbox_inches="tight")
    print(f"   Saved figures to {paper_images_dir}")

    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nL²(SO(3)) Basis:")
    print(f"  - Wigner D-matrix basis up to degree L_max = {L_max}")
    print(f"  - Total basis functions: {M}")
    print(f"\nMixture Model Results:")
    print(f"  - True weights:    {true_weights.numpy()}")
    print(f"  - Learned weights: {model.pi.detach().numpy()}")
    print(f"  - Final MMD²: {history[-1]:.6f}")


if __name__ == "__main__":
    main()
