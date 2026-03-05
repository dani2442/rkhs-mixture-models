"""
Maximum Mean Discrepancy (MMD) for Gaussian Mixtures in Hilbert Spaces.

This package provides:
  - Kernel classes for computing MMD expectations (Gaussian, Polynomial)
  - Hilbert space basis classes for projecting functional data (L2, Graph)
  - Functions for computing MMD between empirical measures and Gaussian mixtures
"""
import torch
from typing import Tuple, Optional, Union

from src.kernel.base import Kernel
from src.kernel.gaussian import GaussianKernel
from src.kernel.polynomial import PolynomialKernel, LinearKernel, QuadraticKernel
from src.spaces.base import HilbertBasis
from src.spaces.L2 import L2Basis, L2CosineBasis, L2FourierBasis, L2TensorBasis2D
from src.spaces.graph import GraphBasis, GraphLaplacianBasis
from src.spaces.graph_embedding import WLHashFingerprint
from src.spaces.Rn import CanonicalBasis, DiscreteCosineBasis
from src.spaces.so3 import SO3Basis, SO3FourierBasis
from src.mixture import GaussianMixtureModel, fit_gaussian_mixture_mmd
from src.data import (
    generate_l2_mixture_data,
    generate_l2_2d_gaussian_data,
    generate_graph_mixture_data,
    generate_l2_sine_cosine_data,
    generate_so3_mixture_data,
    generate_so3_rotation_matrix_data,
    download_ntu_skeleton,
)
from src.visualization import (
    plot_l2_trajectories,
    plot_l2_means_comparison,
    plot_graph_signals,
    plot_graph_means_comparison,
    plot_training_history,
    plot_mixture_weights,
    plot_l2_2d_surface,
    plot_l2_2d_surfaces_grid,
    plot_l2_2d_means_comparison,
    plot_l2_2d_samples_by_component,
    plot_glucodensity_temporal_comparison,
    plot_glucodensity_variance,
    plot_cluster_probabilities_by_group,
    plot_ternary_simplex_evolution,
    plot_ternary_simplex_grid,
    plot_ternary_simplex_interactive,
)


# ============================================================
# MMD^2(P_n, Q) computation for Gaussian mixture Q
# ============================================================

def mmd2_empirical_vs_gaussian_mixture(
    X: torch.Tensor,
    pi: torch.Tensor,
    m: torch.Tensor,
    Kcov: torch.Tensor,
    kernel: Optional[Kernel] = None,
    sigma: Optional[float] = None,
    compute_const_term: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute MMD²(P, Q) where P is the empirical measure of X and Q is a
    Gaussian mixture.

    MMD²(P,Q) = E_{x,x'~P}[κ(x,x')]
               - 2 E_{x~P,y~Q}[κ(x,y)]
               + E_{y,y'~Q}[κ(y,y')]

    With Q = Σ_k π_k N(m_k, K_k):
      E_{x~P,y~Q}[κ] = Σ_k π_k (1/n Σ_i J_{i,k})
      E_{y,y'~Q}[κ] = Σ_{k,s} π_k π_s I_{k,s}

    Args:
        X: Projected data, shape (n, M)
        pi: Mixture weights, shape (K,), should sum to 1
        m: Component means, shape (K, M)
        Kcov: Component covariances, shape (K, M, M)
        kernel: Kernel object. If None, uses GaussianKernel with sigma.
        sigma: Bandwidth for Gaussian kernel (used if kernel is None).
               Deprecated: prefer passing kernel directly.
        compute_const_term: Whether to compute E_{x,x'~P}[κ(x,x')].
                           Set False if only optimizing over Q.

    Returns:
        mmd2: Scalar tensor with MMD² value.
        stats: Dict with 'const', 'Jbar', 'I' for diagnostics.
    """
    device = X.device
    dtype = X.dtype
    n, M = X.shape
    K = pi.shape[0]

    # Handle backward compatibility
    if kernel is None:
        if sigma is None:
            raise ValueError("Either 'kernel' or 'sigma' must be provided")
        kernel = GaussianKernel(sigma=sigma)

    # 1) Constant term: (1/n²) Σ_{i,j} κ(X_i, X_j)
    if compute_const_term:
        gram = kernel.compute_gram_matrix(X)  # (n, n)
        const = gram.mean()
    else:
        const = torch.tensor(0.0, device=device, dtype=dtype)

    # 2) J matrix and its mean over i
    J = kernel.compute_J(X=X, m=m, Kcov=Kcov)  # (n, K)
    Jbar = J.mean(dim=0)  # (K,)

    # 3) I matrix
    I = kernel.compute_I(m=m, Kcov=Kcov)  # (K, K)

    # Mixture terms
    cross = (pi * Jbar).sum()  # scalar
    mixmix = pi @ I @ pi  # scalar

    mmd2 = const - 2.0 * cross + mixmix

    stats = {"const": const, "Jbar": Jbar, "I": I}
    return mmd2, stats


def compute_mmd2_terms(
    X: torch.Tensor,
    m: torch.Tensor,
    Kcov: torch.Tensor,
    kernel: Kernel,
    compute_const_term: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the three terms needed for MMD² optimization.

    This is useful when optimizing mixture weights, as the constant term
    does not depend on the weights.

    Args:
        X: Data, shape (n, M)
        m: Component means, shape (K, M)
        Kcov: Component covariances, shape (K, M, M)
        kernel: Kernel object
        compute_const_term: Whether to compute the data-data term.

    Returns:
        const: E_{x,x'~P}[κ(x,x')], scalar
        Jbar: (1/n) Σ_i J_{i,k}, shape (K,)
        I: I_{k,s}, shape (K, K)
    """
    device = X.device
    dtype = X.dtype

    if compute_const_term:
        gram = kernel.compute_gram_matrix(X)
        const = gram.mean()
    else:
        const = torch.tensor(0.0, device=device, dtype=dtype)

    Jbar = kernel.compute_Jbar(X, m, Kcov)
    I = kernel.compute_I(m, Kcov)

    return const, Jbar, I


__all__ = [
    # Main function
    "mmd2_empirical_vs_gaussian_mixture",
    "compute_mmd2_terms",
    # Mixture model
    "GaussianMixtureModel",
    "fit_gaussian_mixture_mmd",
    # Kernels
    "Kernel",
    "GaussianKernel",
    "PolynomialKernel",
    "LinearKernel",
    "QuadraticKernel",
    # Spaces
    "HilbertBasis",
    "L2Basis",
    "L2CosineBasis",
    "L2FourierBasis",
    "GraphBasis",
    "GraphLaplacianBasis",
    "SO3Basis",
    "SO3FourierBasis",
    "CanonicalBasis",
    "DiscreteCosineBasis",
    "WLHashFingerprint",
    # Data generation
    "generate_l2_mixture_data",
    "generate_l2_2d_gaussian_data",
    "generate_graph_mixture_data",
    "generate_l2_sine_cosine_data",
    "generate_so3_mixture_data",
    "generate_so3_rotation_matrix_data",
    "download_ntu_skeleton",
    # Visualization
    "plot_l2_trajectories",
    "plot_l2_means_comparison",
    "plot_graph_signals",
    "plot_graph_means_comparison",
    "plot_training_history",
    "plot_mixture_weights",
    # Legacy (backward compatibility)
    "gaussian_kernel_J",
    "gaussian_kernel_I",
]
