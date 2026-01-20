"""
Synthetic data generators for testing MMD-based Gaussian mixture fitting.
"""
import torch
import math
from typing import Tuple, Optional, List


def generate_l2_mixture_data(
    n_samples: int,
    n_components: int,
    grid_size: int,
    T: float = 1.0,
    d: int = 1,
    noise_std: float = 0.1,
    component_weights: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate synthetic functional data from a mixture of Gaussian-like processes.

    Each component generates functions of the form:
        f_k(t) = sum_j a_{k,j} * basis_j(t) + noise

    Args:
        n_samples: Number of samples to generate
        n_components: Number of mixture components
        grid_size: Number of time discretization points
        T: End of time interval [0, T]
        d: Spatial dimension (R^d valued functions)
        noise_std: Standard deviation of additive noise
        component_weights: Mixture weights, shape (n_components,)
        seed: Random seed for reproducibility
        device: Torch device
        dtype: Torch dtype

    Returns:
        X: Generated trajectories, shape (n_samples, grid_size, d)
        assignments: True component assignments, shape (n_samples,)
        info: Dictionary with ground truth parameters
    """
    if seed is not None:
        torch.manual_seed(seed)

    t = torch.linspace(0, T, grid_size, device=device, dtype=dtype)

    # Default uniform weights
    if component_weights is None:
        component_weights = torch.ones(n_components, device=device, dtype=dtype) / n_components

    # Sample component assignments
    assignments = torch.multinomial(
        component_weights.expand(n_samples, -1), num_samples=1
    ).squeeze(-1)

    # Generate component-specific base functions
    # Each component has a different characteristic pattern
    base_functions = []
    for k in range(n_components):
        # Create a distinct pattern for each component
        freq_base = 1.0 + k * 0.5  # Different base frequencies
        phase = k * math.pi / n_components  # Different phases
        
        f = torch.zeros(grid_size, d, device=device, dtype=dtype)
        for dim in range(d):
            freq = freq_base * (dim + 1)
            f[:, dim] = torch.sin(2 * math.pi * freq * t / T + phase + dim * math.pi / 4)
            # Add some variation
            f[:, dim] += 0.3 * torch.cos(4 * math.pi * freq * t / T - phase)
        base_functions.append(f)

    # Generate samples
    X = torch.zeros(n_samples, grid_size, d, device=device, dtype=dtype)
    
    for i in range(n_samples):
        k = assignments[i].item()
        # Base function + per-sample variation + noise
        variation = 0.2 * torch.randn(1, device=device, dtype=dtype)
        X[i] = base_functions[k] * (1 + variation)
        X[i] += noise_std * torch.randn(grid_size, d, device=device, dtype=dtype)

    info = {
        "component_weights": component_weights,
        "base_functions": torch.stack(base_functions),  # (K, grid_size, d)
        "t": t,
        "T": T,
        "noise_std": noise_std,
    }

    return X, assignments, info


def generate_graph_mixture_data(
    n_samples: int,
    n_components: int,
    num_nodes: int,
    edge_probability: float = 0.3,
    signal_std: float = 1.0,
    noise_std: float = 0.1,
    component_weights: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Generate synthetic graph signal data from a mixture model.

    Each component has a characteristic smooth signal on the graph,
    with samples being noisy versions of this base signal.

    Args:
        n_samples: Number of samples to generate
        n_components: Number of mixture components
        num_nodes: Number of nodes in the graph
        edge_probability: Probability of edge in random graph
        signal_std: Standard deviation of component base signals
        noise_std: Standard deviation of additive noise
        component_weights: Mixture weights
        seed: Random seed
        device: Torch device
        dtype: Torch dtype

    Returns:
        X: Graph signals, shape (n_samples, num_nodes)
        assignments: Component assignments, shape (n_samples,)
        adjacency: Graph adjacency matrix, shape (num_nodes, num_nodes)
        info: Dictionary with ground truth parameters
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random graph (Erdos-Renyi)
    adjacency = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1).item() < edge_probability:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    # Compute graph Laplacian
    degree = adjacency.sum(dim=1)
    laplacian = torch.diag(degree) - adjacency

    # Eigendecomposition for smooth signal generation
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

    # Default uniform weights
    if component_weights is None:
        component_weights = torch.ones(n_components, device=device, dtype=dtype) / n_components

    # Sample component assignments
    assignments = torch.multinomial(
        component_weights.expand(n_samples, -1), num_samples=1
    ).squeeze(-1)

    # Generate component base signals (smooth on graph = low frequency)
    base_signals = []
    n_low_freq = min(num_nodes // 3, 10)  # Use low-frequency components
    
    for k in range(n_components):
        # Random combination of low-frequency eigenvectors
        coeffs = torch.randn(n_low_freq, device=device, dtype=dtype) * signal_std
        # Decay higher frequencies
        decay = torch.exp(-torch.arange(n_low_freq, device=device, dtype=dtype) * 0.3)
        coeffs = coeffs * decay
        
        # Add component-specific bias
        coeffs[0] = k - n_components / 2  # Different DC levels
        
        signal = eigenvectors[:, :n_low_freq] @ coeffs
        base_signals.append(signal)

    # Generate samples
    X = torch.zeros(n_samples, num_nodes, device=device, dtype=dtype)
    
    for i in range(n_samples):
        k = assignments[i].item()
        X[i] = base_signals[k] + noise_std * torch.randn(num_nodes, device=device, dtype=dtype)

    info = {
        "component_weights": component_weights,
        "base_signals": torch.stack(base_signals),  # (K, num_nodes)
        "laplacian": laplacian,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "noise_std": noise_std,
    }

    return X, assignments, adjacency, info


def generate_l2_sine_cosine_data(
    n_samples: int,
    grid_size: int,
    T: float = 1.0,
    d: int = 2,
    frequencies: List[float] = [1.0, 2.0, 3.0],
    noise_std: float = 0.1,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate data from a mixture of sine/cosine patterns.

    This creates a clear multi-modal distribution where each component
    has a distinct frequency.

    Args:
        n_samples: Number of samples
        grid_size: Time discretization points
        T: Time interval [0, T]
        d: Spatial dimension
        frequencies: List of frequencies for each component
        noise_std: Noise standard deviation
        seed: Random seed
        device: Torch device
        dtype: Torch dtype

    Returns:
        X: Trajectories, shape (n_samples, grid_size, d)
        assignments: Component assignments
        info: Ground truth information
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_components = len(frequencies)
    t = torch.linspace(0, T, grid_size, device=device, dtype=dtype)

    # Equal weights
    weights = torch.ones(n_components, device=device, dtype=dtype) / n_components
    assignments = torch.multinomial(
        weights.expand(n_samples, -1), num_samples=1
    ).squeeze(-1)

    # Base functions for each component
    base_functions = []
    for k, freq in enumerate(frequencies):
        f = torch.zeros(grid_size, d, device=device, dtype=dtype)
        for dim in range(d):
            if dim % 2 == 0:
                f[:, dim] = torch.sin(2 * math.pi * freq * t / T)
            else:
                f[:, dim] = torch.cos(2 * math.pi * freq * t / T)
        base_functions.append(f)

    # Generate samples
    X = torch.zeros(n_samples, grid_size, d, device=device, dtype=dtype)
    for i in range(n_samples):
        k = assignments[i].item()
        # Add amplitude variation
        amplitude = 0.8 + 0.4 * torch.rand(1, device=device, dtype=dtype)
        X[i] = amplitude * base_functions[k]
        X[i] += noise_std * torch.randn(grid_size, d, device=device, dtype=dtype)

    info = {
        "component_weights": weights,
        "base_functions": torch.stack(base_functions),
        "frequencies": frequencies,
        "t": t,
        "T": T,
        "noise_std": noise_std,
    }

    return X, assignments, info
