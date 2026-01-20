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


def generate_so3_mixture_data(
    n_samples: int,
    n_components: int = 3,
    concentration: float = 10.0,
    noise_concentration: float = 50.0,
    component_weights: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate synthetic SO(3) rotation data from a mixture of von Mises-Fisher
    like distributions on the rotation group.

    Each component has a mean rotation, and samples are concentrated around
    this mean according to a concentration parameter.

    Args:
        n_samples: Number of samples to generate
        n_components: Number of mixture components
        concentration: Concentration parameter for component centers (higher = more spread)
        noise_concentration: Concentration parameter for noise around means (higher = less noise)
        component_weights: Mixture weights, shape (n_components,)
        seed: Random seed for reproducibility
        device: Torch device
        dtype: Torch dtype

    Returns:
        X: Generated Euler angles (α, β, γ), shape (n_samples, 3)
        assignments: True component assignments, shape (n_samples,)
        info: Dictionary with ground truth parameters
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Default uniform weights
    if component_weights is None:
        component_weights = torch.ones(n_components, device=device, dtype=dtype) / n_components

    # Sample component assignments
    assignments = torch.multinomial(
        component_weights.expand(n_samples, -1), num_samples=1
    ).squeeze(-1)

    # Generate distinct mean rotations for each component
    # Spread them roughly uniformly on SO(3)
    mean_rotations = []
    for k in range(n_components):
        # Create well-separated mean rotations
        alpha_mean = 2 * math.pi * k / n_components
        beta_mean = math.pi * (0.3 + 0.4 * (k % 2))  # Alternate between different β values
        gamma_mean = math.pi * (k + 0.5) / n_components
        mean_rotations.append(torch.tensor([alpha_mean, beta_mean, gamma_mean], device=device, dtype=dtype))
    
    mean_rotations = torch.stack(mean_rotations)  # (n_components, 3)

    # Generate samples with noise around the mean rotations
    X = torch.zeros(n_samples, 3, device=device, dtype=dtype)
    
    for i in range(n_samples):
        k = assignments[i].item()
        mean = mean_rotations[k]
        
        # Add concentrated noise to each Euler angle
        # Using wrapped normal distribution approximation
        noise_std = 1.0 / math.sqrt(noise_concentration)
        noise = noise_std * torch.randn(3, device=device, dtype=dtype)
        
        euler = mean + noise
        
        # Wrap angles to proper ranges
        # α ∈ [0, 2π), γ ∈ [0, 2π)
        euler[0] = euler[0] % (2 * math.pi)
        euler[2] = euler[2] % (2 * math.pi)
        
        # β ∈ [0, π] - reflect if out of bounds
        euler[1] = euler[1].abs()
        if euler[1] > math.pi:
            euler[1] = 2 * math.pi - euler[1]
        
        X[i] = euler

    info = {
        "component_weights": component_weights,
        "mean_rotations": mean_rotations,  # (K, 3) Euler angles
        "concentration": concentration,
        "noise_concentration": noise_concentration,
    }

    return X, assignments, info


def generate_so3_rotation_matrix_data(
    n_samples: int,
    n_components: int = 3,
    noise_std: float = 0.05,
    component_weights: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate synthetic SO(3) rotation data as rotation matrices.

    Each component has a mean rotation matrix, and samples are noisy
    versions (small perturbations).

    Args:
        n_samples: Number of samples to generate
        n_components: Number of mixture components
        noise_std: Standard deviation for axis-angle noise
        component_weights: Mixture weights, shape (n_components,)
        seed: Random seed
        device: Torch device
        dtype: Torch dtype

    Returns:
        X: Generated rotation matrices, shape (n_samples, 3, 3)
        assignments: True component assignments, shape (n_samples,)
        info: Dictionary with ground truth parameters
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Default uniform weights
    if component_weights is None:
        component_weights = torch.ones(n_components, device=device, dtype=dtype) / n_components

    # Sample component assignments
    assignments = torch.multinomial(
        component_weights.expand(n_samples, -1), num_samples=1
    ).squeeze(-1)

    # Generate distinct mean rotation matrices for each component
    mean_rotations = []
    for k in range(n_components):
        # Create rotation around different axes
        angle = 2 * math.pi * k / n_components
        # Rotate around axis that varies with k
        axis = torch.tensor([
            math.cos(math.pi * k / n_components),
            math.sin(math.pi * k / n_components),
            0.5
        ], device=device, dtype=dtype)
        axis = axis / axis.norm()
        
        # Rodrigues' rotation formula
        R = _axis_angle_to_rotation_matrix(axis, angle)
        mean_rotations.append(R)
    
    mean_rotations = torch.stack(mean_rotations)  # (n_components, 3, 3)

    # Generate samples
    X = torch.zeros(n_samples, 3, 3, device=device, dtype=dtype)
    
    for i in range(n_samples):
        k = assignments[i].item()
        R_mean = mean_rotations[k]
        
        # Small random rotation as noise
        noise_axis = torch.randn(3, device=device, dtype=dtype)
        noise_axis = noise_axis / noise_axis.norm()
        noise_angle = noise_std * torch.randn(1, device=device, dtype=dtype).item()
        
        R_noise = _axis_angle_to_rotation_matrix(noise_axis, noise_angle)
        
        # Compose: R = R_noise @ R_mean
        X[i] = R_noise @ R_mean

    info = {
        "component_weights": component_weights,
        "mean_rotations": mean_rotations,  # (K, 3, 3)
        "noise_std": noise_std,
    }

    return X, assignments, info


def _axis_angle_to_rotation_matrix(
    axis: torch.Tensor, angle: float
) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    
    Args:
        axis: Unit rotation axis, shape (3,)
        angle: Rotation angle in radians
        
    Returns:
        R: Rotation matrix, shape (3, 3)
    """
    device = axis.device
    dtype = axis.dtype
    
    # Skew-symmetric matrix
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=device, dtype=dtype)
    
    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=device, dtype=dtype)
    R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    
    return R
