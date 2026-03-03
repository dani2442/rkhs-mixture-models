"""
Visualization utilities for MMD-based Gaussian mixture fitting.
"""
import os

import torch
import numpy as np
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx


def plot_l2_trajectories(
    X: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    assignments: Optional[torch.Tensor] = None,
    ax: Optional["plt.Axes"] = None,
    dim: int = 0,
    alpha: float = 0.5,
    max_samples: int = 100,
    title: str = "L² Trajectories",
    cmap: str = "tab10",
) -> "plt.Axes":
    """
    Plot L² trajectories colored by component assignment.

    Args:
        X: Trajectories, shape (n, grid_size, d) or (n, grid_size)
        t: Time grid, shape (grid_size,). If None, uses indices.
        assignments: Component assignments, shape (n,)
        ax: Matplotlib axes. If None, creates new figure.
        dim: Which spatial dimension to plot (if d > 1)
        alpha: Line transparency
        max_samples: Maximum number of trajectories to plot
        title: Plot title
        cmap: Colormap name

    Returns:
        Matplotlib axes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    X_np = X.detach().cpu().numpy()
    n = X_np.shape[0]

    # Handle dimensions
    if X_np.ndim == 2:
        X_plot = X_np
    else:
        X_plot = X_np[:, :, dim]

    grid_size = X_plot.shape[1]

    if t is None:
        t_np = np.arange(grid_size)
    else:
        t_np = t.detach().cpu().numpy()

    # Subsample if too many
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
    else:
        indices = np.arange(n)

    # Get colors
    colormap = cm.get_cmap(cmap)
    if assignments is not None:
        assignments_np = assignments.detach().cpu().numpy()
        n_components = int(assignments_np.max()) + 1
        colors = [colormap(k / max(n_components - 1, 1)) for k in range(n_components)]
        
        for i in indices:
            k = int(assignments_np[i])
            ax.plot(t_np, X_plot[i], color=colors[k], alpha=alpha, linewidth=0.8)
    else:
        for i in indices:
            ax.plot(t_np, X_plot[i], alpha=alpha, linewidth=0.8, color="steelblue")

    ax.set_xlabel("t")
    ax.set_ylabel(f"x(t)" if X.ndim == 2 else f"x_{dim}(t)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_l2_means_comparison(
    X: torch.Tensor,
    predicted_means: torch.Tensor,
    basis,
    t: Optional[torch.Tensor] = None,
    true_means: Optional[torch.Tensor] = None,
    assignments: Optional[torch.Tensor] = None,
    predicted_weights: Optional[torch.Tensor] = None,
    dim: int = 0,
    alpha_data: float = 0.3,
    max_samples: int = 50,
    figsize: Tuple[int, int] = (14, 5),
    title: str = "Data vs Predicted Means",
) -> "plt.Figure":
    """
    Compare data trajectories with predicted (and optionally true) component means.

    Args:
        X: Data trajectories, shape (n, grid_size, d)
        predicted_means: Predicted mean coefficients, shape (K, M)
        basis: L2Basis for reconstructing means from coefficients
        t: Time grid
        true_means: True mean functions, shape (K, grid_size, d)
        assignments: Component assignments for data coloring
        predicted_weights: Predicted mixture weights for labeling
        dim: Spatial dimension to plot
        alpha_data: Transparency for data
        max_samples: Max data samples to show
        figsize: Figure size
        title: Main title

    Returns:
        Matplotlib figure
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    X_np = X.detach().cpu().numpy()
    n, grid_size = X_np.shape[0], X_np.shape[1]
    d = X_np.shape[2] if X_np.ndim == 3 else 1

    if t is None:
        t_np = np.arange(grid_size)
    else:
        t_np = t.detach().cpu().numpy()

    # Reconstruct predicted means
    pred_functions = basis.reconstruct(predicted_means).detach().cpu().numpy()
    K_pred = pred_functions.shape[0]

    colormap = cm.get_cmap("tab10")
    colors = [colormap(k / max(K_pred - 1, 1)) for k in range(K_pred)]

    # Left plot: Data + Predicted means
    ax1 = axes[0]
    
    # Plot data
    X_plot = X_np[:, :, dim] if X_np.ndim == 3 else X_np
    indices = np.random.choice(n, min(n, max_samples), replace=False)
    
    if assignments is not None:
        assignments_np = assignments.detach().cpu().numpy()
        for i in indices:
            k = int(assignments_np[i]) % K_pred
            ax1.plot(t_np, X_plot[i], color=colors[k], alpha=alpha_data, linewidth=0.5)
    else:
        for i in indices:
            ax1.plot(t_np, X_plot[i], color="gray", alpha=alpha_data, linewidth=0.5)

    # Plot predicted means
    pred_plot = pred_functions[:, :, dim] if pred_functions.ndim == 3 else pred_functions
    for k in range(K_pred):
        label = f"Pred μ_{k+1}"
        if predicted_weights is not None:
            w = predicted_weights[k].item() if hasattr(predicted_weights[k], 'item') else predicted_weights[k]
            label += f" (π={w:.2f})"
        ax1.plot(t_np, pred_plot[k], color=colors[k], linewidth=3, label=label)

    ax1.set_xlabel("t")
    ax1.set_ylabel(f"x_{dim}(t)")
    ax1.set_title("Data + Predicted Means")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Right plot: Predicted vs True means (if available)
    ax2 = axes[1]
    
    if true_means is not None:
        true_np = true_means.detach().cpu().numpy()
        true_plot = true_np[:, :, dim] if true_np.ndim == 3 else true_np
        K_true = true_plot.shape[0]

        for k in range(K_pred):
            ax2.plot(t_np, pred_plot[k], color=colors[k], linewidth=2, 
                    linestyle="-", label=f"Pred μ_{k+1}")

        for k in range(K_true):
            ax2.plot(t_np, true_plot[k], color=colors[k % len(colors)], 
                    linewidth=2, linestyle="--", label=f"True μ_{k+1}")

        ax2.set_title("Predicted vs True Means")
    else:
        # Just show predicted means more clearly
        for k in range(K_pred):
            ax2.plot(t_np, pred_plot[k], color=colors[k], linewidth=2, 
                    label=f"Pred μ_{k+1}")
        ax2.set_title("Predicted Component Means")

    ax2.set_xlabel("t")
    ax2.set_ylabel(f"x_{dim}(t)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def plot_graph_signals(
    X: torch.Tensor,
    adjacency: torch.Tensor,
    assignments: Optional[torch.Tensor] = None,
    node_positions: Optional[np.ndarray] = None,
    ax: Optional["plt.Axes"] = None,
    sample_idx: int = 0,
    title: str = "Graph Signal",
    cmap: str = "coolwarm",
    node_size: int = 300,
) -> "plt.Axes":
    """
    Visualize a graph signal on the graph structure.

    Args:
        X: Graph signals, shape (n, num_nodes) or (num_nodes,)
        adjacency: Adjacency matrix, shape (num_nodes, num_nodes)
        assignments: Component assignments for title
        node_positions: Pre-computed node positions, shape (num_nodes, 2)
        ax: Matplotlib axes
        sample_idx: Which sample to plot if X is 2D
        title: Plot title
        cmap: Colormap for node colors
        node_size: Size of nodes

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Get signal to plot
    if X.ndim == 2:
        signal = X[sample_idx].detach().cpu().numpy()
    else:
        signal = X.detach().cpu().numpy()

    adj_np = adjacency.detach().cpu().numpy()

    # Create networkx graph
    G = nx.from_numpy_array(adj_np)

    # Compute layout if not provided
    if node_positions is None:
        node_positions = nx.spring_layout(G, seed=42)

    # Draw
    nx.draw_networkx_edges(G, node_positions, ax=ax, alpha=0.3, edge_color="gray")
    nodes = nx.draw_networkx_nodes(
        G, node_positions, ax=ax,
        node_color=signal,
        cmap=cmap,
        node_size=node_size,
        vmin=signal.min(),
        vmax=signal.max(),
    )
    plt.colorbar(nodes, ax=ax, label="Signal value")

    if assignments is not None and X.ndim == 2:
        k = int(assignments[sample_idx].item())
        title = f"{title} (Component {k+1})"

    ax.set_title(title)
    ax.axis("off")

    return ax


def plot_graph_means_comparison(
    X: torch.Tensor,
    predicted_means: torch.Tensor,
    adjacency: torch.Tensor,
    basis,
    true_means: Optional[torch.Tensor] = None,
    predicted_weights: Optional[torch.Tensor] = None,
    node_positions: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 8),
    cmap: str = "coolwarm",
    node_size: int = 200,
) -> "plt.Figure":
    """
    Compare predicted graph signal means with data and optionally true means.

    Args:
        X: Data signals, shape (n, num_nodes)
        predicted_means: Predicted mean coefficients, shape (K, M)
        adjacency: Adjacency matrix
        basis: GraphBasis for reconstructing means
        true_means: True mean signals, shape (K, num_nodes)
        predicted_weights: Predicted mixture weights
        node_positions: Pre-computed layout
        figsize: Figure size
        cmap: Colormap
        node_size: Node size

    Returns:
        Matplotlib figure
    """
    # Reconstruct predicted means
    pred_signals = basis.reconstruct(predicted_means, d=1).detach().cpu().numpy()
    K_pred = pred_signals.shape[0]

    adj_np = adjacency.detach().cpu().numpy()
    G = nx.from_numpy_array(adj_np)

    if node_positions is None:
        node_positions = nx.spring_layout(G, seed=42)

    # Determine layout
    n_cols = K_pred
    n_rows = 2 if true_means is not None else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Global colorbar range
    vmin = min(pred_signals.min(), X.detach().cpu().numpy().min())
    vmax = max(pred_signals.max(), X.detach().cpu().numpy().max())
    if true_means is not None:
        true_np = true_means.detach().cpu().numpy()
        vmin = min(vmin, true_np.min())
        vmax = max(vmax, true_np.max())

    # Plot predicted means
    for k in range(K_pred):
        ax = axes[0, k]
        nx.draw_networkx_edges(G, node_positions, ax=ax, alpha=0.3, edge_color="gray")
        nodes = nx.draw_networkx_nodes(
            G, node_positions, ax=ax,
            node_color=pred_signals[k],
            cmap=cmap,
            node_size=node_size,
            vmin=vmin, vmax=vmax,
        )
        title = f"Pred μ_{k+1}"
        if predicted_weights is not None:
            w = predicted_weights[k].item() if hasattr(predicted_weights[k], 'item') else predicted_weights[k]
            title += f"\n(π={w:.2f})"
        ax.set_title(title)
        ax.axis("off")

    # Plot true means if available
    if true_means is not None:
        for k in range(min(K_pred, true_np.shape[0])):
            ax = axes[1, k]
            nx.draw_networkx_edges(G, node_positions, ax=ax, alpha=0.3, edge_color="gray")
            nx.draw_networkx_nodes(
                G, node_positions, ax=ax,
                node_color=true_np[k],
                cmap=cmap,
                node_size=node_size,
                vmin=vmin, vmax=vmax,
            )
            ax.set_title(f"True μ_{k+1}")
            ax.axis("off")

    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, label="Signal value")

    fig.suptitle("Graph Signal Means Comparison", fontsize=14)

    return fig


def plot_training_history(
    history: List[float],
    ax: Optional["plt.Axes"] = None,
    title: str = "Training History",
    log_scale: bool = True,
) -> "plt.Axes":
    """
    Plot MMD² training history.

    Args:
        history: List of MMD² values per epoch
        ax: Matplotlib axes
        title: Plot title
        log_scale: Use log scale for y-axis

    Returns:
        Matplotlib axes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(history) + 1)
    ax.plot(epochs, history, linewidth=2, color="steelblue")

    if log_scale and min(history) > 0:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD²")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_mixture_weights(
    predicted_weights: torch.Tensor,
    true_weights: Optional[torch.Tensor] = None,
    ax: Optional["plt.Axes"] = None,
    title: str = "Mixture Weights",
) -> "plt.Axes":
    """
    Compare predicted and true mixture weights.

    Args:
        predicted_weights: Predicted π, shape (K,)
        true_weights: True π, shape (K,)
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    pred_np = predicted_weights.detach().cpu().numpy()
    K = len(pred_np)
    x = np.arange(K)
    width = 0.35

    if true_weights is not None:
        true_np = true_weights.detach().cpu().numpy()
        ax.bar(x - width/2, true_np, width, label="True", color="steelblue", alpha=0.7)
        ax.bar(x + width/2, pred_np, width, label="Predicted", color="coral", alpha=0.7)
        ax.legend()
    else:
        ax.bar(x, pred_np, width, color="steelblue", alpha=0.7)

    ax.set_xlabel("Component")
    ax.set_ylabel("Weight (π)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k+1}" for k in range(K)])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_l2_2d_surface(
    X: torch.Tensor,
    s_grid: torch.Tensor,
    t_grid: torch.Tensor,
    ax: Optional["plt.Axes"] = None,
    sample_idx: int = 0,
    dim: int = 0,
    cmap: str = "viridis",
    alpha: float = 0.9,
    title: str = "L² 2D Function",
    xlabel: str = "s",
    ylabel: str = "t",
    zlabel: str = "f(s,t)",
) -> "plt.Axes":
    """
    Plot a single 2D function as a 3D surface mesh.

    Args:
        X: 2D functions, shape (n, grid_size_s, grid_size_t, d) or (grid_size_s, grid_size_t, d)
        s_grid: S meshgrid, shape (grid_size_s, grid_size_t)
        t_grid: T meshgrid, shape (grid_size_s, grid_size_t)
        ax: Matplotlib 3D axes. If None, creates new figure.
        sample_idx: Which sample to plot (if X has batch dimension)
        dim: Which spatial dimension to plot (if d > 1)
        cmap: Colormap for surface
        alpha: Surface transparency
        title: Plot title
        xlabel, ylabel, zlabel: Axis labels

    Returns:
        Matplotlib 3D axes
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Handle input dimensions
    X_np = X.detach().cpu().numpy()
    if X_np.ndim == 4:
        X_plot = X_np[sample_idx, :, :, dim]
    elif X_np.ndim == 3:
        X_plot = X_np[:, :, dim]
    else:
        X_plot = X_np

    S_np = s_grid.detach().cpu().numpy()
    T_np = t_grid.detach().cpu().numpy()

    # Plot surface
    surf = ax.plot_surface(
        S_np, T_np, X_plot,
        cmap=cmap, alpha=alpha, linewidth=0.2, edgecolor="k"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    return ax


def plot_l2_2d_surfaces_grid(
    X: torch.Tensor,
    s_grid: torch.Tensor,
    t_grid: torch.Tensor,
    assignments: Optional[torch.Tensor] = None,
    n_samples: int = 6,
    dim: int = 0,
    figsize: tuple = (15, 10),
    title: str = "L² 2D Functions",
    cmap: str = "tab10",
) -> "plt.Figure":
    """
    Plot multiple 2D functions as a grid of 3D surface meshes.

    Args:
        X: 2D functions, shape (n, grid_size_s, grid_size_t, d)
        s_grid: S meshgrid, shape (grid_size_s, grid_size_t)
        t_grid: T meshgrid, shape (grid_size_s, grid_size_t)
        assignments: Component assignments, shape (n,)
        n_samples: Number of samples to display
        dim: Which spatial dimension to plot (if d > 1)
        figsize: Figure size
        title: Main figure title
        cmap: Colormap for component colors

    Returns:
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X_np = X.detach().cpu().numpy()
    n_total = X_np.shape[0]
    n_samples = min(n_samples, n_total)

    # Determine grid layout
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig = plt.figure(figsize=figsize)

    S_np = s_grid.detach().cpu().numpy()
    T_np = t_grid.detach().cpu().numpy()

    colormap = cm.get_cmap(cmap)

    # Select samples (stratified by component if assignments given)
    if assignments is not None:
        assignments_np = assignments.detach().cpu().numpy()
        n_components = int(assignments_np.max()) + 1
        indices = []
        samples_per_comp = max(1, n_samples // n_components)
        for k in range(n_components):
            comp_indices = np.where(assignments_np == k)[0]
            if len(comp_indices) > 0:
                selected = np.random.choice(
                    comp_indices, min(samples_per_comp, len(comp_indices)), replace=False
                )
                indices.extend(selected)
        indices = indices[:n_samples]
    else:
        indices = np.random.choice(n_total, n_samples, replace=False)

    for plot_idx, sample_idx in enumerate(indices):
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection="3d")

        if X_np.ndim == 4:
            Z = X_np[sample_idx, :, :, dim]
        else:
            Z = X_np[sample_idx]

        # Color by component if available
        if assignments is not None:
            k = int(assignments_np[sample_idx])
            n_comp = int(assignments_np.max()) + 1
            color = colormap(k / max(n_comp - 1, 1))
            surf = ax.plot_surface(
                S_np, T_np, Z,
                color=color, alpha=0.8, linewidth=0.1, edgecolor="k"
            )
            ax.set_title(f"Sample {sample_idx} (k={k+1})", fontsize=10)
        else:
            surf = ax.plot_surface(
                S_np, T_np, Z,
                cmap="viridis", alpha=0.8, linewidth=0.1, edgecolor="k"
            )
            ax.set_title(f"Sample {sample_idx}", fontsize=10)

        ax.set_xlabel("s")
        ax.set_ylabel("t")
        ax.set_zlabel(f"x_{dim}(s,t)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def plot_l2_2d_means_comparison(
    predicted_means: torch.Tensor,
    basis,
    s_grid: torch.Tensor,
    t_grid: torch.Tensor,
    true_means: Optional[torch.Tensor] = None,
    dim: int = 0,
    figsize: tuple = (15, 10),
    title: str = "Mean Functions Comparison",
) -> "plt.Figure":
    """
    Compare predicted and true mean functions for 2D data.

    Args:
        predicted_means: Predicted mean coefficients, shape (K, M)
        basis: L2TensorBasis2D for reconstructing means from coefficients
        s_grid: S meshgrid, shape (grid_size_s, grid_size_t)
        t_grid: T meshgrid, shape (grid_size_s, grid_size_t)
        true_means: True mean functions, shape (K, grid_size_s, grid_size_t, d)
        dim: Spatial dimension to plot
        figsize: Figure size
        title: Main figure title

    Returns:
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Reconstruct predicted means
    pred_functions = basis.reconstruct(predicted_means).detach().cpu().numpy()
    K = pred_functions.shape[0]

    S_np = s_grid.detach().cpu().numpy()
    T_np = t_grid.detach().cpu().numpy()

    # 2 rows: top for predicted, bottom for true (if available)
    n_rows = 2 if true_means is not None else 1
    fig = plt.figure(figsize=figsize)

    colormap = cm.get_cmap("tab10")

    for k in range(K):
        # Predicted mean
        ax = fig.add_subplot(n_rows, K, k + 1, projection="3d")
        Z_pred = pred_functions[k, :, :, dim]
        color = colormap(k / max(K - 1, 1))
        ax.plot_surface(
            S_np, T_np, Z_pred,
            color=color, alpha=0.8, linewidth=0.1, edgecolor="k"
        )
        ax.set_title(f"Pred k={k+1}", fontsize=10)
        ax.set_xlabel("s")
        ax.set_ylabel("t")
        ax.set_zlabel(f"m_{dim}(s,t)")

        # True mean (if available)
        if true_means is not None:
            ax2 = fig.add_subplot(n_rows, K, K + k + 1, projection="3d")
            true_np = true_means.detach().cpu().numpy()
            Z_true = true_np[k, :, :, dim]
            ax2.plot_surface(
                S_np, T_np, Z_true,
                color=color, alpha=0.8, linewidth=0.1, edgecolor="k"
            )
            ax2.set_title(f"True k={k+1}", fontsize=10)
            ax2.set_xlabel("s")
            ax2.set_ylabel("t")
            ax2.set_zlabel(f"m_{dim}(s,t)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def plot_l2_2d_samples_by_component(
    X: torch.Tensor,
    assignments: torch.Tensor,
    s_grid: torch.Tensor,
    t_grid: torch.Tensor,
    dim: int = 0,
    samples_per_component: int = 2,
    figsize: tuple = (15, 8),
    title: str = "Samples by Component",
) -> "plt.Figure":
    """
    Plot samples organized by component assignment.

    Args:
        X: 2D functions, shape (n, grid_size_s, grid_size_t, d)
        assignments: Component assignments, shape (n,)
        s_grid: S meshgrid
        t_grid: T meshgrid
        dim: Spatial dimension to plot
        samples_per_component: Number of samples per component
        figsize: Figure size
        title: Main figure title

    Returns:
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X_np = X.detach().cpu().numpy()
    assignments_np = assignments.detach().cpu().numpy()
    n_components = int(assignments_np.max()) + 1

    S_np = s_grid.detach().cpu().numpy()
    T_np = t_grid.detach().cpu().numpy()

    fig = plt.figure(figsize=figsize)
    colormap = cm.get_cmap("tab10")

    plot_idx = 1
    for k in range(n_components):
        comp_indices = np.where(assignments_np == k)[0]
        n_show = min(samples_per_component, len(comp_indices))
        if n_show == 0:
            continue
        selected = np.random.choice(comp_indices, n_show, replace=False)

        color = colormap(k / max(n_components - 1, 1))

        for idx in selected:
            ax = fig.add_subplot(
                n_components, samples_per_component, plot_idx, projection="3d"
            )
            Z = X_np[idx, :, :, dim]
            ax.plot_surface(
                S_np, T_np, Z,
                color=color, alpha=0.8, linewidth=0.1, edgecolor="k"
            )
            ax.set_title(f"k={k+1}, sample {idx}", fontsize=9)
            ax.set_xlabel("s")
            ax.set_ylabel("t")
            plot_idx += 1

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Glucodensity temporal mixture visualizations
# ---------------------------------------------------------------------------


def plot_glucodensity_temporal_comparison(
    t_grid_np: np.ndarray,
    history_basis: list,
    history_ode: list,
    pi_basis_np: np.ndarray,
    pi_ode_np: np.ndarray,
    recon_basis_np: np.ndarray,
    recon_ode_np: np.ndarray,
    median_treatment_days: float,
    out_dir: str = ".",
    show: bool = True,
) -> "plt.Figure":
    """
    3x2 comparison plot for glucodensity temporal mixture fitting.

    Layout (3 rows x 2 columns):
      Row 0: [Training history]        [Final MMD² bar]
      Row 1: [Basis pi(t)]             [NeuralODE pi(t)]
      Row 2: [Basis mean curves]       [NeuralODE mean curves]

    Temporal axes show approximate treatment days or weeks.
    """
    K = pi_basis_np.shape[1]

    # Map normalized t → treatment days; switch to weeks if > 30 days
    days = t_grid_np * median_treatment_days
    use_weeks = median_treatment_days > 30
    if use_weeks:
        time_axis = days / 7.0
        time_label = "Treatment week (approx.)"
    else:
        time_axis = days
        time_label = "Treatment day (approx.)"

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # --- Row 0, Col 0: Training curves ---
    axes[0, 0].plot(history_basis, label="Basis", lw=2.0)
    axes[0, 0].plot(history_ode, label="NeuralODE", lw=2.0)
    axes[0, 0].set_title("MMD² Training History")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MMD²(avg_t)")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # --- Row 0, Col 1: Final MMD² bar chart ---
    methods = ["Basis", "NeuralODE"]
    final_vals = [history_basis[-1], history_ode[-1]]
    axes[0, 1].bar(methods, final_vals, color=["C0", "C1"], alpha=0.85)
    axes[0, 1].set_title("Final MMD²(avg_t)")
    axes[0, 1].set_ylabel("MMD²")
    axes[0, 1].grid(alpha=0.3, axis="y")

    # --- Row 1, Col 0: pi(t) for Basis ---
    for k in range(K):
        axes[1, 0].plot(time_axis, pi_basis_np[:, k], lw=1.5,
                        label=f"$\\pi_{{{k+1}}}$")
    axes[1, 0].set_title("Basis: $\\pi_k(t)$")
    axes[1, 0].set_xlabel(time_label)
    axes[1, 0].set_ylabel("Weight")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].grid(alpha=0.3)

    # --- Row 1, Col 1: pi(t) for NeuralODE ---
    for k in range(K):
        axes[1, 1].plot(time_axis, pi_ode_np[:, k], lw=1.5,
                        label=f"$\\pi_{{{k+1}}}$")
    axes[1, 1].set_title("NeuralODE: $\\pi_k(t)$")
    axes[1, 1].set_xlabel(time_label)
    axes[1, 1].set_ylabel("Weight")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].grid(alpha=0.3)

    # --- Row 2, Col 0: Basis reconstructed mean intraday curves ---
    hours = np.linspace(0, 24, recon_basis_np.shape[1])
    for k in range(K):
        axes[2, 0].plot(hours, recon_basis_np[k], lw=1.3, label=f"comp {k+1}")
    axes[2, 0].set_title("Basis: Reconstructed mean curves")
    axes[2, 0].set_xlabel("Hour of day")
    axes[2, 0].set_ylabel("Glucose (mg/dL)")
    axes[2, 0].legend(fontsize=7, ncol=2)
    axes[2, 0].grid(alpha=0.3)

    # --- Row 2, Col 1: NeuralODE reconstructed mean intraday curves ---
    for k in range(K):
        axes[2, 1].plot(hours, recon_ode_np[k], lw=1.3, label=f"comp {k+1}")
    axes[2, 1].set_title("NeuralODE: Reconstructed mean curves")
    axes[2, 1].set_xlabel("Hour of day")
    axes[2, 1].set_ylabel("Glucose (mg/dL)")
    axes[2, 1].legend(fontsize=7, ncol=2)
    axes[2, 1].grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "glucodensity_temporal_comparison.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"\nSaved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_glucodensity_variance(
    var_basis_np: np.ndarray,
    var_ode_np: np.ndarray,
    out_dir: str = ".",
    show: bool = True,
) -> "plt.Figure":
    """Variance per coefficient for both models (1x2 subplots)."""
    K = var_basis_np.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for k in range(K):
        axes[0].plot(var_basis_np[k], lw=1.2, label=f"comp {k+1}")
    axes[0].set_title("Basis: Variance per coefficient")
    axes[0].set_xlabel("Coefficient index")
    axes[0].set_ylabel("Variance")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    for k in range(K):
        axes[1].plot(var_ode_np[k], lw=1.2, label=f"comp {k+1}")
    axes[1].set_title("NeuralODE: Variance per coefficient")
    axes[1].set_xlabel("Coefficient index")
    axes[1].set_ylabel("Variance")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "glucodensity_temporal_variance.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_cluster_probabilities_by_group(
    patient_posteriors: dict,
    patient_time_days: dict,
    control_ids: list,
    treatment_ids: list,
    n_time_bins: int = 10,
    out_dir: str = ".",
    show: bool = True,
) -> "plt.Figure":
    """
    Average cluster probability over time for control vs treatment groups.

    Each group gets a subplot showing how cluster membership evolves.
    """
    control_ids_set = set(control_ids)
    treatment_ids_set = set(treatment_ids)

    K = next(iter(patient_posteriors.values())).shape[1]

    # Collect (day, posterior_vector) for each group
    control_data = []
    treatment_data = []
    for pid, posteriors in patient_posteriors.items():
        days = patient_time_days[pid]
        for i in range(len(days)):
            entry = (days[i], posteriors[i])
            if pid in control_ids_set:
                control_data.append(entry)
            elif pid in treatment_ids_set:
                treatment_data.append(entry)

    if not (control_data or treatment_data):
        print("Warning: no data for cluster probability plot.")
        return None

    # Common time range for binning
    all_days_list = [d for d, _ in control_data + treatment_data]
    max_day = max(all_days_list)
    bin_edges = np.linspace(0, max_day + 1e-9, n_time_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    def _bin_and_average(data):
        binned = [[] for _ in range(n_time_bins)]
        for day, post in data:
            b = min(
                int(np.searchsorted(bin_edges, day, side="right")) - 1,
                n_time_bins - 1,
            )
            b = max(0, b)
            binned[b].append(post)
        means = np.full((n_time_bins, K), np.nan)
        stds = np.full((n_time_bins, K), np.nan)
        counts = np.zeros(n_time_bins)
        for b in range(n_time_bins):
            if binned[b]:
                arr = np.array(binned[b])
                means[b] = arr.mean(axis=0)
                stds[b] = arr.std(axis=0)
                counts[b] = len(binned[b])
        return means, stds, counts

    ctrl_means, ctrl_stds, ctrl_counts = _bin_and_average(control_data)
    treat_means, treat_stds, treat_counts = _bin_and_average(treatment_data)

    # Decide axis label (weeks vs days)
    use_weeks = max_day > 30
    if use_weeks:
        time_axis = bin_centers / 7.0
        time_label = "Treatment week"
    else:
        time_axis = bin_centers
        time_label = "Treatment day"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = [plt.cm.tab10(k) for k in range(K)]

    for k in range(K):
        valid_c = ctrl_counts > 0
        if valid_c.any():
            axes[0].plot(
                time_axis[valid_c], ctrl_means[valid_c, k],
                lw=2, color=colors[k], label=f"Cluster {k+1}",
            )
            axes[0].fill_between(
                time_axis[valid_c],
                (ctrl_means[valid_c, k] - ctrl_stds[valid_c, k]).clip(0, 1),
                (ctrl_means[valid_c, k] + ctrl_stds[valid_c, k]).clip(0, 1),
                alpha=0.15, color=colors[k],
            )
        valid_t = treat_counts > 0
        if valid_t.any():
            axes[1].plot(
                time_axis[valid_t], treat_means[valid_t, k],
                lw=2, color=colors[k], label=f"Cluster {k+1}",
            )
            axes[1].fill_between(
                time_axis[valid_t],
                (treat_means[valid_t, k] - treat_stds[valid_t, k]).clip(0, 1),
                (treat_means[valid_t, k] + treat_stds[valid_t, k]).clip(0, 1),
                alpha=0.15, color=colors[k],
            )

    axes[0].set_title("Control group")
    axes[0].set_xlabel(time_label)
    axes[0].set_ylabel("Cluster probability")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Treatment group")
    axes[1].set_xlabel(time_label)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Cluster membership probability over time", fontsize=13)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "glucodensity_cluster_probs_by_group.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_ternary_simplex_evolution(
    patient_posteriors: dict,
    patient_time_norm: dict,
    control_ids: list,
    treatment_ids: list,
    out_dir: str = ".",
    show: bool = True,
) -> "plt.Figure":
    """
    Ternary simplex plot showing each patient's cluster trajectory.

    Requires exactly K=3 clusters.  Each patient traces a path on the
    2-simplex (equilateral triangle).  Control patients are blue,
    treatment patients are red.  Line intensity fades from light (t=0)
    to strong (t=1).
    """
    from matplotlib.lines import Line2D

    control_ids_set = set(control_ids)
    treatment_ids_set = set(treatment_ids)

    K = next(iter(patient_posteriors.values())).shape[1]
    if K != 3:
        raise ValueError(f"Ternary plot requires K=3, got K={K}")

    # Equilateral triangle vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3) / 2.0])
    vertices = np.array([v1, v2, v3])

    fig, ax = plt.subplots(figsize=(10, 9))

    # Draw triangle border
    triangle = plt.Polygon(
        [v1, v2, v3], fill=False, edgecolor="black", linewidth=2,
    )
    ax.add_patch(triangle)

    # Label vertices
    offset = 0.06
    ax.text(v1[0], v1[1] - offset, "Cluster 1", ha="center", fontsize=11,
            fontweight="bold")
    ax.text(v2[0], v2[1] - offset, "Cluster 2", ha="center", fontsize=11,
            fontweight="bold")
    ax.text(v3[0], v3[1] + offset, "Cluster 3", ha="center", fontsize=11,
            fontweight="bold")

    def _plot_trajectory(ax, posteriors, t_norm, base_color, lw=0.5):
        """Plot a single patient's trajectory on the simplex."""
        pts = posteriors @ vertices  # barycentric → 2D, shape (n_w, 2)
        n = len(pts)
        for i in range(n - 1):
            # Alpha increases with time: lighter at t=0, stronger at t=1
            alpha = 0.08 + 0.72 * t_norm[min(i + 1, n - 1)]
            ax.plot(
                pts[i:i + 2, 0], pts[i:i + 2, 1],
                color=base_color, alpha=alpha, linewidth=lw,
            )
        # Start / end markers
        if n > 0:
            ax.plot(pts[0, 0], pts[0, 1], "o", color=base_color,
                    markersize=2, alpha=0.25)
            ax.plot(pts[-1, 0], pts[-1, 1], "s", color=base_color,
                    markersize=3, alpha=0.85)

    for pid, posteriors in patient_posteriors.items():
        t_norm = patient_time_norm[pid]
        if pid in control_ids_set:
            _plot_trajectory(ax, posteriors, t_norm, "tab:blue")
        elif pid in treatment_ids_set:
            _plot_trajectory(ax, posteriors, t_norm, "tab:red")

    # Legend
    legend_elements = [
        Line2D([0], [0], color="tab:blue", lw=2, label="Control"),
        Line2D([0], [0], color="tab:red", lw=2, label="Treatment"),
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
               markersize=6, lw=0, label="Start ($t=0$)"),
        Line2D([0], [0], marker="s", color="gray", markerfacecolor="gray",
               markersize=6, lw=0, label="End ($t=1$)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.15, v3[1] + 0.12)
    ax.set_aspect("equal")
    ax.set_title("Patient evolution in cluster probability simplex",
                 fontsize=13)
    ax.axis("off")

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "glucodensity_ternary_evolution.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
