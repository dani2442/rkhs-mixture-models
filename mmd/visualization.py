"""
Visualization utilities for MMD-based Gaussian mixture fitting.
"""
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
