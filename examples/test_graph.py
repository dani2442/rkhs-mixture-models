"""
Graph Laplacian Heat Trace Visualization

This script generates random Erdős-Rényi graphs using torch_geometric,
visualizes them using networkx, and plots the heat trace function tr(e^{-tL})
where L is the graph Laplacian.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import erdos_renyi_graph, to_networkx
from torch_geometric.data import Data
from scipy.linalg import expm


def compute_laplacian(G: nx.Graph) -> np.ndarray:
    """Compute the Laplacian matrix of a graph."""
    return nx.laplacian_matrix(G).toarray().astype(float)


def compute_heat_trace(L: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    """
    Compute the heat trace tr(e^{-tL}) for a range of t values.
    
    Parameters
    ----------
    L : np.ndarray
        The Laplacian matrix of the graph.
    t_values : np.ndarray
        Array of time values to evaluate the trace at.
    
    Returns
    -------
    np.ndarray
        The trace values tr(e^{-tL}) for each t.
    """
    traces = []
    for t in t_values:
        heat_kernel = expm(-t*t * L)
        traces.append(np.trace(heat_kernel))
    return np.array(traces)


def generate_erdos_renyi_graph(num_nodes: int, edge_prob: float) -> tuple[nx.Graph, Data]:
    """
    Generate an Erdős-Rényi random graph using torch_geometric.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edge_prob : float
        Probability of edge creation between any two nodes.
    
    Returns
    -------
    tuple[nx.Graph, Data]
        NetworkX graph and PyG Data object.
    """
    edge_index = erdos_renyi_graph(num_nodes, edge_prob, directed=False)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)
    return G, data


def main():
    # Configuration
    num_graphs = 4
    num_nodes = 10
    edge_probs = [0.2, 0.4, 0.6, 0.8]
    t_max = 5.0
    t_points = 200
    
    # Time values for heat trace
    t_values = np.linspace(0, t_max, t_points)
    
    # Create figure with two columns
    fig, axes = plt.subplots(num_graphs, 2, figsize=(12, 3 * num_graphs))
    
    # Color palette for heat trace curves
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_graphs))
    
    for i, edge_prob in enumerate(edge_probs):
        # Generate random graph
        G, _ = generate_erdos_renyi_graph(num_nodes, edge_prob)
        
        # Compute Laplacian and heat trace
        L = compute_laplacian(G)
        traces = compute_heat_trace(L, t_values)
        
        # Left column: Graph visualization
        ax_graph = axes[i, 0]
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos, ax=ax_graph,
            node_color=colors[i],
            node_size=400,
            edge_color='#555555',
            width=1.5,
            with_labels=True,
            font_size=10,
            font_color='white',
            font_weight='bold'
        )
        ax_graph.set_title(f'Erdős-Rényi Graph (n={num_nodes}, p={edge_prob})', 
                          fontsize=12, fontweight='bold')
        ax_graph.set_facecolor('#f8f9fa')
        
        # Right column: Heat trace plot
        ax_trace = axes[i, 1]
        ax_trace.plot(t_values, traces, color=colors[i], linewidth=2.5, label=f'p = {edge_prob}')
        ax_trace.fill_between(t_values, traces, alpha=0.2, color=colors[i])
        ax_trace.set_xlabel('t', fontsize=11)
        ax_trace.set_ylabel(r'$\mathrm{tr}(e^{-tL})$', fontsize=11)
        ax_trace.set_title(f'Heat Trace: $\\mathrm{{tr}}(e^{{-tL}})$', fontsize=12, fontweight='bold')
        ax_trace.grid(True, alpha=0.3, linestyle='--')
        ax_trace.set_xlim(0, t_max)
        ax_trace.set_ylim(0, num_nodes + 0.5)
        ax_trace.legend(loc='upper right')
        
        # Add annotation for initial trace value
        ax_trace.annotate(
            f'tr(I) = {num_nodes}', 
            xy=(0, num_nodes), 
            xytext=(0.5, num_nodes - 0.5),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
        )
    
    plt.tight_layout()
    paper_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "images")
    os.makedirs(paper_images_dir, exist_ok=True)
    out_path = os.path.join(paper_images_dir, 'graph_heat_trace.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Figure saved to '{out_path}'")


if __name__ == "__main__":
    main()
