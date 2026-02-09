#!/usr/bin/env python
"""
Example: Training a Gaussian Mixture Model on molecular fingerprints.

This script demonstrates:
1. Loading molecules from MoleculeNet (ESOL dataset)
2. Embedding molecules via WLHashFingerprint -> R^n
3. Projecting embeddings through a Discrete Cosine basis (dimensionality reduction)
4. Training a Gaussian mixture using MMD minimization
5. Visualizing: training history, mixture weights, t-SNE, property distribution
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Batch
from torch_geometric.datasets import MoleculeNet

from examples.test_graph_embedding import WLHashFingerprint
from src import (
    DiscreteCosineBasis,
    GaussianKernel,
    GaussianMixtureModel,
    plot_training_history,
    plot_mixture_weights,
)


def embed_dataset(dataset, encoder: WLHashFingerprint, batch_size: int = 64):
    """Embed all molecules in a dataset using WLHashFingerprint."""
    all_embeddings = []
    n = len(dataset)
    for i in range(0, n, batch_size):
        batch_data = [dataset[j] for j in range(i, min(i + batch_size, n))]
        batch = Batch.from_data_list(batch_data)
        emb = encoder(batch)  # (batch_size, dim)
        all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)


def main():
    # ==========================================
    # Configuration
    # ==========================================
    torch.manual_seed(42)

    # WLHash parameters
    wl_dim = 512
    wl_radius = 2

    # Basis parameters
    R = 64  # Number of DCT coefficients to keep

    # GMM parameters
    n_components = 10

    # Kernel parameters
    sigma_kernel = 2.0

    # Training parameters
    num_epochs = 300
    lr = 0.05

    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 60)
    print("Molecules (ESOL): MMD-based Gaussian Mixture Fitting")
    print("=" * 60)

    # ==========================================
    # Load dataset
    # ==========================================
    print("\n[1] Loading MoleculeNet ESOL dataset...")

    dataset = MoleculeNet(root="data/MoleculeNet", name="ESOL")
    n_molecules = len(dataset)

    # Extract solubility labels (regression target)
    solubility = torch.tensor(
        [dataset[i].y.item() for i in range(n_molecules)], dtype=dtype
    )

    print(f"   {n_molecules} molecules loaded")
    print(f"   Solubility range: [{solubility.min():.2f}, {solubility.max():.2f}]")

    # ==========================================
    # Embed molecules
    # ==========================================
    print("\n[2] Embedding molecules with WLHashFingerprint...")

    encoder = WLHashFingerprint(
        dim=wl_dim, radius=wl_radius, use_edge_attr=True, l2_normalize=True
    )
    X_raw = embed_dataset(dataset, encoder).to(dtype)  # (n_molecules, wl_dim)

    print(f"   Embedding dimension: {wl_dim}")
    print(f"   Embedding shape: {X_raw.shape}")
    print(f"   Non-zero entries per molecule (avg): {(X_raw != 0).float().mean(dim=1).mean():.4f}")

    # ==========================================
    # Project onto DCT basis
    # ==========================================
    print("\n[3] Projecting onto Discrete Cosine basis...")

    basis = DiscreteCosineBasis(n=wl_dim, R=R, device=device, dtype=dtype)
    X = basis.project(X_raw)  # (n_molecules, R)
    M = X.shape[1]

    # Check reconstruction quality
    X_recon = basis.reconstruct(X)
    recon_error = (X_raw - X_recon).norm() / X_raw.norm()

    print(f"   DCT coefficients kept: R = {R} (of {wl_dim})")
    print(f"   Coefficient dimension M = {M}")
    print(f"   Reconstruction relative error: {recon_error:.4f}")

    # ==========================================
    # Create kernel and model
    # ==========================================
    print("\n[4] Setting up model and kernel...")

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
    print("\n[5] Training model by minimizing MMD²...")

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
    # Assign molecules to components
    # ==========================================
    print("\n[6] Assigning molecules to components...")

    with torch.no_grad():
        gamma = model.responsibilities(X)  # (n_molecules, K)
        assignments = gamma.argmax(dim=1)   # (n_molecules,)

    for k in range(n_components):
        mask = assignments == k
        count = mask.sum().item()
        if count > 0:
            sol_k = solubility[mask]
            print(
                f"   Component {k+1}: {count:4d} molecules, "
                f"solubility = {sol_k.mean():.2f} ± {sol_k.std():.2f} "
                f"[{sol_k.min():.2f}, {sol_k.max():.2f}]"
            )

    # ==========================================
    # Visualize results
    # ==========================================
    print("\n[7] Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Training history
    plot_training_history(history, ax=axes[0, 0], title="MMD² Training History")

    # (b) Mixture weights
    plot_mixture_weights(model.pi, ax=axes[0, 1], title="Learned Mixture Weights")

    # (c) Solubility distribution per component
    ax = axes[1, 0]
    colors = plt.cm.Set2(np.linspace(0, 1, n_components))
    for k in range(n_components):
        mask = (assignments == k).numpy()
        if mask.any():
            sol_k = solubility[mask].numpy()
            ax.hist(
                sol_k, bins=25, alpha=0.6, color=colors[k],
                label=f"Component {k+1} (n={mask.sum()})",
                edgecolor="white", linewidth=0.5,
            )
    ax.set_xlabel("log Solubility (mol/L)")
    ax.set_ylabel("Count")
    ax.set_title("Solubility Distribution by Component")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) t-SNE of DCT coefficients, colored by component
    ax = axes[1, 1]
    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(X.detach().numpy())
        for k in range(n_components):
            mask = (assignments == k).numpy()
            if mask.any():
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=[colors[k]], s=12, alpha=0.6,
                    label=f"Component {k+1}",
                )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE of DCT Coefficients")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    except ImportError:
        ax.text(
            0.5, 0.5, "Install scikit-learn\nfor t-SNE visualization",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        ax.set_title("t-SNE (scikit-learn not installed)")

    fig.suptitle("Molecular Mixture Model (ESOL Dataset)", fontsize=15)
    plt.tight_layout()

    # ==========================================
    # Representative molecular graphs per component
    # ==========================================
    print("\n   Creating representative molecule plots...")
    import networkx as nx

    # Element labels from atomic number (covers common organic atoms)
    ATOM_LABELS = {
        1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
        17: "Cl", 35: "Br", 53: "I",
    }
    # CPK-inspired colors with better contrast
    ATOM_COLORS = {
        "C": "#2C2C2C", "N": "#4169E1", "O": "#DC143C", "F": "#32CD32",
        "S": "#DAA520", "P": "#FF6347", "Cl": "#00CED1", "Br": "#8B0000",
        "I": "#9400D3", "H": "#A0A0A0",
    }

    # Distinct, vibrant palette for K components
    COMPONENT_COLORS = [
        "#4C72B0",  # steel blue
        "#DD8452",  # coral orange
        "#55A868",  # sage green
        "#C44E52",  # muted red
        "#8172B3",  # lavender purple
        "#937860",  # warm brown
        "#DA8BC3",  # soft pink
        "#8C8C8C",  # neutral gray
    ]
    # Very light tints for backgrounds (alpha-like via hex)
    COMPONENT_BG = [
        "#E8EEF7",  # light steel blue
        "#FBF0E8",  # light coral
        "#E8F5EC",  # light sage
        "#F7E8E8",  # light rose
        "#EEEBF5",  # light lavender
        "#F2EDE8",  # light tan
        "#F9EDF5",  # light pink
        "#F0F0F0",  # light gray
    ]

    n_representative = 5  # molecules per component
    fig_rep, axes_rep = plt.subplots(
        n_components, n_representative,
        figsize=(3 * n_representative, 2 * n_components),
    )
    if n_components == 1:
        axes_rep = axes_rep[np.newaxis, :]

    with torch.no_grad():
        means = model.mean  # (K, M)

    for k in range(n_components):
        comp_color = COMPONENT_COLORS[k % len(COMPONENT_COLORS)]
        comp_bg = COMPONENT_BG[k % len(COMPONENT_BG)]

        # Indices belonging to component k
        mask_k = (assignments == k).nonzero(as_tuple=True)[0]
        X_k = X[mask_k]  # (n_k, M)

        # Distance to component mean
        dists = (X_k - means[k].unsqueeze(0)).norm(dim=1)
        top_idx = dists.argsort()[:n_representative]
        representative_idx = mask_k[top_idx]

        for j, mol_idx in enumerate(representative_idx):
            ax = axes_rep[k, j]

            # Set per-component background color
            # ax.set_facecolor(comp_bg)
            for spine in ax.spines.values():
                spine.set_visible(False)
                # spine.set_color(comp_color)

            data = dataset[int(mol_idx)]

            # Build networkx graph
            G = nx.Graph()
            num_nodes = data.num_nodes

            # Node labels: use atomic number (data.x argmax) or data.z
            if hasattr(data, "z") and data.z is not None:
                atom_nums = data.z.view(-1).tolist()
            elif data.x is not None:
                atom_nums = data.x.argmax(dim=-1).tolist()
            else:
                atom_nums = [6] * num_nodes  # default to carbon

            node_labels = {}
            node_colors = []
            for i in range(num_nodes):
                label = ATOM_LABELS.get(atom_nums[i], str(atom_nums[i]))
                G.add_node(i)
                node_labels[i] = label
                node_colors.append(ATOM_COLORS.get(label, "#CCCCCC"))

            # Edges
            edge_index = data.edge_index
            src_list = edge_index[0].tolist()
            dst_list = edge_index[1].tolist()
            seen = set()
            for u, v in zip(src_list, dst_list):
                if u < v and (u, v) not in seen:
                    G.add_edge(u, v)
                    seen.add((u, v))

            # Draw with component-aware styling
            pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, num_nodes ** 0.5))
            nx.draw_networkx_edges(
                G, pos, ax=ax, edge_color=comp_color, width=1.8, alpha=0.5,
            )
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color=node_colors,
                node_size=100, edgecolors="#222222", linewidths=1.0,
            )
            nx.draw_networkx_labels(
                G, pos, labels=node_labels, ax=ax,
                font_size=6, font_color="white", font_weight="bold",
            )

            sol_val = solubility[mol_idx].item()
            resp_val = gamma[mol_idx, k].item()
            ax.set_title(
                f"#{int(mol_idx)} · sol={sol_val:.1f} · γ={resp_val:.2f}",
                fontsize=8, color="#333333", pad=4,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # Row label
        axes_rep[k, 0].annotate(
            f"Component {k+1}\n(π={model.pi[k].item():.3f})",
            xy=(-0.18, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", ha="right", va="center",
            rotation=90, color=comp_color,
        )

    fig_rep.suptitle(
        "Most Representative Molecules per Component", fontsize=14, y=1.01,
        fontweight="bold",
    )
    fig_rep.tight_layout()

    # Save figures
    print("\n[8] Saving figures as PDF...")
    paper_images_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "paper", "images",
    )
    os.makedirs(paper_images_dir, exist_ok=True)

    fig.savefig(
        os.path.join(paper_images_dir, "molecule_mixture.pdf"),
        format="pdf", bbox_inches="tight",
    )
    fig_rep.savefig(
        os.path.join(paper_images_dir, "molecule_representatives.pdf"),
        format="pdf", bbox_inches="tight",
    )
    print(f"   Saved to {paper_images_dir}/molecule_mixture.pdf")
    print(f"   Saved to {paper_images_dir}/molecule_representatives.pdf")

    print("\n[9] Displaying plots...")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
