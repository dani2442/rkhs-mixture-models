#!/usr/bin/env python
"""
Example: Training a Gaussian Mixture Model on 3D molecular datasets.

Uses the atomic-datasets library for data loading and py3Dmol for 3D visualization.

This script demonstrates:
1. Loading molecules from atomic-datasets (tmQM, QM9, GEOMDrugs, etc.)
2. Building molecular graphs from 3D coordinates (distance-based edges)
3. Embedding molecules via WLHashFingerprint → R^n (permutation-invariant)
4. Projecting embeddings through a Discrete Cosine basis (dimensionality reduction)
5. Training a Gaussian mixture using MMD minimization
6. Visualizing: training history, mixture weights, t-SNE, property distribution
7. Interactive 3D visualization of representative molecules via py3Dmol
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Batch
from atomic_datasets.wrappers.torch import PyTorchGeometricDataset

from src.spaces.graph_embedding import WLHashFingerprint
from src import (
    DiscreteCosineBasis,
    GaussianKernel,
    GaussianMixtureModel,
    plot_mixture_weights,
    plot_training_history,
)

# ==========================================
# Dataset registry
# ==========================================
DATASET_REGISTRY = {
    "tmqm": {
        "class_name": "tmQM",
        "property_key": "HL_Gap",
        "property_label": "HOMO-LUMO Gap (Ha)",
    },
    "qm9": {
        "class_name": "QM9",
        "property_key": "gap",
        "property_label": "HOMO-LUMO Gap (Ha)",
    },
    "geom_drugs": {
        "class_name": "GEOMDrugs",
        "property_key": None,
        "property_label": None,
    },
    "chembl3d": {
        "class_name": "ChEMBL3D",
        "property_key": None,
        "property_label": None,
    },
    "crossdocked": {
        "class_name": "CrossDocked",
        "property_key": None,
        "property_label": None,
    },
}


# ==========================================
# Graph construction + WLHash embedding
# ==========================================
def build_graph_from_positions(data, bond_cutoff: float = 2.0):
    """
    Create a clean PyG Data with graph structure derived from 3D positions.

    Returns a new Data with only tensor fields that Batch.from_data_list
    can collate (z, edge_index, pos, num_nodes).
    """
    from torch_geometric.data import Data as PyGData

    # Atomic numbers → z
    z = data.atomic_numbers
    if z.dim() > 1:
        z = z.squeeze(-1)

    # Build edges from pairwise distances
    pos = data.pos
    dists = torch.cdist(pos, pos)
    mask = (dists < bond_cutoff) & (dists > 1e-6)
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)

    # Return a *clean* Data — no string fields that break batching
    return PyGData(z=z.long(), pos=pos, edge_index=edge_index)


def embed_dataset_wlhash(
    pyg_dataset,
    indices,
    encoder: WLHashFingerprint,
    bond_cutoff: float = 2.0,
    batch_size: int = 64,
):
    """
    Embed molecules using WLHashFingerprint.

    Constructs molecular graphs from 3D positions (distance cutoff),
    then applies the WL hash embedding in batches.
    """
    all_embeddings = []
    for i in tqdm(range(0, len(indices), batch_size), desc="WLHash embedding"):
        batch_idx = indices[i : i + batch_size]
        batch_data = [
            build_graph_from_positions(pyg_dataset[int(j)], bond_cutoff)
            for j in batch_idx
        ]
        batch = Batch.from_data_list(batch_data)
        emb = encoder(batch)  # (batch_size, wl_dim)
        all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)


# ==========================================
# Coulomb matrix embedding
# ==========================================
def coulomb_matrix_eigenvalues(positions, atomic_numbers, max_atoms):
    """
    Compute sorted Coulomb matrix eigenvalues as a fixed-size molecular fingerprint.

    The Coulomb matrix C is defined as:
        C_{ii} = 0.5 * Z_i^{2.4}           (diagonal)
        C_{ij} = Z_i * Z_j / |r_i - r_j|   (off-diagonal)

    Its sorted eigenvalues form a permutation-invariant, size-consistent
    representation of the molecule's geometry and composition.
    """
    pos = np.asarray(positions, dtype=np.float64)
    z = np.asarray(atomic_numbers, dtype=np.float64)
    n = len(pos)

    diff = pos[:, None, :] - pos[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    zz = np.outer(z, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(dists > 1e-10, zz / dists, 0.0)
    np.fill_diagonal(C, 0.5 * z**2.4)

    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.sort(np.abs(eigvals))[::-1]

    fp = np.zeros(max_atoms, dtype=np.float64)
    k = min(n, max_atoms)
    fp[:k] = eigvals[:k]
    return fp


def embed_dataset_coulomb(dataset, indices, max_atoms):
    """Embed molecules using Coulomb matrix eigenvalues."""
    embeddings = []
    for idx in tqdm(indices, desc="Coulomb embedding"):
        graph = dataset[idx]
        fp = coulomb_matrix_eigenvalues(
            graph["nodes"]["positions"],
            graph["nodes"]["atomic_numbers"],
            max_atoms,
        )
        embeddings.append(fp)
    return torch.tensor(np.stack(embeddings), dtype=torch.float64)


# ==========================================
# Property extraction
# ==========================================
def extract_property(dataset, indices, property_key):
    """Extract a scalar property for a subset of molecules."""
    if property_key is None:
        return None
    values = []
    for idx in indices:
        graph = dataset[idx]
        props = graph.get("properties") or {}
        val = props.get(property_key)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                values.append(float("nan"))
        else:
            values.append(float("nan"))
    return torch.tensor(values, dtype=torch.float64)


# ==========================================
# py3Dmol grid visualization
# ==========================================
def _to_xyz(positions, atom_types, comment=""):
    """Convert positions and atom types to XYZ format string."""
    n = len(positions)
    lines = [str(n), comment]
    for atom, pos in zip(atom_types, positions):
        lines.append(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    return "\n".join(lines)


def create_molecule_grid(
    dataset,
    representative_indices,
    subset_indices,
    gamma,
    property_values,
    property_key,
    n_components,
    n_representative,
    model,
):
    """
    Create a py3Dmol viewergrid of representative molecules per component.

    Returns:
        py3Dmol.view: The viewer object (can be displayed in notebooks or saved).
    """
    import py3Dmol

    cell_w, cell_h = 300, 250
    view = py3Dmol.view(
        width=n_representative * cell_w,
        height=n_components * cell_h,
        viewergrid=(n_components, n_representative),
    )
    view.setBackgroundColor("#ffffff")

    for k in range(n_components):
        for j in range(len(representative_indices[k])):
            local_idx = representative_indices[k][j]
            real_idx = int(subset_indices[local_idx])

            graph = dataset[real_idx]
            positions = graph["nodes"]["positions"]
            atom_types = graph["nodes"]["atom_types"]

            # Build comment string
            resp_val = gamma[local_idx, k].item()
            comment = f"Component {k + 1} | idx={real_idx} | gamma={resp_val:.2f}"
            if property_values is not None:
                prop_val = property_values[local_idx].item()
                comment += f" | {property_key}={prop_val:.3f}"

            xyz_str = _to_xyz(positions, atom_types, comment)

            view.addModel(xyz_str, "xyz", viewer=(k, j))
            view.setStyle(
                {"sphere": {"scale": 0.25}, "stick": {"radius": 0.1}},
                viewer=(k, j),
            )

            # Add a label at the centroid
            center = np.mean(positions, axis=0)
            label_parts = [f"K{k + 1} | γ={resp_val:.2f}"]
            if property_values is not None:
                prop_val = property_values[local_idx].item()
                label_parts.append(f"{property_key}={prop_val:.2f}")

            view.addLabel(
                "\n".join(label_parts),
                {
                    "position": {
                        "x": float(center[0]),
                        "y": float(center[1] + 2.5),
                        "z": float(center[2]),
                    },
                    "fontSize": 9,
                    "fontColor": "white",
                    "backgroundColor": "rgba(0,0,0,0.6)",
                    "backgroundOpacity": 0.5,
                },
                viewer=(k, j),
            )

            view.zoomTo(viewer=(k, j))

    return view


# ==========================================
# Argument parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Gaussian Mixture Model on 3D molecular datasets "
        "using MMD minimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="qm9",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to use.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/atomic_datasets",
        help="Root directory for dataset storage.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--n_subset",
        type=int,
        default=1000,
        help="Number of molecules to sample from the dataset.",
    )
    parser.add_argument(
        "--property",
        type=str,
        default=None,
        help="Property key to analyze per component. If None, uses dataset default.",
    )

    # Embedding
    parser.add_argument(
        "--embedding",
        type=str,
        default="wlhash",
        choices=["wlhash", "coulomb"],
        help="Molecular embedding method.",
    )
    parser.add_argument(
        "--wl_dim", type=int, default=128, help="WLHash fingerprint dimension."
    )
    parser.add_argument(
        "--wl_radius", type=int, default=2, help="WL neighborhood radius."
    )
    parser.add_argument(
        "--bond_cutoff",
        type=float,
        default=2.0,
        help="Distance cutoff (Å) for building molecular graph edges.",
    )

    # Model
    parser.add_argument("--n_components", type=int, default=8, help="GMM components.")
    parser.add_argument("--R", type=int, default=64, help="DCT coefficients to keep.")
    parser.add_argument(
        "--sigma", type=float, default=2.0, help="Gaussian kernel bandwidth."
    )

    # Training
    parser.add_argument("--num_epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Visualization
    parser.add_argument(
        "--n_representative",
        type=int,
        default=5,
        help="Representative molecules per component.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs. Defaults to paper/images/.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display matplotlib plots interactively."
    )

    return parser.parse_args()


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = DATASET_REGISTRY[args.dataset]
    property_key = args.property or config["property_key"]
    property_label = config["property_label"] or property_key or ""

    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 60)
    print(f"3D Molecules ({config['class_name']}): MMD-based Gaussian Mixture Fitting")
    print("=" * 60)

    # ==========================================
    # 1. Load dataset
    # ==========================================
    print(f"\n[1] Loading {config['class_name']} dataset...")

    from atomic_datasets import datasets as ds_module

    DatasetClass = getattr(ds_module, config["class_name"])
    dataset = DatasetClass(root_dir=args.data_dir, split=args.split)

    n_total = len(dataset)
    print(f"    Total molecules in split '{args.split}': {n_total}")

    # Wrap as PyG dataset for WLHash
    pyg_dataset = PyTorchGeometricDataset(dataset)

    # ==========================================
    # 2. Sample subset
    # ==========================================
    n_subset = min(args.n_subset, n_total)
    print(f"\n[2] Sampling {n_subset} molecules from {n_total}...")

    rng = np.random.default_rng(args.seed)
    subset_indices = rng.choice(n_total, size=n_subset, replace=False)
    subset_indices.sort()

    # ==========================================
    # 3. Extract properties
    # ==========================================
    print(f"\n[3] Extracting property '{property_key}'...")

    property_values = extract_property(dataset, subset_indices, property_key)
    if property_values is not None:
        valid_mask = ~torch.isnan(property_values)
        n_valid = valid_mask.sum().item()
        print(f"    Valid property values: {n_valid}/{n_subset}")
        if n_valid > 0:
            vals = property_values[valid_mask]
            print(f"    Range: [{vals.min():.4f}, {vals.max():.4f}]")
            print(f"    Mean ± Std: {vals.mean():.4f} ± {vals.std():.4f}")
    else:
        print("    No property configured for this dataset.")

    # ==========================================
    # 4. Embed molecules
    # ==========================================
    if args.embedding == "wlhash":
        print(f"\n[4] Embedding molecules with WLHashFingerprint...")
        encoder = WLHashFingerprint(
            dim=args.wl_dim,
            radius=args.wl_radius,
            use_edge_attr=False,
            l2_normalize=True,
        )
        X_raw = embed_dataset_wlhash(
            pyg_dataset, subset_indices, encoder, bond_cutoff=args.bond_cutoff
        ).to(dtype)
        embed_dim = args.wl_dim
    else:
        print(f"\n[4] Embedding molecules with Coulomb matrix eigenvalues...")
        # Determine max atoms in subset for fixed-size fingerprint
        n_atoms_list = []
        for idx in tqdm(subset_indices, desc="Scanning atoms", leave=False):
            graph = dataset[idx]
            n_atoms_list.append(int(graph["n_node"][0]))
        max_atoms = max(n_atoms_list)
        print(f"    Atoms per molecule: mean={np.mean(n_atoms_list):.1f}, max={max_atoms}")

        X_raw = embed_dataset_coulomb(dataset, subset_indices, max_atoms)
        # L2-normalize for consistency
        norms = X_raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
        X_raw = X_raw / norms
        embed_dim = max_atoms

    print(f"    Embedding dimension: {embed_dim}")
    print(f"    Embedding shape: {X_raw.shape}")
    print(
        f"    Non-zero entries per molecule (avg): "
        f"{(X_raw != 0).float().mean(dim=1).mean():.4f}"
    )

    # ==========================================
    # 5. Project onto DCT basis
    # ==========================================
    R = min(args.R, embed_dim)  # can't keep more coefficients than embedding dim
    print(f"\n[5] Projecting onto Discrete Cosine basis (R={R})...")

    basis = DiscreteCosineBasis(n=embed_dim, R=R, device=device, dtype=dtype)
    X = basis.project(X_raw)  # (n_subset, R)
    M = X.shape[1]

    # Reconstruction quality
    X_recon = basis.reconstruct(X)
    recon_error = (X_raw - X_recon).norm() / X_raw.norm()

    print(f"    DCT coefficients kept: R = {R} (of {embed_dim})")
    print(f"    Coefficient dimension M = {M}")
    print(f"    Reconstruction relative error: {recon_error:.4f}")

    # ==========================================
    # 6. Create kernel and model
    # ==========================================
    print(f"\n[6] Setting up model and kernel...")

    kernel = GaussianKernel(sigma=args.sigma)

    model = GaussianMixtureModel(
        num_components=args.n_components,
        coeff_dim=M,
        basis=basis,
        covariance_type="diagonal",
        device=device,
        dtype=dtype,
    )
    model.initialize_from_data(X, method="kmeans++")

    print(f"    Kernel: Gaussian (σ={args.sigma})")
    print(f"    Model: {model}")
    print(f"    Initial weights: {model.pi.detach().numpy()}")

    # ==========================================
    # 7. Train the model
    # ==========================================
    print(f"\n[7] Training model by minimizing MMD²...")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Precompute constant term
    gram = kernel.compute_gram_matrix(X)
    const_term = gram.mean()

    history = []
    print_interval = max(1, args.num_epochs // 10)

    for epoch in range(args.num_epochs):
        optimizer.zero_grad()

        mmd2, stats = model.compute_mmd2(X, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()

        full_mmd2 = const_term + mmd2.detach()
        history.append(full_mmd2.item())

        if (epoch + 1) % print_interval == 0:
            print(
                f"    Epoch {epoch + 1:4d}/{args.num_epochs}: "
                f"MMD² = {full_mmd2.item():.6f}, "
                f"π = {model.pi.detach().numpy()}"
            )

    print(f"\n    Final MMD² = {history[-1]:.6f}")
    print(f"    Final weights: {model.pi.detach().numpy()}")

    # ==========================================
    # 8. Assign molecules to components
    # ==========================================
    print(f"\n[8] Assigning molecules to components...")

    with torch.no_grad():
        gamma = model.responsibilities(X)  # (n_subset, K)
        assignments = gamma.argmax(dim=1)  # (n_subset,)

    for k in range(args.n_components):
        mask = assignments == k
        count = mask.sum().item()
        if count > 0:
            msg = f"    Component {k + 1}: {count:4d} molecules"
            if property_values is not None:
                prop_k = property_values[mask]
                valid_k = prop_k[~torch.isnan(prop_k)]
                if len(valid_k) > 0:
                    msg += (
                        f", {property_key} = {valid_k.mean():.3f} ± "
                        f"{valid_k.std():.3f} [{valid_k.min():.3f}, {valid_k.max():.3f}]"
                    )
            print(msg)

    # ==========================================
    # 9. Find representative molecules per component
    # ==========================================
    print(f"\n[9] Finding representative molecules...")

    with torch.no_grad():
        means = model.mean  # (K, M)

    representative_indices = {}
    for k in range(args.n_components):
        mask_k = (assignments == k).nonzero(as_tuple=True)[0]
        if len(mask_k) == 0:
            representative_indices[k] = torch.tensor([], dtype=torch.long)
            continue
        X_k = X[mask_k]
        dists = (X_k - means[k].unsqueeze(0)).norm(dim=1)
        n_rep = min(args.n_representative, len(mask_k))
        top_idx = dists.argsort()[:n_rep]
        representative_indices[k] = mask_k[top_idx]

    # ==========================================
    # 10. Matplotlib visualizations
    # ==========================================
    print(f"\n[10] Creating matplotlib visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Training history
    plot_training_history(history, ax=axes[0, 0], title="MMD² Training History")

    # (b) Mixture weights
    plot_mixture_weights(model.pi, ax=axes[0, 1], title="Learned Mixture Weights")

    # (c) Property distribution per component
    ax = axes[1, 0]
    if property_values is not None:
        colors = plt.cm.Set2(np.linspace(0, 1, args.n_components))
        for k in range(args.n_components):
            mask = (assignments == k).numpy()
            if mask.any():
                vals_k = property_values[mask].numpy()
                valid = vals_k[~np.isnan(vals_k)]
                if len(valid) > 0:
                    ax.hist(
                        valid,
                        bins=25,
                        alpha=0.6,
                        color=colors[k],
                        label=f"Comp {k + 1} (n={mask.sum()})",
                        edgecolor="white",
                        linewidth=0.5,
                    )
        ax.set_xlabel(property_label)
        ax.set_ylabel("Count")
        ax.set_title(f"{property_label} Distribution by Component")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No property configured\nfor this dataset",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Property Distribution (N/A)")

    # (d) t-SNE of DCT coefficients, colored by component
    ax = axes[1, 1]
    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30)
        X_2d = tsne.fit_transform(X.detach().numpy())
        colors = plt.cm.Set2(np.linspace(0, 1, args.n_components))
        for k in range(args.n_components):
            mask = (assignments == k).numpy()
            if mask.any():
                ax.scatter(
                    X_2d[mask, 0],
                    X_2d[mask, 1],
                    c=[colors[k]],
                    s=12,
                    alpha=0.6,
                    label=f"Comp {k + 1}",
                )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE of DCT Coefficients")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    except ImportError:
        ax.text(
            0.5,
            0.5,
            "Install scikit-learn\nfor t-SNE visualization",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("t-SNE (scikit-learn not installed)")

    fig.suptitle(
        f"Molecular Mixture Model ({config['class_name']} — {n_subset} molecules)",
        fontsize=15,
    )
    plt.tight_layout()

    # ==========================================
    # 11. py3Dmol representative molecules
    # ==========================================
    print(f"\n[11] Creating py3Dmol 3D visualization...")

    try:
        view = create_molecule_grid(
            dataset=dataset,
            representative_indices=representative_indices,
            subset_indices=subset_indices,
            gamma=gamma,
            property_values=property_values,
            property_key=property_key,
            n_components=args.n_components,
            n_representative=args.n_representative,
            model=model,
        )
        has_3d_view = True
    except Exception as e:
        print(f"    Warning: py3Dmol visualization failed: {e}")
        has_3d_view = False

    # ==========================================
    # 12. Save outputs
    # ==========================================
    print(f"\n[12] Saving outputs...")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper",
            "images",
        )
    os.makedirs(output_dir, exist_ok=True)

    ds_tag = args.dataset

    # Save matplotlib figure
    pdf_path = os.path.join(output_dir, f"{ds_tag}_mixture.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"    Saved training plots → {pdf_path}")

    # Save py3Dmol HTML
    if has_3d_view:
        html_path = os.path.join(output_dir, f"{ds_tag}_representatives_{args.n_components}.html")
        view.write_html(html_path)
        print(f"    Saved 3D molecule grid → {html_path}")

    if args.show:
        print(f"\n[13] Displaying matplotlib plots...")
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
