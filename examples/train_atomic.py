"""
Helpers for embedding 3D molecules used by the paper synthetic notebook.
"""
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Batch

from src.spaces.graph_embedding import WLHashFingerprint


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


def build_graph_from_positions(data, bond_cutoff: float = 2.0):
    """
    Build a clean PyG Data object whose edges come from a distance cutoff
    on the 3D positions. Strips string fields that break batched collation.
    """
    from torch_geometric.data import Data as PyGData

    z = data.atomic_numbers
    if z.dim() > 1:
        z = z.squeeze(-1)

    pos = data.pos
    dists = torch.cdist(pos, pos)
    mask = (dists < bond_cutoff) & (dists > 1e-6)
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)

    return PyGData(z=z.long(), pos=pos, edge_index=edge_index)


def embed_dataset_wlhash(
    pyg_dataset,
    indices,
    encoder: WLHashFingerprint,
    bond_cutoff: float = 2.0,
    batch_size: int = 64,
):
    all_embeddings = []
    for i in tqdm(range(0, len(indices), batch_size), desc="WLHash embedding"):
        batch_idx = indices[i : i + batch_size]
        batch_data = [
            build_graph_from_positions(pyg_dataset[int(j)], bond_cutoff)
            for j in batch_idx
        ]
        batch = Batch.from_data_list(batch_data)
        emb = encoder(batch)
        all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)


def extract_property(dataset, indices, property_key):
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
