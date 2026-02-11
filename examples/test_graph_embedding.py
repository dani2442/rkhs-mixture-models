import hashlib
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch


class WLHashFingerprint(torch.nn.Module):
    """
    Simple, deterministic graph embedding (ECFP/Morgan-like):
      - no training
      - permutation/isomorphism-invariant (multiset neighborhoods + sorting)
      - fixed-dimensional via hashing trick

    Expects a PyG Data with:
      - either data.z (atomic numbers, shape [num_nodes]) OR data.x (node features)
      - data.edge_index (shape [2, num_edges])
      - optionally data.edge_attr (bond features; int labels or one-hot)
    """
    def __init__(
        self,
        dim: int = 128,
        radius: int = 2,
        use_edge_attr: bool = True,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.radius = int(radius)
        self.use_edge_attr = bool(use_edge_attr)
        self.l2_normalize = bool(l2_normalize)

    @staticmethod
    def _stable_hash(s: str) -> int:
        # Stable across runs/platforms (unlike Python's built-in hash()).
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def _node_labels(data: Data) -> List[int]:
        if hasattr(data, "z") and data.z is not None:
            z = data.z.view(-1)
            return z.to(torch.long).tolist()

        x = data.x
        if x is None:
            raise ValueError("Need data.z or data.x to define initial node labels.")

        if x.dtype in (torch.int64, torch.int32) and x.dim() == 1:
            return x.to(torch.long).tolist()

        # If x is one-hot or dense features, take argmax as a discrete label.
        return x.argmax(dim=-1).to(torch.long).tolist()

    @staticmethod
    def _edge_labels(data: Data, num_edges: int) -> List[int]:
        if (not hasattr(data, "edge_attr")) or (data.edge_attr is None):
            return [0] * num_edges  # single bond-type bucket

        ea = data.edge_attr
        if ea.dtype in (torch.int64, torch.int32) and ea.dim() == 1:
            return ea.to(torch.long).tolist()

        # If one-hot or dense, take argmax.
        return ea.argmax(dim=-1).to(torch.long).tolist()

    @staticmethod
    def _build_adj(
        edge_index: Tensor,
        edge_labels: List[int],
        num_nodes: int,
    ) -> List[List[Tuple[int, int]]]:
        # adjacency list: adj[u] = [(v, bond_label), ...]
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for e, (u, v) in enumerate(zip(src, dst)):
            adj[u].append((v, edge_labels[e]))
        return adj

    def _embed_one(self, data: Data) -> Tensor:
        data = data.cpu()
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        node_lbls = self._node_labels(data)
        edge_lbls = self._edge_labels(data, num_edges) if self.use_edge_attr else [0] * num_edges
        adj = self._build_adj(edge_index, edge_lbls, num_nodes)

        fp = torch.zeros(self.dim, dtype=torch.float32)

        def add_feat(token: str, count: float = 1.0):
            idx = self._stable_hash(token) % self.dim
            fp[idx] += count

        # Radius 0..R: add current node labels into the fingerprint
        for r in range(self.radius + 1):
            for lbl in node_lbls:
                add_feat(f"r={r}|lbl={lbl}")

            if r == self.radius:
                break

            # WL/Morgan update: new label = hash(current label + sorted multiset of (bond, neigh_label))
            new_lbls: List[int] = []
            for u in range(num_nodes):
                neigh = adj[u]
                if neigh:
                    multiset = sorted((b, node_lbls[v]) for (v, b) in neigh)
                    token = f"lbl={node_lbls[u]}|N={multiset}"
                else:
                    token = f"lbl={node_lbls[u]}|N=[]"
                new_lbls.append(self._stable_hash(token))
            node_lbls = new_lbls

        if self.l2_normalize:
            fp = fp / fp.norm(p=2).clamp_min(1e-12)

        return fp

    @torch.no_grad()
    def forward(self, data: Data) -> Tensor:
        # If you pass a Batch, we return [batch_size, dim]
        if isinstance(data, Batch) or (hasattr(data, "batch") and data.batch is not None):
            outs = [self._embed_one(g) for g in data.to_data_list()]
            return torch.stack(outs, dim=0)
        return self._embed_one(data)


# ---- Example usage ----
if __name__ == "__main__":
    from torch_geometric.datasets import MoleculeNet

    # Example: ESOL provides PyG Data objects (with node/edge features)
    dataset = MoleculeNet(root="data/MoleculeNet", name="ESOL")
    encoder = WLHashFingerprint(dim=2048, radius=2, use_edge_attr=True, l2_normalize=True)

    phi0 = encoder(dataset[0])          # [2048]
    phi_batch = encoder(Batch.from_data_list([dataset[0], dataset[1], dataset[2]]))  # [3, 2048]

    print(phi0.shape, phi_batch.shape)