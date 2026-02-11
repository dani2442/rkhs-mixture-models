"""
Graph signal basis representations with Laplacian-based geometry.
"""
import torch
from typing import Optional, Union

from .base import HilbertBasis


class GraphBasis(HilbertBasis):
    """
    Hilbert space basis for graph signals with Laplacian-induced geometry.

    For a graph G=(V,E) with weight matrix W and graph Laplacian L = D - W,
    we equip R^{|V|} with the inner product:

        ⟨f, g⟩_X = f^T (L + αI) g

    where α > 0 is a regularization parameter ensuring positive definiteness.

    The eigenvectors of L provide an orthonormal basis (w.r.t. standard inner
    product) that diagonalizes the Laplacian, with eigenvalues corresponding
    to graph frequencies.

    This class supports:
      - Standard Euclidean projection (identity basis)
      - Graph Fourier Transform (eigenvector basis)
      - Laplacian-weighted inner products
    """

    def __init__(
        self,
        num_nodes: int,
        laplacian: Optional[torch.Tensor] = None,
        weight_matrix: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        num_eigenvectors: Optional[int] = None,
        use_laplacian_inner_product: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            num_nodes: Number of nodes |V| in the graph.
            laplacian: Graph Laplacian matrix L, shape (|V|, |V|).
                       If None, must provide weight_matrix.
            weight_matrix: Adjacency weight matrix W, shape (|V|, |V|).
                          Used to compute L = D - W if laplacian not provided.
            alpha: Regularization parameter for inner product (L + αI).
            num_eigenvectors: Number of Laplacian eigenvectors to use.
                             If None, uses all |V| eigenvectors.
            use_laplacian_inner_product: If True, use ⟨f,g⟩ = f^T(L+αI)g.
                                         If False, use standard Euclidean.
            device: Torch device.
            dtype: Torch dtype.
        """
        super().__init__(device=device, dtype=dtype)
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.use_laplacian_inner_product = use_laplacian_inner_product

        # Build or store the Laplacian
        if laplacian is not None:
            self.L = laplacian.to(device=device, dtype=dtype)
        elif weight_matrix is not None:
            self.L = self._compute_laplacian(weight_matrix)
        else:
            # Default: no graph structure, use identity
            self.L = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)

        # Inner product matrix: M = L + αI
        self.M = self.L + alpha * torch.eye(num_nodes, device=device, dtype=dtype)

        # Compute eigendecomposition for basis
        self.num_eigenvectors = num_eigenvectors or num_nodes
        self._compute_eigenbasis()

    def _compute_laplacian(self, W: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized graph Laplacian L = D - W."""
        W = W.to(device=self.device, dtype=self.dtype)
        D = torch.diag(W.sum(dim=1))
        return D - W

    def _compute_eigenbasis(self):
        """Compute Laplacian eigenvectors as basis."""
        # Eigendecomposition of L (symmetric)
        eigenvalues, eigenvectors = torch.linalg.eigh(self.L)

        # Store leading eigenvectors
        self.eigenvalues = eigenvalues[: self.num_eigenvectors]
        self.U = eigenvectors[:, : self.num_eigenvectors]  # (|V|, R)

        self._M = self.num_eigenvectors

    @property
    def coeff_dim(self) -> int:
        """Return the dimension of the coefficient space."""
        return self._M

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project graph signals onto the Laplacian eigenbasis (Graph Fourier Transform).

        Args:
            X: Graph signals, shape (n, |V|) or (n, |V|, d) for vector-valued.

        Returns:
            coeffs: Fourier coefficients, shape (n, R) or (n, R*d).
        """
        if X.ndim == 2:
            # Scalar signals: (n, |V|)
            n, V = X.shape
            assert V == self.num_nodes, f"Signal dim {V} != num_nodes {self.num_nodes}"
            # GFT: coeffs = U^T @ X^T, then transpose
            return X @ self.U  # (n, R)

        elif X.ndim == 3:
            # Vector-valued signals: (n, |V|, d)
            n, V, d = X.shape
            assert V == self.num_nodes, f"Signal dim {V} != num_nodes {self.num_nodes}"
            # Project each dimension
            # (n, |V|, d) @ (|V|, R) -> need einsum
            C = torch.einsum("nvd,vr->ndr", X, self.U)  # (n, d, R)
            return C.reshape(n, d * self._M)

        else:
            raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")

    def reconstruct(self, coeffs: torch.Tensor, d: int = 1) -> torch.Tensor:
        """
        Reconstruct graph signals from Fourier coefficients (Inverse GFT).

        Args:
            coeffs: Fourier coefficients, shape (n, R) or (n, R*d).
            d: Spatial dimension for vector-valued signals.

        Returns:
            X: Reconstructed signals, shape (n, |V|) or (n, |V|, d).
        """
        n, M = coeffs.shape

        if d == 1:
            assert M == self._M, f"coeffs dim {M} != {self._M}"
            # Inverse GFT: X = U @ coeffs^T, then transpose
            return coeffs @ self.U.T  # (n, |V|)
        else:
            assert M == self._M * d, f"coeffs dim {M} != {self._M * d}"
            C = coeffs.reshape(n, d, self._M)  # (n, d, R)
            X = torch.einsum("ndr,vr->nvd", C, self.U)  # (n, |V|, d)
            return X

    def inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute inner product with optional Laplacian weighting.

        If use_laplacian_inner_product is True:
            ⟨x, y⟩ = x^T (L + αI) y  (in signal domain)

        For coefficient domain, this is more complex due to the
        non-diagonal structure in Fourier domain.
        """
        if not self.use_laplacian_inner_product:
            return super().inner_product(x, y)

        # For Laplacian inner product on coefficients:
        # Need to account for (L + αI) = U @ diag(λ + α) @ U^T
        # In Fourier domain: ⟨x,y⟩_M = Σ_r (λ_r + α) * x_r * y_r
        weights = self.eigenvalues + self.alpha  # (R,)
        return (weights * x * y).sum(dim=-1)


class GraphLaplacianBasis(GraphBasis):
    """
    Convenience class for common graph Laplacian basis construction.

    Supports construction from:
      - Explicit Laplacian matrix
      - Adjacency/weight matrix
      - NetworkX graph
    """

    @classmethod
    def from_adjacency(
        cls,
        adjacency: torch.Tensor,
        alpha: float = 1.0,
        num_eigenvectors: Optional[int] = None,
        normalized: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> "GraphLaplacianBasis":
        """
        Create basis from adjacency matrix.

        Args:
            adjacency: Adjacency matrix W, shape (|V|, |V|).
            alpha: Regularization parameter.
            num_eigenvectors: Number of basis functions.
            normalized: If True, use normalized Laplacian L_sym = D^{-1/2} L D^{-1/2}.
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            GraphLaplacianBasis instance.
        """
        W = adjacency.to(device=device, dtype=dtype)
        num_nodes = W.shape[0]

        # Compute Laplacian
        D = W.sum(dim=1)
        L = torch.diag(D) - W

        if normalized:
            # Symmetric normalized Laplacian
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-10))
            L = D_inv_sqrt @ L @ D_inv_sqrt

        return cls(
            num_nodes=num_nodes,
            laplacian=L,
            alpha=alpha,
            num_eigenvectors=num_eigenvectors,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_edge_list(
        cls,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        num_eigenvectors: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> "GraphLaplacianBasis":
        """
        Create basis from edge list representation.

        Args:
            num_nodes: Number of nodes.
            edge_index: Edge indices, shape (2, num_edges).
            edge_weight: Edge weights, shape (num_edges,). Default: all ones.
            alpha: Regularization parameter.
            num_eigenvectors: Number of basis functions.
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            GraphLaplacianBasis instance.
        """
        edge_index = edge_index.to(device=device)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=device, dtype=dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=dtype)

        # Build adjacency matrix
        W = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
        W[edge_index[0], edge_index[1]] = edge_weight
        # Make symmetric if needed
        W = 0.5 * (W + W.T)

        return cls.from_adjacency(
            adjacency=W,
            alpha=alpha,
            num_eigenvectors=num_eigenvectors,
            device=device,
            dtype=dtype,
        )
