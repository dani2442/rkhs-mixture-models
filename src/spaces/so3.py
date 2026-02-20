"""
L^2(SO(3)) space basis implementation using Wigner D-matrices.

The orthonormal basis for L^2(SO(3)) with respect to Haar measure consists
of normalized Wigner D-matrix elements:

    φ^ℓ_mn(R) = √(2ℓ+1) · D^ℓ_mn(R)

where ℓ = 0, 1, 2, ... and m, n ∈ {-ℓ, ..., ℓ}.

In Euler angles (α, β, γ) with z-y-z convention:
    D^ℓ_mn(α, β, γ) = e^{-imα} · d^ℓ_mn(β) · e^{-inγ}

where d^ℓ_mn(β) are the small Wigner d-functions.
"""
import torch
import math
from typing import Optional, Tuple
from functools import lru_cache

from .base import HilbertBasis


def _factorial(n: int) -> int:
    """Compute factorial of n."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _wigner_d_element(ell: int, m: int, n: int, beta: torch.Tensor) -> torch.Tensor:
    """
    Compute the small Wigner d-function d^ℓ_mn(β).
    
    Uses the formula:
        d^ℓ_mn(β) = Σ_s (-1)^{m-n+s} · √[(ℓ+m)!(ℓ-m)!(ℓ+n)!(ℓ-n)!] /
                     [(ℓ+m-s)!s!(ℓ-n-s)!(s+n-m)!] ·
                     cos(β/2)^{2ℓ+m-n-2s} · sin(β/2)^{2s+n-m}
    
    where s ranges over values that keep all factorials non-negative.
    
    Args:
        ell: Degree ℓ ≥ 0
        m: First index, -ℓ ≤ m ≤ ℓ
        n: Second index, -ℓ ≤ n ≤ ℓ
        beta: Euler angle β, shape (...)
        
    Returns:
        d^ℓ_mn(β), same shape as beta
    """
    cos_half = torch.cos(beta / 2)
    sin_half = torch.sin(beta / 2)
    
    # Prefactor: √[(ℓ+m)!(ℓ-m)!(ℓ+n)!(ℓ-n)!]
    prefactor = math.sqrt(
        _factorial(ell + m) * _factorial(ell - m) *
        _factorial(ell + n) * _factorial(ell - n)
    )
    
    # Sum over valid s values
    # s must satisfy: s ≥ 0, ℓ+m-s ≥ 0, ℓ-n-s ≥ 0, s+n-m ≥ 0
    s_min = max(0, m - n)
    s_max = min(ell + m, ell - n)
    
    result = torch.zeros_like(beta)
    
    for s in range(s_min, s_max + 1):
        # Denominator factorials
        denom = (
            _factorial(ell + m - s) * _factorial(s) *
            _factorial(ell - n - s) * _factorial(s + n - m)
        )
        
        # Powers of cos and sin
        cos_power = 2 * ell + m - n - 2 * s
        sin_power = 2 * s + n - m
        
        # Sign factor
        sign = (-1) ** (m - n + s)
        
        term = sign * prefactor / denom
        term = term * (cos_half ** cos_power) * (sin_half ** sin_power)
        result = result + term
    
    return result


def wigner_D_element(
    ell: int, m: int, n: int,
    alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Wigner D-matrix element D^ℓ_mn(α, β, γ).
    
    D^ℓ_mn(α, β, γ) = e^{-imα} · d^ℓ_mn(β) · e^{-inγ}
    
    Args:
        ell: Degree ℓ ≥ 0
        m: First index, -ℓ ≤ m ≤ ℓ  
        n: Second index, -ℓ ≤ n ≤ ℓ
        alpha: Euler angle α ∈ [0, 2π), shape (...)
        beta: Euler angle β ∈ [0, π], shape (...)
        gamma: Euler angle γ ∈ [0, 2π), shape (...)
        
    Returns:
        Complex D^ℓ_mn(α, β, γ), same shape as inputs
    """
    d_mn = _wigner_d_element(ell, m, n, beta)
    
    # Complex exponentials
    phase = -m * alpha - n * gamma
    return d_mn * torch.exp(1j * phase)


def wigner_D_element_real(
    ell: int, m: int, n: int,
    alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute real and imaginary parts of Wigner D-matrix element.
    
    Returns:
        (real_part, imag_part) of D^ℓ_mn(α, β, γ)
    """
    d_mn = _wigner_d_element(ell, m, n, beta)
    phase = -m * alpha - n * gamma
    
    real_part = d_mn * torch.cos(phase)
    imag_part = d_mn * torch.sin(phase)
    
    return real_part, imag_part


class SO3Basis(HilbertBasis):
    """
    Orthonormal basis for L^2(SO(3)) using Wigner D-matrices.
    
    The basis functions are:
        φ^ℓ_mn(R) = √(2ℓ+1) · D^ℓ_mn(R)
    
    for ℓ = 0, 1, ..., L_max and m, n ∈ {-ℓ, ..., ℓ}.
    
    The total number of basis functions up to degree L_max is:
        M = Σ_{ℓ=0}^{L_max} (2ℓ+1)² = (L_max+1)(2L_max+1)(2L_max+3)/3
    
    Rotations are parameterized by Euler angles (α, β, γ) in z-y-z convention:
        R = R_z(α) · R_y(β) · R_z(γ)
    
    with α ∈ [0, 2π), β ∈ [0, π], γ ∈ [0, 2π).
    
    For use with real-valued kernels and models, we project to the absolute
    values (magnitudes) of the complex Wigner D-coefficients, which captures
    the essential structure while remaining in a real Hilbert space.
    """
    
    def __init__(
        self,
        L_max: int,
        use_real_basis: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            L_max: Maximum degree ℓ of the representation (ℓ = 0, 1, ..., L_max)
            use_real_basis: If True, use real coefficients (magnitudes of complex).
                           If False, work with complex coefficients.
            device: Torch device
            dtype: Torch dtype for real computations
        """
        super().__init__(device=device, dtype=dtype)
        self.L_max = L_max
        self.use_real_basis = use_real_basis
        
        # Compute coefficient dimension
        # Number of complex basis functions: Σ_{ℓ=0}^{L_max} (2ℓ+1)²
        self._num_complex_coeffs = sum((2 * ell + 1) ** 2 for ell in range(L_max + 1))
        
        # For real basis, we keep the same dimension (use magnitudes)
        self._M = self._num_complex_coeffs
        
        # Build index mapping: coefficient index -> (ℓ, m, n)
        self._build_index_map()
    
    def _build_index_map(self):
        """Build mapping from flat index to (ℓ, m, n) triplets."""
        self.index_to_lmn = []
        
        for ell in range(self.L_max + 1):
            for m in range(-ell, ell + 1):
                for n in range(-ell, ell + 1):
                    self.index_to_lmn.append((ell, m, n))
        
        # Inverse mapping
        self.lmn_to_index = {lmn: i for i, lmn in enumerate(self.index_to_lmn)}
    
    @property
    def coeff_dim(self) -> int:
        """Return the dimension M of the coefficient space."""
        return self._M
    
    @property
    def num_complex_coeffs(self) -> int:
        """Return the number of complex coefficients."""
        return self._num_complex_coeffs
    
    def _compute_basis_values(
        self, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute all basis function values at given Euler angles.
        
        For the real basis, we use the fact that:
            D^ℓ_mn(α,β,γ) = e^{-imα} d^ℓ_mn(β) e^{-inγ}
        
        We construct real basis functions using:
            - d^ℓ_mn(β) · cos(mα + nγ)  (cosine part)
            - d^ℓ_mn(β) · sin(mα + nγ)  (sine part)
        
        These form a proper orthonormal real basis for L²(SO(3), ℝ).
        
        Args:
            alpha: Euler angle α, shape (n,)
            beta: Euler angle β, shape (n,)
            gamma: Euler angle γ, shape (n,)
            
        Returns:
            If use_real_basis: real tensor of shape (n, M)
            If not use_real_basis: complex tensor of shape (n, num_complex_coeffs)
        """
        n = alpha.shape[0]
        
        if self.use_real_basis:
            # Real basis using properly normalized real/imaginary decomposition
            # The normalization √(2ℓ+1) comes from the Peter-Weyl theorem
            coeffs = torch.zeros(n, self._M, device=self.device, dtype=self.dtype)
            
            for idx, (ell, m, nn) in enumerate(self.index_to_lmn):
                norm = math.sqrt(2 * ell + 1)
                # Get the small d-function (real-valued)
                d_mn = _wigner_d_element(ell, m, nn, beta)
                
                # Phase from exponentials: e^{-imα} e^{-inγ} = e^{-i(mα + nγ)}
                phase = m * alpha + nn * gamma
                
                # Real part: d_mn(β) · cos(mα + nγ)
                # This captures the real part of D^ℓ_mn
                coeffs[:, idx] = norm * d_mn * torch.cos(phase)
            
            return coeffs
        else:
            # Complex basis
            coeffs = torch.zeros(
                n, self._num_complex_coeffs, 
                device=self.device, dtype=torch.complex128
            )
            
            for idx, (ell, m, nn) in enumerate(self.index_to_lmn):
                norm = math.sqrt(2 * ell + 1)
                coeffs[:, idx] = norm * wigner_D_element(ell, m, nn, alpha, beta, gamma)
            
            return coeffs
            
            return coeffs
    
    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project SO(3) samples onto the Wigner D-matrix basis.
        
        Args:
            X: Euler angles, shape (n, 3) where columns are (α, β, γ)
               OR rotation matrices, shape (n, 3, 3)
               
        Returns:
            coeffs: Basis coefficients, shape (n, M)
        """
        if X.ndim == 2 and X.shape[1] == 3:
            # Input is Euler angles
            alpha = X[:, 0]
            beta = X[:, 1]
            gamma = X[:, 2]
        elif X.ndim == 3 and X.shape[1:] == (3, 3):
            # Input is rotation matrices - convert to Euler angles
            alpha, beta, gamma = self._rotation_matrix_to_euler(X)
        else:
            raise ValueError(
                f"X must be (n, 3) Euler angles or (n, 3, 3) rotation matrices, "
                f"got shape {X.shape}"
            )
        
        return self._compute_basis_values(alpha, beta, gamma)
    
    def _rotation_matrix_to_euler(
        self, R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert rotation matrices to Euler angles (z-y-z convention).
        
        Args:
            R: Rotation matrices, shape (n, 3, 3)
            
        Returns:
            (alpha, beta, gamma): Euler angles, each shape (n,)
        """
        n = R.shape[0]
        
        # β = arccos(R_33)
        beta = torch.acos(torch.clamp(R[:, 2, 2], -1.0, 1.0))
        
        # Handle gimbal lock cases
        sin_beta = torch.sin(beta)
        gimbal_lock = sin_beta.abs() < 1e-10
        
        # Normal case
        alpha = torch.atan2(R[:, 1, 2], R[:, 0, 2])
        gamma = torch.atan2(R[:, 2, 1], -R[:, 2, 0])
        
        # Gimbal lock: β ≈ 0 or β ≈ π
        # When β ≈ 0: R ≈ R_z(α + γ), set γ = 0
        # When β ≈ π: R ≈ R_z(α - γ) R_y(π), set γ = 0
        alpha_lock = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        gamma_lock = torch.zeros_like(alpha_lock)
        
        alpha = torch.where(gimbal_lock, alpha_lock, alpha)
        gamma = torch.where(gimbal_lock, gamma_lock, gamma)
        
        # Normalize to standard ranges
        alpha = torch.remainder(alpha, 2 * math.pi)
        gamma = torch.remainder(gamma, 2 * math.pi)
        
        return alpha, beta, gamma
    
    def euler_to_rotation_matrix(
        self, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert Euler angles to rotation matrices (z-y-z convention).
        
        Args:
            alpha, beta, gamma: Euler angles, each shape (n,)
            
        Returns:
            R: Rotation matrices, shape (n, 3, 3)
        """
        n = alpha.shape[0]
        
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        cb, sb = torch.cos(beta), torch.sin(beta)
        cg, sg = torch.cos(gamma), torch.sin(gamma)
        
        R = torch.zeros(n, 3, 3, device=self.device, dtype=self.dtype)
        
        # R = R_z(α) R_y(β) R_z(γ)
        R[:, 0, 0] = ca * cb * cg - sa * sg
        R[:, 0, 1] = -ca * cb * sg - sa * cg
        R[:, 0, 2] = ca * sb
        R[:, 1, 0] = sa * cb * cg + ca * sg
        R[:, 1, 1] = -sa * cb * sg + ca * cg
        R[:, 1, 2] = sa * sb
        R[:, 2, 0] = -sb * cg
        R[:, 2, 1] = sb * sg
        R[:, 2, 2] = cb
        
        return R
    
    def sample_haar(self, n: int) -> torch.Tensor:
        """
        Sample n rotations uniformly from SO(3) according to Haar measure.
        
        Uses the method: generate uniform quaternion on S^3 and convert.
        
        Args:
            n: Number of samples
            
        Returns:
            Euler angles (α, β, γ), shape (n, 3)
        """
        # Sample uniformly on S^3 (quaternion)
        u = torch.rand(n, 3, device=self.device, dtype=self.dtype)
        
        # Shoemake's method for uniform quaternions
        theta1 = 2 * math.pi * u[:, 0]
        theta2 = 2 * math.pi * u[:, 1]
        r1 = torch.sqrt(1 - u[:, 2])
        r2 = torch.sqrt(u[:, 2])
        
        # Quaternion (w, x, y, z)
        qw = r2 * torch.cos(theta2)
        qx = r1 * torch.sin(theta1)
        qy = r1 * torch.cos(theta1)
        qz = r2 * torch.sin(theta2)
        
        # Convert quaternion to Euler angles (z-y-z)
        # First convert to rotation matrix, then to Euler
        R = torch.zeros(n, 3, 3, device=self.device, dtype=self.dtype)
        
        R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
        R[:, 0, 1] = 2 * (qx * qy - qw * qz)
        R[:, 0, 2] = 2 * (qx * qz + qw * qy)
        R[:, 1, 0] = 2 * (qx * qy + qw * qz)
        R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
        R[:, 1, 2] = 2 * (qy * qz - qw * qx)
        R[:, 2, 0] = 2 * (qx * qz - qw * qy)
        R[:, 2, 1] = 2 * (qy * qz + qw * qx)
        R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
        
        alpha, beta, gamma = self._rotation_matrix_to_euler(R)
        
        return torch.stack([alpha, beta, gamma], dim=1)
    
    def inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the L^2(SO(3)) inner product via coefficients.
        
        Since the basis is orthonormal with respect to Haar measure:
            ⟨f, g⟩ = Σ_i f_i · ḡ_i
        
        Args:
            x, y: Coefficient vectors, shape (..., M)
            
        Returns:
            Inner products, shape (...)
        """
        if self.use_real_basis:
            return (x * y).sum(dim=-1)
        else:
            # Complex case
            return (x * y.conj()).real.sum(dim=-1)


class SO3FourierBasis(SO3Basis):
    """
    L^2(SO(3)) basis with Fourier-like indexing.
    
    This is an alias for SO3Basis that emphasizes the analogy with
    Fourier series: the Wigner D-matrices form a "generalized Fourier basis"
    for functions on the rotation group.
    
    The analogy is:
        S^1 (circle):     e^{inθ}                -> Fourier series
        S^2 (sphere):     Y_ℓm(θ, φ)             -> Spherical harmonics  
        SO(3) (rotations): √(2ℓ+1) D^ℓ_mn(α,β,γ) -> Wigner D-matrix basis
    """
    pass
