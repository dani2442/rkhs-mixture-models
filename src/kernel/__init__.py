"""
Kernel functions for MMD computation on Hilbert spaces.
"""
from .base import Kernel
from .gaussian import GaussianKernel
from .polynomial import PolynomialKernel, LinearKernel, QuadraticKernel

__all__ = [
    "Kernel",
    "GaussianKernel",
    "PolynomialKernel",
    "LinearKernel",
    "QuadraticKernel",
    # Legacy functions for backward compatibility
    "gaussian_kernel_J",
    "gaussian_kernel_I",
]
