import torch

from typing import Tuple
from mmd.kernel.gaussian import gaussian_kernel_J, gaussian_kernel_I

# ============================================================
# 3) Empirical MMD^2(P_n, Q) for Gaussian mixture Q
# ============================================================

def mmd2_empirical_vs_gaussian_mixture(
    X: torch.Tensor,          # (n, M) projected data
    pi: torch.Tensor,         # (K,) mixture weights (should sum to 1)
    m: torch.Tensor,          # (K, M)
    Kcov: torch.Tensor,       # (K, M, M)
    sigma: float,
    compute_const_term: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Returns:
      mmd2: scalar tensor
      stats: dict with const, Jbar, I

    MMD^2(P,Q) = E_{x,x'~P}[k(x,x')]
                -2 E_{x~P,y~Q}[k(x,y)]
                +E_{y,y'~Q}[k(y,y')]

    With Q = sum pi_k N(m_k,K_k),
      E_{x~P,y~Q} = sum_k pi_k * (1/n sum_i J_{i,k})
      E_{y,y'~Q} = sum_{k,s} pi_k pi_s I_{k,s}
    """
    device = X.device
    dtype = X.dtype
    n, M = X.shape
    K = pi.shape[0]

    # 1) constant term: (1/n^2) sum_{i,j} exp(-||x_i-x_j||^2/(2 sigma^2))
    if compute_const_term:
        # pairwise squared distances
        D2 = torch.cdist(X, X, p=2.0) ** 2  # (n,n)
        const = torch.exp(-D2 / (2.0 * sigma * sigma)).mean()
    else:
        const = torch.tensor(0.0, device=device, dtype=dtype)

    # 2) J and its mean over i
    J = gaussian_kernel_J(X=X, m=m, Kcov=Kcov, sigma=sigma)     # (n,K)
    Jbar = J.mean(dim=0)                                        # (K,)

    # 3) I matrix
    I = gaussian_kernel_I(m=m, Kcov=Kcov, sigma=sigma)          # (K,K)

    # mixture terms
    cross = (pi * Jbar).sum()                                   # scalar
    mixmix = pi @ I @ pi                                        # scalar

    mmd2 = const - 2.0 * cross + mixmix

    stats = {"const": const, "Jbar": Jbar, "I": I}
    return mmd2, stats