import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch



def _log_det_from_cholesky(L: torch.Tensor) -> torch.Tensor:
    """
    L: lower Cholesky factor of A (A = L L^T), shape (..., M, M)
    returns log det(A)
    """
    return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)


def gaussian_kernel_J(
    X: torch.Tensor,          # (n, M)
    m: torch.Tensor,          # (K, M)
    Kcov: torch.Tensor,       # (K, M, M) SPD/PSD
    sigma: float,
    eps_jitter: float = 1e-8,
) -> torch.Tensor:
    """
    Compute J_{i,k} = E_{y ~ N(m_k, K_k)} [ exp(-||X_i - y||^2 / (2*sigma^2)) ].

    Closed form:
      J_{i,k} = det(I - 2 alpha K_k)^(-1/2) * exp( alpha * (x-m)^T (I-2 alpha K_k)^(-1) (x-m) )
      alpha = -1/(2*sigma^2) < 0
      I - 2 alpha K_k = I + (1/sigma^2) K_k  (since -2alpha = 1/sigma^2)
    """
    device = X.device
    dtype = X.dtype
    n, M = X.shape
    K = m.shape[0]

    alpha = -1.0 / (2.0 * sigma * sigma)

    J = torch.empty((n, K), device=device, dtype=dtype)

    I_M = torch.eye(M, device=device, dtype=dtype)

    for k in range(K):
        # A_k = I - 2 alpha K_k = I + (1/sigma^2)*K_k
        A = I_M - 2.0 * alpha * Kcov[k]
        # Stabilize if needed
        A = A + eps_jitter * I_M

        L = torch.linalg.cholesky(A)  # (M, M)
        logdetA = _log_det_from_cholesky(L)  # scalar

        diff = X - m[k].unsqueeze(0)  # (n, M)

        # quadratic form: diff^T A^{-1} diff
        # Solve A^{-1} diff^T via Cholesky solve (A X = B)
        # torch.cholesky_solve expects B shape (M, n)
        sol = torch.cholesky_solve(diff.T, L)  # (M, n)
        quad = (diff.T * sol).sum(dim=0)       # (n,)

        logJ = -0.5 * logdetA + alpha * quad
        J[:, k] = torch.exp(logJ)

    return J


def gaussian_kernel_I(
    m: torch.Tensor,          # (K, M)
    Kcov: torch.Tensor,       # (K, M, M)
    sigma: float,
    eps_jitter: float = 1e-8,
) -> torch.Tensor:
    """
    Compute I_{k,s} = E_{y~N(m_k,K_k), y'~N(m_s,K_s)}[ exp(-||y-y'||^2/(2*sigma^2)) ].

    Closed form:
      I_{k,s} = det(I - 2 alpha (K_k+K_s))^(-1/2)
                * exp( alpha * (m_k-m_s)^T (I-2 alpha (K_k+K_s))^{-1} (m_k-m_s) )
    """
    device = m.device
    dtype = m.dtype
    K = m.shape[0]
    M = m.shape[1]

    alpha = -1.0 / (2.0 * sigma * sigma)

    Imat = torch.empty((K, K), device=device, dtype=dtype)
    I_M = torch.eye(M, device=device, dtype=dtype)

    for k in range(K):
        for s in range(K):
            A = I_M - 2.0 * alpha * (Kcov[k] + Kcov[s])
            A = A + eps_jitter * I_M

            L = torch.linalg.cholesky(A)
            logdetA = _log_det_from_cholesky(L)

            dms = (m[k] - m[s]).unsqueeze(1)  # (M,1)
            sol = torch.cholesky_solve(dms, L)  # (M,1)
            quad = (dms * sol).sum()            # scalar

            logI = -0.5 * logdetA + alpha * quad
            Imat[k, s] = torch.exp(logI)

    return Imat