"""Numerically check Lemma (global smoothness): ||Hess_xi F|| <= 8 tr(A).

F(pi,xi) = sum_ks pi_k pi_s I_ks - 2 sum_k pi_k J_k, Gaussian kernel A = sigma^-2 I_M,
xi = (m_k, L_k), K_k = L_k L_k^T.
"""
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def J_I(m, L, X, sigma):
    """Closed forms of Prop. (Gaussian kernel), A = sigma^-2 I_M."""
    K = m.shape[0]
    M = m.shape[1]
    I_M = torch.eye(M, dtype=m.dtype)
    a = 1.0 / sigma ** 2  # A = a I
    Kcov = L @ L.transpose(-1, -2)  # (K,M,M)

    # J_{i,k} = det(I + a K_k)^{-1/2} exp(-1/2 (x-m)^T a (I + a K_k)^{-1} (x-m))
    Jk = []
    for k in range(K):
        Sk = I_M + a * Kcov[k]
        Ls = torch.linalg.cholesky(Sk)
        logdet = 2.0 * torch.log(torch.diagonal(Ls)).sum()
        d = (X - m[k]).T                                  # (M,n)
        sol = torch.cholesky_solve(d, Ls)                  # (M,n)
        q = a * (d * sol).sum(dim=0)                        # (n,)
        Jk.append(torch.exp(-0.5 * logdet - 0.5 * q).mean())
    Jv = torch.stack(Jk)

    Imat = []
    for k in range(K):
        row = []
        for s in range(K):
            Sks = I_M + a * (Kcov[k] + Kcov[s])
            Ls = torch.linalg.cholesky(Sks)
            logdet = 2.0 * torch.log(torch.diagonal(Ls)).sum()
            d = (m[k] - m[s]).unsqueeze(1)
            sol = torch.cholesky_solve(d, Ls)
            q = a * (d * sol).sum()
            row.append(torch.exp(-0.5 * logdet - 0.5 * q))
        Imat.append(torch.stack(row))
    return Jv, torch.stack(Imat)


def check_closed_form_vs_mc(M=3, sigma=0.8, n_mc=4_000_000):
    """Sanity: closed form J_{i,k} equals Monte-Carlo E_{y~N(m,K)} exp(-|x-y|^2/2sigma^2)."""
    m = torch.randn(1, M)
    L = 0.5 * torch.randn(1, M, M)
    X = torch.randn(1, M)
    Jv, _ = J_I(m, L, X, sigma)
    Kc = (L @ L.transpose(-1, -2))[0]
    Lc = torch.linalg.cholesky(Kc + 1e-12 * torch.eye(M))
    y = m[0] + torch.randn(n_mc, M) @ Lc.T
    mc = torch.exp(-((X[0] - y) ** 2).sum(-1) / (2 * sigma ** 2)).mean()
    return Jv.item(), mc.item()


def hess_norm(pi, m0, L0, X, sigma):
    K, M = m0.shape

    def F(flat):
        m = flat[: K * M].reshape(K, M)
        L = flat[K * M:].reshape(K, M, M)
        Jv, Imat = J_I(m, L, X, sigma)
        return pi @ Imat @ pi - 2.0 * (Jv @ pi)

    flat = torch.cat([m0.reshape(-1), L0.reshape(-1)])
    H = torch.autograd.functional.hessian(F, flat)
    return torch.linalg.matrix_norm(H, ord=2).item()


print("closed form vs Monte Carlo (J):  %.6f  vs  %.6f" % check_closed_form_vs_mc())
print()
print(" M   sig    K   regime                ||Hess||      bound 8M/sig^2   ratio")
regimes = {
    "typical":        lambda K, M: (torch.randn(K, M), 0.7 * torch.randn(K, M, M)),
    "tiny cov":       lambda K, M: (torch.randn(K, M), 1e-4 * torch.randn(K, M, M)),
    "huge cov":       lambda K, M: (torch.randn(K, M), 50.0 * torch.randn(K, M, M)),
    "far means":      lambda K, M: (300.0 * torch.randn(K, M), 0.7 * torch.randn(K, M, M)),
    "near-collapse":  lambda K, M: (1e-3 * torch.randn(K, M), 1e-3 * torch.randn(K, M, M)),
    "aniso cov":      lambda K, M: (torch.randn(K, M),
                                    torch.diag_embed(torch.logspace(-6, 3, M)).repeat(K, 1, 1)),
}
worst = 0.0
for M, sigma, K in [(2, 1.0, 2), (3, 0.5, 3), (5, 1.5, 2), (8, 0.9, 4)]:
    bound = 8.0 * M / sigma ** 2
    X = torch.randn(40, M)
    for name, gen in regimes.items():
        r = 0.0
        for _ in range(6):
            m0, L0 = gen(K, M)
            pi = torch.softmax(torch.randn(K), 0)
            r = max(r, hess_norm(pi, m0, L0, X, sigma) / bound)
        worst = max(worst, r)
        print(" %-2d  %-5.1f  %-2d  %-20s  %-11.4f   %-14.2f  %.4f"
              % (M, sigma, K, name, r * bound, bound, r))
print("\nworst ratio ||Hess|| / (8 tr A) over all trials: %.4f  -> bound %s"
      % (worst, "HOLDS" if worst <= 1.0 else "VIOLATED"))
