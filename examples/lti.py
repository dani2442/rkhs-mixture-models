"""
LTI system identification via MMD in L².

We consider the SDE  dx = (Ax + Bu)dt + G dW  with Gaussian initial
conditions, which produces a Gaussian process.  Given sample paths we
estimate A, B, G by minimising MMD² between the empirical measure and a
single‑component Gaussian in the coefficient space of an L² cosine basis.

The mean and covariance of the Gaussian component are obtained analytically
from the current parameter estimates (they satisfy an ODE in time).
"""

import torch
import torch.nn as nn
import torchsde
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import L2CosineBasis, GaussianKernel


# ──────────────────────────────────────────────
# 1.  SDE for sampling paths  dx = (Ax+Bu)dt + GdW
# ──────────────────────────────────────────────
class LinearSDE(nn.Module):
    noise_type = "additive"
    sde_type = "ito"

    def __init__(self, A, B, G, u_fn):
        """
        A : (n, n)   state matrix
        B : (n, m)   input matrix
        G : (n, p)   diffusion matrix
        u_fn: t -> (m,) control signal
        """
        super().__init__()
        self.A = A
        self.B = B
        self.G = G
        self.u_fn = u_fn

    def f(self, t, x):
        # x: (batch, n)
        u = self.u_fn(t)  # (m,)
        return x @ self.A.T + u.unsqueeze(0) @ self.B.T  # (batch, n)

    def g(self, t, x):
        # must return (batch, n, p)
        batch = x.shape[0]
        return self.G.unsqueeze(0).expand(batch, -1, -1)


# ──────────────────────────────────────────────
# 2.  ODE for mean / covariance of the GP
#     dm/dt = A m + B u,   m(0) = m0
#     dP/dt = A P + P A^T + G G^T,   P(0) = Σ0
# ──────────────────────────────────────────────
class MeanCovODE(nn.Module):
    """Deterministic ODE wrapped as an SDE (g=0) so torchsde can integrate."""
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, A, B, G, u_fn, d):
        super().__init__()
        self.A = A
        self.B = B
        self.GGT = G @ G.T
        self.u_fn = u_fn
        self.d = d

    def f(self, t, y):
        d = self.d
        batch = y.shape[0]
        m = y[:, :d]
        P = y[:, d:].view(batch, d, d)
        u = self.u_fn(t)
        if u.dim() == 1:
            u = u.unsqueeze(0).expand(batch, -1)
        dm = m @ self.A.T + u @ self.B.T
        dP = self.A @ P + P @ self.A.T + self.GGT
        return torch.cat([dm, dP.reshape(batch, d * d)], dim=-1)

    def g(self, t, y):
        return torch.zeros_like(y)


# ──────────────────────────────────────────────
# 3.  Compute projected mean & covariance in
#     the L² cosine basis from (A,B,G,m0,Σ0,u)
# ──────────────────────────────────────────────
def compute_projected_mean_cov(A, B, G, m0, Sigma0, u_fn, basis: L2CosineBasis):
    """
    Solve for m(t), P(t) on the basis grid, then project:
      μ_coeff = basis.project(m(t))              (1, M)
      K_coeff[r,r'] = Σ_l  Σ_l'  Φ[l,r] P(t_l,t_l') Φ[l',r'] w[l] w[l']

    For the GP of the LTI system, K(t,s) = Φ(t,min(t,s)) where Φ is the
    state‑transition matrix – but it is easier to use the fact that
      P(t) = Cov[x(t), x(t)]
    and the cross‑covariance is  C(t,s) = e^{A(t-s)} P(s) for t>=s.
    We compute K_coeff via the formula using P(t) only, by noting:
      Σ_{r,q}  K_{(r,d1),(r',d2)} = ∫∫ φ_r(t) C_{d1,d2}(t,s) φ_{r'}(s) ds dt

    For simplicity we just do:
      1) Solve for P(t) on the grid
      2) Approximate the integral by quadrature using the state-transition matrix
         (but the cross-covariance C(t,s)=e^{A(t-s)}P(s) for t>s needs the
          matrix exponential at every pair — expensive).

    Instead we use a simpler approach:  since the paths live in R^d at each
    time, the coefficient covariance is:
       K_coeff[(q,r),(q',r')] = ∫∫ φ_r(t) φ_{r'}(s) C_{q,q'}(t,s) dt ds

    We approximate this by forward‑Euler sampling of C(t,s) on the grid.
    For t>=s:  C(t,s) = e^{A(t-s)} P(s).  We precompute e^{A*dt*k}.

    Returns
    -------
    mu  : (M,)        projected mean coefficients
    Kcov: (M, M)      projected covariance matrix
    """
    d = A.shape[0]
    T = basis.T
    grid_size = basis.grid_size
    R = basis.R
    M = basis.coeff_dim  # R * d

    ts = basis.t             # (L,)
    dt_val = basis.dt
    Phi = basis.Phi          # (L, R)
    w = basis.w              # (L,)

    # --- solve m(t), P(t) ---
    ode = MeanCovODE(A, B, G, u_fn, d)
    y0 = torch.cat([m0.reshape(-1), Sigma0.reshape(-1)]).unsqueeze(0)
    ys = torchsde.sdeint(ode, y0, ts, dt=dt_val, method="euler")  # (L,1,d+d²)
    m_ts = ys[:, 0, :d]                            # (L, d)
    P_ts = ys[:, 0, d:].view(grid_size, d, d)      # (L, d, d)

    # --- projected mean:  μ = basis.project(m(t)) ---
    # m_ts is (L, d), we need (1, L, d) for project
    mu = basis.project(m_ts.unsqueeze(0)).squeeze(0)   # (M,)

    # --- projected covariance ---
    # K_coeff[(q,r),(q',r')] = ∫∫ φ_r(t) φ_{r'}(s) C_{q,q'}(t,s) dt ds
    # with C(t,s) = e^{A(t-s)} P(s)  for t>=s,  C(t,s) = P(t) e^{A^T(s-t)}  for s>=t
    # We symmetrise: C(t,s) = C(s,t)^T
    #
    # Precompute matrix exponentials e^{A * k*dt} for k=0..L-1
    L_ = grid_size
    expA = [torch.eye(d, dtype=A.dtype, device=A.device)]
    eA_dt = torch.matrix_exp(A * dt_val)
    for k in range(1, L_):
        expA.append(expA[-1] @ eA_dt)
    expA = torch.stack(expA, dim=0)   # (L, d, d)

    # Build K_coeff by quadrature
    # weighted_Phi[l, r] = Phi[l, r] * w[l]
    wPhi = Phi * w[:, None]  # (L, R)

    # C(t_i, t_j) for i>=j: expA[i-j] @ P_ts[j]
    # K_coeff[(q,r),(q',r')]
    #   = Σ_i Σ_j  wPhi[i,r] * wPhi[j,r'] * C_{q,q'}(t_i, t_j)
    # We accumulate blocks of size (R, R) for each (q, q') pair.
    Kcov = torch.zeros(M, M, dtype=A.dtype, device=A.device)

    # For efficiency, accumulate  S[r,r'] = Σ_i Σ_j wPhi[i,r]*wPhi[j,r']*C_{qq'}(i,j)
    # by iterating over lag k = |i-j|.
    # C(t_i, t_j) for i>=j:  expA[i-j] @ P_ts[j]
    # contribution at lag k:  Σ_{j=0}^{L-1-k} wPhi[j+k, :] ⊗ wPhi[j, :] * (expA[k] @ P_ts[j])
    # plus symmetric part for k>0

    for k in range(L_):
        # C_lag[j, q, q'] = (expA[k] @ P_ts[j])[q, q']  for j=0..L-1-k
        # shape: (L-k, d, d)
        C_lag = torch.einsum("ab,jbc->jac", expA[k], P_ts[:L_ - k])  # (L-k, d, d)

        # outer_basis[j, r, r'] = wPhi[j+k, r] * wPhi[j, r']
        outer = torch.einsum("ir,jr->ijr", wPhi[k:], wPhi[:L_ - k])
        # actually we need: Σ_j wPhi[j+k,r] * wPhi[j,r'] * C_lag[j,q,q']
        # result shape: (R, R, d, d)
        block = torch.einsum("jr,js,jqd->rsqd", wPhi[k:], wPhi[:L_ - k], C_lag)
        # block[r,r',q,q'] = Σ_j wPhi[j+k,r]*wPhi[j,r']*C_lag[j,q,q']

        # Place into Kcov with index mapping: (q,r) -> q*R + r
        for q in range(d):
            for qp in range(d):
                Kcov[q * R:(q + 1) * R, qp * R:(qp + 1) * R] += block[:, :, q, qp]

        if k > 0:
            # symmetric part: C(t_j, t_{j+k}) = C(t_{j+k}, t_j)^T
            block_T = block.permute(1, 0, 3, 2)  # swap (r,r') and (q,q')
            for q in range(d):
                for qp in range(d):
                    Kcov[q * R:(q + 1) * R, qp * R:(qp + 1) * R] += block_T[:, :, q, qp]

    # Symmetrise for numerical safety
    Kcov = 0.5 * (Kcov + Kcov.T)

    return mu, Kcov


# ──────────────────────────────────────────────
# 4.  MMD² between empirical samples and a
#     single Gaussian N(μ, K) in coeff space
# ──────────────────────────────────────────────
def mmd2_empirical_vs_single_gaussian(X, mu, Kcov, kernel):
    """
    MMD²(P_n, N(μ,K))  with Gaussian kernel.

    X   : (n, M) data coefficients
    mu  : (M,)   Gaussian mean
    Kcov: (M, M) Gaussian covariance
    kernel: GaussianKernel

    Uses the closed‑form expectations from the kernel object
    with K=1 component and π=1.
    """
    m = mu.unsqueeze(0)           # (1, M)
    K = Kcov.unsqueeze(0)         # (1, M, M)
    pi = torch.ones(1, dtype=X.dtype, device=X.device)

    # constant term  (1/n²) Σ_{i,j} κ(x_i, x_j)
    gram = kernel.compute_gram_matrix(X)   # (n, n)
    const = gram.mean()

    # cross term  (1/n) Σ_i E_{y~N}[κ(x_i, y)]
    J = kernel.compute_J(X, m, K)          # (n, 1)
    cross = J.mean()

    # mixture‑mixture  E_{y,y'~N}[κ(y,y')]
    I = kernel.compute_I(m, K)             # (1, 1)
    mixmix = I[0, 0]

    return const - 2.0 * cross + mixmix


# ──────────────────────────────────────────────
# 5.  Trainable LTI model
# ──────────────────────────────────────────────
class TrainableLTI(nn.Module):
    """
    Parameterises A, B, G as nn.Parameters and exposes the
    projected Gaussian (mu, Kcov) in the L² basis.
    """

    def __init__(self, n, m, p, basis, u_fn, m0, Sigma0):
        super().__init__()
        self.n = n
        self.m_in = m
        self.p = p
        self.basis = basis
        self.u_fn = u_fn
        self.m0 = m0
        self.Sigma0 = Sigma0

        # initialise at random stable A
        A_init = torch.randn(n, n)
        A_init = A_init - (torch.linalg.eigvals(A_init).real.max().item() + 0.5) * torch.eye(n)
        self.A = nn.Parameter(A_init)
        self.B = nn.Parameter(torch.randn(n, m) * 0.1)
        # parameterise G via its matrix so that GG^T is automatically PSD
        self.G = nn.Parameter(torch.randn(n, p) * 0.1)

    def projected_gaussian(self):
        """Return (mu, Kcov) in coefficient space."""
        return compute_projected_mean_cov(
            self.A, self.B, self.G,
            self.m0, self.Sigma0, self.u_fn, self.basis,
        )


# ──────────────────────────────────────────────
# 6.  Main script
# ──────────────────────────────────────────────
def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    # --- dimensions ---
    n, m, p = 4, 2, 2          # state, input, noise dims
    R = 7                      # basis functions per spatial dim
    T = 5.0
    dt = 0.1
    grid_size = int(T / dt) + 1
    n_samples = 20

    # --- true system ---
    A_true = torch.randn(n, n)
    A_true = A_true - (torch.linalg.eigvals(A_true).real.max().real + 0.5) * torch.eye(n)
    B_true = torch.randn(n, m)
    G_true = torch.randn(n, p)/n

    m0 = torch.zeros(n)
    Sigma0 = 0.1 * torch.eye(n)

    def u_fn(t):
        return torch.stack([torch.sin(t), torch.cos(2.0 * t)])

    # --- sample paths ---
    ts = torch.linspace(0.0, T, grid_size)
    sde = LinearSDE(A_true, B_true, G_true, u_fn)
    x0 = m0 + torch.randn(n_samples, n) @ torch.linalg.cholesky(Sigma0).T
    paths = torchsde.sdeint(sde, x0, ts, dt=dt, method="euler", dt_min=0.05)  # (L, n_samples, n)
    paths = paths.permute(1, 0, 2)  # (n_samples, L, n)

    print(f"Sampled {n_samples} paths,  shape = {paths.shape}")

    # --- project onto L² cosine basis ---
    basis = L2CosineBasis(T, R, grid_size, d=n)
    X_coeffs = basis.project(paths)  # (n_samples, M) with M = R*n
    print(f"Coefficient dim M = {X_coeffs.shape[1]}")

    # --- true projected Gaussian (for reference) ---
    mu_true, Kcov_true = compute_projected_mean_cov(
        A_true, B_true, G_true, m0, Sigma0, u_fn, basis,
    )

    # --- kernel ---
    sigma_kernel = float(X_coeffs.std()) * 2.0
    kernel = GaussianKernel(sigma=sigma_kernel)
    print(f"Kernel bandwidth σ = {sigma_kernel:.4f}")

    # --- trainable model ---
    model = TrainableLTI(n, m, p, basis, u_fn, m0, Sigma0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # --- training loop ---
    n_epochs = 300
    history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        mu_est, Kcov_est = model.projected_gaussian()
        loss = mmd2_empirical_vs_single_gaussian(X_coeffs, mu_est, Kcov_est, kernel)

        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if (epoch + 1) % 20 == 0:
            # spectral abscissa of current A
            eigs = torch.linalg.eigvals(model.A.data).real.max().item()
            print(f"Epoch {epoch+1:4d}/{n_epochs}  MMD² = {loss.item():.6e}  "
                  f"max Re(λ(A)) = {eigs:.4f}")

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) sample paths (first coordinate)
    ax = axes[0, 0]
    n_plot = min(30, n_samples)
    cmap = plt.cm.Blues
    for i in range(n_plot):
        color = cmap(0.3 + 0.7 * i / max(n_plot - 1, 1))
        ax.plot(ts.numpy(), paths[i, :, 0].numpy(), color=color,
                alpha=0.6, linewidth=0.8)
    ax.set_title("Sample paths (dim 0)")
    ax.set_xlabel("t")

    # (b) training loss
    ax = axes[0, 1]
    ax.semilogy(history)
    ax.set_title("MMD² loss")
    ax.set_xlabel("epoch")
    ax.grid(True)

    # (c) reconstructed mean vs true mean (first coordinate)
    mu_est_final, Kcov_est_final = model.projected_gaussian()
    mean_true_fn = basis.reconstruct(mu_true.unsqueeze(0)).squeeze(0)   # (L, n)
    mean_est_fn = basis.reconstruct(mu_est_final.unsqueeze(0)).squeeze(0)

    colors = plt.cm.tab10.colors
    ax = axes[1, 0]
    for q in range(n):
        c = colors[q % len(colors)]
        ax.plot(ts.numpy(), mean_true_fn[:, q].detach().numpy(),
                color=c, linewidth=2.2, label=f"true dim {q}")
        ax.plot(ts.numpy(), mean_est_fn[:, q].detach().numpy(),
                color=c, linewidth=2.2, linestyle='--', alpha=0.7,
                marker='x', markersize=4, markevery=5,
                label=f"est dim {q}")
    ax.set_title("Mean function")
    ax.set_xlabel("t")
    ax.legend(fontsize=7, ncol=2)

    # (d) diagonal of covariance
    ax = axes[1, 1]
    diag_true = Kcov_true.diag().detach().numpy()
    diag_est = Kcov_est_final.diag().detach().numpy()
    ax.plot(diag_true, color='#1f77b4', linewidth=2, marker='o',
            markersize=4, label='true')
    ax.plot(diag_est, color='#1f77b4', linewidth=2, linestyle='--',
            alpha=0.7, marker='x', markersize=5, label='estimated')
    ax.set_title("Covariance diagonal")
    ax.set_xlabel("coefficient index")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("lti_mmd_estimation.png", dpi=150)
    plt.show()

    # --- print comparison ---
    print("\n--- True vs Estimated parameters ---")
    print(f"A true:\n{A_true}")
    print(f"A est:\n{model.A.data}")
    print(f"\nB true:\n{B_true}")
    print(f"B est:\n{model.B.data}")
    print(f"\nG G^T true:\n{G_true @ G_true.T}")
    print(f"G G^T est:\n{model.G.data @ model.G.data.T}")


if __name__ == "__main__":
    main()