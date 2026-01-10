import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.optimize import linear_sum_assignment
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# Repro / device / dtype
# =========================
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64  # better stability for log-dets


# =========================
# Problem setup
# =========================
T = 20          # number of timesteps
K = 4           # number of mixture components
dt = 1.0 / (T - 1)

ts = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
t1, t2 = ts[:, None], ts[None, :]

# Trapezoid weights to match your functional kernel discretization
w = torch.full((T,), dt, device=device, dtype=dtype)
w[0] = w[-1] = dt / 2.0
sqrt_w = torch.sqrt(w)

# Functional RBF kernel k(x,y) = exp(alpha * ||x-y||_w^2)
sigma_f = 4.0
alpha = -1.0 / (2.0 * sigma_f**2)

# Mixture model (data-generating) lengthscales (for GP covariances)
sigmas = np.array([1.0, 0.5, 2.0, 0.1])  # lengthscales for time kernels
assert len(sigmas) == K

# Number of samples per component for the dataset
num = np.array([50, 100, 40, 50])
assert len(num) == K


# =========================
# Kernels
# =========================
def rbf_time_kernel(t1, t2, sigma: float):
    """RBF kernel on time grid."""
    return torch.exp(-(t1 - t2).pow(2) / (2.0 * sigma**2))


def functional_rbf_from_weighted_dist2(dist2, alpha: float):
    """k = exp(alpha * dist2)"""
    return torch.exp(alpha * dist2)


# =========================
# Generate synthetic dataset: GP mixture
# =========================
means_true = [
    7.0 * torch.sin(2.0 * ts),
    1.0 * torch.zeros(T, device=device, dtype=dtype),
    10.0 * torch.cos(4.0 * ts),
    -10.0 * torch.cos(4.0 * ts),
]

Xs = []
for k in range(K):
    Kk = rbf_time_kernel(t1, t2, sigma=float(sigmas[k])) + 1e-5 * torch.eye(T, device=device, dtype=dtype)
    Xk = MultivariateNormal(means_true[k], covariance_matrix=Kk).sample((int(num[k]),))
    Xs.append(Xk)

X = torch.cat(Xs, dim=0)  # [B, T]
B = X.shape[0]


# =========================
# PCA basis to parametrize means m_k in low rank
# =========================
mean_func = X.mean(dim=0)
X_centered = X - mean_func
C = torch.cov(X_centered.T)  # [T, T]

eigvals_m, eigvecs_m = torch.linalg.eigh(C)
idx = torch.argsort(eigvals_m, descending=True)
eigvecs_m = eigvecs_m[:, idx]

r_mean = min(7, T)  # number of PCA modes for mean parametrization


# =========================
# Precompute E1 = (1/n^2) sum_{i,j} k(X_i, X_j)
# Use fast weighted distance: ||x-y||_w^2 = ||x||_w^2 + ||y||_w^2 - 2<x,y>_w
# =========================
Xw = X * sqrt_w  # [B,T]
norms = (Xw.pow(2)).sum(dim=1)  # [B]
dist2_xx = norms[:, None] + norms[None, :] - 2.0 * (Xw @ Xw.T)  # [B,B]
E1_const = functional_rbf_from_weighted_dist2(dist2_xx, alpha=alpha).mean()


# =========================
# Build weighted covariances for each mixture component:
# Ktilde_k = W^{1/2} K_k W^{1/2}
# Then use eigen-formulas for Gaussian integrals
# =========================
K_raw = torch.stack([rbf_time_kernel(t1, t2, sigma=float(s)) for s in sigmas]).to(device=device, dtype=dtype)
K_raw = K_raw + 1e-5 * torch.eye(T, device=device, dtype=dtype)[None, :, :]  # stabilize

Ktilde = (sqrt_w[None, :, None] * K_raw) * sqrt_w[None, None, :]  # [K,T,T]

# eigenpairs for each Ktilde_k
evals_k, evecs_k = torch.linalg.eigh(Ktilde)  # ascending
evals_k = torch.flip(evals_k, dims=[-1])      # descending
evecs_k = torch.flip(evecs_k, dims=[-1])      # descending

# eigenpairs for each Ktilde_k + Ktilde_s (needed for I_{k,s})
Ksum = Ktilde[:, None, :, :] + Ktilde[None, :, :, :]          # [K,K,T,T]
Ksum_flat = Ksum.reshape(K * K, T, T)                         # [K*K,T,T]
evals_ks, evecs_ks = torch.linalg.eigh(Ksum_flat)             # ascending
evals_ks = torch.flip(evals_ks, dims=[-1]).reshape(K, K, T)   # [K,K,T]
evecs_ks = torch.flip(evecs_ks, dims=[-1]).reshape(K, K, T, T)


# =========================
# Closed-form Gaussian integrals:
# J_k = (1/n) sum_i ∫ k(X_i, y) dN(m_k, K_k)(y)
# I_{k,s} = ∬ k(y, y') dN(m_k,K_k)(y) dN(m_s,K_s)(y')
# with k(x,y)=exp(alpha * ||x-y||_w^2)
# =========================
def get_J(X: torch.Tensor, m: torch.Tensor):
    """
    X: [B,T]
    m: [K,T]
    Returns J: [K]
    """
    # d = W^{1/2} (X_i - m_k)
    d = (X[:, None, :] - m[None, :, :]) * sqrt_w  # [B,K,T]

    # coeff[b,k,r] = <d[b,k,:], evecs_k[k,:,r]>
    coeff = torch.einsum("bkt,ktr->bkr", d, evecs_k)  # [B,K,T]

    denom = (1.0 - 2.0 * alpha * evals_k)[None, :, :]  # [1,K,T]
    log_pref = -0.5 * torch.sum(torch.log(1.0 - 2.0 * alpha * evals_k), dim=-1)  # [K]

    quad = torch.sum(coeff.pow(2) / denom, dim=-1)  # [B,K]
    return torch.mean(torch.exp(log_pref[None, :] + alpha * quad), dim=0)  # [K]


def get_I(m: torch.Tensor):
    """
    m: [K,T]
    Returns I: [K,K]
    """
    d = (m[:, None, :] - m[None, :, :]) * sqrt_w  # [K,K,T]

    d_flat = d.reshape(K * K, T)                 # [K*K,T]
    evecs_flat = evecs_ks.reshape(K * K, T, T)   # [K*K,T,T]
    evals_flat = evals_ks.reshape(K * K, T)      # [K*K,T]

    # coeff = d_flat @ evecs_flat   (row-vector times matrix)
    coeff = torch.bmm(d_flat.unsqueeze(1), evecs_flat).squeeze(1)  # [K*K,T]

    denom = (1.0 - 2.0 * alpha * evals_flat)  # [K*K,T]
    log_pref = -0.5 * torch.sum(torch.log(denom), dim=-1)          # [K*K]
    quad = torch.sum(coeff.pow(2) / denom, dim=-1)                 # [K*K]

    I_flat = torch.exp(log_pref + alpha * quad)  # [K*K]
    return I_flat.reshape(K, K)


def mmd_terms(X: torch.Tensor, m: torch.Tensor, pi: torch.Tensor):
    """
    Returns (E1,E2,E3,I,J) where:
      E1 = 1/n^2 sum_{i,j} k(X_i,X_j)    (constant)
      E2 = sum_k pi_k J_k
      E3 = sum_{k,s} pi_k pi_s I_{k,s}
    """
    J = get_J(X, m)        # [K]
    I = get_I(m)           # [K,K]
    E2 = torch.dot(pi, J)
    E3 = torch.dot(pi, I @ pi)
    return E1_const, E2, E3, I, J


# =========================
# Train (optimize m_k and pi_k)
# =========================
lr = 0.1
lambd_reg = 1e-5
n_steps = 1000

m_scores = nn.Parameter(torch.randn((K, r_mean), device=device, dtype=dtype))
pi_logits = nn.Parameter(torch.randn((K,), device=device, dtype=dtype))

optimizer = torch.optim.RMSprop([m_scores, pi_logits], lr=lr)
pbar = tqdm(range(n_steps))

for step in pbar:
    optimizer.zero_grad()

    # mean parametrization in PCA subspace (+ global mean)
    m = mean_func + m_scores @ eigvecs_m[:, :r_mean].T  # [K,T]
    pi = torch.softmax(pi_logits, dim=0)                # [K]

    E1, E2, E3, I, J = mmd_terms(X, m, pi)

    loss_reg = m_scores.pow(2).mean()
    loss = (E1 - 2.0 * E2 + E3) + lambd_reg * loss_reg

    loss.backward()
    optimizer.step()

    pbar.set_description(
        f"loss={loss.item():.6f} | E1={E1.item():.6f} E2={E2.item():.6f} E3={E3.item():.6f} reg={loss_reg.item():.6f}"
    )

# final params
with torch.no_grad():
    m = mean_func + m_scores @ eigvecs_m[:, :r_mean].T
    pi = torch.softmax(pi_logits, dim=0)


# =========================
# Match learned components to true ones (Hungarian)
# =========================
cost_matrix = torch.zeros((K, K), device=device, dtype=dtype)
for i in range(K):
    for j in range(K):
        cost_matrix[i, j] = torch.mean((means_true[i] - m[j]).pow(2))

row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

print("Assignment (true -> learned):", list(zip(row_ind, col_ind)))
print("True pi:", num / np.sum(num))
print("Estimated pi (reordered):", pi.detach().cpu().numpy()[col_ind])


# =========================
# Plot
# =========================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
cmap = mpl.colormaps.get_cmap("viridis")

# training data
for i in range(len(Xs)):
    ax.plot(
        ts.detach().cpu().numpy(),
        Xs[i].detach().cpu().numpy().T,
        alpha=0.25,
        color=cmap(i / len(Xs)),
        linewidth=1,
    )

# predicted means (reordered)
for i in range(K):
    ax.plot(
        ts.detach().cpu().numpy(),
        m[col_ind[i]].detach().cpu().numpy(),
        color=cmap(i / K),
        lw=3,
        linestyle="--",
        label=f"Predicted {i}",
    )

ax.legend()
ax.set_title("Training data and Predicted means")
ax.set_xlabel("t")
ax.set_ylabel("x(t)")
plt.tight_layout()
plt.savefig("gpm_corrected_means.pdf")
plt.show()
