import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.optimize import linear_sum_assignment


# =========================
# Repro / device / dtype
# =========================
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64  # helps log-det stability


# =========================
# User-style hyperparams
# =========================
T = 20          # timesteps
F = 2           # features (vector-valued process dimension)
K = 4           # mixture components to learn
n = 100         # total samples
c = 4           # true clusters used in generation (can equal K)
lr = 0.1
lambd_reg = 1e-4

sigma_f = 2.0   # functional RBF kernel width
alpha = -1.0 / (2.0 * sigma_f**2)

sigmas = np.array([1.0, 1.0, 1.0, 1.0])  # temporal GP kernel lengthscales per component
assert len(sigmas) == K

R = 64          # requested eigen truncation (will be clipped to <= T)
r = 8           # mean PCA rank


# =========================
# Time grid + trapezoid weights (must match functional kernel discretization)
# =========================
dt = 1.0 / (T - 1)
ts = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
ts_col = ts[:, None]
t1, t2 = ts[:, None], ts[None, :]

w = torch.full((T,), dt, device=device, dtype=dtype)
w[0] = w[-1] = dt / 2.0
sqrt_w = torch.sqrt(w)


# =========================
# Kernels
# =========================
def rbf_time_kernel(t1, t2, sigma: float):
    # t1,t2: [T,1] and [1,T] => [T,T]
    return torch.exp(-(t1 - t2).pow(2) / (2.0 * sigma**2))


def functional_rbf_from_dist2(dist2, alpha: float):
    return torch.exp(alpha * dist2)


# =========================
# Synthetic data: F independent GPs with same temporal kernel inside each cluster
# (i.e. covariance is K_time ⊗ I_F)
# =========================
def generate_stochastic_processes(n_samples, n_features, n_clusters, n_timesteps):
    # distribute samples among clusters
    base = n_samples // n_clusters
    rem = n_samples % n_clusters
    counts = [base + 1 if i < rem else base for i in range(n_clusters)]

    all_X = []
    all_means = []
    all_labels = []

    for cluster_idx in range(n_clusters):
        n_c = counts[cluster_idx]
        if n_c == 0:
            continue

        feature_samples = []
        feature_means = []

        for feature_idx in range(n_features):
            # random mean per cluster/feature
            amp = torch.rand(1, device=device, dtype=dtype) * 10.0 + 2.0
            freq = torch.rand(1, device=device, dtype=dtype) * 3.0 + 0.5
            phase = torch.rand(1, device=device, dtype=dtype) * 2.0 * np.pi
            bias = torch.randn(1, device=device, dtype=dtype) * 5.0

            mean_vec = amp * torch.cos(freq * ts + phase) + bias  # [T]

            # temporal covariance (same across features inside this cluster)
            ls = 2.0  # (keep your choice)
            Kt = rbf_time_kernel(ts_col, ts_col, sigma=float(ls))
            Kt = Kt + 1e-5 * torch.eye(n_timesteps, device=device, dtype=dtype)

            dist = MultivariateNormal(mean_vec, covariance_matrix=Kt)
            samp = dist.sample((n_c,))  # [n_c, T]

            feature_samples.append(samp)
            feature_means.append(mean_vec)

        X_cluster = torch.stack(feature_samples, dim=-1)    # [n_c, T, F]
        means_cluster = torch.stack(feature_means, dim=-1)  # [T, F]

        all_X.append(X_cluster)
        all_means.append(means_cluster)
        all_labels.append(torch.full((n_c,), cluster_idx, dtype=torch.long, device=device))

    X = torch.cat(all_X, dim=0)                 # [N,T,F]
    real_means = torch.stack(all_means, dim=0)  # [n_clusters,T,F]
    labels = torch.cat(all_labels, dim=0)       # [N]

    perm = torch.randperm(X.shape[0], device=device)
    X = X[perm]
    labels = labels[perm]
    return X, labels, real_means, counts


Xs, targets, real_means, samples_per_cluster = generate_stochastic_processes(n, F, c, T)
B = Xs.shape[0]


# =========================
# PCA on flattened (T,F) to parametrize component means m_k
# =========================
def fit_pca_flat(X: torch.Tensor, r: int):
    """
    X: [N,T,F]
    returns mean_flat: [D], top_vecs: [D,r]
    """
    N, T_, F_ = X.shape
    D = T_ * F_
    X_flat = X.reshape(N, D)
    mean_flat = X_flat.mean(dim=0)
    X_centered = X_flat - mean_flat
    C = torch.cov(X_centered.T)  # [D,D]
    evals, evecs = torch.linalg.eigh(C)
    idx = torch.argsort(evals, descending=True)
    evecs = evecs[:, idx]
    r_eff = min(r, D)
    top_vecs = evecs[:, :r_eff]  # [D,r]
    return mean_flat, top_vecs


mean_flat, top_vecs = fit_pca_flat(Xs, r=r)
D = T * F
r_eff = top_vecs.shape[1]


def scores_to_mean(m_scores: torch.Tensor):
    """
    m_scores: [K,r_eff]
    returns m: [K,T,F]
    """
    m_flat = mean_flat[None, :] + m_scores @ top_vecs.T  # [K,D]
    return m_flat.reshape(K, T, F)


# =========================
# Precompute E1 constant (empirical X-X term)
# dist2(x_i,x_j) = sum_t w_t ||x_i(t)-x_j(t)||^2
# =========================
Xw = Xs * sqrt_w[None, :, None]          # [B,T,F]
Xw_flat = Xw.reshape(B, D)               # [B,D]
norms = (Xw_flat.pow(2)).sum(dim=1)      # [B]
dist2_xx = norms[:, None] + norms[None, :] - 2.0 * (Xw_flat @ Xw_flat.T)
E1_const = functional_rbf_from_dist2(dist2_xx, alpha=alpha).mean()


# =========================
# Build weighted temporal covariances per component:
# Ktilde_k = W^{1/2} K_time_k W^{1/2}
# (features assumed independent => Kronecker with I_F, determinant exponent scales by F)
# =========================
K_raw = torch.stack([rbf_time_kernel(t1, t2, sigma=float(s)) for s in sigmas]).to(device=device, dtype=dtype)
K_raw = K_raw + 1e-5 * torch.eye(T, device=device, dtype=dtype)[None, :, :]

Ktilde = (sqrt_w[None, :, None] * K_raw) * sqrt_w[None, None, :]  # [K,T,T]

# eigendecomposition for each Ktilde_k
evals_k, evecs_k = torch.linalg.eigh(Ktilde)  # ascending
evals_k = torch.flip(evals_k, dims=[-1])      # descending
evecs_k = torch.flip(evecs_k, dims=[-1])      # descending

# pairwise sums Ktilde_k + Ktilde_s (NEEDED for I_{k,s})
Ksum = Ktilde[:, None, :, :] + Ktilde[None, :, :, :]      # [K,K,T,T]
Ksum_flat = Ksum.reshape(K * K, T, T)
evals_ks, evecs_ks = torch.linalg.eigh(Ksum_flat)
evals_ks = torch.flip(evals_ks, dims=[-1]).reshape(K, K, T)
evecs_ks = torch.flip(evecs_ks, dims=[-1]).reshape(K, K, T, T)

# truncate eigen-expansion if desired (clip R to <= T)
R_eff = min(R, T)
evals_k_R = evals_k[..., :R_eff]          # [K,R]
evecs_k_R = evecs_k[..., :R_eff]          # [K,T,R]
evals_ks_R = evals_ks[..., :R_eff]        # [K,K,R]
evecs_ks_R = evecs_ks[..., :R_eff]        # [K,K,T,R]


# =========================
# Closed-form integrals (vector-valued, independent features)
# J_k = E_{X~P} E_{Y~N(m_k,K_k)}[exp(alpha||X-Y||_w^2)]
# I_ks = E_{Y~N(m_k,K_k),Y'~N(m_s,K_s)}[exp(alpha||Y-Y'||_w^2)]
# where ||·||_w^2 sums over time weights and feature dimension.
# =========================
def get_J(X: torch.Tensor, m: torch.Tensor):
    """
    X: [B,T,F]
    m: [K,T,F]
    returns J: [K]
    """
    # d[b,k,t,f] = sqrt_w[t]*(X[b,t,f]-m[k,t,f])
    d = (X[:, None, :, :] - m[None, :, :, :]) * sqrt_w[None, None, :, None]  # [B,K,T,F]

    # coeff[b,k,f,r] = sum_t d[b,k,t,f] * evecs_k_R[k,t,r]
    coeff = torch.einsum("bktf,ktr->bkfr", d, evecs_k_R)  # [B,K,F,R]

    denom = (1.0 - 2.0 * alpha * evals_k_R)[None, :, None, :]  # [1,K,1,R]

    # quad[b,k] = sum_f sum_r coeff^2 / denom
    quad = torch.sum(coeff.pow(2) / denom, dim=(2, 3))  # [B,K]

    # log prefactor: det(I - 2α Ktilde_k)^(-F/2)
    log_pref = -0.5 * F * torch.sum(torch.log(1.0 - 2.0 * alpha * evals_k_R), dim=-1)  # [K]

    return torch.mean(torch.exp(log_pref[None, :] + alpha * quad), dim=0)  # [K]


def get_I(m: torch.Tensor):
    """
    m: [K,T,F]
    returns I: [K,K]
    """
    d = (m[:, None, :, :] - m[None, :, :, :]) * sqrt_w[None, None, :, None]  # [K,K,T,F]

    # coeff[k,s,f,r] = sum_t d[k,s,t,f] * evecs_ks_R[k,s,t,r]
    coeff = torch.einsum("kstf,kstr->ksfr", d, evecs_ks_R)  # [K,K,F,R]

    denom = (1.0 - 2.0 * alpha * evals_ks_R)[:, :, None, :]  # [K,K,1,R]
    quad = torch.sum(coeff.pow(2) / denom, dim=(2, 3))       # [K,K]

    log_pref = -0.5 * F * torch.sum(torch.log(1.0 - 2.0 * alpha * evals_ks_R), dim=-1)  # [K,K]

    return torch.exp(log_pref + alpha * quad)  # [K,K]


def mmd_loss_terms(X: torch.Tensor, m: torch.Tensor, pi: torch.Tensor):
    """
    returns (E1,E2,E3,I,J)
    """
    J = get_J(X, m)         # [K]
    I = get_I(m)            # [K,K]
    E2 = torch.dot(pi, J)
    E3 = pi @ I @ pi
    return E1_const, E2, E3, I, J


def mmd_loss_without_E1(I: torch.Tensor, J: torch.Tensor, pi: torch.Tensor):
    # E1 is constant w.r.t (m,pi), so omit in optimization
    return (pi @ I @ pi) - 2.0 * torch.dot(pi, J)


# =========================
# Optimize
# =========================
m_scores = nn.Parameter(torch.randn((K, r_eff), device=device, dtype=dtype) * 2.0)
pi_logits = nn.Parameter(torch.ones((K,), device=device, dtype=dtype))

optimizer = torch.optim.RMSprop([m_scores, pi_logits], lr=lr)
pbar = tqdm(range(1000))

for it in pbar:
    optimizer.zero_grad()

    m = scores_to_mean(m_scores)              # [K,T,F]
    pi = torch.softmax(pi_logits, dim=0)      # [K]

    J = get_J(Xs, m)
    I = get_I(m)

    loss_reg = m_scores.pow(2).mean() * lambd_reg
    loss = mmd_loss_without_E1(I, J, pi) + loss_reg

    loss.backward()
    optimizer.step()

    if it % 10 == 0 or it == 999:
        with torch.no_grad():
            E1, E2, E3, _, _ = mmd_loss_terms(Xs, m, pi)
        pbar.set_description(
            f"loss={loss.item():.6f} | (E1={E1.item():.4f}, E2={E2.item():.4f}, E3={E3.item():.4f}) | reg={loss_reg.item():.6f}"
        )


# =========================
# (Optional) Match learned means to true means (Hungarian)
# =========================
with torch.no_grad():
    m = scores_to_mean(m_scores)                 # [K,T,F]
    pi = torch.softmax(pi_logits, dim=0)         # [K]

# if c != K you can skip matching or match min(c,K)
K_match = min(c, K)
cost = torch.zeros((K_match, K_match), device=device, dtype=dtype)
for i in range(K_match):
    for j in range(K_match):
        cost[i, j] = torch.mean((real_means[i] - m[j]).pow(2))

row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

print("Assignment (true -> learned):", list(zip(row_ind, col_ind)))
true_pi = np.array(samples_per_cluster) / np.sum(samples_per_cluster)
print("True pi:", true_pi[:K_match])
print("Estimated pi (reordered):", pi.detach().cpu().numpy()[col_ind])

cost_matrix = torch.zeros((K, K))
for i in range(K):
    for j in range(K):
        cost_matrix[i, j] = torch.mean((real_means[i] - m[j])**2).item()

row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
for true_idx, learned_idx in zip(row_ind, col_ind):
    mse = cost_matrix[true_idx, learned_idx].item()

estimated_pi = pi.detach().numpy()
print("Estimated pi: ", estimated_pi[col_ind])
print("True pi: ", K/np.sum(K))

fig, axs = plt.subplots(1, F, figsize=(10, 5))
if F == 1: axs = [axs]
cmap = mpl.colormaps.get_cmap('viridis')

# Plot training data
for j in range(F):
    for i in range(len(Xs)):
        axs[j].plot(ts, Xs[i, :, j].detach().numpy().T, alpha=0.3, color=cmap(targets[i]/c), linewidth=1)

# Plot predicted means
for j in range(F):
    for i in range(K):
        axs[j].plot(ts, m[col_ind[i], :, j].detach().numpy(), color=cmap(i/K), lw=3, label=f'Predicted {i}', linestyle='--')

#ax.legend()
#ax.set_title("Training data and Predicted means")
plt.show()