#!/usr/bin/env python
# coding: utf-8

# # Paper Synthetic Experiments: MMD Gaussian Mixtures in Hilbert Spaces
# 
# Reproduces the five synthetic experiments from the appendix:
# 
# 1. **$\mathbb{R}^d$** — classical GMM recovery (histogram vs density).
# 2. **$L^2([0,1];\mathbb{R}^2)$** with $K=5$ Gaussian components.
# 3. **$L^2([0,1]^2;\mathbb{R})$** with $K=3$ and tensor-product cosine basis.
# 4. **$L^2(\mathrm{SO}(3))$** rotation data with Wigner D-matrix basis.
# 5. **Graph signals** with Laplacian-induced geometry.
# 
# All experiments are fit by minimizing $\mathrm{MMD}^2$ with a Gaussian kernel.

# In[ ]:


import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

project_root = os.path.abspath("")
if not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.abspath("..")
sys.path.insert(0, project_root)

from src import (
    CanonicalBasis,
    DiscreteCosineBasis,
    L2CosineBasis,
    SO3Basis,
    GraphLaplacianBasis,
    GaussianKernel,
    GaussianMixtureModel,
    generate_l2_2d_gaussian_data,
    generate_so3_mixture_data,
    generate_graph_mixture_data,
)

# K=5 L^2 data generator lives in examples.train_l2_gaussian
from examples.train_l2_gaussian import generate_l2_gaussian_data

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})

device = torch.device("cpu")
dtype = torch.float64

FIG_DIR = os.path.join(project_root, "paper", "images", "synthetic")
os.makedirs(FIG_DIR, exist_ok=True)


def fit_mmd_gmm(X_coeffs, basis, K, kernel, num_epochs=300, lr=0.1,
                covariance_type="diagonal", seed=0, verbose=False):
    """Fit a Gaussian mixture by MMD^2 minimization, return model + history."""
    torch.manual_seed(seed)
    M = X_coeffs.shape[1]
    model = GaussianMixtureModel(
        num_components=K, coeff_dim=M, basis=basis,
        covariance_type=covariance_type, device=device, dtype=dtype,
    )
    model.initialize_from_data(X_coeffs, method="kmeans++")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gram = kernel.compute_gram_matrix(X_coeffs)
    const_term = gram.mean()
    history = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        mmd2, _ = model.compute_mmd2(X_coeffs, kernel, compute_const_term=False)
        mmd2.backward()
        optimizer.step()
        history.append((const_term + mmd2.detach()).item())
        if verbose and (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f"  epoch {epoch+1:4d}  MMD^2 = {history[-1]:.4e}")
    return model, history


def align_components(true_means, pred_means):
    """Greedy matching from predicted to true components by L2 distance."""
    K = true_means.shape[0]
    tm = true_means.reshape(K, -1).detach().cpu().numpy()
    pm = pred_means.reshape(K, -1).detach().cpu().numpy()
    used = set()
    perm = [-1] * K
    for k in range(K):
        dists = [(np.linalg.norm(tm[k] - pm[j]), j) for j in range(K) if j not in used]
        _, j = min(dists)
        perm[k] = j
        used.add(j)
    return perm


def plot_loss(ax, history, title="Training loss"):
    ax.plot(np.arange(1, len(history) + 1), history, lw=1.8, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathrm{MMD}^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_weights(ax, true_pi, pred_pi, title=r"$\pi$: true vs predicted"):
    K = len(true_pi)
    x = np.arange(K)
    w = 0.38
    tp = true_pi.detach().cpu().numpy() if torch.is_tensor(true_pi) else np.asarray(true_pi)
    pp = pred_pi.detach().cpu().numpy() if torch.is_tensor(pred_pi) else np.asarray(pred_pi)
    ax.bar(x - w/2, tp, w, label="True", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, pp, w, label="Predicted", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k+1}" for k in range(K)])
    ax.set_ylabel(r"$\pi_k$")
    ax.set_title(title)
    ax.set_ylim(0, max(tp.max(), pp.max()) * 1.25)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")


# ## 1. $\mathbb{R}^d$: classical Gaussian Mixture
# 
# Recovery of a 1D Gaussian mixture. Left: training loss. Right: empirical histogram of the data overlaid with the true and learned densities.

# In[ ]:


torch.manual_seed(0)
np.random.seed(0)

# ground truth 1D GMM
K_rd = 3
true_pi_rd = torch.tensor([0.5, 0.3, 0.2], dtype=dtype)
true_mu_rd = torch.tensor([-3.0, 0.5, 4.0], dtype=dtype)
true_sigma_rd = torch.tensor([0.6, 0.9, 0.5], dtype=dtype)

n_rd = 1500
assign_rd = torch.multinomial(true_pi_rd.expand(n_rd, -1), 1).squeeze(-1)
X_rd = (true_mu_rd[assign_rd] + true_sigma_rd[assign_rd] * torch.randn(n_rd, dtype=dtype)).unsqueeze(-1)

basis_rd = CanonicalBasis(n=1, device=device, dtype=dtype)
kernel_rd = GaussianKernel(sigma=1.0)
model_rd, hist_rd = fit_mmd_gmm(X_rd, basis_rd, K=K_rd, kernel=kernel_rd,
                                num_epochs=400, lr=0.05, seed=1)

pred_pi_rd = model_rd.pi.detach()
pred_mu_rd = model_rd.mean.detach().squeeze(-1)
pred_var_rd = model_rd.variance.detach().squeeze(-1)
print("true  pi =", true_pi_rd.numpy(), " mu =", true_mu_rd.numpy(), " sigma =", true_sigma_rd.numpy())
print("pred  pi =", pred_pi_rd.numpy(), " mu =", pred_mu_rd.numpy(), " sigma =", pred_var_rd.sqrt().numpy())


# In[ ]:


def gmm_pdf(x, pi, mu, sigma):
    pi_n = pi.detach().cpu().numpy() if torch.is_tensor(pi) else pi
    mu_n = mu.detach().cpu().numpy() if torch.is_tensor(mu) else mu
    si_n = sigma.detach().cpu().numpy() if torch.is_tensor(sigma) else sigma
    p = np.zeros_like(x)
    for w, m, s in zip(pi_n, mu_n, si_n):
        p += w * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return p

fig1, axes1 = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

plot_loss(axes1[0], hist_rd, title=r"(a) $\mathrm{MMD}^2$ training loss")

ax = axes1[1]
X_np = X_rd.squeeze(-1).numpy()
xx = np.linspace(X_np.min() - 0.5, X_np.max() + 0.5, 500)
ax.hist(X_np, bins=60, density=True, color="lightgray",
        edgecolor="gray", alpha=0.9, label="Samples")
ax.plot(xx, gmm_pdf(xx, true_pi_rd, true_mu_rd, true_sigma_rd),
        lw=2.2, color="steelblue", label="True density")
ax.plot(xx, gmm_pdf(xx, pred_pi_rd, pred_mu_rd, pred_var_rd.sqrt()),
        lw=2.2, color="coral", ls="--", label="Learned density")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.set_title("(b) Histogram vs true/learned density")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

fig1.savefig(os.path.join(FIG_DIR, "rd_gmm_summary.pdf"), format="pdf", bbox_inches="tight")
plt.show()


# ## 2. $L^2([0,1]; \mathbb{R}^2)$ with $K=5$
# 
# Recovery of means and diagonal covariances from $K=5$ non-uniformly weighted Gaussian components in the cosine basis. The four panels show (a) raw trajectories with true means overlaid, (b) true vs predicted mean functions, (c) $\pi$ recovery, (d) training loss.

# In[ ]:


torch.manual_seed(42)

n_l2 = 500
K_l2 = 5
grid_l2 = 100
R_l2 = 15
d_l2 = 2

true_pi_l2 = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10], dtype=dtype)

X_raw_l2, X_coeffs_l2, assign_l2, info_l2 = generate_l2_gaussian_data(
    n_samples=n_l2, n_components=K_l2, grid_size=grid_l2,
    R=R_l2, T=1.0, d=d_l2, component_weights=true_pi_l2,
    seed=42, device=device, dtype=dtype,
)

true_means_l2 = info_l2["true_means"]       # (K, grid, d)
true_mean_coeffs_l2 = info_l2["true_mean_coeffs"]
basis_l2 = info_l2["basis"]
t_l2 = info_l2["t"]

kernel_l2 = GaussianKernel(sigma=1.2)
model_l2, hist_l2 = fit_mmd_gmm(X_coeffs_l2, basis_l2, K=K_l2, kernel=kernel_l2,
                                num_epochs=400, lr=0.1, seed=1)

pred_means_l2 = basis_l2.reconstruct(model_l2.mean.detach())  # (K, grid, d)
perm_l2 = align_components(true_mean_coeffs_l2, model_l2.mean.detach())
print("permutation (pred -> true):", perm_l2)


# In[ ]:


fig2 = plt.figure(figsize=(12, 3), constrained_layout=True)
gs2 = GridSpec(1, 4, figure=fig2)
ax_a = fig2.add_subplot(gs2[0])
ax_b = fig2.add_subplot(gs2[1])
ax_c = fig2.add_subplot(gs2[2])
ax_d = fig2.add_subplot(gs2[3])

colors_l2 = plt.cm.tab10(np.arange(K_l2) % 10)
t_np = t_l2.numpy()
X_np = X_raw_l2.numpy()
assign_np = assign_l2.numpy()
tmeans_np = true_means_l2.numpy()
pmeans_np = pred_means_l2.numpy()

# (a) data trajectories (dim 0) + true means overlaid
for k in range(K_l2):
    mask = assign_np == k
    for j in np.where(mask)[0][:30]:
        ax_a.plot(t_np, X_np[j, :, 0], color=colors_l2[k], alpha=0.18, lw=0.8)
for k in range(K_l2):
    ax_a.plot(t_np, tmeans_np[k, :, 0], color=colors_l2[k], lw=2.2,
              label=f"k={k+1}")
ax_a.set_title("(a) Data (dim 1) with true means")
ax_a.set_xlabel("t")
ax_a.set_ylabel(r"$x_0(t)$")
ax_a.legend(ncol=3, fontsize=8, loc="upper right")

# (b) true vs predicted means (dim 0)
for k in range(K_l2):
    j = perm_l2[k]
    ax_b.plot(t_np, tmeans_np[k, :, 0], color=colors_l2[k], lw=2.0, ls="--",
              label="True" if k == 0 else None)
    ax_b.plot(t_np, pmeans_np[j, :, 0], color=colors_l2[k], lw=2.0,
              label="Predicted" if k == 0 else None)
ax_b.set_title("(b) True (dashed) vs predicted (solid) means")
ax_b.set_xlabel("t")
ax_b.set_ylabel(r"$m_k(t)$")
ax_b.legend(loc="upper right")

# (c) weights
pred_pi_l2 = model_l2.pi.detach()[perm_l2]
plot_weights(ax_c, true_pi_l2, pred_pi_l2, title=r"(c) $\pi$: true vs predicted")

# (d) loss
plot_loss(ax_d, hist_l2, title=r"(d) $\mathrm{MMD}^2$ training loss")

fig2.savefig(os.path.join(FIG_DIR, "l2_k5_summary.pdf"), format="pdf", bbox_inches="tight")
plt.show()


# ## 3. $L^2([0,1]^2;\mathbb{R})$ with $K=3$
# 
# Tensor-product cosine basis on the unit square with $K=3$ components. Layout: top row shows true mean surfaces, bottom row shows predicted. The right column holds (top) training loss and (bottom) predicted vs true $\pi$.

# In[ ]:


torch.manual_seed(42)

n_2d = 400
K_2d = 3
R_s = R_t = 8
true_pi_2d = torch.tensor([0.45, 0.35, 0.20], dtype=dtype)

X_raw_2d, X_coeffs_2d, assign_2d, info_2d = generate_l2_2d_gaussian_data(
    n_samples=n_2d, n_components=K_2d,
    grid_size_s=30, grid_size_t=30, R_s=R_s, R_t=R_t,
    T=1.0, S=1.0, d=1, component_weights=true_pi_2d,
    seed=42, device=device, dtype=dtype,
)

true_means_2d = info_2d["true_means"]         # (K, gs, gt, d)
true_mean_coeffs_2d = info_2d["true_mean_coeffs"]
basis_2d = info_2d["basis"]
s_grid_2d, t_grid_2d = info_2d["s_grid"], info_2d["t_grid"]

kernel_2d = GaussianKernel(sigma=2.0)
model_2d, hist_2d = fit_mmd_gmm(X_coeffs_2d, basis_2d, K=K_2d, kernel=kernel_2d,
                                num_epochs=400, lr=0.1, seed=1)

pred_means_2d = basis_2d.reconstruct(model_2d.mean.detach())  # (K, gs, gt, d)
perm_2d = align_components(true_mean_coeffs_2d, model_2d.mean.detach())
print("permutation (true_k -> pred_idx):", perm_2d)


# In[ ]:


fig3 = plt.figure(figsize=(14, 7), constrained_layout=True)
gs3 = GridSpec(2, K_2d + 1, figure=fig3, width_ratios=[1] * K_2d + [1.1])

S_np = s_grid_2d.numpy()
T_np = t_grid_2d.numpy()
tm_np = true_means_2d.numpy()
pm_np = pred_means_2d.numpy()

zmin = min(tm_np[..., 0].min(), pm_np[..., 0].min())
zmax = max(tm_np[..., 0].max(), pm_np[..., 0].max())

colors_2d = plt.cm.tab10(np.arange(K_2d) % 10)

for k in range(K_2d):
    j = perm_2d[k]
    # top row: true
    ax_t = fig3.add_subplot(gs3[0, k], projection="3d")
    ax_t.plot_surface(S_np, T_np, tm_np[k, :, :, 0],
                      color=colors_2d[k], alpha=0.85, linewidth=0, antialiased=True)
    ax_t.set_title(f"True $m_{{{k+1}}}(s,t)$", fontsize=10)
    ax_t.set_zlim(zmin, zmax)
    ax_t.set_xlabel("s", labelpad=-4)
    ax_t.set_ylabel("t", labelpad=-4)
    ax_t.tick_params(axis="both", pad=-2, labelsize=7)

    # bottom row: predicted
    ax_p = fig3.add_subplot(gs3[1, k], projection="3d")
    ax_p.plot_surface(S_np, T_np, pm_np[j, :, :, 0],
                      color=colors_2d[k], alpha=0.85, linewidth=0, antialiased=True)
    ax_p.set_title(f"Predicted $m_{{{k+1}}}(s,t)$", fontsize=10)
    ax_p.set_zlim(zmin, zmax)
    ax_p.set_xlabel("s", labelpad=-4)
    ax_p.set_ylabel("t", labelpad=-4)
    ax_p.tick_params(axis="both", pad=-2, labelsize=7)

# right column: loss (top) and weights (bottom)
ax_loss = fig3.add_subplot(gs3[0, K_2d])
plot_loss(ax_loss, hist_2d, title=r"$\mathrm{MMD}^2$ training loss")

ax_w = fig3.add_subplot(gs3[1, K_2d])
pred_pi_2d = model_2d.pi.detach()[perm_2d]
plot_weights(ax_w, true_pi_2d, pred_pi_2d, title=r"$\pi$: true vs predicted")

fig3.savefig(os.path.join(FIG_DIR, "l2_2d_k3_summary.pdf"), format="pdf", bbox_inches="tight")
plt.show()


# ## 4. $L^2(\mathrm{SO}(3))$: Rotation Data
# 
# Rotations from a mixture of three concentrated components on $\mathrm{SO}(3)$, projected onto real Wigner D-matrix elements up to $L_{\max}=3$. We show each rotation as the image of the reference direction $[1,0,0]$ on $S^2$, overlaid with true means ($\blacktriangle$) and responsibility-weighted learned component directions ($\bigstar$).
# 

# In[ ]:


torch.manual_seed(42)

n_so3 = 200
K_so3 = 3
L_max = 3

X_euler, assign_so3, info_so3 = generate_so3_mixture_data(
    n_samples=n_so3, n_components=K_so3, noise_concentration=40.0,
    seed=40, device=device, dtype=dtype,
)
true_means_so3 = info_so3["mean_rotations"]   # (K, 3) Euler angles
true_pi_so3 = info_so3["component_weights"]

basis_so3 = SO3Basis(L_max=L_max, use_real_basis=True, device=device, dtype=dtype)
X_coeffs_so3 = basis_so3.project(X_euler)

kernel_so3 = GaussianKernel(sigma=10.0)
model_so3, hist_so3 = fit_mmd_gmm(X_coeffs_so3, basis_so3, K=K_so3, kernel=kernel_so3,
                                  num_epochs=400, lr=0.05, seed=43)

# Align predicted to true components via coefficient-space distance
true_mean_coeffs_so3 = basis_so3.project(true_means_so3)
perm_so3 = align_components(true_mean_coeffs_so3, model_so3.mean.detach())
print("permutation:", perm_so3)
print("true pi:", true_pi_so3.numpy(), " pred pi:", model_so3.pi.detach().numpy())


# In[ ]:


def euler_to_R(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return np.array([
        [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
        [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
        [-sb*cg, sb*sg, cb],
    ])


def sphere_points_from_euler(X_euler_like):
    ref = np.array([1.0, 0.0, 0.0])
    return np.stack([euler_to_R(*row) @ ref for row in np.asarray(X_euler_like)], axis=0)


with torch.no_grad():
    gamma_so3 = model_so3.responsibilities(X_coeffs_so3)

sphere_pts_so3 = sphere_points_from_euler(X_euler.numpy())
true_dirs_so3 = sphere_points_from_euler(true_means_so3.numpy())
pred_dirs_so3 = []
for j in range(K_so3):
    w = gamma_so3[:, j].detach().cpu().numpy()
    p = (w[:, None] * sphere_pts_so3).sum(axis=0)
    pred_dirs_so3.append(p / np.linalg.norm(p))
pred_dirs_so3 = np.stack(pred_dirs_so3, axis=0)

for k in range(K_so3):
    j = perm_so3[k]
    cosang = np.clip(true_dirs_so3[k] @ pred_dirs_so3[j], -1.0, 1.0)
    err_deg = np.degrees(np.arccos(cosang))
    print(f"component {k+1}: sphere-direction error = {err_deg:.2f}°")

fig4 = plt.figure(figsize=(12, 4), constrained_layout=True)
gs4 = GridSpec(1, 3, figure=fig4, width_ratios=[1.1, 1, 1], hspace=0.0)

ax_sphere = fig4.add_subplot(gs4[0, 0], projection="3d")

u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax_sphere.plot_surface(xs, ys, zs, alpha=0.08, color="gray", linewidth=0)

colors_so3 = plt.cm.tab10(np.arange(K_so3) % 10)
assign_np = assign_so3.numpy()
for i, p in enumerate(sphere_pts_so3):
    ax_sphere.scatter(*p, color=colors_so3[assign_np[i]], s=14, alpha=0.65)

for k in range(K_so3):
    ax_sphere.scatter(*true_dirs_so3[k], color=colors_so3[k], s=260, marker="^",
                      edgecolors="black", linewidths=1.4,
                      label=f"True $\\mu_{{{k+1}}}$")
for k in range(K_so3):
    j = perm_so3[k]
    ax_sphere.scatter(*pred_dirs_so3[j], color=colors_so3[k], s=320, marker="*",
                      edgecolors="black", linewidths=1.4,
                      label=f"Pred $\\mu_{{{k+1}}}$")

ax_sphere.set_title("Data on $S^2$")
ax_sphere.set_xlabel("x", labelpad=-4)
ax_sphere.set_ylabel("y", labelpad=-4)
ax_sphere.set_zlabel("z", labelpad=-4)
ax_sphere.set_xticks([])
ax_sphere.set_yticks([])
ax_sphere.set_zticks([])
ax_sphere.legend(loc="upper right", fontsize=7, ncol=2)
ax_sphere.set_box_aspect([1, 1, 1])

ax_loss4 = fig4.add_subplot(gs4[0, 1])
plot_loss(ax_loss4, hist_so3, title=r"$\mathrm{MMD}^2$ training loss")

ax_w4 = fig4.add_subplot(gs4[0, 2])
pred_pi_so3 = model_so3.pi.detach()[perm_so3]
plot_weights(ax_w4, true_pi_so3, pred_pi_so3, title=r"$\pi$: true vs predicted")

fig4.savefig(os.path.join(FIG_DIR, "so3_summary.pdf"), format="pdf", bbox_inches="tight")
plt.show()


# ## 5. Graph Signals
# 
# Graph signals on an Erdős–Rényi graph, projected onto the first $M=15$ Laplacian eigenvectors. The left block shows true (top) vs predicted (bottom) mean signals. The right column holds the training loss (top) and $\pi$ recovery (bottom), vertically aligned with the left block.

# In[ ]:


import networkx as nx

torch.manual_seed(42)

K_g = 3
num_nodes = 30

X_g, assign_g, adjacency_g, info_g = generate_graph_mixture_data(
    n_samples=150, n_components=K_g, num_nodes=num_nodes,
    edge_probability=0.25, signal_std=1.5, noise_std=0.2,
    seed=42, device=device, dtype=dtype,
)
true_means_g = info_g["base_signals"]        # (K, num_nodes)
true_pi_g = info_g["component_weights"]

basis_g = GraphLaplacianBasis.from_adjacency(
    adjacency=adjacency_g, alpha=0.1, num_eigenvectors=15,
    device=device, dtype=dtype,
)
X_coeffs_g = basis_g.project(X_g)

kernel_g = GaussianKernel(sigma=2.0)
model_g, hist_g = fit_mmd_gmm(X_coeffs_g, basis_g, K=K_g, kernel=kernel_g,
                              num_epochs=200, lr=0.1, seed=1)

pred_means_g = basis_g.reconstruct(model_g.mean.detach(), d=1)
true_mean_coeffs_g = basis_g.project(true_means_g)
perm_g = align_components(true_mean_coeffs_g, model_g.mean.detach())


# In[ ]:


G_nx = nx.from_numpy_array(adjacency_g.numpy())
pos = nx.spring_layout(G_nx, seed=42)

tm_g = true_means_g.numpy()
pm_g = pred_means_g.numpy()
vmin = min(tm_g.min(), pm_g.min())
vmax = max(tm_g.max(), pm_g.max())

fig5 = plt.figure(figsize=(14, 6), constrained_layout=True)
gs5 = GridSpec(2, K_g + 1, figure=fig5, width_ratios=[1] * K_g + [1.1])

for k in range(K_g):
    j = perm_g[k]
    ax_t = fig5.add_subplot(gs5[0, k])
    nx.draw_networkx_edges(G_nx, pos, ax=ax_t, alpha=0.3, edge_color="gray")
    nx.draw_networkx_nodes(G_nx, pos, ax=ax_t, node_color=tm_g[k],
                           cmap="coolwarm", vmin=vmin, vmax=vmax, node_size=120)
    ax_t.set_title(f"True $\\mu_{{{k+1}}}$  ($\\pi$={true_pi_g[k]:.2f})")
    ax_t.set_axis_off()

    ax_p = fig5.add_subplot(gs5[1, k])
    nx.draw_networkx_edges(G_nx, pos, ax=ax_p, alpha=0.3, edge_color="gray")
    nodes = nx.draw_networkx_nodes(G_nx, pos, ax=ax_p, node_color=pm_g[j],
                                   cmap="coolwarm", vmin=vmin, vmax=vmax, node_size=120)
    ax_p.set_title(f"Predicted $\\mu_{{{k+1}}}$  ($\\pi$={model_g.pi[j].item():.2f})")
    ax_p.set_axis_off()

ax_loss5 = fig5.add_subplot(gs5[0, K_g])
plot_loss(ax_loss5, hist_g, title=r"$\mathrm{MMD}^2$ training loss")

ax_w5 = fig5.add_subplot(gs5[1, K_g])
pred_pi_g = model_g.pi.detach()[perm_g]
plot_weights(ax_w5, true_pi_g, pred_pi_g, title=r"$\pi$: true vs predicted")

fig5.savefig(os.path.join(FIG_DIR, "graph_summary.pdf"), format="pdf", bbox_inches="tight")
plt.show()


# ## 6. Linear SDE in $L^2$: System Identification
# 
# We identify the matrices $(A, B, G)$ of a linear SDE
# $dx = (Ax + Bu)\,dt + G\,dW$ by minimizing $\mathrm{MMD}^2$ between the empirical
# measure of sampled trajectories and a **single** Gaussian component whose mean
# and covariance in the $L^2$ cosine basis are obtained analytically from the
# ODE for $(m(t), P(t))$. The Gaussian kernel uses a bandwidth $\sigma \propto
# \mathrm{std}(X)$.
# 

# In[ ]:


import torchsde
from examples.lti import (
    LinearSDE, compute_projected_mean_cov,
    mmd2_empirical_vs_single_gaussian, TrainableLTI,
)

torch.manual_seed(0)

# --- dimensions ---
n_lti, m_lti, p_lti = 4, 2, 2
R_lti = 7
T_lti = 5.0
dt_lti = 0.1
grid_lti = int(T_lti / dt_lti) + 1
n_samples_lti = 20

# --- true system (stable A) ---
A_true = torch.randn(n_lti, n_lti, dtype=dtype)
A_true = A_true - (torch.linalg.eigvals(A_true).real.max().real + 0.5) * torch.eye(n_lti, dtype=dtype)
B_true = torch.randn(n_lti, m_lti, dtype=dtype)
G_true = torch.randn(n_lti, p_lti, dtype=dtype) / n_lti

m0_lti = torch.zeros(n_lti, dtype=dtype)
Sigma0_lti = 0.1 * torch.eye(n_lti, dtype=dtype)

def u_fn_lti(t):
    return torch.stack([torch.sin(t), torch.cos(2.0 * t)]).to(dtype)

# --- simulate paths ---
ts_lti = torch.linspace(0.0, T_lti, grid_lti, dtype=dtype)
sde = LinearSDE(A_true, B_true, G_true, u_fn_lti)
x0 = m0_lti + torch.randn(n_samples_lti, n_lti, dtype=dtype) @ torch.linalg.cholesky(Sigma0_lti).T
paths = torchsde.sdeint(sde, x0, ts_lti, dt=dt_lti, method="euler", dt_min=0.05)
paths = paths.permute(1, 0, 2)  # (n_samples, L, n)
print("paths:", tuple(paths.shape))

# --- L^2 cosine basis ---
basis_lti = L2CosineBasis(T_lti, R_lti, grid_lti, d=n_lti, dtype=dtype)
X_coeffs_lti = basis_lti.project(paths)
print("M =", X_coeffs_lti.shape[1])

# --- reference: true projected Gaussian ---
mu_true, Kcov_true = compute_projected_mean_cov(
    A_true, B_true, G_true, m0_lti, Sigma0_lti, u_fn_lti, basis_lti,
)

sigma_lti = float(X_coeffs_lti.std()) * 2.0
kernel_lti = GaussianKernel(sigma=sigma_lti)

# --- train ---
model_lti = TrainableLTI(n_lti, m_lti, p_lti, basis_lti, u_fn_lti, m0_lti, Sigma0_lti).to(dtype)
optimizer = torch.optim.Adam(model_lti.parameters(), lr=1e-2)
n_epochs_lti = 400
hist_lti = []
for epoch in range(n_epochs_lti):
    optimizer.zero_grad()
    mu_est, Kcov_est = model_lti.projected_gaussian()
    loss = mmd2_empirical_vs_single_gaussian(X_coeffs_lti, mu_est, Kcov_est, kernel_lti)
    loss.backward()
    optimizer.step()
    hist_lti.append(loss.item())
    if (epoch + 1) % (n_epochs_lti // 5) == 0:
        print(f"  epoch {epoch+1:3d}  MMD^2 = {loss.item():.3e}")


# In[ ]:


fig6, axes6 = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)

# (a) sample paths (dim 0)
ax = axes6[0]
cmap_b = plt.cm.Blues
for i in range(n_samples_lti):
    ax.plot(ts_lti.numpy(), paths[i, :, 0].detach().numpy(),
            color=cmap_b(0.3 + 0.7 * i / max(n_samples_lti - 1, 1)),
            alpha=0.65, lw=0.9)
ax.set_title("(a) Sample paths (state dim 0)")
ax.set_xlabel("t"); ax.set_ylabel(r"$x_0(t)$")

# (b) loss
plot_loss(axes6[1], hist_lti, title=r"(b) $\mathrm{MMD}^2$ training loss")

# (c) mean function: true vs estimated
mu_est_f, Kcov_est_f = model_lti.projected_gaussian()
mean_true_fn = basis_lti.reconstruct(mu_true.unsqueeze(0)).squeeze(0)
mean_est_fn = basis_lti.reconstruct(mu_est_f.unsqueeze(0)).squeeze(0)
ax = axes6[2]
colors_n = plt.cm.tab10.colors
for q in range(n_lti):
    c = colors_n[q % len(colors_n)]
    ax.plot(ts_lti.numpy(), mean_true_fn[:, q].detach().numpy(),
            color=c, lw=2.0, label=f"true dim {q}")
    ax.plot(ts_lti.numpy(), mean_est_fn[:, q].detach().numpy(),
            color=c, lw=1.6, ls="--", alpha=0.85,
            label=f"est dim {q}")
ax.set_title("(c) Mean function $m(t)$: true vs estimated")
ax.set_xlabel("t")
ax.legend(ncol=2, fontsize=7)

# (d) covariance diagonal
ax = axes6[3]
diag_true = Kcov_true.diag().detach().numpy()
diag_est = Kcov_est_f.diag().detach().numpy()
idx = np.arange(len(diag_true))
ax.plot(idx, diag_true, marker="o", ms=4, lw=1.8, color="steelblue", label="true")
ax.plot(idx, diag_est, marker="x", ms=5, lw=1.8, ls="--", color="coral", label="estimated")
ax.set_title("(d) Projected covariance diagonal")
ax.set_xlabel("coefficient index")
ax.legend()

fig6.savefig(os.path.join(FIG_DIR, "lti_summary.pdf"), format="pdf", bbox_inches="tight")
plt.show()

print("max Re(lambda(A_est)) =", torch.linalg.eigvals(model_lti.A.data).real.max().item())
print("max Re(lambda(A_true)) =", torch.linalg.eigvals(A_true).real.max().item())


# ## 7. 3D Molecular Data (QM9)
# 
# We embed 3D QM9 molecules with a permutation-invariant WL-hash fingerprint, project the fingerprints onto a DCT basis, and fit a Gaussian mixture in coefficient space. The summary panel shows training behaviour, learned weights, HOMO-LUMO gap distributions, and a t-SNE of the coefficient vectors. The companion figure reports the most representative molecules per learned component, with components arranged as columns.
# 

# In[ ]:


from pathlib import Path

from atomic_datasets import datasets as atomic_ds_module
from atomic_datasets.wrappers.torch import PyTorchGeometricDataset
from examples.train_atomic import (
    DATASET_REGISTRY as ATOMIC_DATASETS,
    extract_property,
    embed_dataset_wlhash,
)
from src.spaces.graph_embedding import WLHashFingerprint

torch.manual_seed(42)
np.random.seed(42)

atomic_cfg = ATOMIC_DATASETS["qm9"]
property_label_atomic = atomic_cfg["property_label"]

n_atomic = 500
K_atomic = 5
wl_dim_atomic = 128
R_atomic = 64
atomic_epochs = 150

atomic_root = Path(project_root) / "data" / "atomic_datasets"
dataset_atomic = atomic_ds_module.QM9(root_dir=str(atomic_root), split="train")
pyg_atomic = PyTorchGeometricDataset(dataset_atomic)

subset_atomic = np.random.default_rng(42).choice(len(dataset_atomic), size=n_atomic, replace=False)
subset_atomic.sort()
property_atomic = extract_property(dataset_atomic, subset_atomic, atomic_cfg["property_key"])
valid_atomic = property_atomic[~torch.isnan(property_atomic)]
print("QM9 subset:", len(subset_atomic), "molecules")
print("HOMO-LUMO gap range:", float(valid_atomic.min()), float(valid_atomic.max()))

encoder_atomic = WLHashFingerprint(
    dim=wl_dim_atomic,
    radius=2,
    use_edge_attr=False,
    l2_normalize=True,
)
X_raw_atomic = embed_dataset_wlhash(
    pyg_atomic,
    subset_atomic,
    encoder_atomic,
    bond_cutoff=2.0,
).to(dtype)

basis_atomic = DiscreteCosineBasis(n=wl_dim_atomic, R=R_atomic, device=device, dtype=dtype)
X_coeffs_atomic = basis_atomic.project(X_raw_atomic)
M_atomic = X_coeffs_atomic.shape[1]
print("M =", M_atomic)

kernel_atomic = GaussianKernel(sigma=2.0)
model_atomic, hist_atomic = fit_mmd_gmm(
    X_coeffs_atomic,
    basis_atomic,
    K=K_atomic,
    kernel=kernel_atomic,
    num_epochs=atomic_epochs,
    lr=0.05,
    seed=42,
    verbose=True,
)

with torch.no_grad():
    gamma_atomic = model_atomic.responsibilities(X_coeffs_atomic)
    assign_atomic = gamma_atomic.argmax(dim=1)
print("cluster sizes:", [int((assign_atomic == k).sum()) for k in range(K_atomic)])


# In[ ]:


ATOM_COLORS = {
    "H": "#B8B8B8",
    "C": "#2C2C2C",
    "N": "#2B5DB1",
    "O": "#D84A4A",
    "F": "#2BA84A",
}
ATOM_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
}
ATOM_SIZES = {
    "H": 36,
    "C": 90,
    "N": 96,
    "O": 102,
    "F": 86,
}


def canonicalize_positions(positions):
    pts = np.asarray(positions, dtype=np.float64)
    pts = pts - pts.mean(axis=0, keepdims=True)
    if pts.shape[0] >= 3:
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        pts = pts @ vh.T
    if pts[:, 0].sum() < 0:
        pts[:, 0] *= -1
    if pts[:, 1].sum() < 0:
        pts[:, 1] *= -1
    return pts


def atomic_bonds(positions, atom_types, slack=0.25, scale=1.15):
    pts = np.asarray(positions, dtype=np.float64)
    bonds = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ri = ATOM_RADII.get(str(atom_types[i]), 0.8)
            rj = ATOM_RADII.get(str(atom_types[j]), 0.8)
            cutoff = scale * (ri + rj) + slack
            if np.linalg.norm(pts[i] - pts[j]) <= cutoff:
                bonds.append((i, j))
    return bonds


def set_equal_3d(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins + 1e-8)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_atomic_structure(ax, positions, atom_types):
    pts = canonicalize_positions(positions)
    for i, j in atomic_bonds(pts, atom_types):
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            color="#8a8a8a",
            lw=1.1,
            alpha=0.85,
        )
    for atom in sorted(set(atom_types)):
        mask = np.array(atom_types) == atom
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            pts[mask, 2],
            s=ATOM_SIZES.get(str(atom), 80),
            c=ATOM_COLORS.get(str(atom), "#cccccc"),
            edgecolors="black",
            linewidths=0.35,
            alpha=0.95,
        )
    set_equal_3d(ax, pts)
    ax.view_init(elev=18, azim=-62)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#dddddd")


fig_atomic, axes_atomic = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
plot_loss(axes_atomic[0, 0], hist_atomic, title=r"(a) $\mathrm{MMD}^2$ training loss")

ax = axes_atomic[0, 1]
pi_atomic = model_atomic.pi.detach().cpu().numpy()
x = np.arange(K_atomic)
ax.bar(x, pi_atomic, color="coral", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"k={k+1}" for k in range(K_atomic)])
ax.set_ylabel(r"$\pi_k$")
ax.set_title("(b) Learned mixture weights")
ax.set_ylim(0, pi_atomic.max() * 1.25)
ax.grid(True, alpha=0.3, axis="y")

ax = axes_atomic[1, 0]
comp_colors_atomic = plt.cm.Set2(np.linspace(0, 1, K_atomic))
for k in range(K_atomic):
    mask = (assign_atomic == k).cpu().numpy()
    if mask.any():
        vals_k = property_atomic[mask].detach().cpu().numpy()
        vals_k = vals_k[~np.isnan(vals_k)]
        if len(vals_k) > 0:
            ax.hist(
                vals_k,
                bins=25,
                alpha=0.62,
                color=comp_colors_atomic[k],
                label=f"k={k+1} (n={mask.sum()})",
                edgecolor="white",
                linewidth=0.5,
            )
ax.set_xlabel(property_label_atomic)
ax.set_ylabel("count")
ax.set_title("(c) HOMO-LUMO gap distribution")
ax.legend(fontsize=8)

ax = axes_atomic[1, 1]
from sklearn.manifold import TSNE

tsne_atomic = TSNE(
    n_components=2,
    random_state=42,
    perplexity=min(30, len(subset_atomic) - 1),
    init="pca",
)
X_atomic_2d = tsne_atomic.fit_transform(X_coeffs_atomic.detach().cpu().numpy())
for k in range(K_atomic):
    mask = (assign_atomic == k).cpu().numpy()
    if mask.any():
        ax.scatter(
            X_atomic_2d[mask, 0],
            X_atomic_2d[mask, 1],
            c=[comp_colors_atomic[k]],
            s=16,
            alpha=0.7,
            label=f"k={k+1}",
        )
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("(d) t-SNE of DCT coefficients")
ax.legend(fontsize=8, loc="best")

fig_atomic.savefig(
    os.path.join(FIG_DIR, "atomic_qm9_summary.pdf"),
    format="pdf",
    bbox_inches="tight",
)
plt.show()

n_rep_atomic = 3
fig_atomic_rep = plt.figure(figsize=(3.1 * K_atomic, 3.0 * n_rep_atomic), facecolor="white")
with torch.no_grad():
    means_atomic = model_atomic.mean

for k in range(K_atomic):
    mask_k = (assign_atomic == k).nonzero(as_tuple=True)[0]
    if len(mask_k) == 0:
        continue
    X_k = X_coeffs_atomic[mask_k]
    dists = (X_k - means_atomic[k].unsqueeze(0)).norm(dim=1)
    top_idx = dists.argsort()[:min(n_rep_atomic, len(mask_k))]
    reps = mask_k[top_idx]
    for row in range(n_rep_atomic):
        ax = fig_atomic_rep.add_subplot(
            n_rep_atomic,
            K_atomic,
            row * K_atomic + k + 1,
            projection="3d",
            facecolor="white",
        )
        if row < len(reps):
            local_idx = reps[row].item()
            data_idx = int(subset_atomic[local_idx])
            mol = dataset_atomic[data_idx]
            plot_atomic_structure(ax, mol["nodes"]["positions"], mol["nodes"]["atom_types"])
            gap_val = float(property_atomic[local_idx])
            resp_val = gamma_atomic[local_idx, k].item()
            ax.text2D(
                0.03,
                0.03,
                f"gap={gap_val:.3f}\n$\\gamma$={resp_val:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                color="#333333",
            )
        else:
            ax.set_axis_off()
        if row == 0:
            ax.set_title(f"k={k+1}\n$\\pi$={model_atomic.pi[k].item():.2f}", fontsize=9, pad=8)
        if k == 0:
            ax.text2D(
                -0.18,
                0.5,
                f"rep {row+1}",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=9,
                color="#444444",
            )

fig_atomic_rep.suptitle("Representative QM9 molecules per component", fontsize=14, y=0.99)
plt.tight_layout(rect=[0.03, 0, 1, 0.97])
fig_atomic_rep.savefig(
    os.path.join(FIG_DIR, "atomic_qm9_representatives.pdf"),
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
)
plt.show()


# ## 8. NTU RGB+D Skeleton Sequences
# 
# Action-level mixture modeling on real human motion data. Each sequence is
# resampled to a fixed grid, mean-centered, projected onto the $L^2$ cosine
# basis, and fit with a $K=5$ Gaussian mixture. Visualizations summarize
# training behavior, cluster sizes, action composition of each cluster, and a
# t-SNE of the coefficient space.
# 
# 

# In[ ]:


import glob, random
from pathlib import Path

from examples.visualize_ntu_skeleton import (
    parse_skeleton_file, get_action_label, plot_skeleton_frame,
)
from examples.train_ntu_skeleton import (
    skeleton_to_trajectory, resample_trajectory, trajectory_to_joints,
)

torch.manual_seed(42); random.seed(42); np.random.seed(42)

n_samples_ntu = 150
grid_ntu = 64
d_ntu = 75
R_ntu = 10
K_ntu = 5
T_ntu = 1.0

DATA_ROOT = Path(project_root) / "data" / "ntu_rgbd_skeleton"
files = sorted(glob.glob(str(DATA_ROOT / "**/*.skeleton"), recursive=True))
if not files:
    print(f"No skeleton files at {DATA_ROOT}; downloading NTU dataset ...")
    from src.data import download_ntu_skeleton
    download_ntu_skeleton(root=str(DATA_ROOT))
    files = sorted(glob.glob(str(DATA_ROOT / "**/*.skeleton"), recursive=True))
print(f"Found {len(files)} skeleton files")
files = random.sample(files, min(n_samples_ntu, len(files)))

trajs, actions, kept = [], [], []
for fp in files:
    try:
        frames = parse_skeleton_file(fp)
        if len(frames) < 2:
            continue
        t = skeleton_to_trajectory(frames)
        t = resample_trajectory(t, grid_ntu)
        trajs.append(t)
        actions.append(get_action_label(fp))
        kept.append(fp)
    except Exception:
        continue

X_raw_np = np.stack(trajs, axis=0)
n_ntu = X_raw_np.shape[0]
print("Loaded:", X_raw_np.shape)

# center
mean_traj = X_raw_np.mean(axis=0)
X_centered_np = X_raw_np - mean_traj[None, :, :]
X_raw_ntu = torch.tensor(X_centered_np, dtype=dtype)

basis_ntu = L2CosineBasis(T=T_ntu, R=R_ntu, grid_size=grid_ntu, d=d_ntu, dtype=dtype)
X_coeffs_ntu = basis_ntu.project(X_raw_ntu)
M_ntu = X_coeffs_ntu.shape[1]
print("M =", M_ntu)

kernel_ntu = GaussianKernel(sigma=2.0)
model_ntu, hist_ntu = fit_mmd_gmm(
    X_coeffs_ntu, basis_ntu, K=K_ntu, kernel=kernel_ntu,
    num_epochs=250, lr=0.05, seed=1, verbose=True,
)

with torch.no_grad():
    gamma_ntu = model_ntu.responsibilities(X_coeffs_ntu)
    assign_ntu = gamma_ntu.argmax(dim=1)
print("cluster sizes:", [int((assign_ntu == k).sum()) for k in range(K_ntu)])


# In[ ]:


# Representative skeletons: components as columns
n_rep = 3
fig8 = plt.figure(figsize=(3.0 * K_ntu, 3.2 * n_rep), facecolor="white")

with torch.no_grad():
    means_ntu = model_ntu.mean

for k in range(K_ntu):
    mask_k = (assign_ntu == k).nonzero(as_tuple=True)[0]
    if len(mask_k) == 0:
        continue
    X_k = X_coeffs_ntu[mask_k]
    dists = (X_k - means_ntu[k].unsqueeze(0)).norm(dim=1)
    top_idx = dists.argsort()[1:1+min(n_rep, len(mask_k))]
    reps = mask_k[top_idx]
    for row in range(n_rep):
        ax = fig8.add_subplot(
            n_rep,
            K_ntu,
            row * K_ntu + k + 1,
            projection="3d",
            facecolor="white",
        )
        if row < len(reps):
            idx = reps[row].item()
            traj_disp = X_centered_np[idx] + mean_traj
            joints_seq = trajectory_to_joints(traj_disp)
            snaps = np.linspace(0, grid_ntu - 1, 6, dtype=int)
            for si, fidx in enumerate(snaps):
                t_frac = si / max(len(snaps) - 1, 1)
                plot_skeleton_frame(
                    ax,
                    joints_seq[fidx],
                    alpha=0.2 + 0.8 * t_frac,
                    linewidth=1.0 + 2.0 * t_frac,
                    joint_size=6 + 12 * t_frac,
                )
            ax.text2D(
                0.03,
                0.03,
                f"{actions[idx]}\n$\\gamma$={gamma_ntu[idx, k].item():.2f}",
                transform=ax.transAxes,
                fontsize=7,
                color="#333333",
            )
            ax.tick_params(colors="#666666", labelsize=6)
            ax.set_xlabel("X", fontsize=7, color="#666666", labelpad=2)
            ax.set_ylabel("Z", fontsize=7, color="#666666", labelpad=2)
            ax.set_zlabel("Y", fontsize=7, color="#666666", labelpad=2)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor("#dddddd")
            ax.grid(True, alpha=0.12, color="#bbbbbb")
            ax.view_init(elev=15, azim=-60)
        else:
            ax.set_axis_off()
        if row == 0:
            ax.set_title(f"k={k+1}\n$\\pi$={model_ntu.pi[k].item():.2f}", fontsize=9, pad=8)
        if k == 0:
            ax.text2D(
                -0.18,
                0.5,
                f"rep {row+1}",
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                ha="right",
                va="center",
                rotation=90,
                color="#444444",
            )

fig8.suptitle("Representative NTU skeleton sequences per component", fontsize=14, y=0.99)
plt.tight_layout(rect=[0.03, 0, 1, 0.97])
fig8.savefig(
    os.path.join(FIG_DIR, "ntu_representatives.pdf"),
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
)
plt.show()


# ## Summary
# 
# | Experiment | $n$ | $K$ | $M$ | Final $\mathrm{MMD}^2$ |
# | --- | --- | --- | --- | --- |
# | $\mathbb{R}^d$ (1D GMM) | 1500 | 3 | 1 | see above |
# | $L^2([0,1];\mathbb{R}^2)$ | 500 | 5 | 30 | see above |
# | $L^2([0,1]^2)$ | 400 | 3 | 64 | see above |
# | $L^2(\mathrm{SO}(3))$ | 200 | 3 | 84 | see above |
# | Graph signals | 150 | 3 | 15 | see above |
# | Linear SDE (LTI) | 20 | 1 | 28 | see above |
# | QM9 molecules | 500 | 5 | 64 | see above |
# | NTU skeletons | 150 | 5 | 750 | see above |
# 

# In[ ]:


print("Final MMD^2:")
print(f"  R^d         : {hist_rd[-1]:.4e}")
print(f"  L^2 K=5     : {hist_l2[-1]:.4e}")
print(f"  L^2(0,1)^2  : {hist_2d[-1]:.4e}")
print(f"  SO(3)       : {hist_so3[-1]:.4e}")
print(f"  Graph       : {hist_g[-1]:.4e}")
print(f"  LTI         : {hist_lti[-1]:.4e}")
print(f"  Atomic QM9  : {hist_atomic[-1]:.4e}")
print(f"  NTU skeleton: {hist_ntu[-1]:.4e}")

