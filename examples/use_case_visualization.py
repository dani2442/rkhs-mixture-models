#!/usr/bin/env python
# coding: utf-8

# # Glucodensity Temporal Mixture: Walkthrough
# 
# Temporal mixture models on CGM data with three representations:
# 1. **L² curves** — intraday glucose on a cosine basis
# 2. **Correlation matrices** — inter-hour correlation in Sym(24)
# 3. **Graph signals** — hourly glucose on a correlation-derived graph
# 
# Each is fit with time-varying cluster weights π(t) via MMD² minimisation.

# In[ ]:


import os, sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

project_root = os.path.abspath("")
if not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.abspath("..")
sys.path.insert(0, project_root)

IMAGES_DIR = Path(project_root) / "paper" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

from examples.train_glucodensity_temporal import (
    CONTROL_IDS, TREATMENT_IDS,
    load_and_preprocess_cgm, compute_sliding_windows,
    build_training_representation, run_experiment,
)
from examples.train_glucodensity_correlation import (
    compute_sliding_window_correlations,
    build_temporal_coefficients,
    estimate_sigma_median_heuristic,
)
from examples.train_glucodensity_correlation_graph import (
    compute_sliding_window_hourly, compute_mean_correlation,
    threshold_to_adjacency,
)
from src import GaussianKernel, GraphLaplacianBasis, SymmetricMatrixBasis
from src.spaces import H1CosineBasis, L2CosineBasis
from src.temporal_mixture import BasisLogitsTimeWeights, NeuralODETimeWeights

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({"figure.dpi": 100, "font.size": 11})
torch.manual_seed(42); np.random.seed(42)
device, dtype = torch.device("cpu"), torch.float64

K = 2        # mixture components
N_BINS = 16  # temporal bins
EPOCHS = 400
LR = 0.01

csv_path = os.path.join(project_root, "data/glucodensities/cgm_all_patients.csv")
patient_data = load_and_preprocess_cgm(csv_path, max_prop_missing=0.20, block_size=4, verbose=True)


# ## 1. Data: mean hourly glucose surface
# 
# 3D view of glucose across 24 hours and treatment weeks, split by group.

# In[ ]:


curves, t_indices, pids, wdays, _ = compute_sliding_windows(
    patient_data, window_size=4, stride=1, verbose=True
)
hourly_all = curves.reshape(curves.shape[0], 24, 12).mean(axis=2)
weeks = t_indices / 7.0

ctrl_mask = np.array([p in CONTROL_IDS for p in pids])
treat_mask = np.array([p in TREATMENT_IDS for p in pids])

def make_surface(mask, n_bins=15):
    w, h = weeks[mask], hourly_all[mask]
    edges = np.linspace(w.min(), w.max() + 0.01, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, np.array([
        h[(w >= edges[i]) & (w < edges[i + 1])].mean(0) for i in range(n_bins)
    ])

fig = plt.figure(figsize=(16, 6))
hours = np.arange(24)
for i, (mask, title) in enumerate([(ctrl_mask, "Control"), (treat_mask, "Treatment")]):
    centers, surf = make_surface(mask)
    H, W = np.meshgrid(hours, centers)
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    ax.plot_surface(H, W, surf, cmap="viridis", alpha=0.85, edgecolor="k", linewidth=0.15)
    ax.set(xlabel="Hour of day", ylabel="Treatment week", zlabel="Glucose (mg/dL)")
    ax.set_title(title)
    ax.view_init(elev=25, azim=-50)
plt.suptitle("Mean hourly glucose over treatment course", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# In[ ]:


# 1. Choose your colormap
cmap = plt.get_cmap('viridis') 

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

centers, surf_ctrl = make_surface(ctrl_mask)
centers, surf_treat = make_surface(treat_mask)

# 2. Loop through each column and plot
for i in range(surf_ctrl.shape[0]):
    # Use i / (num_lines - 1) to get a value between 0 and 1 for the cmap

    color = cmap(i / max(1, surf_ctrl.shape[0] - 1))
    axes[0].plot(hours, surf_ctrl[i, :], color=color, label=f'{centers[i]:.1f}')

    axes[1].plot(hours, surf_treat[i, :], color=color)

    axes[0].set_ylim(120, 190)
    axes[1].set_ylim(120, 190)

    axes[0].set_xlabel('Hour of day')
    axes[0].set_ylabel('Glucose (mg/dL)')
    axes[0].set_title('Control')
    axes[1].set_xlabel('Hour of day')
    axes[1].set_title('Treatment')
    axes[0].legend(title='Week', loc='upper left')

    axes[2].set_xlabel('Hour of day')
    axes[2].set_title('Treatment - Control')
    axes[2].plot(hours, surf_treat[i, :]-surf_ctrl[i, :], color=color, label=f'{centers[i]:.1f}')


# In[ ]:


hourly_all.shape


# ## 2. Temporal mixture on L² curves
# 
# Intraday glucose profiles projected onto a cosine basis.
# We compare **NeuralODE** vs **Cosine basis** for the time-varying weights π(t).

# In[ ]:


rep = build_training_representation(
    curves, t_indices, n_time_bins=N_BINS, r_s=8, device=device, dtype=dtype, verbose=True, space_metric="h1"
)
X_time, t_grid, mask = rep["X_time"], rep["t_grid"], rep["mask"]
kernel = GaussianKernel(sigma=0.8) #sigma=max(1e-8, rep["sigma_auto"]))

def train_and_extract(name, tw_model):
    model, history, pi_t = run_experiment(
        name, tw_model, X_time, kernel, K, rep["coeff_dim"],
        EPOCHS, LR, device, dtype, mask, verbose=True,
    )
    with torch.no_grad():
        means = model.mean.cpu() * rep["coeff_std"].cpu() + rep["coeff_mean"].cpu()
        recon = rep["space_basis"].reconstruct(means).squeeze(-1).numpy()

        # Function-space std: propagate covariance through de-standardization + basis reconstruction
        cov_std = model.covariance.cpu()  # (K, M, M)
        s = rep["coeff_std"].cpu()        # (M,)
        M = cov_std.shape[1]
        I_M = torch.eye(M, dtype=s.dtype)
        B_raw = rep["space_basis"].reconstruct(I_M).squeeze(-1)  # (M, L)
        B_eff = B_raw * s[:, None]  # effective basis scaled by coeff_std
        std_curves = np.zeros((K, B_raw.shape[1]))
        for k in range(K):
            var_f = ((cov_std[k] @ B_eff) * B_eff).sum(0)  # (L,)
            std_curves[k] = torch.sqrt(torch.clamp(var_f, min=0)).numpy()
    return model, history, pi_t.cpu().numpy(), recon, std_curves

# NeuralODE
ode_model, ode_h, ode_pi, ode_m, ode_std = train_and_extract(
    "NeuralODE", NeuralODETimeWeights(t_grid, K, hidden_dim=64, device=device, dtype=dtype)
)
# Cosine basis
phi = L2CosineBasis(T=1.0, R=6, grid_size=rep["L_t"], d=1, device=device, dtype=dtype)
bas_model, bas_h, bas_pi, bas_m, bas_std = train_and_extract(
    "CosineBasis", BasisLogitsTimeWeights(phi.Phi, K, device=device, dtype=dtype)
)

# --- Plot both models ---
t_days = t_grid.cpu().numpy() * (rep["t_max_days"] - rep["t_min_days"]) + rep["t_min_days"]
t_weeks = t_days / 7.0
slot_x = np.linspace(0, 24, ode_m.shape[1])

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for row, (h, pi, m, sd, name) in enumerate([
    (ode_h, ode_pi, ode_m, ode_std, "NeuralODE"),
    (bas_h, bas_pi, bas_m, bas_std, "CosineBasis"),
]):
    axes[row, 0].plot(h, lw=2)
    axes[row, 0].set(title=f"{name}: MMD² loss", xlabel="Epoch")
    axes[row, 0].grid(alpha=0.3)

    for k in range(K):
        axes[row, 1].plot(t_weeks, pi[:, k], lw=2, label=f"Cluster {k + 1}")
    axes[row, 1].set(title=f"{name}: π(t)", xlabel="Treatment week", ylabel="Probability", ylim=(0, 1))
    axes[row, 1].legend(fontsize=8)
    axes[row, 1].grid(alpha=0.3)

    for k in range(K):
        axes[row, 2].plot(slot_x, m[k], lw=2, label=f"Cluster {k + 1}")
        axes[row, 2].fill_between(slot_x, m[k] - sd[k], m[k] + sd[k], alpha=0.2)
    axes[row, 2].set(title=f"{name}: mean glucodensity", xlabel="Hour of day", ylabel="Glucose")
    axes[row, 2].legend(fontsize=8)
    axes[row, 2].grid(alpha=0.3)

plt.suptitle("Temporal mixture on L² glucodensity curves", fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:


curves, t_indices, pids, wdays, _ = compute_sliding_windows(
    patient_data, window_size=4, stride=1, verbose=True
)
hourly_all = curves.reshape(curves.shape[0], 24, 12).mean(axis=2)
weeks = t_indices / 7.0

def make_surface(n_bins=15):
    w, h = weeks, hourly_all
    edges = np.linspace(w.min(), w.max() + 0.01, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, np.array([
        h[(w >= edges[i]) & (w < edges[i + 1])].mean(0) for i in range(n_bins)
    ])

# Predicted mean surface from the CosineBasis model: mean(t) = sum_k pi_k(t) * mu_k
# slot_x_model = np.linspace(0, 24, bas_m.shape[1])
# bas_m_hourly = np.array([np.interp(np.arange(24), slot_x_model, bas_m[k]) for k in range(K)])
# pred_surface = bas_pi @ bas_m_hourly  # [L_t, 24]

slot_x_model = np.linspace(0, 24, ode_m.shape[1])
ode_m_hourly = np.array([np.interp(np.arange(24), slot_x_model, ode_m[k]) for k in range(K)])
pred_surface = ode_pi @ ode_m_hourly  # [L_t, 24]


fig = plt.figure(figsize=(5, 10))
hours = np.arange(24)

# Left: empirical mean
centers, surf = make_surface()
H, W = np.meshgrid(hours, centers)
ax = fig.add_subplot(2, 1, 1, projection="3d")
ax.plot_surface(H, W, surf, cmap="viridis", alpha=0.85, edgecolor="k", linewidth=0.15)
ax.set(xlabel="Hour of day", ylabel="Treatment week", zlabel="Glucose (mg/dL)")
ax.set_title("Empirical mean")
ax.view_init(elev=25, azim=-60)

# Right: predicted mean from temporal mixture model
H_pred, W_pred = np.meshgrid(hours, t_weeks)
ax2 = fig.add_subplot(2, 1, 2, projection="3d")
ax2.plot_surface(H_pred, W_pred, pred_surface, cmap="viridis", alpha=0.85, edgecolor="k", linewidth=0.15)
ax2.set(xlabel="Hour of day", ylabel="Treatment week", zlabel="Glucose (mg/dL)")
ax2.set_title("Predicted mean (NeuralODE)")
ax2.view_init(elev=25, azim=-60)

plt.suptitle("Mean hourly glucose: Empirical vs Predicted", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# In[ ]:


from examples.train_glucodensity_temporal import compute_all_patient_posteriors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec

# ── Patient posteriors ────────────────────────────────────────────────────────
patient_posteriors, _, patient_time_days_dict = compute_all_patient_posteriors(
    curves=curves,
    t_indices=t_indices,
    patient_ids=pids,
    window_days=wdays,
    model=ode_model,
    space_basis=rep["space_basis"],
    coeff_mean=rep["coeff_mean"],
    coeff_std=rep["coeff_std"],
    t_grid=t_grid,
    t_min_days=rep["t_min_days"],
    t_max_days=rep["t_max_days"],
    device=device,
    dtype=dtype,
)

# ── Bin cluster probabilities per group ──────────────────────────────────────
N_BINS_GRP = 12
ctrl_set = set(CONTROL_IDS)
treat_set = set(TREATMENT_IDS)

ctrl_data, treat_data = [], []
for pid, post in patient_posteriors.items():
    for i, day in enumerate(patient_time_days_dict[pid]):
        entry = (day, post[i])
        if pid in ctrl_set:
            ctrl_data.append(entry)
        elif pid in treat_set:
            treat_data.append(entry)

all_days = [d for d, _ in ctrl_data + treat_data]
bin_edges = np.linspace(min(all_days), max(all_days) + 1e-9, N_BINS_GRP + 1)
bin_weeks = ((bin_edges[:-1] + bin_edges[1:]) / 2.0) / 7.0

def _bin_mean(data):
    binned = [[] for _ in range(N_BINS_GRP)]
    for day, post in data:
        b = max(0, min(int(np.searchsorted(bin_edges, day, side="right")) - 1, N_BINS_GRP - 1))
        binned[b].append(post)
    means = np.full((N_BINS_GRP, K), np.nan)
    for b in range(N_BINS_GRP):
        if binned[b]:
            means[b] = np.array(binned[b]).mean(axis=0)
    return means

ctrl_means_grp = _bin_mean(ctrl_data)
treat_means_grp = _bin_mean(treat_data)

# ── Color scheme ──────────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#d62728", "#1f77b4"]   # red, blue
MEAN_COLORS = ["#e377c2", "#17becf"]      # pink, cyan

cmap = matplotlib.colormaps.get_cmap("tab20")
slot_x_fine = np.linspace(0, 24, ode_m.shape[1])

# ── Figure layout: 2 rows × 3 cols ───────────────────────────────────────────
# Col 1: empirical 3D
# Col 2: predicted 3D
# Col 3: top=π(t), bottom=group membership / cluster means
fig = plt.figure(figsize=(20*0.7, 6.5*0.7))
gs = gridspec.GridSpec(
    2, 4,
    width_ratios=[1.2, 1.2, 0.1, 0.9],  # third entry is spacer width
    height_ratios=[1, 1],
    wspace=0.19,
    hspace=0.40,
)

ax_emp   = fig.add_subplot(gs[:, 0], projection="3d")
ax_pred  = fig.add_subplot(gs[:, 1], projection="3d")
# gs[:, 2] is intentionally unused as spacer
ax_pi    = fig.add_subplot(gs[0, 3])
ax_group = fig.add_subplot(gs[1, 3])

# ── Col 1: Empirical 3D surface ──────────────────────────────────────────────
H_emp, W_emp = np.meshgrid(hours, centers)
ax_emp.plot_surface(
    H_emp, W_emp, surf,
    cmap="viridis",
    alpha=0.85,
    edgecolor="k",
    linewidth=0.1
)
ax_emp.set_xlabel("Hour")
ax_emp.set_ylabel("Week")
ax_emp.set_zlabel("Glucose (mg/dL)")
ax_emp.set_title("Empirical mean", y=0.98 ) # move a little lower
ax_emp.view_init(elev=25, azim=-60)
ax_emp.set_zlim(120, 180)

# ── Col 2: Predicted 3D surface ──────────────────────────────────────────────
H_pred, W_pred = np.meshgrid(hours, t_weeks)
ax_pred.plot_surface(
    H_pred, W_pred, pred_surface,
    cmap="viridis",
    alpha=0.85,
    edgecolor="k",
    linewidth=0.1
)
ax_pred.set_xlabel("Hour")
ax_pred.set_ylabel("Week")
ax_pred.set_zlabel("Glucose (mg/dL)")
ax_pred.set_title("Predicted mean (NeuralODE)", y=0.98)
ax_pred.view_init(elev=25, azim=-60)
ax_pred.set_zlim(120, 180)

# ── Col 3, Row 1: π(t) ───────────────────────────────────────────────────────
for k in range(K):
    ax_pi.plot(
        t_weeks,
        ode_pi[:, k],
        lw=2.5,
        color=cmap(6 + k),
        label=fr"$\pi_{{{k + 1}}}(t)$"
    )

ax_pi.set_title(r"$\pi(t)$: cluster weights over time")
ax_pi.set_xlabel("Treatment week", labelpad=-3)
ax_pi.set_ylabel("Probability")
ax_pi.set_ylim(0, 1)
ax_pi.grid(alpha=0.0)


# ── Col 3, Row 2: group membership probabilities ─────────────────────────────
for k in range(K):
    ok = ~np.isnan(ctrl_means_grp[:, k])
    ax_pi.plot(
        bin_weeks[ok],
        ctrl_means_grp[ok, k],
        lw=2.5,
        color=cmap(6 + k),
        linestyle=":",
        label=fr"$\overline{{\gamma}}_{{\mathrm{{control}}, {k + 1}}}(t)$"
    )

for k in range(K):
    ok = ~np.isnan(treat_means_grp[:, k])
    ax_pi.plot(
        bin_weeks[ok],
        treat_means_grp[ok, k],
        lw=2.5,
        color=cmap(6 + k),
        linestyle="--",
        label=fr"$\overline{{\gamma}}_{{\mathrm{{treatment}}, {k + 1}}}(t)$"
    )

#ax_pi.legend(fontsize=9, ncols=3)

handles, labels = ax_pi.get_legend_handles_labels()

# Create the customized legend
ax_pi.legend(
    handles[:2],       # Ensures only \pi_1 and \pi_2 are included
    labels[:2],        
    #fontsize=9,
    ncols=2,              # Two columns
    frameon=False,        # Removes the legend frame/box
    columnspacing=0.5,     # Sets spacing between columns to 0.5
    loc="center left",
    fontsize=12
)
# ax_group.set_title("Cluster membership by group")
# ax_group.set_xlabel("Treatment week")
# ax_group.set_ylabel("Probability")
# ax_group.set_ylim(0, 1)
# ax_group.grid(alpha=0.3)
# ax_group.legend(fontsize=8, ncols=1)

# bottom-right: cluster means
for k in range(K):
    ax_group.plot(slot_x_fine, ode_m[k], lw=2.5, color=cmap(k), label=fr"$m_{{{k + 1}}}(x)$")
    #ax_group.fill_between(slot_x_fine, ode_m[k] - ode_std[k], ode_m[k] + ode_std[k],  color=cmap(1), alpha=0.2)
ax_group.set_title("Cluster mean intraday glucose")
ax_group.set_xlabel("Hour of day")
ax_group.set_ylabel("Glucose (mg/dL)")
ax_group.grid(alpha=0.0)
ax_group.legend(fontsize=12, loc="upper left", frameon=False, ncols=2, columnspacing=0.5)

#plt.suptitle("NeuralODE temporal mixture", fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = IMAGES_DIR / "model_summary_v3.pdf"

plt.savefig(out_path, dpi=300, bbox_inches="tight")


# ## 2.1 Total Variation Distance Between Groups
# 
# We quantify the divergence between the control and treatment arms via the **total variation (TV) distance** between their group-level posterior distributions at each time bin:
# 
# $$\mathrm{TV}(t) = \tfrac{1}{2}\|\bar{\gamma}_{\mathrm{treat}}(t) - \bar{\gamma}_{\mathrm{ctrl}}(t)\|_1$$
# 
# where $\bar{\gamma}_{g,k}(t)$ is the average posterior probability of cluster $k$ across all patients in group $g$ at time $t$.
# TV = 0 means identical cluster usage; TV = 1 means complete separation.

# In[ ]:


# ── Total Variation distance between control and treatment ────────────────────
# Reuses ctrl_means_grp, treat_means_grp, bin_weeks from the cell above.

tv_distance = 0.5 * np.abs(treat_means_grp - ctrl_means_grp).sum(axis=1)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

# Left: TV(t) over time
valid = ~np.isnan(tv_distance)
axes[0].plot(bin_weeks[valid], tv_distance[valid], "o-", lw=2, color="#2ca02c")
axes[0].fill_between(bin_weeks[valid], 0, tv_distance[valid], alpha=0.15, color="#2ca02c")
axes[0].set_xlabel("Treatment week")
axes[0].set_ylabel("TV distance")
axes[0].set_title("Total Variation distance: treatment vs. control")
axes[0].set_ylim(0, max(0.3, tv_distance[valid].max() * 1.2))
axes[0].axhline(0, color="gray", lw=0.5)
axes[0].grid(alpha=0.3)

# Annotate start and end values
axes[0].annotate(
    f"TV = {tv_distance[valid][0]:.3f}",
    xy=(bin_weeks[valid][0], tv_distance[valid][0]),
    xytext=(bin_weeks[valid][0] + 3, tv_distance[valid].max() * 0.8),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=10,
)
axes[0].annotate(
    f"TV = {tv_distance[valid][-1]:.3f}",
    xy=(bin_weeks[valid][-1], tv_distance[valid][-1]),
    xytext=(bin_weeks[valid][-1] - 10, tv_distance[valid].max() * 0.5),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=10,
)

# Right: summary statistics table
stats = {
    "Metric": [
        "TV at baseline",
        "TV at end of trial",
        "Max TV",
        "Mean TV",
        "Week of max TV",
    ],
    "Value": [
        f"{tv_distance[valid][0]:.4f}",
        f"{tv_distance[valid][-1]:.4f}",
        f"{tv_distance[valid].max():.4f}",
        f"{tv_distance[valid].mean():.4f}",
        f"{bin_weeks[valid][np.argmax(tv_distance[valid])]:.1f}",
    ],
}
axes[1].axis("off")
table = axes[1].table(
    cellText=list(zip(stats["Metric"], stats["Value"])),
    colLabels=["Metric", "Value"],
    loc="center",
    cellLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

plt.tight_layout()
plt.savefig(IMAGES_DIR / "tv_distance.pdf", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nTV distance per time bin:")
for w, tv in zip(bin_weeks[valid], tv_distance[valid]):
    print(f"  Week {w:5.1f}: TV = {tv:.4f}")


# ## 3. Correlation matrix model
# 
# Sliding-window 24×24 inter-hour correlation matrices, embedded in Sym(24) via scaled vech.

# In[ ]:


K=2
corr_mats, corr_t, corr_pids, corr_wd, _ = compute_sliding_window_correlations(
    patient_data, window_size=10, stride=1, shrinkage=0.1, verbose=True
)
sym_basis = SymmetricMatrixBasis(n=24, device=device, dtype=dtype)
corr_coeffs = sym_basis.project(
    torch.tensor(corr_mats, device=device, dtype=dtype)
).detach().cpu().numpy()

corr_rep = build_temporal_coefficients(corr_coeffs, corr_t, N_BINS, device, dtype)
sigma_c = estimate_sigma_median_heuristic(corr_rep["X_time"], corr_rep["mask"])
sigma_c = 1.0


# In[ ]:


K=3
corr_mats, corr_t, corr_pids, corr_wd, _ = compute_sliding_window_correlations(
    patient_data, window_size=10, stride=1, shrinkage=0.1, verbose=True
)
sym_basis = SymmetricMatrixBasis(n=24, device=device, dtype=dtype)
corr_coeffs = sym_basis.project(
    torch.tensor(corr_mats, device=device, dtype=dtype)
).detach().cpu().numpy()

corr_rep = build_temporal_coefficients(corr_coeffs, corr_t, N_BINS, device, dtype)
sigma_c = estimate_sigma_median_heuristic(corr_rep["X_time"], corr_rep["mask"])
sigma_c = 5.5
EPOCHS = 200
LR = 1e-2
model_c, hist_c, pi_c = run_experiment(
    "CorrODE",
    NeuralODETimeWeights(corr_rep["t_grid"], K, hidden_dim=64, device=device, dtype=dtype),
    corr_rep["X_time"], GaussianKernel(sigma=max(1e-8, sigma_c)),
    K, corr_rep["coeff_dim"], EPOCHS, LR, device, dtype, corr_rep["mask"], verbose=True,
)

with torch.no_grad():
    recon_corr = sym_basis.reconstruct(
        model_c.mean.cpu() * corr_rep["coeff_std"].cpu() + corr_rep["coeff_mean"].cpu()
    ).numpy()

t_weeks_c = (
    corr_rep["t_grid"].cpu().numpy()
    * (corr_rep["t_max_days"] - corr_rep["t_min_days"]) + corr_rep["t_min_days"]
) / 7.0


# In[ ]:


import matplotlib.pyplot as plt



# 1. Use layout="constrained" but we will tune its padding below

fig = plt.figure(figsize=(3 + 3 * K, 2.8))



# 2. Set hspace to 0.0

gs = fig.add_gridspec(nrows=2, ncols=1 + K, hspace=0.5, wspace=0.15)



# --- COLUMN 1, ROW 1: MMD² loss ---

ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(hist_c, lw=2)

ax1.set(title="Training loss (MMD²)", ylabel="Loss")

ax1.grid(alpha=0.3)



# REMOVE x-axis labels and tick labels to allow ax2 to slide up

ax1.set_xticks([0, 200])

# ax1.set_xlabel("")



# --- COLUMN 1, ROW 2: π(t) ---

ax2 = fig.add_subplot(gs[1, 0])

for k in range(K):

    ax2.plot(t_weeks_c, pi_c.cpu().numpy()[:, k], lw=2, label=f"$\pi_{k + 1}(t)$")



# REMOVE the title and move it to a ylabel or an annotation inside the plot

# This prevents the title text from pushing ax1 away.

ax2.set(xlabel="Treatment week", ylabel="Probability", ylim=(0, 1))

ax2.legend(fontsize=10, loc='upper center', ncols=3, frameon=False, handlelength=1., columnspacing=0.5)

ax2.grid(alpha=0.0)

ax2.set_title(r"$\pi(t)$: cluster weights")



# --- COLUMNS 2 to 1+K: Correlation Heatmaps ---

for k in range(K):

    # These span both rows (gs[:, k+1])

    ax_k = fig.add_subplot(gs[:, k + 1])

    im = ax_k.imshow(recon_corr[k], cmap="RdBu_r", vmin=-1, vmax=1,

                    origin="lower", aspect='auto')

    ax_k.set_xticks([]); ax_k.set_yticks([])

    ax_k.set(title=fr"$m_{k + 1}$", xlabel="Hours")



fig.set_constrained_layout_pads(h_pad=0.01, w_pad=0.1)



# Align the labels vertically

fig.align_ylabels([ax1, ax2])

#plt.suptitle("Training Correlation Matrices", fontsize=16, y=1.03)

fig.savefig(IMAGES_DIR / "corr_ode_training.pdf", dpi=300, bbox_inches="tight")

plt.show()


# ## 4. Patient correlation network with graph Laplacian features
#
# Replace the `SymmetricMatrixBasis` representation with a fixed patient graph
# and a Laplacian eigenbasis.
#
# Pipeline:
# 1. Compute per-patient sliding-window average curves $x_i(t)$
# 2. Build patient-to-patient similarity matrices $S(t) \in \mathbb{R}^{N \times N}$
# 3. Construct a fixed patient graph from the time-averaged similarity matrix
# 4. Project each $S(t)$ as a graph signal onto a `GraphLaplacianBasis`
# 5. Bin in time, fit the temporal GMM with MMD, and reconstruct cluster means
# 6. Visualize the learned patient graph and network snapshots → `patient_graph.pdf`

# In[ ]:


WINDOW_SIZE = 21
STRIDE = 1
MIN_PATIENT_PRESENCE = 0.4
N_TIME_BINS = 10
N_COMPONENTS = 3

GRAPH_EDGE_PERCENTILE = 85
R_GRAPH = 4
GRAPH_ALPHA = 1.0
NORMALIZED_LAPLACIAN = False

PG_EPOCHS = 200
PG_LR = 0.01
MODEL_TYPE = "basis"
R_PI = 3
ODE_HIDDEN = 16
SIGMA_MULT = 4.0
SEED = 42

EDGE_THRESHOLD_PERCENTILE = 2
SPACE_METRIC = "h1"
R_S = 16

torch.manual_seed(SEED)
np.random.seed(SEED)

patient_data_pg = load_and_preprocess_cgm(csv_path=csv_path, verbose=True)
print(f"\nPatients loaded: {len(patient_data_pg)}")

N_SLOTS = 288
patient_window_curves = {}
for pid, days_matrix in patient_data_pg.items():
    n_days = days_matrix.shape[0]
    pw = {}
    for w in range(0, n_days - WINDOW_SIZE + 1, STRIDE):
        window = days_matrix[w : w + WINDOW_SIZE]
        avg_curve = np.nanmean(window, axis=0)
        nan_mask = np.isnan(avg_curve)
        if nan_mask.all():
            continue
        if nan_mask.any():
            valid_idx = np.where(~nan_mask)[0]
            avg_curve = np.interp(np.arange(N_SLOTS), valid_idx, avg_curve[valid_idx])
        pw[w] = avg_curve
    if pw:
        patient_window_curves[pid] = pw


# In[ ]:


all_w = sorted({w for pw in patient_window_curves.values() for w in pw})
w_pids = {w: set() for w in all_w}
for pid, pw in patient_window_curves.items():
    for w in pw:
        w_pids[w].add(pid)

n_total_w = len(all_w)
patient_pres = {
    pid: sum(1 for w in all_w if pid in w_pids[w]) / n_total_w
    for pid in patient_window_curves
}
common_pids = sorted(pid for pid, frac in patient_pres.items() if frac >= MIN_PATIENT_PRESENCE)
N_pat = len(common_pids)
pid_to_idx = {pid: i for i, pid in enumerate(common_pids)}

valid_w = [w for w in all_w if all(pid in w_pids[w] for pid in common_pids)]
pg_t_indices = np.array([w + (WINDOW_SIZE - 1) / 2.0 for w in valid_w])

ctrl_set_pg, treat_set_pg = set(CONTROL_IDS), set(TREATMENT_IDS)

basis_cls = L2CosineBasis if SPACE_METRIC == "l2" else H1CosineBasis
curve_basis = basis_cls(T=1.0, R=R_S, grid_size=N_SLOTS, d=1, device=device, dtype=dtype)

pairwise_distances = np.zeros((len(valid_w), N_pat, N_pat))
for idx, w in enumerate(valid_w):
    curves_raw = np.stack([patient_window_curves[pid][w] for pid in common_pids])
    curves_t = torch.tensor(curves_raw, device=device, dtype=dtype).unsqueeze(-1)
    with torch.no_grad():
        coeffs = curve_basis.project(curves_t).cpu().numpy()
    pairwise_distances[idx] = squareform(pdist(coeffs, metric="euclidean"))

global_max_dist = pairwise_distances.max()
sim_matrices = 1.0 - pairwise_distances / (global_max_dist + 1e-10)


# In[ ]:


avg_sim = sim_matrices.mean(axis=0)
np.fill_diagonal(avg_sim, 0.0)

avg_off_diag = avg_sim[~np.eye(N_pat, dtype=bool)]
graph_threshold = np.percentile(avg_off_diag, GRAPH_EDGE_PERCENTILE)
edge_mask = threshold_to_adjacency(avg_sim, threshold=graph_threshold, absolute=False)
graph_weights = avg_sim * edge_mask
n_edges = int(np.count_nonzero(np.triu(edge_mask, k=1)))

graph_rank = N_pat if R_GRAPH is None else min(R_GRAPH, N_pat)
graph_basis = GraphLaplacianBasis.from_adjacency(
    adjacency=torch.tensor(graph_weights, device=device, dtype=dtype),
    alpha=GRAPH_ALPHA,
    num_eigenvectors=graph_rank,
    normalized=NORMALIZED_LAPLACIAN,
    device=device,
    dtype=dtype,
)

sim_t = torch.tensor(sim_matrices, device=device, dtype=dtype)
coeffs_all = graph_basis.project(sim_t).detach().cpu().numpy()


# In[ ]:


pg_rep = build_temporal_coefficients(
    coeffs=coeffs_all,
    t_indices=pg_t_indices,
    n_time_bins=N_TIME_BINS,
    device=device,
    dtype=dtype,
    verbose=True,
)
X_time_pg, t_grid_pg, mask_pg = pg_rep["X_time"], pg_rep["t_grid"], pg_rep["mask"]

sigma_auto = estimate_sigma_median_heuristic(X_time_pg, mask_pg)
sigma_pg = max(1e-8, sigma_auto * SIGMA_MULT)
kernel_pg = GaussianKernel(sigma=sigma_pg)

if MODEL_TYPE == "basis":
    time_basis = L2CosineBasis(
        T=1.0, R=R_PI, grid_size=pg_rep["L_t"], d=1, device=device, dtype=dtype
    )
    twm = BasisLogitsTimeWeights(
        basis_matrix=time_basis.Phi,
        num_components=N_COMPONENTS,
        device=device,
        dtype=dtype,
    )
    model_name = "PatientGraphBasis"
else:
    twm = NeuralODETimeWeights(
        t_grid=t_grid_pg,
        num_components=N_COMPONENTS,
        hidden_dim=ODE_HIDDEN,
        device=device,
        dtype=dtype,
    )
    model_name = "PatientGraphODE"

print(f"Training {model_name} ({PG_EPOCHS} epochs)...")
pg_model, pg_history, pg_pi_t = run_experiment(
    name=model_name,
    time_weight_model=twm,
    X_time=X_time_pg,
    kernel=kernel_pg,
    n_components=N_COMPONENTS,
    coeff_dim=pg_rep["coeff_dim"],
    num_epochs=PG_EPOCHS,
    lr=PG_LR,
    device=device,
    dtype=dtype,
    mask=mask_pg,
    verbose=True,
)


# In[ ]:


pi_t_np = pg_pi_t.cpu().numpy()
t_np = t_grid_pg.detach().cpu().numpy()
day_range = pg_rep["t_max_days"] - pg_rep["t_min_days"]
t_days_pg = t_np * day_range + pg_rep["t_min_days"]
use_weeks = t_days_pg.max() > 30
t_axis = t_days_pg / 7.0 if use_weeks else t_days_pg
t_label_pg = "Treatment week" if use_weeks else "Treatment day"
K_pg = N_COMPONENTS

ctrl_indices = [pid_to_idx[p] for p in common_pids if p in ctrl_set_pg]
treat_indices = [pid_to_idx[p] for p in common_pids if p in treat_set_pg]
communities = [comm for comm in (ctrl_indices, treat_indices) if comm]

layout_weights = graph_weights.copy()
if np.count_nonzero(layout_weights) == 0:
    layout_weights = avg_sim.copy()
np.fill_diagonal(layout_weights, 0.0)
G_full = nx.from_numpy_array(layout_weights)

if len(communities) >= 2:
    supergraph = nx.path_graph(len(communities))
    superpos = nx.spring_layout(supergraph, scale=1, seed=SEED)
    centers_pg = [superpos[i] for i in range(len(communities))]
    pos = {}
    for center, comm in zip(centers_pg, communities):
        uniform_subg = nx.complete_graph(comm)
        pos.update(
            nx.spring_layout(
                uniform_subg,
                center=center,
                seed=SEED,
                weight=None,
                k=1.5 / max(np.sqrt(len(comm)), 1),
            )
        )
else:
    pos = nx.spring_layout(
        G_full,
        seed=SEED,
        weight=None,
        k=2.5 / max(np.sqrt(N_pat), 1),
        iterations=200,
    )

node_colors = []
for pid in common_pids:
    if pid in ctrl_set_pg:
        node_colors.append("#4A90D9")
    elif pid in treat_set_pg:
        node_colors.append("#D94A4A")
    else:
        node_colors.append("#999999")

off_diag_sim_all = sim_matrices[:, ~np.eye(N_pat, dtype=bool)]
threshold = np.percentile(off_diag_sim_all, 100 - EDGE_THRESHOLD_PERCENTILE)
n_graph_times = 4
graph_indices = np.linspace(0, len(valid_w) - 1, n_graph_times, dtype=int)

fig = plt.figure(figsize=(16, 3.5))
outer = fig.add_gridspec(2, 2, width_ratios=(1.0, 4.0), hspace=0.4, wspace=0.05)

ax_loss = fig.add_subplot(outer[0, 0])
ax_loss.plot(pg_history, lw=2, color="tab:blue")
ax_loss.set_xticks([0, 200])
ax_loss.set_title("Training loss (MMD²)")
ax_loss.set_ylabel("Loss", fontsize=13)
ax_loss.grid(alpha=0.3)

ax_pi_pg = fig.add_subplot(outer[1, 0])
for k in range(K_pg):
    ax_pi_pg.plot(t_axis, pi_t_np[:, k], lw=2, label=fr"$\pi_{{{k + 1}}}(t)$")
ax_pi_pg.set_title(r"$\pi(t)$: cluster weights")
ax_pi_pg.set_xlabel(t_label_pg, fontsize=13)
ax_pi_pg.set_ylabel("Probability", fontsize=13)
ax_pi_pg.set_ylim(0, 1.2)
ax_pi_pg.grid(alpha=0.0)
ax_pi_pg.legend(
    fontsize=13,
    frameon=False,
    ncols=3,
    columnspacing=0.4,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    handlelength=0.7,
)

graph_gs = outer[:, 1].subgridspec(1, n_graph_times, wspace=0.08)
graph_axes = [fig.add_subplot(graph_gs[0, i]) for i in range(n_graph_times)]

for ax, t_idx in zip(graph_axes, graph_indices):
    S = sim_matrices[t_idx]
    t_day = pg_t_indices[t_idx]

    G = nx.Graph()
    G.add_nodes_from(range(N_pat))
    edges, edge_w = [], []
    for i in range(N_pat):
        for j in range(i + 1, N_pat):
            if S[i, j] >= threshold:
                G.add_edge(i, j)
                edges.append((i, j))
                edge_w.append(S[i, j])

    if edge_w:
        ew = np.array(edge_w)
        ew_norm = (ew - ew.min()) / (ew.max() - ew.min() + 1e-9)
        edge_colors = [plt.cm.Greens(0.3 + 0.7 * v) for v in ew_norm]
        edge_widths = 0.8 + 2.0 * ew_norm
    else:
        edge_colors, edge_widths = [], []

    if len(communities) >= 2:
        for comm, bg_color in zip(communities[:2], ("#DDEEFF", "#FFDDDD")):
            comm_pos = np.array([pos[n] for n in comm])
            cx, cy = comm_pos.mean(axis=0)
            rx = comm_pos[:, 0].std() * 2.5 + 0.15
            ry = comm_pos[:, 1].std() * 2.5 + 0.15
            ax.add_patch(
                Ellipse(
                    (cx, cy),
                    width=1.5 * rx,
                    height=1.5 * ry,
                    color=bg_color,
                    alpha=0.25,
                    zorder=0,
                )
            )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=25, alpha=0.9,
        edgecolors="black", linewidths=0.5,
    )
    if edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges, edge_color=edge_colors, width=edge_widths, alpha=0.7,
        )

    title = f"Week {t_day / 7:.1f}" if use_weeks else f"Day {t_day:.0f}"
    ax.text(0.55, 0.9, title, transform=ax.transAxes, fontsize=15, va="top")
    ax.set_axis_off()

legend_elements = [
    Patch(facecolor="#4A90D9", edgecolor="black", label="Control"),
    Patch(facecolor="#D94A4A", edgecolor="black", label="Treatment"),
    Line2D([0], [0], color=plt.cm.Greens(0.85), lw=2.5, label="Similar (high)"),
    Line2D([0], [0], color=plt.cm.Greens(0.35), lw=1.0, label="Less similar"),
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.65, 0.1),
    fontsize=15,
    ncols=4,
    frameon=False,
)

plt.tight_layout()
fig.subplots_adjust(top=0.84)
fig.savefig(IMAGES_DIR / "patient_graph.pdf", dpi=100, bbox_inches="tight")
plt.show()

# In[ ]:




