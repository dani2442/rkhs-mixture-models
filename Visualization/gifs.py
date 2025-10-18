
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import imageio
from tqdm import tqdm

# =====================================================================
# Funciones para visualización GIF
# =====================================================================

def gif(history_weights,
            history_means,
            history_covariances,
            history_X=None,           
            means_gt=None, 
            vars_gt=None,
            t_grid=None, 
            fps=10,
            evol_weights='path',      # 'path' | 'barplot'
            bg_mode='histogram',      # 'histogram' | 'gaussian'
            folder="Results",
            filename="gmm_evolution.gif"):
    """
    Animated visualisation of a time-evolving GMM.

    Parameters:
    -----------
    history_weights : list
        List of weight arrays for each time step
    history_means : list
        List of mean arrays for each time step
    history_covariances : list
        List of covariance arrays for each time step
    history_X : list, optional
        List of data arrays for each time step (needed if bg_mode == 'histogram')
    means_gt : list, optional
        Ground truth means (needed if bg_mode == 'gaussian')
    vars_gt : list, optional
        Ground truth variances (needed if bg_mode == 'gaussian')
    t_grid : array, optional
        Time grid for evolution plots
    fps : int, optional
        Frames per second for the GIF
    evol_weights : str, optional
        Mode for visualizing weight evolution ('path' | 'barplot')
    bg_mode : str, optional
        Background visualization mode ('histogram' | 'gaussian')
    folder : str, optional
        Folder to save the GIF
    filename : str, optional
        Filename for the GIF
    """
    # -----------------------------------------------------------------
    # 0)  Early exit & basic constants
    # -----------------------------------------------------------------
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path) and (t_grid is not None and len(t_grid) > 1e5):
        print("GIF already exists →", full_path)
        return

    n_frames = len(history_weights)
    K = len(history_weights[0])
    cmap = plt.cm.get_cmap('viridis', K)

    # -----------------------------------------------------------------
    # 1)  x-axis limits
    # -----------------------------------------------------------------
    if bg_mode == 'histogram' and history_X is not None:
        all_X = np.vstack(history_X)
        x_min = all_X[:, 0].min()
        x_max = all_X[:, 0].max()
        margin = 0.1 * (x_max - x_min)
        top_xlim = (x_min - margin, x_max + margin)

    elif bg_mode == 'gaussian' and means_gt is not None:
        # FIX: Handle the structure [[m1]*d, [m2]*d, [m3]*d]
        all_means = []
        for t_means in means_gt:  # For each time
            for cluster_mean in t_means:  # For each cluster
                if isinstance(cluster_mean, list):
                    all_means.append(cluster_mean[0])  # Get first element
                else:
                    all_means.append(cluster_mean)
        
        max_std = np.sqrt(max(vars_gt))
        top_xlim = (min(all_means) - 5 * max_std,
                    max(all_means) + 5 * max_std)
    else:
        raise ValueError("Inconsistent background setup.")

    # -----------------------------------------------------------------
    # 2)  y-axis limits - Evaluate every mixture on the same fine grid
    # -----------------------------------------------------------------
    x_grid = np.linspace(top_xlim[0], top_xlim[1], 400)
    global_max = 0.0
    for f in range(n_frames):
        pdf_f = np.zeros_like(x_grid)
        for k in range(K):
            # Extract mean properly, handling both float and array/list cases
            if hasattr(history_means[f][k], '__len__'):
                μ = history_means[f][k][0]  # If it's a list/array, get first element
            else:
                μ = history_means[f][k]     # If it's a float, use directly
            
            # Extract variance properly, handling different formats
            if hasattr(history_covariances[f][k], '__len__'):
                if hasattr(history_covariances[f][k][0], '__len__'):
                    σ2 = history_covariances[f][k][0][0]  # For 2D array format
                else:
                    σ2 = history_covariances[f][k][0]     # For 1D array format
            else:
                σ2 = history_covariances[f][k]            # For scalar format
                
            pdf_f += history_weights[f][k] * norm.pdf(x_grid, loc=μ, scale=np.sqrt(σ2))
        global_max = max(global_max, pdf_f.max())

    top_ylim = (0.0, global_max * 1.10)   # 10% head-room

    # -----------------------------------------------------------------
    # 3)  bottom-panel limits
    # -----------------------------------------------------------------
    if evol_weights == 'barplot':
        bottom_xlim = (-0.5, K - 0.5)
    else:
        bottom_xlim = (t_grid[0], t_grid[-1]) if t_grid is not None else (0, n_frames - 1)
    bottom_ylim = (0, 1)

    # -----------------------------------------------------------------
    # 4)  Prepare figure
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    axT = fig.add_subplot(gs[0])          # top panel
    axB = fig.add_subplot(gs[1])          # bottom panel
    frames = []

    # Optional: repeat short history_X to match n_frames (avoids len mismatch)
    if bg_mode == 'histogram' and history_X is not None and len(history_X) < n_frames:
        rep = int(np.ceil(n_frames / len(history_X)))
        history_X = (history_X * rep)[:n_frames]

    # -----------------------------------------------------------------
    # 5)  Build each frame
    # -----------------------------------------------------------------
    for f in tqdm(range(n_frames), desc="Building GMM GIF"):
        axT.clear(); axB.clear()
        axT.set_xlim(top_xlim); axT.set_ylim(top_ylim)
        axB.set_xlim(bottom_xlim); axB.set_ylim(bottom_ylim)

        # ---------- background ----------
        if bg_mode == 'histogram':
            Xf = np.asarray(history_X[f])[:, 0]
            axT.hist(Xf, bins=50, density=True, color='lightgray', alpha=0.8)
        else:   # gaussian GT
            # FIX: Calculate GT as proper GMM, not average
            frame_idx = min(f, len(means_gt) - 1)  # Handle shorter GT
            pdf = np.zeros_like(x_grid)
            n_clusters_gt = len(means_gt[frame_idx])
            
            # Assume equal weights for GT (as in data generation)
            weight_gt = 1.0 / n_clusters_gt
            
            for k in range(n_clusters_gt):
                # Extract mean for cluster k
                if isinstance(means_gt[frame_idx][k], list):
                    μ_gt = means_gt[frame_idx][k][0]  # First dimension
                else:
                    μ_gt = means_gt[frame_idx][k]
                
                # Add component to mixture
                pdf += weight_gt * norm.pdf(x_grid, μ_gt, np.sqrt(vars_gt[frame_idx]))
            
            axT.fill_between(x_grid, 0, pdf, color='lightgray', alpha=0.4)
            axT.plot(x_grid, pdf, color='k', lw=2)

        # ---------- draw mixture components ----------
        for k in range(K):
            # Extract mean
            if hasattr(history_means[f][k], '__len__'):
                μ = history_means[f][k][0]
            else:
                μ = history_means[f][k]
            
            # Extract variance
            if hasattr(history_covariances[f][k], '__len__'):
                if hasattr(history_covariances[f][k][0], '__len__'):
                    σ2 = history_covariances[f][k][0][0]
                else:
                    σ2 = history_covariances[f][k][0]
            else:
                σ2 = history_covariances[f][k]
            
            pdf_k = history_weights[f][k] * norm.pdf(x_grid, μ, np.sqrt(σ2))
            axT.fill_between(x_grid, 0, pdf_k, color=cmap(k), alpha=0.3)
            axT.plot(x_grid, pdf_k, color=cmap(k), lw=1)
            axT.scatter([μ], [0], s=70*history_weights[f][k], color=cmap(k), marker='s')

        # ---------- bottom panel ----------
        if evol_weights == 'barplot':
            axB.bar(np.arange(K), history_weights[f], color=[cmap(k) for k in range(K)])
        else:   # evol_weights == 'path'
            x_axis = np.arange(f+1) if t_grid is None else t_grid[:f+1]
            for k in range(K):
                traj = [history_weights[i][k] for i in range(f+1)]
                axB.plot(x_axis, traj, color=cmap(k), marker='o', ms=1)

        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)

    plt.close(fig)
    imageio.mimsave(full_path, frames, fps=fps)
    print(f"[INFO] GMM GIF saved → {full_path}")