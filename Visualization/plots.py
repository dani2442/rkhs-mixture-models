import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
from tqdm import tqdm

from Utils.IO_utils import load_experiment, order_results
from Utils.test_utils import compute_gmm_density_mc, compute_gt_density_mc
from data import compute_gmm_density

def plot_fit(diff_ws, diff_mus, diff_covs, mmds,
             X, final_weights, final_means, final_covariances,
             best_index, folder="Results", filename='fit.png',
             log_scale=False):
    """
    Plot the evolution of parameter differences and the final Gaussian mixture.
    Colors of components are assigned in increasing order of their means.
    """
    # create folder if missing
    os.makedirs(folder, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # --- Left panel: error evolution ---
    iters = np.arange(len(diff_mus))
    ax1 = axes[0]
    ax1.plot(iters, diff_ws, label='Weights diff')
    ax1.plot(iters, diff_mus, label='Means diff')
    ax1.plot(iters, diff_covs, label='Covariance diff')
    ax1.plot(iters, mmds, label='Squared MMD error')
    ax1.set(xlabel='Iteration', ylabel='Error (log scale)' if log_scale else 'Error',
            title='Error Evolution')
    ax1.legend(); ax1.grid(True)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if log_scale:
        ax1.set_yscale('log')

    # --- Right panel: final mixture ---
    ax2 = axes[1]
    K = len(final_weights)
    if K == 3:
        base_colors = ["tab:blue", "tab:green", "tab:orange"][:K]
    else:
        cmap = cm.get_cmap("viridis", K)
        base_colors = [cmap(i) for i in range(K)]

    # only for 1D data
    if X.shape[1] == 1:
        ax2.hist(X[:, 0], bins=50, density=True,
                 alpha=0.5, label='Empirical Data')
        # grid for density
        x_grid = np.linspace(-8,30, 5000)

        # sort components by their means
        means_1d = np.array([m if np.isscalar(m) else m[0]
                              for m in final_means])
        order = np.argsort(means_1d)

        total_density = np.zeros_like(x_grid)
        for idx, k in enumerate(order):
            w = final_weights[k]
            mu = final_means[k]
            cov = final_covariances[k]
            # compute density
            comp = w * multivariate_normal.pdf(x_grid, mean=mu, cov=cov)
            total_density += comp
            color = base_colors[idx]
            # fill, line, and marker
            ax2.fill_between(x_grid, 0, comp, color=color, alpha=0.3)
            ax2.plot(x_grid, comp, color=color, lw=max(1, 2*w))
            ax2.scatter(mu, 0, s=int(70*w), color=color, marker='s')
        # mixture density
        ax2.plot(x_grid, total_density, 'r--', lw=2, label='Mixture')
        ax2.set(xlabel='x', ylabel='Density')

    ax2.set_title(f'Optimal Gaussian Mixture (Iter {best_index})')
    fig.tight_layout()
    out_path = os.path.join(folder, filename)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    
def _grid_1d(X: np.ndarray,
             means: np.ndarray,
             covs: np.ndarray,
             n: int = 400) -> np.ndarray:
    """
    Build a 1-D grid from the minimum data point to the maximum mean,
    extended by ±4·σ_max on both sides.
    """
    x_min = X.min()
    x_max = X.max()
    max_sigma = np.sqrt(np.max(covs))
    lo = min(x_min, means.min()) - 4 * max_sigma
    hi = max(x_max, means.max()) + 4 * max_sigma
    return np.linspace(lo, hi, n)


def plot_mixture(X: np.ndarray,
                    weights: np.ndarray,
                    means: np.ndarray,
                    covs: np.ndarray,
                    folder: str = "Results",
                    filename: str = "mixture.png",
                    figsize: tuple = (10, 5)) -> None:
    """
    Figure that shows the 1-D Gaussian mixture
    (components + total density + data histogram).

    """
    if X.shape[1] != 1:
        print("plot_mixture_1d is for d = 1 only.")
        return

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    K = len(weights)
    # --- special colour mapping when K == 3 -------------------------
    if K == 3:
        order = np.argsort(means.ravel())          # left → right
        base_colors = ["tab:blue", "tab:green", "tab:orange"]
        colours = {k: base_colors[i] for i, k in enumerate(order)}
        cmap_fn = lambda k: colours[k]
    else:
        cmap = cm.get_cmap("viridis", K)
        cmap_fn = lambda k: cmap(k)

    # --- grid & densities ------------------------------------------
    x_grid = _grid_1d(X, means, covs)
    total  = np.zeros_like(x_grid)

    # --- figure ----------------------------------------------------
    plt.figure(figsize=figsize)
    plt.hist(X[:, 0], bins=50, density=True,
             alpha=0.5, label="Empirical data", color="lightgray")

    for k in range(K):
        mu  = means[k]
        var = covs[k] if np.ndim(covs[k]) == 0 else covs[k][0]
        pdf_k = weights[k] * multivariate_normal.pdf(x_grid,
                                                     mean=mu,
                                                     cov=var)
        total += pdf_k
        lw = max(1, 2 * weights[k])            
        plt.fill_between(x_grid, 0, pdf_k,
                         color=cmap_fn(k), alpha=0.5)
        plt.plot(x_grid, pdf_k, color=cmap_fn(k), lw=lw)
        plt.scatter(mu, 0, s=80 * weights[k],
                    color=cmap_fn(k), marker='s')

    plt.plot(x_grid, total, 'k--', lw=2, label="Mixture density")
    plt.xlabel('x')
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
   
# Plotting functions for NODE training loss and trajectory

def plot_training(loss_history, log = True, folder=None):
    """
    Plot the evolution of the training loss over epochs.
    
    Parameters:
      loss_history (list): List of loss values per epoch.
    log (bool): If True, plot on a logarithmic scale.
      folder (str): Directory to save the plot.
    """
    print(f"Plotting training loss history.")
    # Simulate a progress bar while "processing" loss history.
    for _ in tqdm(loss_history, desc="Processing training loss history", leave=False):
        pass  # This loop only serves to display a progress bar.
    
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.title("Training Loss Evolution")
    if folder is not None:
        filename = os.path.join(folder, "node_training.png")
        plt.savefig(filename)
        print(f"Training plot saved.")
        plt.close()
    else:
        plt.show()
        plt.close()
        
def add_multicolored_title(ax, base_text="Weight Trajectory - Components",
                          comp_colors=['#1a659e', '#ffa62b', '#7ae582']):
    """
    Add a title to a plot with different colored component numbers,
    placed above the axes so they never overlap the data.
    Adjusted vertical placement to match other plots.
    """
    
    # Get figure and adjust top margin to leave space for title
    fig = ax.get_figure()  

    # Title lower down
    fig.subplots_adjust(top=0.75)

    # 1) Draw base text with fig.text, track Text instance for bbox
    #    Use y=0.94 for vertical placement
    t = fig.text(0.5, 0.9, base_text,
                 ha='center', va='bottom', fontsize=22)

    # 2) Force draw to populate the renderer and get bounding box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer)

    # 3) Transform the right edge of base text to figure coords
    inv = fig.transFigure.inverted()
    x_base_end, _ = inv.transform((bb.x1, bb.y0))

    # 4) Define labels and small horizontal shift
    labels = ['1', '/', '2', '/', '3']
    colors = [comp_colors[0], 'black', comp_colors[1], 'black', comp_colors[2]]

    # 5) Place each symbol in figure coords, just to the right of base text
    gap = 0.007
    dx = 0.03
    for i, (lab, col) in enumerate(zip(labels, colors)):
        fig.text(x_base_end + gap + i*dx, 0.9, lab,
                 ha='left', va='bottom', fontsize=22, color=col)



def plot_traj(t_grid, t_i, pred_traj, sample_data=None, K=3,
            folder=None, filename=None):
    """
    Plot predicted trajectories against observed data points.
    
    Args:
        t_grid (ndarray): Time grid for predictions (length T).
        t_i (ndarray): Observation times (length n_i).
        pred_traj (ndarray): Predicted trajectories, shape (T, K).
        sample_data (ndarray): Observed data, shape (n_i, K).
        K (int): Number of mixture components.
        legend (bool): Whether to add a legend.
        folder (str): Directory to save the plot (if provided).
        filename (str): Base filename without extension.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Ensure t_i is an array
    if t_i.ndim == 0:
        t_i = np.array([t_i])
        
    if sample_data is not None:
        # If sample_data has one row but multiple t_i, repeat the row
        if sample_data.shape[0] == 1 and pred_traj.shape[0] > 1:
            sample_data = np.repeat(sample_data, len(t_i), axis=0)

    # 1) Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # 2) Plot each component with consistent colors
    colors = ['#1a659e', '#ffa62b', '#7ae582'] if K == 3 else [None] * K
    for k in range(K):
        ax.plot(t_grid, pred_traj[:, k],
                color=colors[k], label=f'pred w_{k+1}')
        if sample_data is not None:
            ax.scatter(t_i, sample_data[:, k],
                    color=colors[k], marker='x', label=f'data w_{k+1}')

    # 3) Set axis labels and tick formatting
    ax.set_xlabel('Time', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 4) Add the multicolored title if there are exactly 3 components
    if K == 3:
        add_multicolored_title(ax,
            base_text="Weight Trajectory - Components",
            comp_colors=colors)

    # 5) Adjust layout to leave space for the title
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # 6) Save or display the figure
    if folder is not None:
        fname = 'trajectories.png' if filename is None else f"{filename}.png"
        fig.savefig(os.path.join(folder, fname),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


            
def plot_L2err(L2errs, L2errs_ts, t_grid, t_i, path):
    """
    Plot L2 errors over time for synthetic data (solo L2, sin KL).
    
    Parameters:
      L2errs: List/array of L2 error values corresponding to t_grid.
      L2errs_ts: List/array of L2 error values at specific time instants (t_i).
      t_grid: 1D array of all time points.
      t_i: 1D array of specific time instants where markers should be plotted.
      path: Path to save the plot.
    """
    plt.figure(figsize=(10, 4))
    
    # Plot L2 error
    plt.plot(t_grid, L2errs, 'b-', label='L2 Error (NODE)', linewidth=2)
    # Add large black diamond markers at specific time instants.
    plt.plot(t_i, L2errs_ts, 'D', markersize=6, color='black', label='L2 Error (MMD fitting)')
    
    plt.title('L2 Error Evolution', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('L2 Error', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(path, dpi=600)
    plt.close()
    print(f"L2 error plot saved: {path}")
    
# ------------------------------------------------------------------
# Plot   L2   +   MMD   empirical errors
# ------------------------------------------------------------------
def plot_L2MMDerr(L2_node, MMD_node,
                  L2_fit,  MMD_fit,
                  t_grid,  t_i,
                  path):
    """
    Two-panel log-scale plot.

    * Top  : L2 distance KDE ↔ GMM
    * Bottom : MMD distance   X ↔ GMM

    Black diamonds show the coarse instants *t_i* (weights from the
    MMD-fitting step); continuous lines show the NODE trajectory
    evaluated on the fine grid *t_grid*.

    Parameters
    ----------
    L2_node , MMD_node : list / ndarray, len == len(t_grid)
    L2_fit  , MMD_fit  : list / ndarray, len == len(t_i)
    t_grid  : 1-D array of fine time instants (NODE)
    t_i     : 1-D array of coarse time instants (MMD-fit)
    path    : full file-name where the PNG is saved (600 dpi)
    """

    plt.figure(figsize=(10, 6))

    # -------- L2 panel -------------------------------------------
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t_grid, L2_node, 'b-',  label='L2 (NODE)')
    ax1.plot(t_i,    L2_fit,  'Dk',  ms=4, label='L2 (MMD fit)')
    ax1.set_yscale('log')
    ax1.set_ylabel('L2 distance')
    ax1.set_title('Empirical L2 and MMD errors')
    ax1.legend()
    ax1.grid(alpha=.4, which='both')

    # -------- MMD panel ------------------------------------------
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(t_grid, MMD_node, 'r-',  label='MMD (NODE)')
    ax2.plot(t_i,    MMD_fit,  'Dk',  ms=4, label='MMD (MMD fit)')
    ax2.set_yscale('log')
    ax2.set_xlabel('time')
    ax2.set_ylabel('MMD distance')
    ax2.legend()
    ax2.grid(alpha=.4, which='both')

    plt.tight_layout()

    # high-resolution save
    plt.savefig(path, dpi=600)
    plt.close()


def plot_sumweights(t_grid, traj_weights, folder=None):
    """
    Plot the trajectory of the sum of weights over time.
    
    Parameters:
        t_grid (numpy.ndarray): Time grid for the x-axis.
            Should be a 1D array of shape (n_timesteps,).
        traj_weights (numpy.ndarray): Trajectory weights of shape (n_timesteps, n_components)
            or a 1D array of shape (n_timesteps,) if already summed.
        folder (str, optional): Directory to save the plot.
    """

    # Ensure t_grid is 1D.
    t_grid = np.array(t_grid).flatten()
    
    # If traj_weights is 2D, compute the sum over components.
    if traj_weights.ndim > 1:
        sum_weights = np.sum(traj_weights, axis=1)
    else:
        sum_weights = traj_weights

    # Create the plot.
    plt.figure(figsize=(12, 6))
    plt.plot(t_grid, sum_weights, label='Sum of Weights', color='blue', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Sum of Weights')
    plt.title('Trajectory of Sum of Weights Over Time')
    plt.grid(True)
    plt.ylim(0, 2)  
    plt.legend()
    
    if folder is not None:
        filename = os.path.join(folder, "sum_weights.png")
        plt.savefig(filename)
        print("Sum of weights plot saved.")
    plt.close()

    
def _read_csv_simple(path):
    """Load a CSV, fill NaNs by linear interpolation row-wise, return a 2D ndarray."""
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    data = np.atleast_2d(data)
    if np.isnan(data).any():
        x = np.arange(data.shape[1])
        for r in range(data.shape[0]):
            mask = ~np.isnan(data[r])
            if mask.sum() >= 2:
                data[r, ~mask] = np.interp(x[~mask], x[mask], data[r, mask])
    return data


def plot_all_trajs(patient_ids, root_folder, name, alpha=0.8, dist="L2"):
    """
    Plot weight trajectories for each mixture component across multiple patients,
    after aligning components by matching their fitted means.
    Adds mean trajectory (dashed line) and standard deviation band.

    patient_ids : list of patient IDs
    root_folder : base directory containing 'ids={pid}' subfolders
    name        : prefix for saved image files
    alpha       : fraction of trajectories to plot (0 < alpha <= 1)
    dist        : 'L2' or 'L1' distance metric for selection
    """
    

    # 1) Load all results
    results = []
    for pid in patient_ids:
        patient_dir = os.path.join(root_folder, f"ids={pid}")
        if not os.path.isdir(patient_dir):
            continue

        nodes = sorted(d for d in os.listdir(patient_dir) if d.startswith("node_"))
        if not nodes:
            print(f"[WARN] No node_* in {patient_dir}")
            continue
        node_dir = os.path.join(patient_dir, nodes[-1])

        res = load_experiment(node_dir, "result_node.json")
        if not res or "traj_weights" not in res or "means" not in res:
            print(f"[WARN] Invalid result_node.json in {node_dir}")
            continue

        results.append(res)

    if not results:
        raise RuntimeError("No valid results found.")

    # 2) Use first patient's means as reference
    ref_means = np.array(results[0]["means"])
    if ref_means.ndim == 1:
        ref_means = ref_means[:, None]
    K = ref_means.shape[0]

    # 3) Align all trajectories to reference
    aligned_trajs = []
    for res in results:
        means = np.array(res["means"])
        if means.ndim == 1:
            means = means[:, None]

        traj = np.array(res["traj_weights"])
        if traj.ndim == 3 and traj.shape[1] == 1:
            traj = traj[:, 0, :]

        if traj.ndim != 2 or traj.shape[1] != K:
            print(f"[WARN] Skipping reordering for shape mismatch: traj {traj.shape}, expected K={K}")
            aligned_trajs.append(traj)
            continue

        if means.shape[1] == 1:
            order = np.argsort(means.flatten())
        else:
            cost = np.linalg.norm(ref_means[:, None, :] - means[None, :, :], axis=2)
            _, order = linear_sum_assignment(cost)

        aligned_trajs.append(traj[:, order])

    # 4) Prepare time grid
    T = aligned_trajs[0].shape[0]
    t_grid = np.linspace(0, 1.0, T)

    # 5) Determine component colors based on name
    if name == "Treatment":
        # All trajectories in Treatment use color #ad2831
        comp_colors = ["#ad2831"] * K
    elif name == "Control":
        # All trajectories in Control use color #588157
        comp_colors = ["#588157"] * K

    # Plot each component separately
    for k in range(K):
        comp_trajs = np.stack([
            traj[:, k].reshape(-1)
            for traj in aligned_trajs
            if traj.ndim == 2 and traj.shape[1] > k and traj.shape[0] == T
        ])

        if comp_trajs.shape[0] == 0:
            print(f"[WARN] No valid trajectories for component {k}")
            continue

        mean_traj = comp_trajs.mean(axis=0)
        if dist == "L2":
            dists = np.linalg.norm(comp_trajs - mean_traj, axis=1)
        else:
            dists = np.abs(comp_trajs - mean_traj).sum(axis=1)

        m = int(np.ceil(alpha * len(comp_trajs)))
        sel = np.argsort(dists)[:m]
        sel_trajs = comp_trajs[sel]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        # Plot individual trajectories
        for tr in sel_trajs:
            ax.plot(t_grid, tr, lw=1, color=comp_colors[k])
        # Plot mean trajectory (darker shade)
        darker_color = tuple(max(0, c - 0.3) for c in plt.cm.colors.to_rgb(comp_colors[k]))
        ax.plot(t_grid, mean_traj, lw=3, color=darker_color)

        ax.set_xlabel("Time", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_title(f"Component {k+1} ({name})", fontsize=22)
        ax.set_ylim(0.0, 0.8)
        plt.tight_layout()

        out_file = os.path.join(root_folder, f"{name}_comp{k+1}.png")
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved {out_file} ({m}/{len(comp_trajs)} trajectories)")


    


def plot_timesignal(csv_files, out_png="timesignal.png"):
    """
    Plots every row from the given CSV files as a thin, semi-transparent blue line.
    csv_files can be either a list of paths or a single string.
    """
    
    # allow a single string
    if isinstance(csv_files, str):
        csv_files = [csv_files]

    rows = []
    for fp in csv_files:
        if not os.path.isfile(fp):
            print(f"[WARN] file not found: {fp}")
            continue
        rows.extend(_read_csv_simple(fp))

    if not rows:
        raise RuntimeError("No valid data found in the provided CSVs.")

    plt.figure(figsize=(10, 5))
    for row in rows:
        plt.plot(row, color="#1f77b4", alpha=0.25, lw=0.6)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title(f"Time signal", fontsize=22)  # Changed from set_title to title
    plt.xlabel("Number of measurements", fontsize=22)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_boxplots(treatment_ids, control_ids, root_folder, alpha=0.8, dist="L2"):
    """
    Plot paired boxplots of weight differences (before-after) for each component,
    comparing Treatment vs Control in the same figure.

    treatment_ids : list of patient IDs for Treatment group
    control_ids   : list of patient IDs for Control group
    root_folder   : base directory containing 'ids={pid}' subfolders
    alpha, dist   : as before
    
    """
    
    def load_aligned(patient_ids):
        from scipy.optimize import linear_sum_assignment
        # Load and align trajectories same as in plot_all_trajs
        results = []
        for pid in patient_ids:
            patient_dir = os.path.join(root_folder, f"ids={pid}")
            if not os.path.isdir(patient_dir):
                continue
            nodes = sorted(d for d in os.listdir(patient_dir) if d.startswith("node_"))
            if not nodes:
                continue
            node_dir = os.path.join(patient_dir, nodes[-1])
            res = load_experiment(node_dir, "result_node.json")
            if not res or "traj_weights" not in res or "means" not in res:
                continue
            results.append(res)
        if not results:
            raise RuntimeError("No valid results found.")
        ref_means = np.array(results[0]["means"])
        if ref_means.ndim==1:
            ref_means = ref_means[:,None]
        K = ref_means.shape[0]
        aligned = []
        for res in results:
            means = np.array(res["means"])
            if means.ndim==1:
                means = means[:,None]
            traj = np.array(res["traj_weights"])
            if traj.ndim==3 and traj.shape[1]==1:
                traj = traj[:,0,:]
            if traj.ndim!=2 or traj.shape[1]!=K:
                continue
            if means.shape[1]==1:
                order = np.argsort(means.flatten())
            else:
                cost = np.linalg.norm(ref_means[:,None,:]-means[None,:,:],axis=2)
                _,order = linear_sum_assignment(cost)
            aligned.append(traj[:,order])
        return np.stack(aligned)

    # Load aligned trajectories
    traj_t = load_aligned(treatment_ids)
    traj_c = load_aligned(control_ids)
    T, K = traj_t.shape[1], traj_t.shape[2] if traj_t.ndim==3 else (traj_t.shape[1],1)
    # Define colors
    cols = {"Treatment":"#ad2831", "Control":"#588157"}

    for k in range(K):
        # compute differences
        dif_t = traj_t[:,0,k] - traj_t[:,-1,k] if traj_t.ndim==3 else traj_t[:,0] - traj_t[:,-1]
        dif_c = traj_c[:,0,k] - traj_c[:,-1,k] if traj_c.ndim==3 else traj_c[:,0] - traj_c[:,-1]
        # plot
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        data = [dif_t, dif_c]
        positions = [1,2]
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, 
                       medianprops=dict(color='black', linewidth=2))
        # style
        for idx, grp in enumerate(["Treatment","Control"]):
            color = cols[grp]
            bp['boxes'][idx].set_facecolor(color)
            bp['boxes'][idx].set_alpha(0.6)
            for element in ['whiskers','caps','fliers']:
                for obj in bp[element][2*idx:2*idx+2]:
                    obj.set_color(color)
        # labels
        ax.set_xticks(positions)
        ax.set_xticklabels(["Treatment","Control"], fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(-0.5,0.5)
        ax.set_ylabel("", fontsize=0)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f"Weight Difference - Component {k+1}", fontsize=22)
        plt.tight_layout()
        out_file = os.path.join(root_folder, f"boxplot_comp{k+1}.png")
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved {out_file} with combined boxplot for component {k+1}")
        

def plot_quantilecurves(treatment_ids, control_ids, root_folder, quantiles=[0.1,0.25,0.5,0.75,0.9]):
    """
    Plot quantile curves (difference from initial) for each component,
    comparing Treatment vs Control in same figure.

    treatment_ids, control_ids : lists of patient IDs
    root_folder : results directory
    quantiles : list of quantile levels (0-1), default [0.1,0.25,0.5,0.75,0.9]
    """
    
    from scipy.optimize import linear_sum_assignment

    def load_diff(ids, k):
        # load and align trajectories, then return matrix of difference-from-initial (n_patients x T)
        results = []
        for pid in ids:
            pd_dir = os.path.join(root_folder, f"ids={pid}")
            if not os.path.isdir(pd_dir):
                continue
            nodes = sorted(d for d in os.listdir(pd_dir) if d.startswith("node_"))
            if not nodes:
                continue
            res = load_experiment(os.path.join(pd_dir, nodes[-1]), "result_node.json")
            if not res or "traj_weights" not in res or "means" not in res:
                continue
            results.append(res)
        if not results:
            raise RuntimeError("No valid results found.")
        
        # set up reference component means
        ref = np.array(results[0]["means"])
        if ref.ndim == 1:
            ref = ref[:, None]
        K = ref.shape[0]
        
        diffs = []
        for res in results:
            means = np.array(res["means"])
            traj = np.array(res["traj_weights"])
            if traj.ndim == 3 and traj.shape[1] == 1:
                traj = traj[:, 0, :]
            # ensure means has shape (K,1) when needed
            if means.ndim == 1:
                means = means[:, None]
            # compute optimal matching to reference
            if means.shape[1] > 1:
                cost = np.linalg.norm(ref[:, None, :] - means[None, :, :], axis=2)
                _, order = linear_sum_assignment(cost)
                traj = traj[:, order]
            # compute difference from initial time point
            diff = traj[:, k] - traj[0, k]
            diffs.append(diff)
        return np.stack(diffs), K

    # get number of components from treatment group
    sample, K = load_diff(treatment_ids, 0)
    T = sample.shape[1]
    t_grid = np.linspace(0, 1.0, T)
    cols = {"Treatment": "#ad2831", "Control": "#588157"}

    # loop over each component
    for k in range(K):
        dif_t, _ = load_diff(treatment_ids, k)
        dif_c, _ = load_diff(control_ids,   k)
        # compute quantiles across patients at each time point
        q_t = np.quantile(dif_t, quantiles, axis=0)
        q_c = np.quantile(dif_c, quantiles, axis=0)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        for i, q in enumerate(quantiles):
            # plot treatment quantile curves
            ax.plot(
                t_grid, q_t[i],
                lw=3,
                linestyle='-' if q == 0.5 else '--',
                alpha=0.9,
                color=cols['Treatment'],
                label=f"T {int(q*100)}th" if k == 0 else None
            )
            # plot control quantile curves
            ax.plot(
                t_grid, q_c[i],
                lw=3,
                linestyle='-' if q == 0.5 else '--',
                alpha=0.9,
                color=cols['Control'],
                label=f"C {int(q*100)}th" if k == 0 else None
            )

        ax.set_xlabel("Time", fontsize=22)
        ax.tick_params(labelsize=20)
        ax.set_ylim(-0.2, 0.2) 
        ax.set_title(f" Weight Difference - Quantile Curves - Component {k+1}", fontsize=22)
        plt.tight_layout()
        fig.savefig(os.path.join(root_folder, f"quantile_curves_comp{k+1}.png"), dpi=300)
        plt.close(fig)
        print(f"Saved file with quantile curves for component {k+1}")

def plot_gmm_heatmap(means, covs, weights, output_file, cmap, grid_size=400, title="Predicted Density Heatmap"):
    # --- debug prints ---
    x_axis, y_axis = compute_grid(means, means, covs, n=grid_size) 
    X, Y = np.meshgrid(x_axis, y_axis)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    density = np.zeros(grid_points.shape[0], dtype=float)

    for w, mu, cov in zip(weights, means, covs):
        density += w * multivariate_normal.pdf(grid_points, mean=mu, cov=cov)

    Z = density.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(10,5), dpi=300)
    contour = plt.contourf(X, Y, Z, levels=300, cmap=cmap)
    plt.title(title, fontsize=22)
    
    # Formatear colorbar con notación científica y 2 cifras significativas
    from matplotlib.ticker import ScalarFormatter
    cbar = plt.colorbar(contour)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    cbar.formatter = formatter
    cbar.update_ticks()
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

 
def compute_grid(X, means, covariances, n=400):
    """
    Compute a grid for plotting density functions.
    
    For 1D: returns a 1D numpy array.
    For d>=2: returns a tuple of 1D arrays (one per dimension).
    
    Parameters:
        X (np.ndarray): Data array of shape (n_samples, d) or (n_samples,) for 1D.
        means (array-like): Mixture means, shape: (K, d) or (K,) for 1D.
        covariances (array-like): Covariance matrices. For 1D: (K, d); for d>=2: (K, d, d).
        n (int): Number of points per dimension for constructing the grid.
    
    Returns:
        For 1D: a numpy array.
        For d>=2: a tuple of 1D arrays.
    """
    X = np.atleast_2d(X)         # Ensure X is 2D.
    means = np.atleast_2d(means)   # Ensure means is 2D.
    d = X.shape[1]               # Determine the dimensionality.
    
    if d == 1:
        # For 1D, compute min and max using both data and means, and add a margin.
        x_min = min(X[:, 0].min(), means[:, 0].min())
        x_max = max(X[:, 0].max(), means[:, 0].max())
        margin = 0.1 * (x_max - x_min)
        return np.linspace(x_min - margin, x_max + margin, n)
    else:
        grid_list = []
        cov = np.array(covariances)
        for j in range(d):
            col_data = X[:, j]
            col_means = means[:, j]
            # If covariance is 3-dimensional, extract the j-th diagonal element.
            if cov.ndim == 3:
                col_cov = cov[:, j, j]
            else:
                col_cov = cov[:, j]
            # Define lower and upper bounds for the grid in this dimension.
            lower = min(col_data.min(), col_means.min()) - 3 * np.sqrt(col_cov.max())
            upper = max(col_data.max(), col_means.max()) + 3 * np.sqrt(col_cov.max())
            points = np.linspace(lower, upper, n)
            grid_list.append(points)
        return tuple(grid_list)

def plot_bands(result, Xs, results_folder, combo):
    """Generate uncertainty band plots for all time slices."""
    
    aligned = order_results(result, combo.get("n_bags", 0))
    band_dir = os.path.join(results_folder, "Bands")
    os.makedirs(band_dir, exist_ok=True)
    
    # Extract keys for all bags
    weight_keys = sorted(k for k in aligned if k.startswith("best_weights"))
    mean_keys = sorted(k for k in aligned if k.startswith("best_means"))
    cov_keys = sorted(k for k in aligned if k.startswith("best_covariances"))
    
    dist = combo.get("dist", "L2")
    alpha = combo.get("alpha_uq", 0.8)
    quantiles = [0.25, 0.5, 0.75]
    
    # Create initial vs final comparison plot (done once)
    if Xs:
        x0, x1 = Xs[0][:, 0].min(), Xs[0][:, 0].max()
        margin = 0.1 * (x1 - x0)
        x_grid = np.linspace(x0 - margin, x1 + margin, 1000)
        
        ts_final = np.array(aligned[weight_keys[0]]).shape[0] - 1
        
        # Compute densities for initial and final
        D0, D1 = [], []
        for w_key, m_key, c_key in zip(weight_keys, mean_keys, cov_keys):
            W = np.array(aligned[w_key])
            M = np.atleast_2d(aligned[m_key]).astype(float)
            C = np.array(aligned[c_key], dtype=float)
            
            D0.append(compute_gmm_density(x_grid, W[0], M, C))
            D1.append(compute_gmm_density(x_grid, W[ts_final], M, C))
        
        D0, D1 = np.vstack(D0), np.vstack(D1)
        
        # Plot initial vs final
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        colors = {"Initial": "#f72585", "Final": "#780116"}
        
        for label, D in [("Initial", D0), ("Final", D1)]:
            center = D.mean(0)
            lower, upper = D.min(0), D.max(0)
            ax.plot(x_grid, center, color=colors[label], lw=1.5)
            ax.fill_between(x_grid, lower, upper, color=colors[label], alpha=0.5)
        
        ax.set_xlabel('Glucose concentration, mg/dL', fontsize=22)
        ax.set_title("Predicted Density", fontsize=24, pad=20)
        ax.tick_params(labelsize=20)
        
        handles = [Patch(color=col, label=label) for label, col in colors.items()]
        ax.legend(handles=handles, fontsize=14, frameon=False)
        
        fig.tight_layout()
        fig.savefig(os.path.join(band_dir, 'Bands_InitFinalx.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create individual time slice plots
    for ts, X in enumerate(Xs):
        x0, x1 = X[:, 0].min(), X[:, 0].max()
        margin = 0.1 * (x1 - x0)
        x_grid = np.linspace(x0 - margin, x1 + margin, 1000)
        
        # Compute densities at this time step
        densities = []
        for w_key, m_key, c_key in zip(weight_keys, mean_keys, cov_keys):
            W = np.array(aligned[w_key])
            w_ts = W[ts]
            M = np.atleast_2d(aligned[m_key]).astype(float)
            C = np.array(aligned[c_key], dtype=float)
            densities.append(compute_gmm_density(x_grid, w_ts, M, C))
        
        D = np.vstack(densities)
        center = D.mean(0)
        
        # Compute confidence bands
        dists = (np.linalg.norm(D - center, axis=1) if dist=="L2" 
                else np.sum(np.abs(D - center), axis=1))
        m = D.shape[0]
        keep = np.argsort(dists)[:int(np.ceil(alpha * m))]
        sel = D[keep]
        lower, upper = sel.min(0), sel.max(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.plot(x_grid, center, color='navy', lw=0.5, label='Mean Density')
        ax.fill_between(x_grid, lower, upper, color='skyblue', alpha=0.6,
                        label=f'Bands ({len(keep)}/{m})')
        
        if quantiles:
            cmap = plt.get_cmap("viridis")
            for i, q in enumerate(sorted(quantiles)):
                pct = np.percentile(sel, q * 100, axis=0)
                clr = cmap(i/(len(quantiles)-1)) if len(quantiles)>1 else 'red'
                ax.plot(x_grid, pct, '--', lw=2, label=f'{int(q*100)}th pct', color=clr)
        
        ax.tick_params(labelsize=20)
        ax.set_title(f"Density at t={ts}", fontsize=22, pad=15)
        ax.legend(fontsize=14)
        fig.tight_layout()
        
        fig.savefig(os.path.join(band_dir, f'ConfidenceBands_{ts}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print("Uncertainty bands plotted.")

def plot_model_vs_gt(ws_ts, mus, covs, mus_gt_ts, vars_gt_ts, t_i, folder, combo, n_points=1000):
    """Generate 1D comparative plots between ground truth and GMM model across time points."""
    d = combo["d"]
    
    if d != 1:
        print(f"Skipping visualization for d={d} (only 1D supported)")
        return
    
    fig = plt.figure(figsize=(15, 10))
    n_rows, n_cols = 3, 4
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for i, t in enumerate(t_i):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        
        current_mus_gt = mus_gt_ts[i]
        current_var_gt = vars_gt_ts[i]
        current_weights = ws_ts[i]
        
        # Determine plot range
        all_means = np.concatenate([np.atleast_2d(current_mus_gt), np.atleast_2d(mus)])
        min_mean, max_mean = np.min(all_means), np.max(all_means)
        max_std = max(np.sqrt(current_var_gt), np.sqrt(np.max([np.max(cov) for cov in covs])))
        x_min, x_max = min_mean - 4 * max_std, max_mean + 4 * max_std
        
        x_points = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
        
        # Compute and plot densities
        gt_density = compute_gt_density_mc(x_points, current_mus_gt, current_var_gt, d)
        gmm_density = compute_gmm_density_mc(x_points, current_weights, mus, covs)
        
        ax.plot(x_points, gt_density, 'r-', linewidth=2, label='Ground Truth')
        ax.plot(x_points, gmm_density, 'b--', linewidth=2, label='GMM Model')
        
        # Plot individual components
        gt_weights = np.ones(len(current_mus_gt)) / len(current_mus_gt)
        for j, mu in enumerate(current_mus_gt):
            component_density = compute_gmm_density_mc(
                x_points, [1.0], [mu], [np.eye(d) * current_var_gt]) * gt_weights[j]
            ax.plot(x_points, component_density, 'r:', alpha=0.5)
        
        for j, (w, mu, cov) in enumerate(zip(current_weights, mus, covs)):
            component_density = compute_gmm_density_mc(x_points, [1.0], [mu], [cov]) * w
            ax.plot(x_points, component_density, 'b:', alpha=0.5)
        
        ax.set_title(f't = {t:.2f}')
        if i >= (n_rows-1) * n_cols:
            ax.set_xlabel('x')
        if i % n_cols == 0:
            ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
    
    fig.suptitle('Ground Truth vs GMM Model Comparison (1D)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(folder, "model_vs_ground_truth_1d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate time evolution summary
    _plot_time_evolution(ws_ts, mus, mus_gt_ts, t_i, folder)
    print(f"1D plots saved to {folder}")

def _plot_time_evolution(ws_ts, mus, mus_gt_ts, t_i, folder):
    """Generate time evolution summary plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine number of ground truth clusters from first time step
    n_clusters_gt = len(mus_gt_ts[0]) if isinstance(mus_gt_ts[0], list) else 1
    
    # Plot ground truth means evolution (first dimension)
    for cluster_idx in range(n_clusters_gt):
        cluster_means = []
        for t_idx in range(len(t_i)):  # Iterate only over actual time indices
            means_at_t = mus_gt_ts[t_idx]
            # Extract the mean for this cluster at this time
            if isinstance(means_at_t[cluster_idx], list):
                # If it's [[m1]*d, [m2]*d, [m3]*d] structure
                cluster_means.append(means_at_t[cluster_idx][0])
            else:
                # If it's a scalar
                cluster_means.append(means_at_t[cluster_idx])
        
        ax.plot(t_i, cluster_means, 'r-o', 
               label=f'GT Cluster {cluster_idx+1}' if cluster_idx == 0 else None)
    
    # Plot model means as horizontal lines
    for k, mu in enumerate(mus):
        # Handle the case where mu might be a list/array or a scalar
        mu_val = mu[0] if hasattr(mu, '__len__') else mu
        ax.axhline(y=mu_val, color='b', linestyle='--', 
                  label=f'Model Component {k+1}' if k == 0 else None)
    
    # Plot weights as scatter points
    max_weight = max([max(w) for w in ws_ts])
    for k, mu in enumerate(mus):
        # Handle the case where mu might be a list/array or a scalar
        mu_val = mu[0] if hasattr(mu, '__len__') else mu
        weight_sizes = [500 * w[k] / max_weight for w in ws_ts]
        
        ax.scatter(t_i, [mu_val] * len(t_i), s=weight_sizes, 
                  alpha=0.5, color='blue', edgecolors='black')
    
    ax.legend(loc='upper left')
    ax.set_xlabel('Time')
    ax.set_ylabel('First Dimension Mean Value')
    ax.set_title('Evolution of Means and Weights Over Time')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(folder, "time_evolution_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()