import numpy as np
import os
import matplotlib.pyplot as plt
from data import compute_gmm_density


def compute_bag_densities(params, x_points, plot=True, results_folder=None, ts=0):
    """
    Compute GMM densities for each bag at specified points.
    
    Args:
        params: dict containing keys 'best_weights_bagi', 'best_meansi', 'best_covariancesi'
        x_points: 1D array of evaluation points
        plot: whether to plot each bag density
        results_folder: path to save the plot if plot=True
    
    Returns:
        densities: array of shape (n_bags, len(x_points))
    """
    weight_keys = sorted([k for k in params if k.startswith('best_weights')])
    mean_keys = sorted([k for k in params if k.startswith('best_means')])
    cov_keys = sorted([k for k in params if k.startswith('best_covariances')])
    
    dens_list = []
    for w_k, mu_k, cov_k in zip(weight_keys, mean_keys, cov_keys):
        w_arr = np.ravel(params[w_k]).astype(float)
        mu_arr = np.atleast_2d(params[mu_k]).astype(float)
        if mu_arr.shape[1] == 1 and w_arr.size > 1:
            mu_arr = mu_arr.reshape(-1, 1)
        cov_arr = np.atleast_2d(params[cov_k]).astype(float)
        if cov_arr.ndim == 2 and cov_arr.shape[1] == 1 and w_arr.size > 1:
            cov_arr = cov_arr.reshape(-1, 1)
        dens_list.append(compute_gmm_density(x_points, w_arr, mu_arr, cov_arr))
    
    densities = np.vstack(dens_list)
    
    if plot and results_folder:
        plt.figure(figsize=(10, 5), dpi=300)
        for i, density in enumerate(densities):
            plt.plot(x_points, density, alpha=0.3, label=f'Bag {i+1}')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.savefig(os.path.join(results_folder, 'BagDensities.png'))
        plt.close()
    
    return densities


def sample_points(x_grid, n=1000):
    """Uniformly sample n points over the range of x_grid."""
    return np.sort(np.random.uniform(x_grid[0], x_grid[-1], size=n))


def get_confidence_bands(params, x_grid, dist, alpha, n_MC=1000):
    """
    Generate Monte Carlo points, central density, pointwise confidence bands and selected bag densities.
    
    Args:
        params: bag parameters as in compute_bag_densities
        x_grid: 1D domain array
        dist: 'L2' or 'L1'
        alpha: fraction of bags to include in band
        n_MC: Monte Carlo sample size
    
    Returns:
        x_MC, center, lower, upper, selection
    """
    x_MC = sample_points(x_grid, n_MC)
    dens = compute_bag_densities(params, x_MC, plot=False)
    center = dens.mean(axis=0)
    V = x_grid[-1] - x_grid[0]
    
    if dist == 'L2':
        dists = np.sqrt(V / n_MC * np.sum((dens - center)**2, axis=1))
    else:
        dists = V / n_MC * np.sum(np.abs(dens - center), axis=1)
    
    k = int(np.ceil(alpha * dens.shape[0]))
    idx = np.argsort(dists)[:k]
    selection = dens[idx]
    lower = selection.min(axis=0)
    upper = selection.max(axis=0)
    
    return x_MC, center, lower, upper, selection