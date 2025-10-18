import numpy as np
import os
from scipy.stats import gaussian_kde
from data import generate_data
from Utils.IO_utils import save_json
from Utils.UQ_utils import get_confidence_bands

### ---------------------- Function to run the confidence bands ------------------- ###

import numpy as np
import os
from scipy.stats import gaussian_kde
from data import generate_data
from Utils.IO_utils import save_json
from Utils.UQ_utils import get_confidence_bands


def run_bands(result, combo, folder, options, seed_sim, factor_size):
    """Study the performance of static confidence bands with the model."""
    
    combo["n_points"] = combo["n_points"] * factor_size
    print(f"Studying performance of {combo['alpha_uq']}-confidence bands for "
          f"{combo['m_sim']} new densities with {combo['n_points']} samples...")
    
    # Get parameters
    dist = combo.get("dist", "L2")
    alpha = combo.get("alpha_uq", 0.8)
    m_sim = combo.get("m_sim", 100)
    x_grid = np.linspace(-10, 10, 1000)
    n_MC = 1000
    
    # Get confidence bands
    x_MC, center, lower, upper, _ = get_confidence_bands(result, x_grid, dist, alpha, n_MC)
    V = x_grid[-1] - x_grid[0]
    
    # Generate simulated densities
    sim_list = []
    for i in range(m_sim):
        data = generate_data(combo['distribution'], combo, seed_sim + i)
        pts = data[:, 0] if data.ndim > 1 else data
        kde = gaussian_kde(pts)
        sim_list.append(kde(x_MC))
    sim = np.vstack(sim_list)
    
    # Compute distances and coverage statistics
    if dist == 'L2':
        dists = np.sqrt(V / n_MC * np.sum((sim - center)**2, axis=1))
    else:
        dists = V / n_MC * np.sum(np.abs(sim - center), axis=1)
    
    threshold = np.max(dists[np.argsort(dists)[:int(np.ceil(alpha * sim.shape[0]))]])
    pct_inside = np.mean(dists <= threshold) * 100
    pct_full = np.mean(np.all((sim >= lower) & (sim <= upper), axis=1)) * 100
    
    result_stats = {
        'percentage_inside': pct_inside,
        'percentage_full_inside': pct_full
    }
    
    os.makedirs(folder, exist_ok=True)
    save_json(result_stats, folder, 'model_vs_bands.json')
    
    return result_stats