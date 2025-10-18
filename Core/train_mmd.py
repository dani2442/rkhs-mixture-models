import numpy as np
import time
import torch
from Utils.MMD_utils import (init_params, grad_update, compute_mmd_closed_form, update_local_weights)


def mmd_gmm_fit(X, K, gamma=1.0, max_iter=100, tol_rel_param=1e-6, tol_rel_err=1e-5, tol_abs=1e-4, 
                lr=0.01, grad_steps=20, verbose=False, 
                init_weights=None, init_means=None, init_covs=None, 
                fixed_means=None, fixed_covs=None, lambda_ridge=0.0):
    """
    Fit a Gaussian Mixture Model using an MMD-based objective.
    
    X is a list of datasets (one per time instant) and global fitting 
    with local weight estimation is performed using an alternating optimization strategy:
      1. Update local weights for each dataset via quadratic programming.
      2. a) Update global parameters (means and covariances) using weighted moment matching
         b) Update global parameters using gradient descent 
    
    Returns:
      Tuple containing histories of parameters, best iteration index, total fitting time, device info, and iteration count.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] Using device: {device}")
    
    start = time.time()
    
    # Global initialization: concatenate all datasets.
    X_all = np.concatenate(X, axis=0)
    _, d = X_all.shape
    
    #Initialize global means and covariances.
    local_weights, means, covs = init_params(X_all, K, init_weights, init_means, init_covs, len(X))
    freeze_mu  = fixed_means is not None
    freeze_cov = fixed_covs  is not None
    if freeze_mu:   means = np.asarray(fixed_means, dtype=float).copy()
    if freeze_cov:  covs  = np.asarray(fixed_covs , dtype=float).copy()

    
    # Initialize history lists.
    hist_means, hist_covs, hist_errmeans, hist_errcovs, hist_errweights, mmd_list = [], [], [], [], [], []             
    history_ws = []  # History of local weights per dataset.
    best_index, best_mmd, prev_mmd = -1, np.inf, None
    
    if verbose:
        from tqdm import tqdm
        iter_range = tqdm(range(max_iter), desc="Iterations")
    else:
        iter_range = range(max_iter)
        
    Gamma = []
    
    # Main iteration loop.
    for it in iter_range:
        # Store old parameters for convergence checks.
        means_old, covs_old, local_weights_old = means.copy(), covs.copy(), [lw.copy() for lw in local_weights]

        # Step 1: Update local weights for each dataset  
        for i, X_i in enumerate(X):
                if isinstance(gamma, (list, tuple, np.ndarray)):
                    Gamma.append(gamma[i]**2*np.eye(d))
                    local_weights[i] = update_local_weights(X_i, means, covs, Gamma[i], lambda_ridge, local_weights[i])
                else: 
                    Gamma = (gamma**2) * np.eye(d)
                    local_weights[i] = update_local_weights(X_i, means, covs, Gamma, lambda_ridge, local_weights[i])
        
        # Step 2: Update global means and covariances using gradient descent.
        means, covs = grad_update(X, local_weights, means, covs, np.mean(gamma), grad_steps, lr, device)

        # Compute parameter changes.
        diff_means = 0.0 if freeze_mu  else np.linalg.norm(means - means_old)
        diff_cov = 0.0 if freeze_cov else np.linalg.norm(covs  - covs_old)
        diff_w = [np.linalg.norm(lw - lw_old) for lw, lw_old in zip(local_weights, local_weights_old)]

        # Compute total squared MMD error over all series.
        total_sq_mmd = 0.0
        for i, X_i in enumerate(X):
            total_sq_mmd += compute_mmd_closed_form(X_i, local_weights[i], means, covs, gamma[i])
        
        
        # Store errors in history.
        hist_errmeans.append(diff_means), hist_errcovs.append(diff_cov), hist_errweights.append(diff_w), mmd_list.append(total_sq_mmd)
        
        if total_sq_mmd < best_mmd:
            best_mmd = total_sq_mmd
            best_index = it
        if verbose:
            print(f"Iter {it+1}: Δmeans={diff_means:.6f}, Δcovs={diff_cov:.6f}, (avg) Δweights={np.mean(diff_w):.6f}, (Global) sq MMD={total_sq_mmd:.6f}")
        hist_means.append(means.copy()), hist_covs.append(covs.copy()), history_ws.append(local_weights.copy())
        
        
        # Check convergence.
        if diff_means < tol_rel_param or diff_cov < tol_rel_param or (np.mean(diff_w) < tol_rel_param if np.mean(diff_w)>0 else False):
            if verbose:
                print(f"[INFO] Convergence reached at iteration {it} because of relative parameter error.")
            break
        if prev_mmd is not None and abs(total_sq_mmd - prev_mmd) < tol_rel_err:
            if verbose:
                print(f"[INFO] Convergence reached at iteration {it} because of relative MMD^2 error.")
            break
        if total_sq_mmd < tol_abs:
            if verbose:
                print(f"[INFO] Convergence reached at iteration {it} because of absolute MMD^2 error.")
            break

        prev_mmd = total_sq_mmd

    fit_time = time.time() - start
    if verbose:
        print(f"[INFO] Fitting time: {fit_time:.4f} sec")
    
    return (best_index, history_ws, hist_means, hist_covs,
            hist_errweights, hist_errmeans, hist_errcovs, mmd_list,
             fit_time, device, it)