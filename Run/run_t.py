import torch
from Utils.IO_utils import load_experiment, save_json

def run_test(combo, best_means, best_covs, ts_weights, pred_traj,
             node_folder, Xs, options):
    """
    Evaluate model performance using L2 error metrics.
    
    For synthetic data: Compare against ground truth using multiple L2 methods.
    For real data: Use empirical L2 and MMD metrics.
    """
    
    # Check for cached results
    existing_result = load_experiment(node_folder, "result_error.json")
    if existing_result is not None and options.get("load_cached_test", False):
        return existing_result
    
    # Setup time grids
    t_i = torch.linspace(0.0, combo["T"], combo["t_steps"])
    t_grid = torch.linspace(0.0, combo["T"], int(combo["T"] / combo["dt"]) + 1)
    
    results = {}
    
    if not options.get("load_data", False):
        # Synthetic data path
        results = _process_synthetic_data(combo, best_means, best_covs, ts_weights, pred_traj,
                                        t_grid, t_i, node_folder)
    else:
        # Real data path
        results = _process_real_data(best_means, best_covs, ts_weights, pred_traj,
                                   t_grid, t_i, node_folder, combo, Xs)
    
    # Save results
    save_json(data=results, folder=node_folder, filename='result_error.json')
    
    return results


def _process_synthetic_data(combo, best_means, best_covs, ts_weights, pred_traj,
                           t_grid, t_i, node_folder):
    """Process synthetic data with ground truth comparison."""
    from data import get_distrparams
    from Core.test_mmd import testL2, testL2_adaptive, testL2_importance, testL2_analytical
    from Visualization.plots import plot_model_vs_gt
    
    # Get ground truth parameters
    means_gt, vars_gt = get_distrparams(combo, combo["T"], len(t_grid))
    means_gtts, vars_gtts = get_distrparams(combo, combo["T"], combo["t_steps"])
    
    # Generate visualization (only for 1D)
    print("Generating visualization of model vs ground truth...")
    plot_model_vs_gt(ts_weights, best_means, best_covs, means_gtts, vars_gtts, 
                    t_i.cpu().numpy(), node_folder, combo)
    
    # Define test methods with their labels
    test_methods = [
        ("Standard Monte Carlo", testL2, "standard"),
        ("Adaptive Integration", testL2_adaptive, "adaptive"),
        ("Importance Sampling", testL2_importance, "importance"),
        ("Analytical", testL2_analytical, "analytical")
    ]
    
    results = {}
    method_results = {}
    
    # Run all test methods
    for method_name, test_func, key_prefix in test_methods:
        print(f"Computing L2 errors with {method_name.lower()} method...")
        
        result_tuple = test_func(
            pred_traj, ts_weights, best_means, best_covs,
            means_gt, vars_gt, means_gtts, vars_gtts,
            t_grid=t_grid, t_i=t_i, folder=node_folder, combo=combo
        )
        
        # Unpack results consistently
        (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts) = result_tuple
        
        # Store results with prefixed keys
        prefix = f"{key_prefix}_" if key_prefix != "standard" else ""
        results[f"{prefix}histL2"] = L2errs
        results[f"{prefix}histL2fit"] = L2errs_ts
        results[f"{prefix}avgL2"] = avgL2
        results[f"{prefix}stdL2"] = stdL2
        results[f"{prefix}avgL2_ts"] = avgL2_ts
        results[f"{prefix}stdL2_ts"] = stdL2_ts
        
        # Store for comparison table
        method_results[method_name] = {"L2_avg": avgL2, "L2_std": stdL2}
    
    # Create comparison table and relative differences
    results["comparison"] = method_results
    _compute_relative_differences(results, method_results)
    
    # Save comparison table
    _save_comparison_table(method_results, results, node_folder)
    
    return results


def _process_real_data(best_means, best_covs, ts_weights, pred_traj,
                      t_grid, t_i, node_folder, combo, Xs):
    """Process real data with empirical metrics."""
    from Core.test_mmd import testL2MMDemp
    
    print("Computing empirical L2 and MMD errors...")
    
    result_tuple = testL2MMDemp(
        pred_traj, ts_weights, best_means, best_covs, data=Xs,
        t_grid=t_grid.cpu().numpy(), t_i=t_i.cpu().numpy(),
        folder=node_folder, combo=combo
    )
    
    # Unpack empirical results
    (histL2, histMMD, histL2fit, histMMDfit,
     avgL2, stdL2, avgMMD, stdMMD,
     avgL2_ts, stdL2_ts, avgMMD_ts, stdMMD_ts) = result_tuple
    
    return {
        "histL2": histL2, "histMMD": histMMD,
        "histL2fit": histL2fit, "histMMDfit": histMMDfit,
        "avgL2": avgL2, "stdL2": stdL2,
        "avgMMD": avgMMD, "stdMMD": stdMMD,
        "avgL2_ts": avgL2_ts, "stdL2_ts": stdL2_ts,
        "avgMMD_ts": avgMMD_ts, "stdMMD_ts": stdMMD_ts
    }


def _compute_relative_differences(results, method_results):
    """Compute relative differences using analytical method as baseline."""
    if "Analytical" in method_results and method_results["Analytical"]["L2_avg"] is not None:
        baseline = method_results["Analytical"]["L2_avg"]
        
        # Compute relative differences for each method
        for method_name, method_data in method_results.items():
            if method_name != "Analytical":
                key_name = method_name.lower().replace(" ", "_")
                rel_diff = (method_data["L2_avg"] - baseline) / baseline * 100
                results[f"rel_diff_{key_name}"] = rel_diff

def _save_comparison_table(method_results, results, node_folder):
    """Save comparison table for synthetic data methods."""
    comparison_table = {
        "Method": list(method_results.keys()),
        "L2_avg": [data["L2_avg"] for data in method_results.values()],
        "L2_std": [data["L2_std"] for data in method_results.values()]
    }
    
    # Add relative differences if available
    rel_diff_keys = [key for key in results.keys() if key.startswith("rel_diff_")]
    if rel_diff_keys:
        rel_diffs = []
        for method_name in method_results.keys():
            if method_name == "Analytical":
                rel_diffs.append(0.0)  # Baseline
            else:
                key_name = method_name.lower().replace(" ", "_")
                rel_diff_key = f"rel_diff_{key_name}"
                rel_diffs.append(results.get(rel_diff_key, None))
        
        comparison_table["L2_rel_diff"] = rel_diffs
    
    save_json(data=comparison_table, folder=node_folder, filename='error_comparison.json')