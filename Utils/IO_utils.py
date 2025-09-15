"""
IO utilities and experiment management for machine learning experiments.

Provides functionality for parameter validation, folder management, result caching,
and data serialization with support for numpy types and bagging experiments.
"""

import json
import os
import glob
import math
from typing import Dict, Any, Iterable, Tuple, Optional, List
import numpy as np
import pandas as pd


# ==============================================================================
# JSON Encoding for NumPy Types
# ==============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types and ranges."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, range):
            return list(obj)
        return super().default(obj)


# ==============================================================================
# Parameter Validation
# ==============================================================================

def validate_params(params_data: Dict[str, Any], params_uq: Dict[str, Any], 
                   options: Dict[str, Any]) -> None:
    """
    Validate experiment parameters and update options accordingly.
    
    Args:
        params_data: Data generation parameters
        params_uq: Uncertainty quantification parameters  
        options: Experiment options
        
    Raises:
        ValueError: If parameters are invalid
    """
    load_data = options.get("load_data", False)
    
    # Handle bagging and mean-variance constraints
    if params_uq["n_bags"] > 1:
        options["fix_mv"] = False
    if options.get("fix_mv", False):
        params_uq["n_bags"] = 1
        
    # Configure dimensionality for loaded data based on derivatives
    if load_data:
        if params_data.get("add_deriv", False) and params_data.get("add_second_deriv", False):
            params_data["d"] = 3
        elif params_data.get("add_deriv", False) or params_data.get("add_second_deriv", False):
            params_data["d"] = 2
        else:
            params_data["d"] = 1
    else:
        options["fix_mv"] = False
        
        # Validate n_points parameter for synthetic data
        n_points = params_data.get("n_points")
        if not isinstance(n_points, list) or not all(isinstance(x, int) for x in n_points):
            raise ValueError("n_points must be a list of integers")
        
        # Ensure sufficient points for clustering
        n_clusters = params_data.get("n_clusters", 1)
        for n_pts in n_points:
            if n_pts < n_clusters:
                raise ValueError(f"Invalid combination: n_points ({n_pts}) < n_clusters ({n_clusters})")


# ==============================================================================
# Folder Management and Caching
# ==============================================================================

def set_base_folder(options: Dict[str, Any], params_data: Dict[str, Any], 
                   params_optim: Dict[str, Any], params_uq: Dict[str, Any],
                   params_node: Dict[str, Any], *, 
                   ignore_keys: Optional[Iterable[str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create or reuse directory structure based on experiment parameters.
    
    Implements intelligent caching by comparing parameter signatures and reusing
    existing runs when parameters match.
    
    Args:
        options: Experiment options
        params_data: Data parameters
        params_optim: Optimization parameters
        params_uq: Uncertainty quantification parameters
        params_node: Neural ODE parameters
        ignore_keys: Parameters to exclude from signature matching
        
    Returns:
        Updated options and params_data dictionaries
    """
    ignore_keys = set(ignore_keys or [])

    # Determine parent directory based on data type
    parent_dir = "Results_MedicalData" if options.get("load_data") else "Results_TimeSeries"
    os.makedirs(parent_dir, exist_ok=True)

    # Merge and categorize parameters
    merged_params = {**options, **params_data, **params_optim, **params_uq, **params_node}
    fixed_params, grid_params = _split_parameters(merged_params)

    # Create signature for matching (exclude ignored keys)
    signature = {k: v for k, v in fixed_params.items() if k not in ignore_keys}

    # Find or create run folder
    run_folder = _find_or_create_run_folder(parent_dir, signature, fixed_params)

    # Handle CSV subfolder for medical data without grid sweep
    if options.get("load_data") and not grid_params:
        run_folder = _create_csv_subfolder(run_folder, params_data)

    # Update options with folder information
    options.update({
        "parent_folder": os.path.abspath(parent_dir),
        "base_folder": os.path.abspath(run_folder),
        "fixed_params": fixed_params,
        "grid_params": grid_params
    })
    
    return options, params_data


def _split_parameters(merged_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split parameters into fixed (scalar) and grid (sweep) categories."""
    fixed_params = {}
    grid_params = {}
    
    for key, value in merged_params.items():
        if isinstance(value, list) and len(value) > 1:
            grid_params[key] = value
        else:
            fixed_params[key] = value[0] if isinstance(value, list) else value
    
    return fixed_params, grid_params


def _find_or_create_run_folder(parent_dir: str, signature: Dict[str, Any], 
                              fixed_params: Dict[str, Any]) -> str:
    """Find existing run folder or create new one based on signature matching."""
    # Search for existing matching run
    for folder_name in os.listdir(parent_dir):
        if not folder_name.startswith("run"):
            continue
            
        fixed_params_path = os.path.join(parent_dir, folder_name, "fixed_params.json")
        if not os.path.isfile(fixed_params_path):
            continue
            
        try:
            with open(fixed_params_path) as f:
                cached_params = json.load(f)
            
            # Check if cached parameters contain all signature items
            if all(cached_params.get(k) == v for k, v in signature.items()):
                run_folder = os.path.join(parent_dir, folder_name)
                print(f"[INFO] Using cached run folder → {run_folder}")
                return run_folder
        except Exception:
            continue
    
    # Create new run folder
    run_index = 1
    while os.path.exists(os.path.join(parent_dir, f"run{run_index}")):
        run_index += 1
    
    run_folder = os.path.join(parent_dir, f"run{run_index}")
    os.makedirs(run_folder, exist_ok=True)
    
    with open(os.path.join(run_folder, "fixed_params.json"), "w") as f:
        json.dump(fixed_params, f, indent=2, sort_keys=True)
    
    print(f"[INFO] Created new run folder → {run_folder}")
    return run_folder


def _create_csv_subfolder(run_folder: str, params_data: Dict[str, Any]) -> str:
    """Create CSV-specific subfolder for medical data experiments."""
    patient_ids = params_data.get("ids", [])
    ids_str = "-".join(map(str, patient_ids))
    block_size = params_data.get("block_size", "unknown")
    
    csv_folder = os.path.join(run_folder, f"Id_{ids_str}_blocksize_{block_size}")
    os.makedirs(csv_folder, exist_ok=True)
    return csv_folder


def create_folder(options: Dict[str, Any], params: Dict[str, Any]) -> str:
    """
    Create experiment-specific folder based on varying parameters.
    
    Args:
        options: Global experiment options containing grid_params
        params: Current experiment parameters (scalars)
        
    Returns:
        Full path to experiment directory
    """
    folder_parts = []
    
    # Include only parameters that vary across experiments
    for key, grid_values in options["grid_params"].items():
        if isinstance(grid_values, list) and len(grid_values) > 1:
            current_value = params.get(key)
            folder_parts.append(f"{key}={current_value}")
    
    folder_name = "_".join(folder_parts) if folder_parts else "default"
    return os.path.join(options["base_folder"], folder_name)


# ==============================================================================
# Node Folder Management
# ==============================================================================

def check_nodefolder(combo_folder: str, combo: Dict[str, Any], 
                    options: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Find or create node folder with parameter matching and caching support.
    
    Args:
        combo_folder: Parent folder for node experiments
        combo: Current experiment parameters
        options: Experiment options
        
    Returns:
        Tuple of (node_folder_path, cached_results_or_None)
    """
    # Find existing node folders
    existing_folders = glob.glob(os.path.join(combo_folder, "node_*"))
    existing_folders = sorted(existing_folders, 
                            key=lambda f: int(os.path.basename(f).split('_')[-1]))
    
    # Search for matching parameters
    for folder in existing_folders:
        json_path = os.path.join(folder, "result_node.json")
        if not os.path.exists(json_path):
            continue
        
        try:
            with open(json_path, 'r') as f:
                cached_params = json.load(f)
            
            if _parameters_match(cached_params, combo):
                if options.get("load_cached_node", False):
                    print("Loading cached node results")
                    return folder, cached_params
                else:
                    return folder, None
        except (json.JSONDecodeError, KeyError):
            continue
    
    # Create new node folder
    next_index = len(existing_folders) + 1
    new_folder = os.path.join(combo_folder, f"node_{next_index}")
    os.makedirs(new_folder, exist_ok=True)
    
    return new_folder, None


def _parameters_match(cached_params: Dict[str, Any], new_params: Dict[str, Any]) -> bool:
    """Check if parameter sets are equivalent within numerical tolerance."""
    for key, new_value in new_params.items():
        old_value = cached_params.get(key)
        
        # Handle numpy arrays
        if isinstance(new_value, np.ndarray) or isinstance(old_value, np.ndarray):
            try:
                if not np.allclose(new_value, old_value, rtol=1e-8, atol=1e-10):
                    return False
            except (TypeError, ValueError):
                return False
        
        # Handle floating point numbers
        elif isinstance(new_value, (float, np.floating)) and isinstance(old_value, (float, np.floating)):
            if not math.isclose(float(new_value), float(old_value), rel_tol=1e-8, abs_tol=1e-10):
                return False
        
        # Handle other types with exact comparison
        elif new_value != old_value:
            return False
    
    return True


# ==============================================================================
# Result Management and Bagging
# ==============================================================================

def save_fit(folder: str, bag_results: Dict[str, List], combo: Dict[str, Any], 
            options: Dict[str, Any], device: str) -> Dict[str, Any]:
    """
    Aggregate bagging results, align GMM components, and save to JSON.
    
    Args:
        folder: Output directory
        bag_results: Dictionary of lists containing results from each bag
        combo: Experiment parameters
        options: Experiment options
        device: Computation device used
        
    Returns:
        Aggregated results dictionary
    """
    # Convert dict-of-lists to list-of-dicts (one per bag)
    n_bags = len(bag_results.get("seed", []))
    bags = []
    
    for i in range(n_bags):
        bag = {}
        for key, values in bag_results.items():
            bag[key] = values[i] if i < len(values) else None
        bags.append(bag)

    # Extract valid values for averaging
    valid_values = _extract_valid_values(bags)
    
    # Build aggregated result
    aggregated = _build_aggregated_result(combo, options, valid_values, device)
    
    # Add individual bag results
    for i, bag in enumerate(bags):
        for key, value in bag.items():
            aggregated[f"{key} (bag {i+1})"] = value

    # Align GMM components across bags and update averages
    aggregated = order_results(aggregated, n_bags)

    # Save results
    os.makedirs(folder, exist_ok=True)
    save_json(data=aggregated, folder=folder, filename="result_fit.json")
    
    return aggregated


def _extract_valid_values(bags: List[Dict[str, Any]]) -> Dict[str, List]:
    """Extract non-None values from bags for averaging."""
    keys = ["min_training_error", "best_index", "best_weights", "best_means", 
            "best_covariances", "fit_time", "convergence_iter", "test_error"]
    
    valid_values = {}
    for key in keys:
        valid_values[key] = [bag[key] for bag in bags if bag.get(key) is not None]
    
    return valid_values


def _build_aggregated_result(combo: Dict[str, Any], options: Dict[str, Any], 
                           valid_values: Dict[str, List], device: str) -> Dict[str, Any]:
    """Build aggregated result dictionary with averaged values."""
    aggregated = {**combo, **options, "device": device}
    
    # Compute averages for each metric
    if valid_values["min_training_error"]:
        aggregated["avg_training_error"] = np.mean(valid_values["min_training_error"])
    if valid_values["best_index"]:
        aggregated["avg_index"] = int(np.mean(valid_values["best_index"]))
    if valid_values["best_weights"]:
        aggregated["avg_weights"] = np.mean(valid_values["best_weights"], axis=0).tolist()
    if valid_values["best_means"]:
        aggregated["avg_means"] = np.mean(valid_values["best_means"], axis=0).tolist()
    if valid_values["best_covariances"]:
        aggregated["avg_covariances"] = np.mean(valid_values["best_covariances"], axis=0).tolist()
    if valid_values["fit_time"]:
        aggregated["fit_time"] = np.mean(valid_values["fit_time"])
    if valid_values["convergence_iter"]:
        aggregated["convergence_iter"] = int(np.mean(valid_values["convergence_iter"]))
    if valid_values["test_error"] and all(e is not None for e in valid_values["test_error"]):
        aggregated["avg_test_error"] = np.mean(valid_values["test_error"])
    
    return aggregated


def order_results(result_dict: Dict[str, Any], n_bags: int) -> Dict[str, Any]:
    """
    Align GMM components across bags using optimal assignment.
    
    Reorders components in all bags to match bag 1's ordering, then updates
    averaged parameters accordingly.
    
    Args:
        result_dict: Aggregated results containing individual bag results
        n_bags: Number of bags to align
        
    Returns:
        Updated result_dict with aligned components
    """
    from scipy.optimize import linear_sum_assignment
    
    # Use bag 1 as reference
    reference_means = np.atleast_2d(result_dict["best_means (bag 1)"])
    if reference_means.ndim == 1:
        reference_means = reference_means[:, None]

    aligned_means, aligned_weights, aligned_covs = [], [], []

    for bag_idx in range(1, n_bags + 1):
        # Get current bag parameters
        means = np.atleast_2d(result_dict[f"best_means (bag {bag_idx})"])
        if means.ndim == 1:
            means = means[:, None]
        weights = np.array(result_dict[f"best_weights (bag {bag_idx})"])
        covariances = np.array(result_dict[f"best_covariances (bag {bag_idx})"])

        # Compute optimal component assignment
        if means.shape[1] == 1:  # 1D case: sort by mean value
            assignment = np.argsort(means.flatten())
        else:  # Multi-dimensional: use Hungarian algorithm
            cost_matrix = np.linalg.norm(reference_means[:, None, :] - means[None, :, :], axis=2)
            _, assignment = linear_sum_assignment(cost_matrix)

        # Apply permutation
        aligned_means.append(means[assignment])
        aligned_weights.append(weights[:, assignment] if weights.ndim == 2 else weights[assignment])
        aligned_covs.append(covariances[assignment])

        # Update result dictionary
        result_dict[f"best_means (bag {bag_idx})"] = means[assignment].tolist()
        result_dict[f"best_weights (bag {bag_idx})"] = (
            weights[:, assignment] if weights.ndim == 2 else weights[assignment]
        ).tolist()
        result_dict[f"best_covariances (bag {bag_idx})"] = covariances[assignment].tolist()

    # Update averaged parameters
    result_dict["avg_means"] = np.mean(aligned_means, axis=0).tolist()
    result_dict["avg_weights"] = np.mean(aligned_weights, axis=0).tolist()
    result_dict["avg_covariances"] = np.mean(aligned_covs, axis=0).tolist()

    return result_dict


# ==============================================================================
# Data Serialization
# ==============================================================================

def save_json(data: Dict[str, Any], folder: str, filename: str) -> None:
    """
    Save data to JSON file with numpy type support.
    
    Args:
        data: Dictionary to save
        folder: Output directory
        filename: JSON filename
    """
    file_path = os.path.join(folder, filename)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    print(f"Results saved as {filename}")


def save_data_csv(experiments: Dict[str, Any], output_dir: str, 
                 filename: str = "generated_data.csv") -> None:
    """
    Save experiment data summary as CSV with JSON-encoded cells.
    
    Creates a matrix where rows=seeds, columns=n_points, and cells contain
    JSON-encoded lists of generated datasets.
    
    Args:
        experiments: Dictionary of experiment results
        output_dir: Output directory
        filename: CSV filename
    """
    # Collect experiment records
    records, seeds, n_points_values = [], set(), set()
    
    for experiment_info in experiments.values():
        combo = experiment_info["combo"]
        seed = combo.get("seed")
        n_points = combo.get("n_points")
        
        if seed is None or n_points is None:
            continue
            
        seeds.add(seed)
        n_points_values.add(n_points)
        
        # Convert numpy arrays to lists for JSON serialization
        cell_data = [dataset.tolist() for dataset in experiment_info["data"]]
        records.append((seed, n_points, cell_data))

    if not records:
        print("[WARN] save_data_csv: No records found to save")
        return

    # Create and populate DataFrame
    seeds_sorted = sorted(seeds)
    n_points_sorted = sorted(n_points_values)
    df = pd.DataFrame(index=seeds_sorted, columns=n_points_sorted, dtype=object)

    for seed, n_points, cell_data in records:
        df.at[seed, n_points] = cell_data

    # JSON-encode populated cells
    for column in df.columns:
        df[column] = df[column].apply(lambda v: json.dumps(v) if isinstance(v, list) else "")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.index.name = "seed"
    df.to_csv(output_path)
    print(f"[INFO] Saved generated data summary to {output_path}")


def load_experiment(base_folder: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Load experiment results from JSON file.
    
    Args:
        base_folder: Directory containing the file
        filename: JSON filename
        
    Returns:
        Loaded results dictionary or None if file doesn't exist/is invalid
    """
    file_path = os.path.join(base_folder, filename)
    
    if not os.path.isfile(file_path):
        return None
    
    # Check for empty file
    if os.stat(file_path).st_size == 0:
        print(f"Warning: {file_path} is empty")
        return None
    
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to decode JSON from {file_path}: {e}")
        return None


def aggregated_to_bag_results(aggregated: Dict[str, Any]) -> Dict[str, List]:
    """
    Convert aggregated results back to bag_results format.
    
    Args:
        aggregated: Aggregated results dictionary
        
    Returns:
        Dictionary with lists of values per bag
    """
    keys = ["seed", "min_training_error", "test_error", "fit_time",
            "convergence_iter", "best_index", "best_weights",
            "best_means", "best_covariances", "traj_weights",
            "avgL2err", "stdL2err"]
    
    bag_results = {key: [] for key in keys}
    
    bag_index = 1
    while f"seed (bag {bag_index})" in aggregated:
        for key in keys:
            bag_results[key].append(aggregated.get(f"{key} (bag {bag_index})"))
        bag_index += 1
    
    return bag_results