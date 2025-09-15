"""
Medical data processing module for time series analysis with derivatives.

Provides robust finite difference calculations on uneven grids with missing data,
block-based data loading, and patient filtering utilities.
"""

import os
import glob
import math
import random
import csv
from typing import Dict, List, Sequence, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd


# ==============================================================================
# Finite Difference Calculations for Uneven Grids
# ==============================================================================

def _compute_first_derivative(values: List[float], index: int) -> float:
    """
    Compute first derivative at given index using robust finite differences.
    
    Handles uneven grids and missing data (NaN values) by:
    - Using centered differences when both neighbors exist (2nd order accurate)
    - Falling back to one-sided differences when only one neighbor exists
    - Accounting for actual spacing between points
    
    Args:
        values: List of function values (may contain NaN for missing data)
        index: Index at which to compute derivative
        
    Returns:
        First derivative value or NaN if computation impossible
    """
    n = len(values)
    if math.isnan(values[index]):
        return math.nan

    # Find nearest valid neighbors
    left_idx = index - 1
    while left_idx >= 0 and math.isnan(values[left_idx]):
        left_idx -= 1
    
    right_idx = index + 1
    while right_idx < n and math.isnan(values[right_idx]):
        right_idx += 1

    # Centered difference (2nd order accurate)
    if left_idx >= 0 and right_idx < n:
        h1, h2 = index - left_idx, right_idx - index
        f_left, f_center, f_right = values[left_idx], values[index], values[right_idx]
        return ((h2**2) * (f_center - f_left) + (h1**2) * (f_right - f_center)) / (h1 * h2 * (h1 + h2))

    # One-sided differences (1st order accurate)
    if right_idx < n:
        return (values[right_idx] - values[index]) / (right_idx - index)
    if left_idx >= 0:
        return (values[index] - values[left_idx]) / (index - left_idx)

    return 0.0  # Isolated point


def _compute_second_derivative(values: List[float], index: int) -> float:
    """
    Compute second derivative at given index using robust finite differences.
    
    Uses 3-point stencil with nearest available neighbors to maintain accuracy
    even with missing data.
    
    Args:
        values: List of function values (may contain NaN for missing data)
        index: Index at which to compute derivative
        
    Returns:
        Second derivative value or NaN if computation impossible
    """
    n = len(values)
    if math.isnan(values[index]):
        return math.nan

    # Find two nearest valid neighbors (may be on same side)
    offsets = []
    for offset in range(1, n):
        if index - offset >= 0 and not math.isnan(values[index - offset]):
            offsets.append(-offset)
        if index + offset < n and not math.isnan(values[index + offset]):
            offsets.append(offset)
        if len(offsets) >= 2:
            break

    if len(offsets) < 2:
        return 0.0  # Insufficient data

    offsets.sort()
    left_idx = index + offsets[0]
    right_idx = index + offsets[1]

    h1, h2 = index - left_idx, right_idx - index
    f_left, f_center, f_right = values[left_idx], values[index], values[right_idx]
    
    return 2 * ((f_left - f_center) / h1 - (f_center - f_right) / h2) / (h1 + h2)


def _process_derivatives(values: List[float], include_first: bool, include_second: bool) -> List[List[float]]:
    """
    Process row values to include derivatives as requested.
    
    Args:
        values: Original function values
        include_first: Whether to include first derivatives
        include_second: Whether to include second derivatives
        
    Returns:
        List of augmented values [value, deriv1, deriv2] as appropriate
    """
    if include_first or include_second:
        first_derivs = [_compute_first_derivative(values, i) for i in range(len(values))]
        if include_second:
            second_derivs = [_compute_second_derivative(values, i) for i in range(len(values))]

    result = []
    for i, val in enumerate(values):
        if math.isnan(val):
            continue
            
        entry = [val]
        if include_first:
            entry.append(first_derivs[i])
        if include_second:
            entry.append(second_derivs[i])
        result.append(entry)
    
    return result


def _save_derivative_csv(df: pd.DataFrame, output_path: str, include_first: bool, include_second: bool) -> None:
    """Save CSV with derivative information included in each cell."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(df.columns)
        
        for row_idx, row in df.iterrows():
            new_row = []
            for col_name in df.columns:
                val = row[col_name]
                
                if pd.isna(val):
                    new_row.append("")
                    continue
                
                col_vals = df[col_name].tolist()
                
                if include_second:
                    d1 = _compute_first_derivative(col_vals, row_idx)
                    d2 = _compute_second_derivative(col_vals, row_idx)
                    formatted_val = f"[{val}, {d1 if not pd.isna(d1) else 'NaN'}, {d2 if not pd.isna(d2) else 'NaN'}]"
                elif include_first:
                    d1 = _compute_first_derivative(col_vals, row_idx)
                    formatted_val = f"[{val}, {d1 if not pd.isna(d1) else 'NaN'}]"
                else:
                    formatted_val = str(val)
                
                new_row.append(formatted_val)
            
            writer.writerow(new_row)


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_patient_blocks(patient_id: int, block_size: int, *, folder: str = ".", 
                       add_first_deriv: bool = False, add_second_deriv: bool = False, 
                       min_blocks: int = 0) -> Dict[int, List[List]]:
    """
    Load patient data and split into blocks with optional derivative calculation.
    
    Args:
        patient_id: Patient ID corresponding to CSV filename
        block_size: Number of rows per block
        folder: Directory containing CSV files
        add_first_deriv: Include first derivatives
        add_second_deriv: Include second derivatives (implies first derivatives)
        min_blocks: Minimum blocks required (return empty if not met)
        
    Returns:
        Dictionary with patient_id as key and list of blocks as value
    """
    csv_path = os.path.join(folder, f"{patient_id}.csv")
    if not os.path.isfile(csv_path):
        return {}

    # Read data
    df = pd.read_csv(csv_path, dtype=float)
    
    # Save derivative CSV if requested
    if add_first_deriv or add_second_deriv:
        suffix = "_diff1" if add_first_deriv and not add_second_deriv else "_diff2" if add_second_deriv and not add_first_deriv else "_diff12"
        output_path = os.path.join(folder, f"{patient_id}{suffix}.csv")
        _save_derivative_csv(df, output_path, add_first_deriv, add_second_deriv)

    # Process rows
    processed_rows = []
    for row_tuple in df.itertuples(index=False, name=None):
        # Convert to float, handle NaN
        values = [float(v) if not (isinstance(v, float) and math.isnan(v)) else math.nan for v in row_tuple]
        
        # Remove trailing NaNs
        while values and math.isnan(values[-1]):
            values.pop()
        
        if not values:  # Skip empty rows
            continue
        
        # Process with derivatives
        row_data = _process_derivatives(values, add_first_deriv, add_second_deriv)
        if row_data:  # Only add non-empty processed rows
            processed_rows.append(row_data)

    # Create complete blocks
    num_complete_blocks = len(processed_rows) // block_size
    blocks = [processed_rows[i * block_size:(i + 1) * block_size] for i in range(num_complete_blocks)]

    # Check minimum block requirement
    if min_blocks > 0 and len(blocks) < min_blocks:
        return {}

    return {patient_id: blocks}


def aggregate_patient_data(*, n_patients: Optional[int] = None, 
                          patient_ids: Optional[List[int]] = None,
                          folder: str = ".", seed: Optional[int] = None,
                          add_first_deriv: bool = False, 
                          add_second_deriv: bool = False) -> np.ndarray:
    """
    Aggregate data from multiple patients with optional derivative calculation.
    
    Args:
        n_patients: Number of patients to randomly select
        patient_ids: Specific patient IDs to use (overrides n_patients)
        folder: Directory containing CSV files
        seed: Random seed for patient selection
        add_first_deriv: Include first derivatives
        add_second_deriv: Include second derivatives
        
    Returns:
        Array of shape (N, F) where F = 1 + add_first_deriv + add_second_deriv
    """
    rng = random.Random(seed)

    # Discover available patient IDs
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    available_ids = sorted(
        int(os.path.splitext(os.path.basename(path))[0])
        for path in csv_files 
        if os.path.basename(path)[:-4].isdigit()
    )

    # Select patient IDs
    if patient_ids is not None:
        selected_ids = [pid for pid in patient_ids if pid in available_ids]
        if not selected_ids:
            raise ValueError("None of the specified patient IDs were found")
    else:
        if not n_patients or n_patients <= 0:
            raise ValueError("n_patients must be positive when patient_ids not specified")
        if n_patients > len(available_ids):
            raise ValueError(f"Requested {n_patients} patients but only {len(available_ids)} available")
        selected_ids = rng.sample(available_ids, k=n_patients)

    # Collect features from all selected patients
    all_features = []
    
    for patient_id in selected_ids:
        csv_path = os.path.join(folder, f"{patient_id}.csv")
        try:
            df = pd.read_csv(csv_path, dtype=float)
        except FileNotFoundError:
            print(f"Warning: {csv_path} not found - skipped")
            continue

        for row_tuple in df.itertuples(index=False, name=None):
            # Convert and clean values
            values = [float(v) if not (isinstance(v, float) and math.isnan(v)) else math.nan for v in row_tuple]
            while values and math.isnan(values[-1]):
                values.pop()
            if not values:
                continue

            # Process with derivatives
            row_features = _process_derivatives(values, add_first_deriv, add_second_deriv)
            all_features.extend(row_features)

    if not all_features:
        raise RuntimeError("No valid numeric data found in selected patient files")

    return np.asarray(all_features, dtype=float)


# ==============================================================================
# Patient Filtering Utilities
# ==============================================================================

def filter_patient_ids(summary_csv: str, 
                      conditions: Dict[str, Union[Any, Sequence[Any], Tuple[Any, Any]]]) -> List[int]:
    """
    Filter patient IDs based on summary table conditions.
    
    Args:
        summary_csv: Path to patient summary CSV file
        conditions: Dictionary of {column: condition} where condition can be:
            - Single value: exact match
            - List/tuple: value must be in the collection
            - 2-tuple of numbers: inclusive range [min, max]
            
    Returns:
        List of patient IDs meeting all conditions
        
    Raises:
        ValueError: If PtID column missing or specified column not found
    """
    df = pd.read_csv(summary_csv)
    if "PtID" not in df.columns:
        raise ValueError("'PtID' column not found in summary CSV")

    # Start with all rows matching
    mask = pd.Series(True, index=df.index)

    for column, condition in conditions.items():
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV")

        # Check if condition is a numeric range (2-tuple of numbers)
        if (isinstance(condition, (list, tuple)) and len(condition) == 2 and 
            all(isinstance(x, (int, float)) for x in condition)):
            min_val, max_val = condition
            mask &= df[column].between(min_val, max_val, inclusive="both")
        # Check if condition is a collection of values
        elif isinstance(condition, (list, tuple, set)):
            mask &= df[column].isin(condition)
        # Single value exact match
        else:
            mask &= df[column] == condition

    return df.loc[mask, "PtID"].astype(int).tolist()


# ==============================================================================
# Backward Compatibility Aliases
# ==============================================================================

def load_datamed(idx: int, n: int, *, folder: str = ".", add_deriv: bool = False, 
                add_second_deriv: bool = False, threshold: bool = False) -> Dict[int, List[List]]:
    """Legacy function name for load_patient_blocks."""
    min_blocks = 3 if threshold else 0
    return load_patient_blocks(idx, n, folder=folder, add_first_deriv=add_deriv, 
                              add_second_deriv=add_second_deriv, min_blocks=min_blocks)


def load_aggregate_datamed(*, n_ids: Optional[int] = None, 
                          ids_explicit: Optional[List[int]] = None,
                          folder: str = ".", seed: Optional[int] = None,
                          add_deriv: bool = False, add_second_deriv: bool = False) -> np.ndarray:
    """Legacy function name for aggregate_patient_data."""
    return aggregate_patient_data(n_patients=n_ids, patient_ids=ids_explicit, 
                                 folder=folder, seed=seed, add_first_deriv=add_deriv, 
                                 add_second_deriv=add_second_deriv)


def filter_ids(summary_csv: str, 
               conditions: Dict[str, Union[Any, Sequence[Any], Tuple[Any, Any]]]) -> List[int]:
    """Legacy function name for filter_patient_ids."""
    return filter_patient_ids(summary_csv, conditions)