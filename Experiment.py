import os
import numpy as np
from itertools import product
from Run.run_f import run_fitting
from Run.run_n import run_node
from Run.run_t import run_test
from data import generate_data, compute_gammaopt
from Utils.IO_utils import create_folder, save_data_csv
from load import load_datamed, load_aggregate_datamed

# ----------------------------------------------------------------------------- 
# Experiment : one hyper-parameter configuration  ➜  one complete run
# -----------------------------------------------------------------------------

class Experiment:
    """
    Hold everything related to ONE experiment:
      – fixed hyper-parameter combo
      – the time-series datasets generated or loaded
      – the γ* found for every time-step
      – the results dict that comes back from fitting / NODE
    """
    def __init__(self, exp_id, combo, data, gamma):
        self.exp_id = exp_id
        self.combo  = combo          # fixed params
        self.data   = data           # list of arrays (len = t_steps)
        self.gamma  = gamma          # list of γ* (len = t_steps)
        self.results = None
        self.folder  = None

    # ------------------------------------------------------------------ #
    # run : launch fitting   →   launch NODE   →   collect every result
    # ------------------------------------------------------------------ #
    def run(self, options):
        """
        Run complete time-series experiment:
        1) Fit weights/means/covariances via MMD-GMM
        2) Train Neural ODE for weight trajectories
        3) Evaluate and test the model
        """
        
        # 0) ----------------------------------------------------------------
        print(f"[Experiment]  {self.exp_id}")
        self.folder = create_folder(options, self.combo)
        os.makedirs(self.folder, exist_ok=True)
        
        # 1) MMD-GMM fitting
        fit_res = run_fitting(self.combo, options, self.folder, self.data, self.gamma)
        all_trajs = []
        all_errs = []

        m  = fit_res["avg_means"]
        c  = fit_res["avg_covariances"]
        w  = fit_res["avg_weights"]
            
        # 2) Train Neural ODE and evaluate
        traj, f = run_node(self.combo, self.data, m, c, w, self.folder, options)
        errs = run_test(self.combo, m, c, w, traj, f, self.data, options)
        all_trajs.append(traj)
        all_errs.append(errs)

        # 3) Store results
        fit_res["traj_weights"] = all_trajs
        fit_res["test_errors"] = all_errs
        
        self.results = fit_res
        print(f"Experiment {self.exp_id} finished. Results saved in {self.folder}")
        return self.results

def prepare_experiments(params_data,
                        params_optim,
                        params_uq,
                        params_node,
                        options):
    """
    Build the dict {exp_id: {"combo", "data", "gamma"}} consumed later by
    `Experiment`.  Works for
        • purely synthetic runs          (params_load is None)
        • CSV time-series runs   (params_load is a dict from get_params)
    """

    # ------------------------------------------------------------------ #
    # 1. LOADED CSV BRANCH
    # ------------------------------------------------------------------ #
    folder = options["base_folder"]
    
    params_load  = options.get("load_data", False)
    if params_load:
        print(f"[prepare_experiments]  Loading CSV data from «{params_data['csv_folder']}»")

        d            = params_data["d"]
        T            = params_node["T"]               
        block_size   = params_data["block_size"]
        ids  = params_data["ids"]
        csv_folder   = params_data["csv_folder"]
        dif1=params_data["add_deriv"]
        dif2=params_data["add_second_deriv"]
        if options.get("fix_mv", False) and params_uq["n_bags"] == 1:
            from Utils.MMD_utils import init_params
            subset_ids = [13, 62, 377]
            print("Aggregating data for initial means and variances")
            X_batch = load_aggregate_datamed(ids_explicit = subset_ids, folder="data", seed=42, add_deriv = dif1, add_second_deriv = dif2)
            _, params_data["means"], params_data["covs"] = init_params(X_batch, params_optim["K"][0])
            print(f"Fitted Means: {params_data['means']})  ||  Fitted Variances: {params_data['covs']}")
 
        experiments  = {}
        discarded   = 0
        not_found   = 0

        for id in ids:
            # Load the data blocks for this ID
            blocks = load_datamed(id, n=block_size, folder=csv_folder,
                                add_deriv=dif1, add_second_deriv=dif2, threshold=options["threshold"]).get(id, [])
            # Add this before processing a block to see its structure

            # Skip if no valid blocks found
            if not blocks:
                discarded += 1
                continue
            if blocks == {}:
                not_found += 1
                continue
            
            # Process each block
            data_list, gamma_list = [], []
            for blk in blocks:
                try:
                    # Extract all points as a flat list
                    all_points = []
                    
                    for row in blk:
                        for item in row:
                            if isinstance(item, list):
                                all_points.append(item)
                            else:
                                all_points.append([float(item)])
                    
                    # Only proceed if we have enough points for distance calculation
                    if len(all_points) >= 2:
                        # Convert to numpy array
                        X_t = np.array(all_points, dtype=float)
                        
                        # Add to our data collection
                        data_list.append(X_t)
                        
                        # Calculate optimal gamma
                        gamma_list.append(compute_gammaopt(X_t))
                    else:
                        print(f"Warning: Block for ID {id} has insufficient points ({len(all_points)})")
                
                except Exception as e:
                    print(f"Error processing block for ID {id}: {e}")

            # ---------------------------------------------------------------------

            # ------- build combo: use first values in list-type hyper-params ------
            merged = {**params_data, **params_optim, **params_uq, **params_node}
            combo  = {k: (v[0] if isinstance(v, list) else v) for k, v in merged.items()}
            combo.update({
                "ids" : id,
                "t_steps"    : len(data_list),   # override with real #blocks
                "T"          : T,
                "n_points"   : data_list[0].shape[0]   # size of each block
            })

            exp_id = f"Id{id}"
            experiments[exp_id] = {"combo": combo,
                                   "data" : data_list,
                                   "gamma": gamma_list}
            
        print("=" * 50)
        print(f"[prepare_experiments] Discarded {discarded} id(s) with <3 blocks of size {block_size} (<30 days)")
        print(f"[prepare_experiments] Not found {not_found} id(s) in folder {csv_folder}")
        print(f"[prepare_experiments] Total experiments: {len(experiments)}")
        print("=" * 50)

        return experiments

    # ------------------------------------------------------------------ #
    # 2. SYNTHETIC BRANCH  (identical logic to previous version)
    # ------------------------------------------------------------------ #
    merged   = {**params_data, **params_optim, **params_uq, **params_node}
    ts_keys  = {"means", "vars"}

    fixed_keys   = [k for k in merged if k not in ts_keys]
    fixed_params = {k: (merged[k] if isinstance(merged[k], list) else [merged[k]]) for k in fixed_keys}
    fixed_combos = [dict(zip(fixed_keys, vals)) for vals in product(*fixed_params.values())]

    experiments = {}
    T       = params_node.get("T", 0)
    t_steps = params_data.get("t_steps", 1)
    d       = params_data["d"]

    for i, combo in enumerate(fixed_combos, start=1):
        exp_id = f"exp_{i}"
        seed   = combo.get("seed")

        if T == 0:
            ts_params = {"means": params_data["means"][0], "vars": params_data["vars"][0]}
            X     = generate_data("gaussian", d, {**combo, **ts_params}, seed=seed)
            gamma = compute_gammaopt(X)
            experiments[exp_id] = {"combo": combo, "data": [X], "gamma": [gamma]}
        else:
            data_list, gamma_list = [], []
            for t in range(t_steps):
                ts_params = {"means": params_data["means"][t], "vars": params_data["vars"][t]}
                X_t   = generate_data("gaussian", d, {**combo, **ts_params}, seed=seed)
                data_list.append(X_t)
                gamma_list.append(compute_gammaopt(X_t))
            experiments[exp_id] = {"combo": combo, "data": data_list, "gamma": gamma_list}

    if folder is not None:
        save_data_csv(experiments, output_dir=folder, filename="data.csv")

    return experiments


