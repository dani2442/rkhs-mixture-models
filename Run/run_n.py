from Utils.IO_utils import check_nodefolder, save_json
from Visualization.gifs import gif
from Visualization.plots import plot_training, plot_traj, plot_sumweights
import numpy as np

            
            
# ----------------------------------------------------------------------------- 
# run_node : train / load a Neural-ODE  and (optionally) make GIFs
# -----------------------------------------------------------------------------
def run_node(combo, data, best_means, best_covs, ts_weights,
             combo_folder, options):
    """
    Train or load the Neural-ODE that drives the weight trajectories
    ẇ = fθ(w,t).  

    The function returns a tuple (trajectory, folder).
    """
    # ------------------------------------------------------------------ #
    # 0)  House-keeping: folders, device, tensors
    # ------------------------------------------------------------------ #
    import torch
    from torch import optim
    import torch.optim.lr_scheduler as lr_sched
    from Core.train_node import Trainer

    node_folder, cached = check_nodefolder(combo_folder, combo, options)

    K      = combo["K"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(combo["seed_node"])

    # observation tensor (T, 1, K)
    obs_w  = torch.tensor(np.stack(ts_weights), dtype=torch.float32,
                          device=device).unsqueeze(1)
    x0     = obs_w[0]
    t_i    = torch.linspace(0.0, combo["T"],  combo["t_steps"], device=device)
    t_grid = torch.linspace(0.0, combo["T"],
                            int(combo["T"] / combo["dt"]) + 1, device=device)

    # ------------------------------------------------------------------ #
    # 1)  Train or load the NODE
    # ------------------------------------------------------------------ #
    
    if cached is not None:
        print(">> Using cached NODE trajectory")
        pred_traj   = np.asarray(cached["traj_weights"])
        
        loss_history = None
    
    else:
        if combo.get("use_replicator", False):
            # Choose between standard NODE and Replicator NODE
            from Core.replicator_node import ReplicatorNeuralODE
            model = ReplicatorNeuralODE(
                K, combo["hidden_dim"],
                non_linearity = combo["non_linearity"],
                N_layers      = combo["N_layers"],
                T             = combo["T"],
                step_size     = combo["dt"],
                method        = combo["method"]
            ).to(device)
            print(">> Using Replicator Neural ODE (simplex-preserving dynamics)")
            
            # Optional: Check simplex invariance on initial condition
            with torch.no_grad():
                check_result = model.check_simplex_invariance(x0, t_i)
                print(f"   Simplex check - Max sum deviation: {check_result['max_sum_deviation']:.2e}")
                print(f"   Min value: {check_result['min_value']:.2e}")
                print(f"   Valid trajectory: {check_result['is_valid']}")
        else:
            from Core.node import NeuralODE
            model = NeuralODE(
                K, combo["hidden_dim"],
                non_linearity = combo["non_linearity"],
                N_layers      = combo["N_layers"],
                T             = combo["T"],
                step_size     = combo["dt"],
                method        = combo["method"]
            ).to(device)
            print(">> Using standard Neural ODE with post-hoc normalization")

        opt  = optim.Adam(model.parameters(), lr=combo["lr"])
        sch  = lr_sched.ReduceLROnPlateau(opt, mode="min",
                                          factor=0.5, patience=50)
        sch = None
        trainer = Trainer(model, opt, sch, device,
                          loss_str     = combo["loss_func"],
                          reg_lambda   = combo["lambda_node"],
                          print_freq   = combo["print_freq"],
                          max_epochs   = combo["max_epochs"],
                          tol          = combo["tol_abs"],
                          tolrel       = combo["tol_rel"])

        trainer.train(t_i, obs_w, x0)
        loss_history = trainer.loss_history

        if loss_history:                                  # plot only if we trained
            plot_training(loss_history, folder=node_folder)

        model.eval()
        with torch.no_grad():
            pred_traj = model.flow(x0, eval_times=t_grid).cpu().numpy()
        combo["traj_weights"] = pred_traj.tolist()  # Convert to list for JSON
        save_json(data=combo, folder=node_folder, filename='result_node.json')

    # ------------------------------------------------------------------ #
    # 2)  Always make the basic diagnostic plots
    # ------------------------------------------------------------------ #
    pred_np   = pred_traj[:, 0, :]                          # (T,K)
    plot_traj(t_grid.cpu().numpy(), t_i.cpu().numpy(),
              pred_np, obs_w.cpu().numpy()[:, 0, :],
              K, folder=node_folder)
    plot_sumweights(t_grid.cpu().numpy(), pred_np, folder=node_folder)
    
    # ------------------------------------------------------------------ #
    # 3)  Optional: make the GIFs
    # ------------------------------------------------------------------ #
    if not options.get("load_data", False):
        from data import get_distrparams

        # FIX: Get ground truth for original time steps, then interpolate
        means_gt_coarse, vars_gt_coarse = get_distrparams(combo, combo["T"], combo["t_steps"])
        
        # Interpolate ground truth to match fine time grid
        t_coarse = np.linspace(0, combo["T"], combo["t_steps"])
        t_fine = t_grid.cpu().numpy()
        
        # Simple interpolation for means and variances
        means_gt_fine = []
        for t_idx in range(len(t_fine)):
            t_val = t_fine[t_idx]
            # Find interpolation indices
            idx = np.searchsorted(t_coarse, t_val)
            if idx == 0:
                means_gt_fine.append(means_gt_coarse[0])
            elif idx >= len(t_coarse):
                means_gt_fine.append(means_gt_coarse[-1])
            else:
                # Linear interpolation
                alpha = (t_val - t_coarse[idx-1]) / (t_coarse[idx] - t_coarse[idx-1])
                means_t = []
                for k in range(len(means_gt_coarse[0])):
                    if isinstance(means_gt_coarse[0][k], list):
                        mean_interp = [
                            (1-alpha) * means_gt_coarse[idx-1][k][d] + alpha * means_gt_coarse[idx][k][d]
                            for d in range(len(means_gt_coarse[0][k]))
                        ]
                    else:
                        mean_interp = (1-alpha) * means_gt_coarse[idx-1][k] + alpha * means_gt_coarse[idx][k]
                    means_t.append(mean_interp)
                means_gt_fine.append(means_t)
        
        # Interpolate variances
        vars_gt_fine = np.interp(t_fine, t_coarse, vars_gt_coarse)
        
        # ---------- optional GIFs with analytic background ----------
        if options.get("gif", False) and combo["d"] == 1:
            gif(pred_np, [best_means]*len(pred_np), [best_covs]*len(pred_np),
                means_gt = means_gt_fine, 
                vars_gt = vars_gt_fine,
                bg_mode  = "gaussian",
                t_grid   = t_grid.cpu().numpy(),
                fps      = min(30, combo["T"] / combo["dt"]),
                folder   = node_folder,
                filename = "gmm_evolution.gif")
    else:
        # ---------- optional histogram-background GIF ----------
        if options.get("gif", False) and combo["d"] == 1:
            gif(pred_np, [best_means]*len(pred_np), [best_covs]*len(pred_np),
                history_X = data,
                bg_mode   = "histogram",
                evol_weights = "path",
                t_grid    = t_grid.cpu().numpy(),
                fps       = min(30, combo["T"] / combo["dt"]),
                folder    = node_folder,
                filename  = "gmm_evolution.gif")
      
    return pred_np, node_folder