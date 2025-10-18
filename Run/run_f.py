import os
import numpy as np
from Utils.IO_utils import (
    load_experiment, save_fit,
    aggregated_to_bag_results, order_results
)
from Visualization.log import log_experiment
from Core.train_mmd import mmd_gmm_fit
from Core.test_mmd import test
from Visualization.plots import plot_fit, plot_mixture, plot_bands
from sklearn.model_selection import KFold

def run_fitting(combo, options, results_folder, Xs, test_gamma=None):
    """
    Run one full MMD-GMM fit in one of three modes:
      1) single fit
      2) bagging (bootstrap + OOB)
      3) k-fold cross-validation
    """
    log_experiment(combo, phase=2, options = options)
    existing = load_experiment(results_folder, "result_fit.json")

    # If we can resume from cache
    if existing is not None and options.get("load_cached_fit", False):
        if "seed (bag 1)" in existing:
            i = 1
            while f"seed (bag {i})" in existing:
                i += 1
            done = i - 1
        if done >= combo["n_bags"]:
            print(f"Cached {done}/{combo['n_bags']} bags.")
            existing = order_results(existing, combo["n_bags"])
            if combo.get("d",0)==1 and combo.get("n_bags",0)>1:
                plot_bands(existing, Xs, results_folder, combo)
            return existing
        print(f"Resuming experiment: {done} bags done, continuing from bag {done+1}.")
        bag_results = aggregated_to_bag_results(existing)
        start_idx = done
    else:
        # initialize an empty results accumulator
        bag_results = {
            "seed": [], "min_training_error": [], "test_error": [],
            "fit_time": [], "convergence_iter": [], "best_index": [],
            "best_weights": [], "best_means": [], "best_covariances": []
        }
        start_idx = 0
    

    # Decide mode
    do_bagging = combo.get("n_bags",1) > 1
    do_cv      = (combo.get("n_bags",1)==1) and options.get("test", False)
    do_single  = not do_bagging and not do_cv

    # 1) Single fit mode
    if do_single:
        print("Single fit mode.")
        seed = combo.get("seed_split",0)

        (best_i, hist_ws, hist_m, hist_c, hist_ews, hist_em, hist_ec,
         hist_e, fit_time, device, conv_it) = mmd_gmm_fit(
            Xs, K=combo["K"], gamma=test_gamma,
            max_iter=combo["max_iter"],
            tol_rel_param=combo["tol_rel_param"],
            tol_rel_err=combo["tol_rel_err"],
            tol_abs=combo["tol_abs"],
            lr=combo["lr"], grad_steps=combo["grad_steps"],
            verbose=options.get("verbose",False),
            fixed_means=(combo["means"] if options.get("fix_mv") else None),
            fixed_covs=(combo["covs"] if options.get("fix_mv") else None),
            lambda_ridge=combo.get("lambda_ridge",0)
        )
        test_error = None
        log_experiment(combo, phase=3,
                       errors=hist_e, best_index=best_i,
                       conv_it=conv_it, test_error=test_error)
        # Plot
        folder = os.path.join(results_folder, "bag_1")
        os.makedirs(folder, exist_ok=True)
        
        for i, data in enumerate(Xs):
            current_ws = hist_ws[best_i][i] 
            current_ews = [err[i] for err in hist_ews] 
            if combo["d"]<=1 and not options.get("fix_mv"):
                plot_fit(
                        current_ews, hist_em, hist_ec, hist_e,
                        data, current_ws, hist_m[best_i], hist_c[best_i],
                        best_i, folder=folder,
                        filename=f"fit_ds{i+1}.png", log_scale=True
                    )
            elif combo["d"]<=1:
                plot_mixture(
                    data, current_ws, hist_m[best_i], hist_c[best_i],
                    folder=folder,
                    filename=f"mix_ds{i+1}.png"
                )
        
        # Save as if it were “one bag”
        bag_results["seed"].append(seed)
        bag_results["min_training_error"].append(float(hist_e[best_i]))
        bag_results["test_error"].append(test_error)
        bag_results["fit_time"].append(fit_time)
        bag_results["convergence_iter"].append(conv_it)
        bag_results["best_index"].append(best_i)
        bag_results["best_weights"].append(hist_ws[best_i])
        bag_results["best_means"].append(hist_m[best_i].tolist())
        bag_results["best_covariances"].append(hist_c[best_i].tolist())
        aggregated = save_fit(results_folder, bag_results, combo, options, str(device))
        return aggregated

    # 2) Bagging mode
    if do_bagging:
        print(f"Bagging mode: {combo['n_bags']} bags.")
        end_idx = combo["n_bags"]
        for b in range(start_idx, end_idx):
            print(f"\nStarting fitting of bag {b + 1}/{combo['n_bags']}...")
            seed = combo.get("seed_split",0)+b
            rng  = np.random.RandomState(seed)
            X_train, X_oob = [], []
            for arr in Xs:
                idx = rng.choice(len(arr), len(arr), replace=True)
                train = arr[idx]
                mask  = np.ones(len(arr),bool)
                mask[np.unique(idx)] = False
                X_train.append(train)
                X_oob.append(arr[mask])
            # fit
            (best_i, hist_ws, hist_m, hist_c, hist_ews, hist_em, hist_ec,
                hist_e, fit_time, device, conv_it) = mmd_gmm_fit(
                X_train, K=combo["K"], gamma=test_gamma,
                max_iter=combo["max_iter"],
                tol_rel_param=combo["tol_rel_param"],
                tol_rel_err=combo["tol_rel_err"],
                tol_abs=combo["tol_abs"],
                lr=combo["lr"], grad_steps=combo["grad_steps"],
                verbose=options.get("verbose",False),
                fixed_means=(combo["means"] if options.get("fix_mv") else None),
                fixed_covs=(combo["covs"] if options.get("fix_mv") else None),
                lambda_ridge=combo.get("lambda_ridge",0)
            )
            # OOB test
            test_error = test(
                ws=hist_ws[best_i], mus=hist_m[best_i], covs=hist_c[best_i],
                X_test=X_oob, gamma=test_gamma
            ) if options.get("test",False) else None
            log_experiment(combo, phase=3,
                           errors=hist_e, best_index=best_i,
                           conv_it=conv_it, test_error=test_error)
            # per-bag plots
            bag_folder = os.path.join(results_folder,f"bag_{b+1}")
            os.makedirs(bag_folder, exist_ok=True)
            
            for i,data in enumerate(X_train):
                current_ws = hist_ws[best_i][i] 
                current_ews = [err[i] for err in hist_ews] 
                if combo["d"]<=2 and not options.get("fix_mv"):
                    plot_fit(
                        current_ews, hist_em, hist_ec, hist_e,
                        data, current_ws, hist_m[best_i], hist_c[best_i],
                        best_i, folder=bag_folder,
                        filename=f"fit_bag{b+1}_ds{i+1}.png", log_scale=True
                    )
                elif combo["d"]<=1:
                    plot_mixture(
                        data, current_ws, hist_m[best_i], hist_c[best_i],
                        folder=bag_folder,
                        filename=f"mix_bag{b+1}_ds{i+1}.png"
                    )
            
            # append + save
            bag_results["seed"].append(seed)
            bag_results["min_training_error"].append(float(hist_e[best_i]))
            bag_results["test_error"].append(test_error)
            bag_results["fit_time"].append(fit_time)
            bag_results["convergence_iter"].append(conv_it)
            bag_results["best_index"].append(best_i)
            bag_results["best_weights"].append(hist_ws[best_i])
            bag_results["best_means"].append(hist_m[best_i].tolist())
            bag_results["best_covariances"].append(hist_c[best_i].tolist())
            aggregated = save_fit(results_folder, bag_results, combo, options, str(device))
            print(f"Bag {b+1}/{combo['n_bags']} done.")
        if combo["d"]==1 and combo["n_bags"]>1:
            plot_bands(aggregated, Xs, results_folder, combo)
        return aggregated

    # 3) CV mode
    if do_cv:
        k = options.get("cv_folds", 5)
        print(f"Cross-validation mode: {k} folds.")
        kf = KFold(n_splits=k, shuffle=True, random_state=combo.get("seed_split", 0))
        splits = [list(kf.split(X_i)) for X_i in Xs]
        for fold in range(k):
            seed = combo.get("seed_split", 0) + fold

            # 2) Build train/test series for this fold
            X_train = [
                Xs[i][splits[i][fold][0]]  # take training rows of block i
                for i in range(len(Xs))
            ]
            X_test = [
                Xs[i][splits[i][fold][1]]  # take testing rows of block i
                for i in range(len(Xs))
            ]
            # fit
            (best_i, hist_ws, hist_m, hist_c, hist_ews, hist_em, hist_ec,
                hist_e, fit_time, device, conv_it) = mmd_gmm_fit(
                X_train, K=combo["K"], gamma=test_gamma,
                max_iter=combo["max_iter"],
                tol_rel_param=combo["tol_rel_param"],
                tol_rel_err=combo["tol_rel_err"],
                tol_abs=combo["tol_abs"],
                lr=combo["lr"], grad_steps=combo["grad_steps"],
                verbose=options.get("verbose",False),
                fixed_means=(combo["means"] if options.get("fix_mv") else None),
                fixed_covs=(combo["covs"] if options.get("fix_mv") else None),
                lambda_ridge=combo.get("lambda_ridge",0)
            )
            # test
            test_error = test(
                ws=hist_ws[best_i], mus=hist_m[best_i], covs=hist_c[best_i],
                X_test=X_test, gamma=test_gamma
            )
            log_experiment(combo, phase=3,
                           errors=hist_e, best_index=best_i,
                           conv_it=conv_it, test_error=test_error)
            
            # fold plots
            cv_folder = os.path.join(results_folder,f"bag_{fold+1}")
            os.makedirs(cv_folder, exist_ok=True)
            for i,data in enumerate(X_train):
                current_ws = hist_ws[best_i][i] 
                current_ews = [err[i] for err in hist_ews] 
                if combo["d"]<=1 and not options.get("fix_mv"):
                    plot_fit(
                        current_ews, hist_em, hist_ec, hist_e,
                        data, current_ws, hist_m[best_i], hist_c[best_i],
                        best_i, folder=cv_folder,
                        filename=f"fit_fold{fold+1}_ds{i+1}.png", log_scale=True
                    )
                elif combo["d"]<=1:
                    plot_mixture(
                        data, current_ws, hist_m[best_i], hist_c[best_i],
                        folder=cv_folder,
                        filename=f"mix_fold{fold+1}_ds{i+1}.png"
                    )
            # append + save
            bag_results["seed"].append(seed)
            bag_results["min_training_error"].append(float(hist_e[best_i]))
            bag_results["test_error"].append(test_error)
            bag_results["fit_time"].append(fit_time)
            bag_results["convergence_iter"].append(conv_it)
            bag_results["best_index"].append(best_i)
            bag_results["best_weights"].append(hist_ws[best_i])
            bag_results["best_means"].append(hist_m[best_i].tolist())
            bag_results["best_covariances"].append(hist_c[best_i].tolist())
            aggregated = save_fit(results_folder, bag_results, combo, options, str(device))
            print(f"Fold {fold+1}/{k} done.")
        return aggregated

    # Should never get here
    raise RuntimeError("run_fitting: unexpected fit mode")


