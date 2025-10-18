# ──────────────────────────────────────────────────────────────────────
#  Glucose-curve visualisation utilities
# ──────────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# 1. Low-level helpers
# ---------------------------------------------------------------------
def _collect_trajs(run_folder: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Return {patient_id : (t_vector , W)}  where
        • t_vector has been normalised to [0, 10]
        • W has shape (T, K)
    """
    trajs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    id_folders = sorted(glob(os.path.join(run_folder, "ids=*")))

    for id_dir in id_folders:
        # -------- patient ID from  "ids=<ID>" ------------------------
        try:
            pid = int(os.path.basename(id_dir).split("ids=")[1])
        except (IndexError, ValueError):
            continue

        # -------- latest  node_<j>  ----------------------------------
        node_dirs = glob(os.path.join(id_dir, "node_*"))
        if not node_dirs:
            continue
        node_latest = max(node_dirs,
                          key=lambda p: int(os.path.basename(p).split("_")[1]))

        res_file = os.path.join(node_latest, "result_node.json")
        if not os.path.isfile(res_file):
            continue

        # -------- load JSON ------------------------------------------
        with open(res_file) as fh:
            js = json.load(fh)
        W_raw = np.asarray(js["traj_weights"])        # expected (T, 1, K) or (T,K)

        # build / normalise the time vector
        if "time_grid" in js:
            t_raw = np.asarray(js["time_grid"], dtype=float)
        else:
            t_raw = np.linspace(0.0, 1.0, W_raw.shape[0])   # dummy, already 0-1

        # map *any* t_raw to the unit interval
        t = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min())

        # remove dummy axis  → (T, K)
        W = W_raw.squeeze(1) if W_raw.ndim == 3 else W_raw
        trajs[pid] = (t, W)
    return trajs



def _percentile_band(arr: np.ndarray, pct: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return lower/upper percentile trajectories (shape (T,)) that include
    `pct` percent of the samples in *arr* (shape = (#samples, T)).
    """
    lower = np.percentile(arr, (100 - pct) / 2, axis=0)
    upper = np.percentile(arr, 100 - (100 - pct) / 2, axis=0)
    return lower, upper


# ---------------------------------------------------------------------
# 2. Public plotting functions
# ---------------------------------------------------------------------
def plot_glucosecurves(i: int) -> None:
    """
    Overlay **all** NODE trajectories found under

        Results_MedicalData/run<i>/Id_*/node_* /

    Blue / green / orange for component 1 / 2 / 3.
    """
    base_run = os.path.join("Results_MedicalData", f"run{i}")
    if not os.path.isdir(base_run):
        raise FileNotFoundError(f"Run folder not found: {base_run}")

    trajs = _collect_trajs(base_run)
    if not trajs:
        print("[plot_glucosecurves] No trajectories found.")
        return

    colors = ["tab:blue", "tab:green", "tab:orange"]

    plt.figure(figsize=(12, 6))
    for pid, (t, W) in trajs.items():
        for k in range(W.shape[1]):                  # K components
            plt.plot(t, W[:, k],
                     color=colors[k % 3],
                     alpha=0.30, lw=0.8)

    plt.xlim(0, 1.0);  plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.title(f"All NODE trajectories – run{i}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_glucosebands(i: int, band_pct: float = 90.0) -> None:
    """
    Plot the mean NODE trajectory per component plus an **uncertainty
    band** that contains `band_pct` % of all curves.

    Colours: blue / green / orange as in *plot_glucosecurves*.
    """
    base_run = os.path.join("Results_MedicalData", f"run{i}")
    trajs = _collect_trajs(base_run)
    if not trajs:
        print("[plot_glucosebands] No trajectories found.")
        return

    # ----------- gather by component ---------------------------------
    #  dict  comp_k ➜ list of  W[:, k]  (each has length T_k)
    comp_samples: Dict[int, List[np.ndarray]] = {}
    time_vectors: Dict[int, np.ndarray] = {}

    for _, (t, W) in trajs.items():
        for k in range(W.shape[1]):
            comp_samples.setdefault(k, []).append(W[:, k])
            time_vectors[k] = t                      # all t are equal

    colors = ["tab:blue", "tab:green", "tab:orange"]
    plt.figure(figsize=(12, 6))

    for k, samples in comp_samples.items():
        t = time_vectors[k]
        curves = np.vstack(samples)                 # (#curves, T)

        mean = curves.mean(axis=0)
        low , up = _percentile_band(curves, band_pct)

        plt.plot(t, mean, color=colors[k], lw=2, label=f"Comp {k+1}")
        plt.fill_between(t, low, up, color=colors[k], alpha=0.2)

    plt.xlim(0, 1.0);  plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.title(f"{band_pct:.0f}% uncertainty bands – run{i}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


