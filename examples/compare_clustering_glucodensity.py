"""
Glucodensity clustering helpers shared by the benchmark scripts.
"""
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Patient-level posterior and group divergence utilities
# ---------------------------------------------------------------------------

def compute_patient_posteriors_from_labels(
    labels: np.ndarray,
    patient_ids: list,
    n_clusters: int,
) -> dict:
    """Convert hard cluster labels to per-patient posterior distributions."""
    posteriors = {}
    unique_patients = sorted(set(patient_ids))
    for pid in unique_patients:
        idx = [i for i, p in enumerate(patient_ids) if p == pid]
        if not idx:
            continue
        counts = np.bincount([labels[i] for i in idx], minlength=n_clusters).astype(float)
        counts /= counts.sum()
        posteriors[pid] = counts
    return posteriors


def compute_patient_posteriors_from_soft(
    soft_labels: np.ndarray,
    patient_ids: list,
) -> dict:
    """Average soft cluster posteriors per patient."""
    posteriors = {}
    unique_patients = sorted(set(patient_ids))
    for pid in unique_patients:
        idx = [i for i, p in enumerate(patient_ids) if p == pid]
        if not idx:
            continue
        posteriors[pid] = soft_labels[idx].mean(axis=0)
    return posteriors


def build_patient_level_features(
    X: torch.Tensor,
    patient_ids: list,
    mode: str,
) -> tuple[torch.Tensor, list]:
    """Aggregate window features into one feature vector per patient."""
    if mode == "window":
        return X, list(patient_ids)

    X_np = X.detach().cpu().numpy()
    unique_patients = sorted(set(patient_ids))
    patient_features = []

    for pid in unique_patients:
        idx = [i for i, p in enumerate(patient_ids) if p == pid]
        X_pid = X_np[idx]
        mean_feat = X_pid.mean(axis=0)

        if mode == "patient_mean":
            feat = mean_feat
        elif mode == "patient_meanstd":
            feat = np.concatenate([mean_feat, X_pid.std(axis=0)])
        else:
            raise ValueError(f"Unknown patient feature mode: {mode}")

        patient_features.append(feat)

    X_patient = torch.tensor(
        np.asarray(patient_features),
        device=X.device,
        dtype=X.dtype,
    )
    return X_patient, unique_patients


def expand_patient_soft_labels(
    patient_soft_labels: np.ndarray,
    patient_index_ids: list,
    window_patient_ids: list,
) -> np.ndarray:
    """Broadcast patient-level soft assignments back to all windows."""
    soft_by_patient = {
        pid: patient_soft_labels[i]
        for i, pid in enumerate(patient_index_ids)
    }
    return np.asarray([soft_by_patient[pid] for pid in window_patient_ids])


def compute_tv_divergence(posteriors: dict, control_ids: list, treatment_ids: list) -> float:
    """TV distance between mean control and treatment group posteriors."""
    ctrl = [posteriors[pid] for pid in control_ids if pid in posteriors]
    treat = [posteriors[pid] for pid in treatment_ids if pid in posteriors]
    if not ctrl or not treat:
        return float("nan")
    ctrl_mean = np.mean(ctrl, axis=0)
    treat_mean = np.mean(treat, axis=0)
    return 0.5 * float(np.sum(np.abs(ctrl_mean - treat_mean)))


def estimate_dbscan_eps(X_np: np.ndarray, min_samples: int, percentile: float = 90.0) -> float:
    """
    Estimate a good DBSCAN eps from the k-NN distance distribution.
    Sorts distances to the (min_samples)-th nearest neighbor and returns
    the given percentile — a standard data-adaptive heuristic for eps.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_np)
    distances, _ = nbrs.kneighbors(X_np)
    knn_dists = np.sort(distances[:, -1])  # distance to k-th neighbor, sorted
    return float(np.percentile(knn_dists, percentile))


def resolve_gaussian_sigma(
    X: torch.Tensor,
    sigma: float,
    sigma_scale: float,
    max_samples: int,
) -> float:
    """Return the user-provided sigma or a median-heuristic bandwidth."""
    if sigma > 0:
        return float(sigma)

    with torch.no_grad():
        if X.shape[0] > max_samples:
            sample_idx = torch.randperm(X.shape[0], device=X.device)[:max_samples]
            X_sigma = X[sample_idx]
        else:
            X_sigma = X

        pairwise_dist = torch.cdist(X_sigma, X_sigma)
        iu = torch.triu_indices(pairwise_dist.shape[0], pairwise_dist.shape[1], offset=1)
        upper = pairwise_dist[iu[0], iu[1]]
        positive = upper[upper > 0]
        median_sigma = torch.median(positive).item() if positive.numel() > 0 else 1.0
    return max(float(sigma_scale) * float(median_sigma), 1e-6)
