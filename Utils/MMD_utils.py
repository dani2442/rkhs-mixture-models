import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize


def gaussian_kernel(X, Y, gamma):
    """
    Compute the Gaussian (RBF) kernel matrix between two sets of points.
    """
    diff = X[:, None, :] - Y[None, :, :]
    dists = np.sum(diff**2, axis=2)
    return np.exp(-0.5 * dists / (gamma**2))

def compute_I(mu_k, Sigma_k, mu_l, Sigma_l, Gamma):
    """
    Compute I_{k,l} = E_{X ~ N(mu_k, Sigma_k)} E_{X' ~ N(mu_l, Sigma_l)}[ k(X, X') ]
    using the closed-form expression.
    """
    
    Sigma_sum = Sigma_k + Sigma_l + Gamma
    det_ratio = np.sqrt(np.linalg.det(Gamma)) / np.sqrt(np.linalg.det(Sigma_sum))
    diff = mu_k - mu_l
    exp_term = np.exp(-0.5 * diff.T @ inv(Sigma_sum) @ diff)
    return det_ratio * exp_term

def compute_J(mu_k, Sigma_k, X_i, Gamma):
    """
    Compute J_{k,i} = E_{X ~ N(mu_k, Sigma_k)}[ k(X, X_i) ]
    using the closed-form expression.
    """
    Sigma_sum = Sigma_k + Gamma
    det_ratio = np.sqrt(np.linalg.det(Gamma)) / np.sqrt(np.linalg.det(Sigma_sum))
    diff = X_i - mu_k
    exp_term = np.exp(-0.5 * diff.T @ inv(Sigma_sum) @ diff)
    return det_ratio * exp_term

def KMeans(X, n, d, K):
    """
    Initialize weights, means, and covariances via k-means clustering.
    """
    try:
        from sklearn.cluster import KMeans as SKKMeans
        kmeans = SKKMeans(n_clusters=K, random_state=0).fit(X)
        labels = kmeans.labels_
        means = kmeans.cluster_centers_
    except ImportError:
        indices = np.random.choice(n, K, replace=False)
        means = X[indices]
        labels = np.zeros(n, dtype=int)
    weights = np.zeros(K)
    covariances = np.zeros((K, d, d))
    for k in range(K):
        cluster_data = X[labels == k]
        if cluster_data.shape[0] < 2:
            weights[k] = 1.0 / K
            covariances[k] = np.eye(d)
        else:
            weights[k] = cluster_data.shape[0] / n
            cov = np.cov(cluster_data, rowvar=False, ddof=0) + 1e-6 * np.eye(d)
            covariances[k] = cov if d > 1 else cov.reshape(1, 1)
    return weights, means, covariances

def compute_mmd_closed_form(X, weights, means, covariances, gamma):
    """
    Compute the squared RKHS distance (MMD^2) between the empirical distribution of X 
    and the Gaussian mixture Q defined by weights, means, and covariances, using closed-form expressions.
    """
    means=np.asarray(means)
    covariances=np.asarray(covariances)
    n, d = X.shape
    Gamma = gamma**2 * np.eye(d)
    # Term 3: Empirical term.
    K_XX = gaussian_kernel(X, X, gamma)
    term3 = K_XX.sum() / (n * n)
    
    # Term 1: Mixture-Mixture term.
    K_comp = len(weights)
    term1 = 0.0
    
    for k in range(K_comp):
        for l in range(K_comp):
            I_kl = compute_I(means[k], covariances[k], means[l], covariances[l], Gamma)
            term1 += weights[k] * weights[l] * I_kl
            
    # Term 2: Mixture-Empirical cross term.
    term2 = 0.0
    for i in range(n):
        for k in range(K_comp):
            J_ki = compute_J(means[k], covariances[k], X[i], Gamma)
            term2 += weights[k] * J_ki
    term2 = (2.0 / n) * term2
    
    return term1 - term2 + term3

def init_params(X, K, init_w=None, init_means=None, init_covs=None, t_steps=1):
    """Initialize parameters using provided values or KMeans clustering."""
    if init_means is not None and init_w is not None and init_covs is not None:
        return init_w.copy(), init_means.copy(), init_covs.copy()
    else:
        # KMeans should return weights, means, covariances.
        w, means, covs = KMeans(X, *X.shape, K)
        w = [w.copy() for _ in range(t_steps)]
        return w, means, covs

def update_local_weights(X, means, covs, Gamma, lambda_ridge, w_old):
    """
    Update local weights using quadratic programming.

    Args:
        X (ndarray): Data for a single dataset (n_samples, d).
        means (list): List of global cluster means (each a vector of length d).
        covs (list): List of global cluster covariances (each a (d,d) matrix or a scalar for 1D).
        Gamma (ndarray): A d x d matrix (typically (gamma**2)*I).
        lambda_ridge: Regularization parameter for each weight, provided as a list/number.
                      The list can be of length 1 (apply same value to all weights) or length K (each weight gets its own value).
        w_old (ndarray): Initial weight vector (expected length K).

    Returns:
        w_new (ndarray): Updated and normalized weight vector (length K).
    """
    K = len(means)
    n = X.shape[0]
    
    # Ensure w_old is a one-dimensional numpy array.
    w_old = np.asarray(w_old).flatten()
    if w_old.size == 1:
        w_old = np.ones(K) / K
    elif w_old.size != K:
        raise ValueError(f"Expected initial weights of length {K}, but got {w_old.size}.")

    # Compute the interaction matrix I_mat.
    I_mat = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            I_mat[k, l] = compute_I(means[k], covs[k], means[l], covs[l], Gamma)
    
    # Compute the data-dependent vector J_vec.
    J_vec = np.zeros(K)
    for k in range(K):
        J_sum = 0.0
        for x in X:
            J_sum += compute_J(means[k], covs[k], x, Gamma)
        J_vec[k] = J_sum / n  # Average over the dataset.
    
    # Define the quadratic objective function.
    # The ridge term is computed elementwise:
    #   If lambda_ridge is a number or has length 1: reg = lambda_ridge[0] * sum(w**2)
    #   Else if length K: reg = sum_{i=1}^K lambda_ridge[i] * (w_i)**2
    def objective(w):
        w = np.asarray(w)  # Ensure w is a numpy array.
        if isinstance(lambda_ridge, list):
            if len(lambda_ridge) == 1:
                reg = lambda_ridge[0] * np.sum(w**2)
            elif len(lambda_ridge) == K:
                reg = np.sum(np.array(lambda_ridge) * (w**2))
            else:
                raise ValueError(f"lambda_ridge length must be 1 or K={K}, got {len(lambda_ridge)}")
        else:
            reg = lambda_ridge * np.sum(w**2)
        return w @ I_mat @ w - 2 * (w @ J_vec) + reg
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * K

    res = minimize(objective, w_old, method='SLSQP', constraints=constraints, bounds=bounds)
    if res.success:
        w_new = np.maximum(res.x, 0)  # Clip negatives.
        w_new /= np.sum(w_new)        # Normalize.
        
        return w_new
    else:
        print("Local weight optimization failed. Keeping previous weights.")
        return w_old

def grad_update(X, w, means, covs, gamma, grad_steps, lr, device, eps_val=1e-2):
    """
    M-step (gradient): update global means and covariances using gradient descent (PyTorch).
    
    X is a list of datasets (one per time instant) and w is a list of local weight vectors.
    
    The loss minimized is the sum of the MMD losses over all datasets:
      L = \sum_{i} MMD^2(P_{X^{(i)}}, \sum_{k=1}^{K} \alpha^{(i)}_k N(\mu_k, \Sigma_k)),
    where in the static case the sum has a single term.
    """
    import torch

    # Use first dataset to get dimensionality.
    d = X[0].shape[1]
    K_comp = len(means)
    
    # Prepare tensors for global means.
    means_t = torch.tensor(means, dtype=torch.float32, device=device, requires_grad=True)
    # Prepare Cholesky factors for covariances.
    A_list = []
    for k in range(K_comp):
        cov_k = covs[k].copy()
        np.fill_diagonal(cov_k, np.maximum(np.diag(cov_k), 1e-3))
        try:
            L = np.linalg.cholesky(cov_k - 1e-6 * np.eye(d))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov_k + 1e-6 * np.eye(d))
        A_k = torch.tensor(L, dtype=torch.float32, device=device, requires_grad=True)
        A_list.append(A_k)
    
    params = [means_t] + A_list
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    Gamma_t = (gamma**2) * torch.eye(d, device=device)
    
    # Define closed-form functions for MMD terms.
    def closed_form_I(mu_k, cov_k, mu_l, cov_l, Gamma):
        Sigma_sum = cov_k + cov_l + Gamma
        L = torch.linalg.cholesky(Sigma_sum)
        inv_Sigma = torch.cholesky_inverse(L)
        det_Sigma = torch.prod(torch.diag(L))**2
        det_ratio = torch.sqrt(torch.det(Gamma)) / torch.sqrt(det_Sigma)
        diff = mu_k - mu_l
        exp_term = torch.exp(-0.5 * (diff.unsqueeze(0) @ inv_Sigma @ diff.unsqueeze(1)).squeeze())
        return det_ratio * exp_term

    def closed_form_J(mu, cov, X, Gamma):
        Sigma_sum = cov + Gamma
        L = torch.linalg.cholesky(Sigma_sum)
        inv_Sigma = torch.cholesky_inverse(L)
        det_Sigma = torch.prod(torch.diag(L))**2
        det_ratio = torch.sqrt(torch.det(Gamma)) / torch.sqrt(det_Sigma)
        diff = X - mu
        exponents = -0.5 * (diff @ inv_Sigma * diff).sum(dim=1)
        return det_ratio * torch.exp(exponents)
    
    # Unified gradient descent loop over all datasets.
    for _ in range(grad_steps):
        optimizer.zero_grad()
        total_loss = 0.0
        # Loop over each dataset and its corresponding local weight vector.
        for X_i, w_i in zip(X, w):
            X_i_t = torch.tensor(X_i, dtype=torch.float32, device=device)
            n_i = X_i_t.shape[0]
            # Compute kernel term for data.
            X_diff = X_i_t.unsqueeze(1) - X_i_t.unsqueeze(0)
            sq_dists = (X_diff**2).sum(dim=2)
            K_XX = torch.exp(-0.5 * sq_dists / (gamma**2))
            term3 = K_XX.sum() / (n_i * n_i)
            # Convert local weights for this dataset to tensor.
            w_i_t = torch.tensor(w_i, dtype=torch.float32, device=device)
            # Compute term1: model self-interaction.
            term1 = 0.0
            for k in range(K_comp):
                cov_k = A_list[k] @ A_list[k].T + eps_val * torch.eye(d, device=device)
                for l in range(K_comp):
                    cov_l = A_list[l] @ A_list[l].T + eps_val * torch.eye(d, device=device)
                    term1 += w_i_t[k] * w_i_t[l] * closed_form_I(means_t[k], cov_k, means_t[l], cov_l, Gamma_t)
            # Compute term2: cross-interaction with data.
            term2 = 0.0
            for k in range(K_comp):
                cov_k = A_list[k] @ A_list[k].T + eps_val * torch.eye(d, device=device)
                J_k = closed_form_J(means_t[k], cov_k, X_i_t, Gamma_t)
                term2 += w_i_t[k] * J_k.sum()
            term2 = (2.0 / n_i) * term2
            total_loss += (term1 - term2 + term3)
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
    
    # Extract updated parameters.
    new_means = means_t.detach().cpu().numpy()
    new_covs = []
    for A in A_list:
        A_np = A.detach().cpu().numpy()
        new_covs.append(A_np @ A_np.T + eps_val * np.eye(d))
    new_covs = np.array(new_covs)
    return new_means, new_covs
