
import json
import nbformat

# Load the notebook
with open('notebooks/train_gpm_v4.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Get the cells
cells = nb['cells']

# Cell 2: Parameters
cells[1]['source'] = """
# B = 16 # Batch size
T = 20 # Number of timesteps
F = 12 # Numbers of features
K = 4 # Number of mixture components
r = 128 # Estimate the covariance matrix
R = 128 # Number of eigenvalues
r = 4 # Number of eigenvalues in the mean approximation
dt = 1/(T-1)
ts = torch.linspace(0, 1, T)
#X = torch.randn((B, T, F))

lr = 0.1
lambd_reg = 0

sigma = 5 # Functional RBF
alpha=-1/(2*sigma**2)

sigmas = np.array([1, 0.3, 2, 0.1]) # Mixture model data
assert len(sigmas)==K
"""

# Cell 3: rbf_fkernel
cells[2]['source'] = """
def rbf_fkernel(x1, x2, dt, alpha=-1.0): # (B, T, F) or (B, T)
    # The .sum(-1) is the change to handle F features
    y = dt*torch.square(x1-x2).sum(-1) # (B, B, T)
    norm = (y[..., 0] / 2 + y[..., 1:-1].sum(dim=-1) + y[..., -1] / 2) # (B, B)
    return torch.exp(alpha*norm)

def rbf_kernel(x1, x2, sigma): # (B, F) or (B)
    norm = torch.square(x1-x2) #.sum(-1) # (B, B)
    return torch.exp(-1/(2*sigma**2)*norm)
"""

# Cell 4: Data Generation
cells[3]['source'] = """
t1, t2 = ts[:, None], ts[None, :]

num = np.array([50, 100, 40, 50]) # Num samples for dataset
means_1d = [5*torch.sin(2*ts), torch.zeros(T), 10*torch.cos(4*ts), -10*torch.cos(4*ts)]
means = []
for m in means_1d:
    m_f = torch.zeros(T, F)
    m_f[:, 0] = m
    means.append(m_f)

Xs = []
for i in range(K):
    cov = rbf_kernel(t1, t2, sigma=sigmas[i]) + 1e-5 * torch.eye(T)
    # Sample each feature independently
    samples_f = []
    for f in range(F):
        mean_f = means[i][:, f]
        # The first feature has a non-zero mean, others are zero.
        # All features share the same covariance structure.
        s = MultivariateNormal(mean_f, cov).sample((num[i],))
        samples_f.append(s.unsqueeze(-1))
    Xs.append(torch.cat(samples_f, dim=-1))

X = torch.concatenate(Xs, dim=0)
B, T, F = X.shape
"""

# Cell 5: Plotting
cells[4]['source'] = """
fig, ax = plt.subplots(figsize=(10, 5))
cmap = mpl.colormaps.get_cmap('viridis')
for i in range(len(Xs)):
    # Plot only the first feature
    ax.plot(ts, Xs[i][:, :, 0].detach().numpy().T, alpha=0.5, color=cmap(i/len(Xs)))

legend_elements = [Line2D([0], [0], color=cmap(i/len(Xs)), lw=2, label=f'Group {i}') 
                   for i in range(len(Xs))]
ax.legend(handles=legend_elements)
ax.set_title("Training data (first feature)")
plt.show()
"""

# Cell 6: PCA
cells[5]['source'] = """
X_flat = X.reshape(B, T*F)
mean_func = X_flat.mean(axis=0)
X_centered = X_flat #- mean_func
C = X_centered.T.cov()
# Eigen-decomposition
m_eigvals, m_eigvecs = torch.linalg.eigh(C)

# Sort descending by variance
idx = torch.argsort(m_eigvals, descending=True)
m_eigvals = m_eigvals[idx]
m_eigvecs = m_eigvecs[:, idx]
scores = X_centered @ m_eigvecs

#k = 3  # number of components
X_recon_flat = scores[:, :r] @ m_eigvecs[:, :r].T #+ mean_func
X_recon = X_recon_flat.reshape(B, T, F)
"""

# Cell 7: Plotting reconstruction
cells[6]['source'] = """
cmap = mpl.colormaps['tab10']

plt.figure(figsize=(8, 4))
rand_ind = np.random.choice(X.shape[0], size=5, replace=False)
for i, idx in enumerate(rand_ind):
    color = cmap(i)
    # Plot only the first feature
    plt.plot(ts, X_recon[idx, :, 0], color=color)
    plt.plot(ts, X[idx, :, 0], '--', color=color, alpha=0.6,)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Real vs Reconstructed Time Series (first feature)')
legend_elements = [
    Line2D([0], [0], color='k', lw=2, label='Reconstructed'),
    Line2D([0], [0], color='k', lw=2, ls='--', label='Real')
]
plt.legend(handles=legend_elements, loc='best')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
"""

# Cell 8: get_J and get_I
cells[7]['source'] = """
""" X: [B, T*F]
    m: [K, T*F]
    evals: [K, R]
    evecs: [K, T, R]
    K_math: [K, T, T]
"""

def get_J(X_flat, m_flat, evals, evecs, alpha):
    # p1 is raised to power F
    p1 = torch.prod(1 - 2*alpha*evals, dim=-1)**(-F/2) # [K] 
    
    X = X_flat.reshape(B, T, F)
    m = m_flat.reshape(K, T, F)
    
    a1 = X[:, None, :, :] - m[None, :, :, :] # [B, K, T, F]
    
    proj = torch.einsum('bktf,ktr->bkfr', a1, evecs) # [B, K, F, R] 
    b1 = torch.square(proj) * dt 
    
    c1 = b1 / (1-2*alpha*evals[None, :, None, :]) # [B, K, F, R]
    
    # Sum over R and F
    exp_argument = torch.sum(c1, dim=(-1, -2)) # [B, K]
    d1 = p1[None, :] * torch.exp(alpha*exp_argument) # [B, K]
    
    return torch.mean(d1, dim=0)


def get_I(X, m, evals, evecs, alpha):
    # p2 is raised to power F
    p2 = torch.prod(1 - 2*alpha*(evals[None, :, :] + evals[:, None, :]),dim=-1)**(-F/2)
    
    a2 = (m[None, :, :, :] - m[:, None, :, :]) # [K, K, T, F]

    # The original b2 was problematic. Assuming common eigenvectors for pairs for simplicity
    # This part of the math is likely an approximation and might need revision
    # For now, we project onto each component's eigenvectors and sum
    proj_i = torch.einsum('ijtf,itr->ijfr', a2, evecs)
    proj_j = torch.einsum('ijtf,jtr->ijfr', a2, evecs)
    
    #This is a simplification
    c2_i = torch.square(proj_i)*dt / (1-2*alpha*evals[None, :, None, :])
    c2_j = torch.square(proj_j)*dt / (1-2*alpha*evals[:, None, None, :])
    
    exp_argument = torch.sum(c2_i + c2_j, dim=(-1, -2))

    return p2*torch.exp(alpha*exp_argument)


def MMD(X, I, J, pi):
    E1 = torch.mean(rbf_fkernel(X[None, :, :], X[:, None, :], dt, alpha=alpha))
    E2 = torch.dot(pi,J)
    E3 = torch.dot(pi,I @ pi)
    return E1, E2, E3
"""

# Cell 11: Training
cells[10]['source'] = """
m_scores = nn.Parameter(torch.randn((K,r)))
pi_logits = nn.Parameter(torch.randn((K,)))

# Note that m_eigvecs are from the PCA on the flattened data
m_flat = m_scores @ m_eigvecs[:, :r].T  # [K, T*F]

# We need a different set of eigvecs for the MMD calculation, from the time kernel
kernels = [lambda x1, x2, s=s: rbf_kernel(x1, x2, sigma=s) for s in sigmas]
K_mat = torch.stack([kernel(t1, t2) for kernel in kernels]) * dt # [K, T, T]
evals_t, evecs_t = torch.linalg.eigh(K_mat)
evals_t, evecs_t = evals_t[..., -R:], evecs_t[..., -R:]
evals_t, evecs_t = torch.flip(evals_t, dims=[-1]), torch.flip(evecs_t, dims=[-1])


J = get_J(X_flat, m_flat, evals_t, evecs_t, alpha)
I = get_I(X, m_flat.reshape(K,T,F), evals_t, evecs_t, alpha)
E1, E2, E3 = MMD(X, I, J, torch.softmax(pi_logits, dim=0))
MMD(X, I, J, torch.softmax(pi_logits, dim=0))
"""

# Cell 12: Training loop
cells[11]['source'] = """
optimizer = torch.optim.Adam([m_scores, pi_logits], lr=lr)
pbar = tqdm(range(1000))
for i in pbar:
    optimizer.zero_grad()
    
    m_flat = m_scores @ m_eigvecs[:, :r].T  # [K, T*F]
    m = m_flat.reshape(K, T, F)
    
    J = get_J(X_flat, m_flat, evals_t, evecs_t, alpha)
    I = get_I(X, m, evals_t, evecs_t, alpha)
    
    pi = torch.softmax(pi_logits, dim=0)
    E1, E2, E3 = MMD(X, I, J, pi)
    
    loss_reg = m_scores.square().mean()
    loss = (E1 - 2*E2 + E3) + loss_reg*lambd_reg
    
    loss.backward()
    optimizer.step()
    pbar.set_description(f"Total loss: {loss.item():.4f}, E1: {E1.item():.4f}, E2: {E2.item():.4f}, E3: {E3.item():.4f}, loss_reg: {loss_reg.item():.4f}")
"""

# Cell 13: cost_matrix
cells[12]['source'] = """
cost_matrix = torch.zeros((K, K))
m_learned = m_scores @ m_eigvecs[:, :r].T.reshape(K, T, F)
for i in range(K):
    for j in range(K):
        # Compare only the first feature for cost matrix
        cost_matrix[i, j] = torch.mean((means[i][:,0] - m_learned[j][:,0])**2).item()

row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
for true_idx, learned_idx in zip(row_ind, col_ind):
    mse = cost_matrix[true_idx, learned_idx].item()
"""

# Cell 15: Plot predicted means
cells[14]['source'] = """
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
cmap = mpl.colormaps.get_cmap('viridis')
m_learned = m_scores @ m_eigvecs[:, :r].T.reshape(K, T, F)

# Plot training data (first feature)
for i in range(len(Xs)):
    ax.plot(ts, Xs[i][:, :, 0].detach().numpy().T, alpha=0.3, color=cmap(i/len(Xs)), linewidth=1)

# Plot predicted means (first feature)
for i in range(K):
    ax.plot(ts, m_learned[col_ind[i], :, 0].detach().numpy(), color=cmap(i/K), lw=3, label=f'Predicted {i}', linestyle='--')

ax.legend()
ax.set_title("Training data and Predicted means (first feature)")
plt.show()
"""

# Cell 16: pi comparison
cells[15]['source'] = """
# 3. Compara pi
reordered_pi = pi.detach().numpy()[col_ind]
true_pi = num/np.sum(num)

print(f"Estimated pi (reordered): {reordered_pi}")
print(f"True pi:                  {true_pi}")
print(f"Aboslute Error:           {np.abs(reordered_pi - true_pi)}")
print(f"Mean Absolute Error:           {np.mean(np.abs(reordered_pi - true_pi)):.6f}")

# 4. Visualiza
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
m_learned = m_scores @ m_eigvecs[:, :r].T.reshape(K, T, F)
for i in range(K):
    # Plot first feature
    axes[i].plot(ts, means[i][:,0].numpy(), 'k-', lw=3, label='True', alpha=0.7)
    learned_idx = col_ind[i]
    axes[i].plot(ts, m_learned[learned_idx, :, 0].detach().numpy(), 'r--', lw=2, label='Learned')
    axes[i].set_title(f'Grupo {i} (pi_true={true_pi[i]:.3f}, pi_est={reordered_pi[i]:.3f})')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim((-10,10))

plt.tight_layout()
plt.show()
"""

# Filter out the cells that are not code or markdown
b['cells'] = [c for c in cells if c.cell_type in ['code', 'markdown']]

# Write the modified notebook to a new file
with open('notebooks/train_gpm_v4_generalized.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Notebook generalized and saved to notebooks/train_gpm_v4_generalized.ipynb")
