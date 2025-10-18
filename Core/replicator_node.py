#  Advantages over Standard NODE:

# Guaranteed simplex constraint: No need for post-hoc projection
# Better numerical stability: The dynamics respect the geometry of the problem
# Theoretical guarantees: Based on well-studied replicator dynamics

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from Utils.NODE_utils import init_weights

activations = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
}

def MLP(in_dim, out_dim, hidden_dim, num_layers, non_linearity):
    """
    Helper function to build an MLP (reused from node.py).
    """
    layers = []
    if num_layers <= 0:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(non_linearity)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(non_linearity)
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class ReplicatorDynamics(nn.Module):
    """
    Neural Replicator Dynamics that ensures the vector field is tangent to the simplex.
    
    The dynamics follow: ẋᵢ = xᵢ(fᵢ(x) - ∑ⱼ xⱼfⱼ(x))
    
    This ensures:
    1. ∑ᵢ ẋᵢ = 0 (tangent to simplex)
    2. If xᵢ = 0, then ẋᵢ ≥ 0 (no escape from boundary)
    """
    def __init__(self, input_dim, hidden_dim, non_linearity='relu', N_layers=2):
        super(ReplicatorDynamics, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.non_linearity = activations[non_linearity]
        
        # MLP to compute fitness functions f(x,t)
        # Takes [x, t] as input and outputs K fitness values
        self.fitness_net = MLP(input_dim + 1, input_dim, hidden_dim, N_layers, self.non_linearity)
        
        # Small constant for numerical stability
        self.eps = 1e-8

    def forward(self, t, x):
        """
        Compute replicator dynamics: ẋᵢ = xᵢ(fᵢ(x,t) - φ(x,t))
        where φ(x,t) = ∑ⱼ xⱼfⱼ(x,t) is the average fitness.
        
        Parameters:
            t (float or torch.Tensor): Current time
            x (torch.Tensor): Current state on simplex, shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Time derivative that keeps x on the simplex
        """
        batch_size = x.shape[0]
        t_tensor = t * torch.ones(batch_size, 1, device=x.device)
        
        # Concatenate state and time
        x_input = torch.cat([x, t_tensor], dim=1)
        
        # Compute fitness functions f(x,t)
        fitness = self.fitness_net(x_input)
        
        # Compute average fitness: φ = ∑ᵢ xᵢfᵢ
        avg_fitness = (x * fitness).sum(dim=1, keepdim=True)
        
        # Replicator equation: ẋᵢ = xᵢ(fᵢ - φ)
        dx_dt = x * (fitness - avg_fitness)
        
        # Additional safety: ensure boundary conditions
        # If xᵢ is very small and fᵢ - φ < 0, set ẋᵢ = 0
        mask = (x < self.eps) & ((fitness - avg_fitness) < 0)
        dx_dt = torch.where(mask, torch.zeros_like(dx_dt), dx_dt)
        
        return dx_dt


class ReplicatorSemiflow(nn.Module):
    """
    Solves the replicator ODE using adjoint method for memory efficiency.
    """
    def __init__(self, dynamics, T=1.0, step_size=0.01, method='rk4'):
        super(ReplicatorSemiflow, self).__init__()
        self.dynamics = dynamics
        self.T = T
        self.dt = step_size
        self.method = method
        self.eps = 1e-8

    def forward(self, x, eval_times=None):
        """
        Solves the replicator ODE starting from initial state x.
        
        Parameters:
            x (torch.Tensor): Initial state on simplex (batch_size, input_dim)
            eval_times (torch.Tensor, optional): Times at which to evaluate the solution
        
        Returns:
            torch.Tensor: Solution trajectory on the simplex
        """
        
        if eval_times is None:
            integration_time = torch.tensor([0.0, self.T]).to(x)
        else:
            integration_time = eval_times.to(x)
        
        # Solve ODE using adjoint method
        trajectory = odeint_adjoint(
            self.dynamics, x, integration_time, 
            method=self.method, options={'step_size': self.dt}
        )
        
        return trajectory if eval_times is not None else trajectory[1]


class ReplicatorNeuralODE(nn.Module):
    """
    Neural ODE with replicator dynamics for probability vector evolution.
    
    This ensures the solution always remains on the probability simplex
    by using dynamics that are tangent to the simplex.
    """
    def __init__(self, input_dim, hidden_dim, non_linearity='relu', N_layers=2,
                 T=1.0, step_size=0.01, method='rk4'):
        """
        Parameters:
            input_dim (int): Dimension of probability vector (K components)
            hidden_dim (int): Hidden layer size for fitness network
            non_linearity (str): Activation function ('tanh', 'relu', 'sigmoid')
            N_layers (int): Number of layers in fitness network
            T (float): Total integration time
            step_size (float): Integration step size
            method (str): ODE solver method ('euler', 'rk4', 'dopri5')
        """
        super(ReplicatorNeuralODE, self).__init__()
        self.input_dim = input_dim
        self.T = T
        self.dt = step_size
        self.method = method
        self.activation_type = non_linearity
        
        # Instantiate replicator dynamics
        self.dynamics = ReplicatorDynamics(input_dim, hidden_dim, non_linearity, N_layers)
        self.flow = ReplicatorSemiflow(self.dynamics, T, step_size, method)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, self.activation_type))
    
    def forward(self, x, eval_times=None):
        """
        Evolve probability vector using replicator dynamics.
        
        Parameters:
            x (torch.Tensor): Initial probability vector
            eval_times (torch.Tensor, optional): Evaluation times
            
        Returns:
            torch.Tensor: Evolved probability vectors
        """
        return self.flow(x, eval_times)
    
    def check_simplex_invariance(self, x, eval_times=None, tolerance=1e-6):
        """
        Verify that the solution remains on the simplex throughout evolution.
        
        Returns:
            dict: Dictionary with max deviation from simplex and boundary violations
        """
        with torch.no_grad():
            trajectory = self.forward(x, eval_times)
            
            # Check sum = 1
            sums = trajectory.sum(dim=-1)
            max_sum_deviation = torch.abs(sums - 1.0).max().item()
            
            # Check non-negativity
            min_value = trajectory.min().item()
            
            # Check boundary behavior
            # For components near 0, check if derivative points inward
            small_mask = trajectory < tolerance
            if small_mask.any() and eval_times is not None:
                # Compute derivatives at boundary
                derivatives = []
                for i in range(len(eval_times)):
                    dx_dt = self.dynamics(eval_times[i], trajectory[i])
                    derivatives.append(dx_dt)
                derivatives = torch.stack(derivatives)
                
                # Check if derivatives at boundary are non-negative
                boundary_violations = (derivatives[small_mask] < -tolerance).sum().item()
            else:
                boundary_violations = 0
            
            return {
                'max_sum_deviation': max_sum_deviation,
                'min_value': min_value,
                'boundary_violations': boundary_violations,
                'is_valid': max_sum_deviation < tolerance and min_value > -tolerance
            }


# Compatibility wrapper to use with existing training infrastructure
class NeuralODE(ReplicatorNeuralODE):
    """
    Alias for ReplicatorNeuralODE to maintain compatibility with existing code.
    """
    pass