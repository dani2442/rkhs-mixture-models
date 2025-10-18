import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from Utils.NODE_utils import init_weights, normalize

# Dictionary of activation functions.
activations = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
}

def MLP(in_dim, out_dim, hidden_dim, num_layers, non_linearity):
    """
    Helper function to build an MLP.
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

class Dynamics(nn.Module):
    """
    Continuous-time dynamics for the Neural ODE: f(t, x) = dx/dt.
    
    Uses time-concatenated input: f([x, t]) -> dx/dt
    """
    def __init__(self, input_dim, hidden_dim, non_linearity='relu', N_layers=2):
        super(Dynamics, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.non_linearity = activations[non_linearity]
        
        # MLP takes [x, t] as input
        self.net = MLP(input_dim + 1, input_dim, hidden_dim, N_layers, self.non_linearity)

    def forward(self, t, x):
        """
        Compute the dynamics f(t, x) = dx/dt.
        
        Parameters:
            t (float or torch.Tensor): Current time.
            x (torch.Tensor): Current state, shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Time derivative of x.
        """
        batch_size = x.shape[0]
        t_tensor = t * torch.ones(batch_size, 1, device=x.device)
        x_input = torch.cat([x, t_tensor], dim=1)
        return self.net(x_input)

class Semiflow(nn.Module):
    """
    Solves the Neural ODE using adjoint method for memory efficiency.
    """
    def __init__(self, dynamics, T=1.0, step_size=0.01, method='rk4'):
        super(Semiflow, self).__init__()
        self.dynamics = dynamics
        self.T = T
        self.dt = step_size
        self.method = method

    def forward(self, x, eval_times=None):
        """
        Solves the ODE starting from initial state x.
        
        Parameters:
            x (torch.Tensor): Initial state (batch_size, input_dim).
            eval_times (torch.Tensor, optional): Times at which to evaluate the solution.
        
        Returns:
            torch.Tensor: The solution at the given time points.
        """
        if eval_times is None:
            integration_time = torch.tensor([0.0, self.T]).to(x)
        else:
            integration_time = eval_times.to(x)
            
        # Use adjoint method for memory efficiency
        out = odeint_adjoint(self.dynamics, x, integration_time, 
                           method=self.method, options={'step_size': self.dt})
        
        # Apply normalization/project on the simplex to ensure weights sum to 1
        out = normalize(out)
        
        return out if eval_times is not None else out[1]

class NeuralODE(nn.Module):
    """
    Neural ODE for continuous-time weight evolution with manual normalization.
    """
    def __init__(self, input_dim, hidden_dim, non_linearity='relu', N_layers=2,
                 T=1.0, step_size=0.01, method='rk4'):
        """
        Parameters:
            input_dim (int): Dimension of the state (number of mixture components).
            hidden_dim (int): Hidden layer size.
            non_linearity (str): Activation function ('tanh', 'relu', 'sigmoid').
            N_layers (int): Number of layers in the MLP.
            T (float): Total integration time.
            step_size (float): Integration step size.
            method (str): Integration method ('euler', 'rk4', 'rk45').
        """
        super(NeuralODE, self).__init__()
        self.input_dim = input_dim
        self.T = T
        self.dt = step_size
        self.method = method
        self.activation_type = non_linearity
        
        # Instantiate the dynamics
        self.dynamics = Dynamics(input_dim, hidden_dim, non_linearity, N_layers)
        self.flow = Semiflow(self.dynamics, T, step_size, method)
        
        # Apply weight initialization
        self.apply(lambda m: init_weights(m, self.activation_type))
            
    def forward(self, x, eval_times=None):
        """
        Performs a forward pass through the Neural ODE.
        
        Parameters:
            x (torch.Tensor): Input state.
            eval_times (torch.Tensor, optional): Evaluation times.
            
        Returns:
            torch.Tensor: The solution with manual normalization applied.
        """
        return self.flow(x, eval_times)