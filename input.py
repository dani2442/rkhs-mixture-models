from data import get_distrparams
from Utils.IO_utils import validate_params, set_base_folder


def get_params():
    options = {
        "gif": True,                        # Generate GIF of trajectories
        "gif_error": False,                  # Generate GIF of error trajectories
        "load_cached_fit": True,            # Load cached data in fitting global parameters if available
        "load_cached_node": True,           # Load cached data in fitting neural ode if available
        "load_cached_test": False,           # Load cached data in loss test if available
        "verbose": True,                    # Verbose output during training
        "test": False,                      # Test the model for cross-validation
        "load_data": False,                  # Load data from CSV files instead of generating synthetic data
        "fix_mv": True                    # Fix the mean and variance of the GMMs (by fitting a random subset)
    }
    
    if options["load_data"]:
        options["threshold"] = False           # Discard data with less than 3 blocks
        params_data = {
                    "csv_folder": "csvs",
                    #"ids": [13, 62, 377],
                    "ids": list(range(1, 346)),     # Generate all integers from 1 to 503
                    "block_size": 10,               # number of rows to aggregate
                    "add_deriv": False,             # Add first derivatives to the data
                    "add_second_deriv": False,      # Add second derivatives to the data
                    "T": 1.0                       # Total time for time series experiments; if T > 0, a time series experiment is performed.
                    }
        
    else:
        options["threshold"] = False
        params_data = {
            "distribution": 'gaussian',         # Distribution type: 'gaussian', 'student_t', 'uniform', 'poisson', 
            "d": 2,                             # Dimension of the distribution
            "n_points": [5000],                  # Number of sample points per time step
            "n_clusters": 1,                    # Number of clusters for the Gaussian mixture model
            "seed": [1],         # Seed for the data generation
            "t_steps": 11,                      # Number of time steps
            "T": 1.0,                           # Total time for time series experiments; if T > 0, a time series experiment is performed.
        }

            
    params_optim = {
        "seed_split": 42,                   # Seed for splitting data into train/test sets
        "max_iter": 20,                     # Maximum iterations for MMD-GMM fitting
        "tol_rel_param": 1e-9,              # Relative tolerance for parameter convergence
        "tol_rel_err": 1e-7,                # Relative tolerance for error convergence
        "tol_abs": 1e-6,                    # Absolute tolerance for convergence
        "K": [7],                          # Number of Gaussian components
        "update_method": ['gradient'],      # Options: 'gradient' or 'heuristic'
        "lr": [0.01],                       # Learning rate for the MMD-GMM fitting
        "grad_steps": [10],                 # Number of gradient substeps for each step of the MMD-GMM fitting
        "gamma": [5.0],                     # Bandwidth for the MMD-kernel 
        "lambda_ridge": [[0.01]],            # Regularization parameter. Each value of the list can be a number/list of size 1 
                                            # (same for all K) or a list of size K (different for each K).
    }
    
    # Uncertainty quantification parameters
    params_uq = {
        "n_bags": 1,                        # Number of bagging iterations (>1 enables bagging)
        "alpha_uq": 0.95,                   # Confidence level for uncertainty quantification
        "dist": "L2",                       # Distance metric for band comparison
        "m_sim": 1000                       # Number of simulations for confidence bands
    }   
    
    # Neural ODE parameters
    params_node = {
        "T": 1.0,                 # Total time for the neural ODE
        "dt": 0.01,          # Time step for the neural ODE
        "seed_node": 42,         # Seed for the neural ODE
        "hidden_dim": 256,           # Hidden dimension for the neural ODE
        "non_linearity": "relu",    # Non-linearity for the neural ODE
        "N_layers": 2,              # Number of hidden layers in the neural ODE
        "method": "rk4",            # ODE solver method (e.g., 'rk4', 'dopri5')
        "lr": 1e-3,                 # Learning rate for the neural ODE
        "loss_func": "MSE",         # Loss function for the neural ODE
        "lambda_node": 0.0,        # Regularization parameter for the neural ODE
        "print_freq": 50,       # Frequency of printing training progress   
        "max_epochs": 10000,         # Maximum epochs for training the neural ODE
        "tol_abs": 1e-5,            # Absolute tolerance for convergence
        "tol_rel": 1e-8,        # Relative tolerance for convergence
        "n_MC": 10000,               # Number of Monte Carlo samples for the neural ODE
        "use_replicator": True     # Use replicator dynamics for the neural ODE
    }

    # Validate input parameters
    validate_params(params_data, params_uq, options)      
    
    # Set up result directories
    options, params_data = set_base_folder(
        options, params_data, params_optim, params_uq, params_node, 
        ignore_keys=["load_cached_fit", "load_cached_node", "load_cached_test", 
                    "load_data", "gif", "verbose", "test", 
                    "ids", ("alpha_uq", "dist", "m_sim") if params_uq["n_bags"] > 1 else None]
    )

    # Generate synthetic distribution parameters if not loading external data
    if not options["load_data"]:
        params_data["means"], params_data["vars"] = get_distrparams(
            params_data, params_node["T"], params_data["t_steps"]
        )
            
    return options, params_data, params_optim, params_uq, params_node