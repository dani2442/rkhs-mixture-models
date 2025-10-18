import copy
import time
import torch
import torch.nn as nn


class Trainer:
    """
    Trainer class for optimizing Neural ODE models to interpolate time series.
    
    Minimizes MSE between predicted and observed trajectories with optional regularization.
    """
    
    def __init__(self, model, optimizer, scheduler, device, loss_str="MSE", 
                 reg_lambda=0.0, print_freq=10, max_epochs=1000, 
                 tol=1e-6, tolrel=1e-4, patience=200):
        """
        Initialize trainer.
        
        Args:
            model: Neural ODE model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler  
            device: Training device
            loss_str: Loss function
            reg_lambda: L2 regularization coefficient
            print_freq: Print frequency for training progress
            max_epochs: Maximum training epochs
            tol: Absolute tolerance for early stopping
            tolrel: Relative tolerance for early stopping
            patience: Epochs to wait before early stopping
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Configure loss function
        if loss_str is None or loss_str == 'MSE':
            self.loss_func = nn.MSELoss()
        elif loss_str == 'L1':
            self.loss_func = nn.L1Loss()
        elif loss_str == 'LogLikelihood':
            # Determine state dimension from model
            self.loss_func = LogLikelihoodLoss(3, 2, device) # TODO
            self.use_log_likelihood = True
        else:
            raise ValueError(f"Unknown loss function: {loss_str}")
        
        self.reg_lambda = reg_lambda
        self.print_freq = print_freq
        self.max_epochs = max_epochs
        self.tol = tol
        self.tolrel = tolrel
        self.patience = patience
        
        self.best_loss = float('inf')
        self.best_state = None
        self.loss_history = []

    def _compute_regularization(self):
        """Compute L2 regularization terms."""
        reg_loss = 0.0
        reg_loss += self.reg_lambda * sum(torch.sum(param ** 2) for param in self.model.parameters())
        return reg_loss

    def _compute_total_loss(self, x_pred, x_obs):
        """Compute total loss including regularization."""
        loss = self.loss_func(x_pred, x_obs)
        reg_loss = self._compute_regularization()
        return loss + reg_loss

    def _should_stop_early(self, current_loss, prev_loss, patience_counter):
        """Check early stopping conditions."""
        # Absolute tolerance
        if current_loss < self.tol:
            return True, f"absolute tolerance ({self.tol})"
        
        # Patience exceeded
        if patience_counter >= self.patience:
            return True, f"no improvement in {self.patience} epochs"
        
        # Relative tolerance
        if prev_loss is not None:
            abs_diff = abs(current_loss - prev_loss)
            rel_diff = abs_diff / (abs(prev_loss) + 1e-12)
            if rel_diff < self.tolrel:
                return True, f"relative tolerance ({self.tolrel})"
        
        return False, None

    def train(self, t_train, x_obs, initial_state, batch_size=32, shuffle=True):
        """
        Train the Neural ODE model using mini-batch learning.
        
        Args:
            t_train: Time points tensor [n_timesteps]
            x_obs: Observed trajectory [n_timesteps, n_samples, state_dim]
            initial_state: Initial state [n_samples, state_dim]
            batch_size: Size of mini-batches (default: 32)
            shuffle: Whether to shuffle data between epochs (default: True)
        
        Returns:
            Best training loss
        """
        self.model.train()
        
        # Move data to device
        t_train = t_train.to(self.device)
        x_obs = x_obs.to(self.device)
        initial_state = initial_state.to(self.device)
        
        n_samples = x_obs.shape[1]  # Number of samples/trajectories
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        prev_loss = None
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            epoch_losses = []
            
            # Create indices for this epoch
            if shuffle:
                indices = torch.randperm(n_samples, device=self.device)
            else:
                indices = torch.arange(n_samples, device=self.device)
            
            # Mini-batch training loop
            for batch_idx in range(n_batches):
                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract mini-batch data
                x_obs_batch = x_obs[:, batch_indices, :]  # [n_timesteps, batch_size, state_dim]
                initial_state_batch = initial_state[batch_indices, :]  # [batch_size, state_dim]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                x_pred_batch = self.model(initial_state_batch, eval_times=t_train)
                batch_loss = self._compute_total_loss(x_pred_batch, x_obs_batch)
                
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_losses.append(batch_loss.item())
            
            # Calculate average epoch loss
            current_loss = sum(epoch_losses) / len(epoch_losses)
            self.loss_history.append(current_loss)
            
            # Update best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % self.print_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} - Loss: {current_loss:.10f} - LR: {lr:.6f} - Batches: {n_batches}")
            
            # Check early stopping
            should_stop, reason = self._should_stop_early(current_loss, prev_loss, patience_counter)
            if should_stop:
                print(f"Early stopping at epoch {epoch} due to {reason}.")
                break
            
            prev_loss = current_loss
        
        # Load best model state
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        fit_time = time.time() - start_time
        print(f"Training completed. Best Loss: {self.best_loss:.10f}")
        print(f"Fitting time: {fit_time:.4f} sec")
        print(f"Total samples: {n_samples}, Batch size: {batch_size}, Batches per epoch: {n_batches}")
        
        return self.best_loss

    def evaluate(self, t_eval, x_obs, initial_state):
        """
        Evaluate model without parameter updates.
        
        Args:
            t_eval: Time points for evaluation
            x_obs: Observed trajectory  
            initial_state: Initial state
        
        Returns:
            Loss value
        """
        self.model.eval()
        with torch.no_grad():
            t_eval = t_eval.to(self.device)
            x_obs = x_obs.to(self.device)
            initial_state = initial_state.to(self.device)
            
            x_pred = self.model(initial_state, eval_times=t_eval)
            total_loss = self._compute_total_loss(x_pred, x_obs)
            
        return total_loss.item()