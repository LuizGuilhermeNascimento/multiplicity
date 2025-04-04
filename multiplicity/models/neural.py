import torch
import torch.nn as nn
import numpy as np
from typing import List
from .base import BasePredictor

class NeuralPredictor(BasePredictor):
    """Neural network based predictor implementation
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output dimensions
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using the neural network
        
        Args:
            x: Input features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples, output_dim)
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            return self.model(x_tensor).numpy()
    
    def update(self, x: np.ndarray, y: np.ndarray, event_mask: np.ndarray, phi: np.ndarray) -> None:
        """Update network weights based on disagreement event
        
        Args:
            x: Input features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples, output_dim)
            event_mask: Binary mask array of shape (n_samples,) indicating disagreement region
            phi: Update direction array of shape (output_dim,)
        """
        self.model.train()
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)
        event_mask_tensor = torch.FloatTensor(event_mask)
        
        # Reshape phi to match output dimensions
        phi = phi.reshape(1, -1)  # Make phi 2D
        phi_tensor = torch.FloatTensor(phi)
        
        def closure():
            self.optimizer.zero_grad()
            pred = self.model(x_tensor)
            # Update predictions in event region
            loss = torch.mean(torch.square(
                pred + phi_tensor * event_mask_tensor.unsqueeze(1) - y_tensor
            ))
            loss.backward()
            return loss
            
        self.optimizer.step(closure) 