"""
Low-dimensional observation encoder for bimanual manipulation.

This module provides specialized encoders for processing low-dimensional observations
in bimanual manipulation tasks, focusing on capturing relationships between dual arms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class LowDimMLPEncoder(nn.Module):
    """
    MLP encoder for low-dimensional observations.
    
    Processes proprioceptive information through a configurable MLP network.
    Data processing is automatically determined by observation types.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 512],
                 output_dim: int = 256,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 **kwargs):
        """
        Args:
            input_dim: Total input dimension (all low-dim observations concatenated)
            hidden_dims: List of hidden layer dimensions
            output_dim: Final output feature dimension
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'swish')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build main MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.main_encoder = nn.Sequential(*layers)
    
    def forward(self, low_dim_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            low_dim_obs: Concatenated low-dimensional observations [B, input_dim]
            
        Returns:
            Encoded features [B, output_dim]
        """
        # Main encoding only (no relation encoding)
        encoded = self.main_encoder(low_dim_obs)
        return encoded
    
    def output_shape(self) -> List[int]:
        """Return the output shape of the encoder."""
        return [self.output_dim]


def create_low_dim_encoder(**kwargs) -> nn.Module:
    """
    Create a low-dimensional MLP encoder.
    
    Args:
        **kwargs: Arguments passed to the LowDimMLPEncoder constructor
        
    Returns:
        LowDimMLPEncoder instance
    """
    return LowDimMLPEncoder(**kwargs)
