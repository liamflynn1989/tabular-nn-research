"""
Base classes for tabular neural network models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class TabularModel(nn.Module, ABC):
    """
    Abstract base class for tabular neural network models.
    
    All implementations should inherit from this class and implement
    the forward method.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        task: str = "regression",
    ):
        """
        Args:
            d_in: Number of input features
            d_out: Number of output dimensions
            task: "regression" or "classification"
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.task = task
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
            
        Returns:
            Output tensor of shape (batch_size, d_out)
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (applies softmax for classification).
        """
        out = self.forward(x)
        if self.task == "classification":
            return torch.softmax(out, dim=-1)
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NumericalEmbedding(nn.Module):
    """
    Embed numerical features into a higher-dimensional space.
    
    Options:
    - Linear: Simple linear projection
    - Periodic: Periodic embeddings (sin/cos)
    - PLR: Piecewise Linear Encoding
    """
    
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        embedding_type: str = "linear",
        n_frequencies: int = 48,
    ):
        """
        Args:
            n_features: Number of input features
            d_embedding: Embedding dimension per feature
            embedding_type: "linear", "periodic", or "plr"
            n_frequencies: Number of frequencies for periodic embedding
        """
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.embedding_type = embedding_type
        
        if embedding_type == "linear":
            self.embedding = nn.Linear(1, d_embedding)
        elif embedding_type == "periodic":
            self.n_frequencies = n_frequencies
            # Learnable frequencies
            self.frequencies = nn.Parameter(
                torch.randn(n_features, n_frequencies)
            )
            self.linear = nn.Linear(n_frequencies * 2, d_embedding)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Embedded tensor of shape (batch_size, n_features, d_embedding)
        """
        if self.embedding_type == "linear":
            # (batch, features) -> (batch, features, 1) -> (batch, features, d_embedding)
            return self.embedding(x.unsqueeze(-1))
        
        elif self.embedding_type == "periodic":
            # Periodic embedding with learnable frequencies
            # x: (batch, features) -> (batch, features, 1)
            x = x.unsqueeze(-1)
            # frequencies: (features, n_freq) -> (1, features, n_freq)
            freqs = self.frequencies.unsqueeze(0)
            # Compute sin and cos features
            angles = 2 * torch.pi * x * freqs  # (batch, features, n_freq)
            periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            return self.linear(periodic)


class MLPBlock(nn.Module):
    """
    Standard MLP block with optional normalization and dropout.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: Optional[str] = "batchnorm",
    ):
        super().__init__()
        
        layers = [nn.Linear(d_in, d_out)]
        
        if normalization == "batchnorm":
            layers.append(nn.BatchNorm1d(d_out))
        elif normalization == "layernorm":
            layers.append(nn.LayerNorm(d_out))
        
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "silu":
            layers.append(nn.SiLU())
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLP(TabularModel):
    """
    Simple MLP baseline for tabular data.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        task: str = "regression",
    ):
        super().__init__(d_in, d_out, task)
        
        layers = []
        current_d = d_in
        
        for _ in range(n_blocks):
            layers.append(MLPBlock(current_d, d_block, dropout))
            current_d = d_block
        
        layers.append(nn.Linear(d_block, d_out))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
