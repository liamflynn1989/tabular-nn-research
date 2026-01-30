"""
TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling

Paper: https://arxiv.org/abs/2410.24210
Authors: Yury Gorishniy et al. (Yandex Research)
Venue: ICLR 2025

Key Idea:
    TabM efficiently imitates an ensemble of MLPs by using parameter-efficient 
    ensembling (similar to BatchEnsemble). Multiple "virtual" MLPs share most 
    of their parameters but produce diverse predictions through learned 
    scaling factors.
    
Architecture:
    - Shared MLP backbone
    - Per-head scaling factors (r_i and s_i) for each layer
    - Output: average of all head predictions
    
Reference implementation: https://github.com/yandex-research/tabm
"""

import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TabularModel


class TabMLinear(nn.Module):
    """
    Linear layer with BatchEnsemble-style parameter-efficient ensembling.
    
    Instead of K separate weight matrices W_k, we have:
        W_k = W ⊙ (r_k ⊗ s_k^T)
    
    Where:
        - W: Shared weight matrix (d_out, d_in)
        - r_k: Per-head input scaling (d_in,)
        - s_k: Per-head output scaling (d_out,)
        - ⊗: Outer product
        - ⊙: Hadamard (element-wise) product
    
    This gives K "virtual" linear layers with only O(K * (d_in + d_out)) 
    additional parameters instead of O(K * d_in * d_out).
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        
        # Shared weight matrix
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        
        # Per-head scaling factors (initialized around 1)
        self.r = nn.Parameter(torch.empty(n_heads, d_in))   # Input scaling
        self.s = nn.Parameter(torch.empty(n_heads, d_out))  # Output scaling
        
        if bias:
            self.bias = nn.Parameter(torch.empty(d_out))
        else:
            self.register_parameter("bias", None)
        
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize shared weight with Kaiming
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Initialize scaling factors around 1 with small variance
        # This ensures ensemble members start similar and diverge during training
        nn.init.normal_(self.r, mean=1.0, std=0.1)
        nn.init.normal_(self.s, mean=1.0, std=0.1)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_heads, d_in)
               or (batch_size, d_in) if n_heads == 1
               
        Returns:
            Output tensor of shape (batch_size, n_heads, d_out)
        """
        # Handle case where input doesn't have head dimension
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.n_heads, -1)
        
        batch_size = x.shape[0]
        
        # x: (batch, heads, d_in)
        # r: (heads, d_in) -> (1, heads, d_in)
        # Multiply input by per-head input scaling
        x_scaled = x * self.r.unsqueeze(0)  # (batch, heads, d_in)
        
        # Apply shared linear transformation
        # weight: (d_out, d_in)
        # x_scaled: (batch, heads, d_in)
        out = torch.einsum("bhi,oi->bho", x_scaled, self.weight)  # (batch, heads, d_out)
        
        # Apply per-head output scaling
        # s: (heads, d_out) -> (1, heads, d_out)
        out = out * self.s.unsqueeze(0)  # (batch, heads, d_out)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class TabMBlock(nn.Module):
    """
    Single block of TabM: Linear -> BatchNorm -> Activation -> Dropout
    
    Uses TabMLinear for parameter-efficient ensembling.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.linear = TabMLinear(d_in, d_out, n_heads)
        self.norm = nn.BatchNorm1d(d_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_heads, d_in)
            
        Returns:
            (batch_size, n_heads, d_out)
        """
        # Linear with per-head scaling
        out = self.linear(x)  # (batch, heads, d_out)
        
        # BatchNorm expects (batch, features), so we reshape
        batch_size, n_heads, d_out = out.shape
        out = out.reshape(batch_size * n_heads, d_out)
        out = self.norm(out)
        out = out.reshape(batch_size, n_heads, d_out)
        
        # Activation and dropout
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class TabM(TabularModel):
    """
    TabM: Parameter-Efficient Ensemble MLP for Tabular Data
    
    Key features:
    1. Multiple "virtual" MLPs share most parameters
    2. Per-head scaling factors create ensemble diversity
    3. Final prediction is average across heads
    
    This achieves ensemble-like performance with much lower compute/memory.
    
    Example:
        >>> model = TabM(d_in=10, d_out=1, n_blocks=3, d_block=256, n_heads=16)
        >>> x = torch.randn(32, 10)
        >>> out = model(x)  # Shape: (32, 1)
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_blocks: int = 3,
        d_block: int = 256,
        n_heads: int = 16,
        dropout: float = 0.1,
        activation: str = "relu",
        task: str = "regression",
    ):
        """
        Args:
            d_in: Number of input features
            d_out: Number of output dimensions
            n_blocks: Number of MLP blocks (depth)
            d_block: Hidden dimension
            n_heads: Number of ensemble heads (virtual MLPs)
            dropout: Dropout rate
            activation: Activation function ("relu", "gelu", "silu")
            task: "regression" or "classification"
        """
        super().__init__(d_in, d_out, task)
        
        self.n_heads = n_heads
        
        # Build MLP blocks
        blocks = []
        current_d = d_in
        
        for _ in range(n_blocks):
            blocks.append(
                TabMBlock(current_d, d_block, n_heads, dropout, activation)
            )
            current_d = d_block
        
        self.blocks = nn.ModuleList(blocks)
        
        # Output layer (also with per-head scaling)
        self.output = TabMLinear(d_block, d_out, n_heads)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_all_heads: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
            return_all_heads: If True, return predictions from all heads
            
        Returns:
            If return_all_heads:
                Tensor of shape (batch_size, n_heads, d_out)
            Else:
                Tensor of shape (batch_size, d_out) (averaged across heads)
        """
        # Expand input for all heads: (batch, d_in) -> (batch, n_heads, d_in)
        x = x.unsqueeze(1).expand(-1, self.n_heads, -1)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x)
        
        # Output layer
        out = self.output(x)  # (batch, n_heads, d_out)
        
        if return_all_heads:
            return out
        
        # Average across heads
        return out.mean(dim=1)  # (batch, d_out)
    
    def get_ensemble_diversity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity of ensemble predictions.
        
        Useful for understanding how different the virtual MLPs are.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
            
        Returns:
            Standard deviation across heads, shape (batch_size, d_out)
        """
        all_heads = self.forward(x, return_all_heads=True)
        return all_heads.std(dim=1)


def create_tabm(
    d_in: int,
    d_out: int,
    size: str = "medium",
    task: str = "regression",
) -> TabM:
    """
    Factory function to create TabM with preset configurations.
    
    Args:
        d_in: Number of input features
        d_out: Number of output dimensions
        size: Model size ("small", "medium", "large")
        task: "regression" or "classification"
        
    Returns:
        Configured TabM model
    """
    configs = {
        "small": {"n_blocks": 2, "d_block": 128, "n_heads": 8, "dropout": 0.1},
        "medium": {"n_blocks": 3, "d_block": 256, "n_heads": 16, "dropout": 0.1},
        "large": {"n_blocks": 4, "d_block": 512, "n_heads": 32, "dropout": 0.15},
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    return TabM(d_in=d_in, d_out=d_out, task=task, **configs[size])
