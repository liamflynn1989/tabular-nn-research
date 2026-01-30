"""
Feature-aware Modulation for Learning from Temporal Tabular Data

Paper: https://arxiv.org/abs/2512.03678
Authors: Hao-Run Cai, Han-Jia Ye
Venue: NeurIPS 2025

Key Idea:
    Temporal distribution shifts cause concept drift in tabular data.
    This method conditions feature representations on temporal context
    using FiLM-style modulation (Feature-wise Linear Modulation).
    
    By learning time-dependent scale and shift parameters, the model
    can adapt feature statistics across temporal stages while maintaining
    a shared backbone for generalization.

Architecture:
    - Temporal Encoder: Encodes time indices into embeddings
    - FiLM Layers: Modulate hidden representations based on time
    - Backbone: Standard MLP with FiLM-modulated blocks
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TabularModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal information.
    
    Similar to Transformer positional encodings but applied to
    temporal indices in tabular data.
    """
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)
    
    def forward(self, time_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_idx: Tensor of time indices, shape (batch_size,)
            
        Returns:
            Positional encodings, shape (batch_size, d_model)
        """
        return self.pe[time_idx]


class TemporalEncoder(nn.Module):
    """
    Encodes temporal information into a latent representation.
    
    Combines:
    1. Sinusoidal positional encoding (captures periodic patterns)
    2. Learned MLP transformation (captures complex temporal patterns)
    """
    
    def __init__(
        self,
        d_time: int,
        d_hidden: int = 64,
        max_time: int = 10000,
        use_positional: bool = True,
    ):
        """
        Args:
            d_time: Output dimension of temporal encoding
            d_hidden: Hidden dimension of MLP
            max_time: Maximum time index supported
            use_positional: Whether to use sinusoidal encoding
        """
        super().__init__()
        self.d_time = d_time
        self.use_positional = use_positional
        
        if use_positional:
            self.positional = PositionalEncoding(d_hidden, max_time)
            self.mlp = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_time),
            )
        else:
            # Learnable embedding table
            self.embedding = nn.Embedding(max_time, d_time)
    
    def forward(self, time_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_idx: Time indices, shape (batch_size,) 
                     Can be int indices or normalized floats [0, 1]
            
        Returns:
            Temporal encoding, shape (batch_size, d_time)
        """
        if self.use_positional:
            # Handle both integer and float time inputs
            if time_idx.dtype == torch.float:
                # Assume normalized [0, 1] -> scale to indices
                time_idx = (time_idx * (self.positional.pe.shape[0] - 1)).long()
            pos_enc = self.positional(time_idx)
            return self.mlp(pos_enc)
        else:
            return self.embedding(time_idx)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Modulates features using learned scale (gamma) and shift (beta)
    parameters that are conditioned on temporal context.
    
    output = gamma * input + beta
    
    Reference: Perez et al., "FiLM: Visual Reasoning with a General 
    Conditioning Layer", AAAI 2018
    """
    
    def __init__(
        self,
        d_feature: int,
        d_condition: int,
        modulation_type: str = "scale_shift",
    ):
        """
        Args:
            d_feature: Dimension of features to modulate
            d_condition: Dimension of conditioning signal (temporal encoding)
            modulation_type: Type of modulation
                - "scale_shift": gamma * x + beta (full FiLM)
                - "scale": gamma * x (scaling only)
                - "shift": x + beta (shifting only)
        """
        super().__init__()
        self.d_feature = d_feature
        self.modulation_type = modulation_type
        
        # Predict modulation parameters from conditioning signal
        if modulation_type in ["scale_shift", "scale"]:
            self.gamma_net = nn.Linear(d_condition, d_feature)
        if modulation_type in ["scale_shift", "shift"]:
            self.beta_net = nn.Linear(d_condition, d_feature)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize to identity transformation (gamma=1, beta=0)."""
        if hasattr(self, "gamma_net"):
            nn.init.zeros_(self.gamma_net.weight)
            nn.init.ones_(self.gamma_net.bias)
        if hasattr(self, "beta_net"):
            nn.init.zeros_(self.beta_net.weight)
            nn.init.zeros_(self.beta_net.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.
        
        Args:
            x: Features to modulate, shape (batch_size, d_feature)
            condition: Conditioning signal, shape (batch_size, d_condition)
            
        Returns:
            Modulated features, shape (batch_size, d_feature)
        """
        if self.modulation_type == "scale_shift":
            gamma = self.gamma_net(condition)
            beta = self.beta_net(condition)
            return gamma * x + beta
        elif self.modulation_type == "scale":
            gamma = self.gamma_net(condition)
            return gamma * x
        elif self.modulation_type == "shift":
            beta = self.beta_net(condition)
            return x + beta
        else:
            raise ValueError(f"Unknown modulation type: {self.modulation_type}")


class TemporalMLPBlock(nn.Module):
    """
    MLP block with temporal FiLM modulation.
    
    Structure: Linear -> FiLM -> Norm -> Activation -> Dropout
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_time: int,
        dropout: float = 0.0,
        activation: str = "relu",
        modulation_type: str = "scale_shift",
    ):
        super().__init__()
        
        self.linear = nn.Linear(d_in, d_out)
        self.film = FiLMLayer(d_out, d_time, modulation_type)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch_size, d_in)
            time_encoding: Temporal encoding, shape (batch_size, d_time)
            
        Returns:
            Output features, shape (batch_size, d_out)
        """
        x = self.linear(x)
        x = self.film(x, time_encoding)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TemporalTabularModel(TabularModel):
    """
    Tabular model with feature-aware temporal modulation.
    
    Handles temporal distribution shifts by conditioning feature
    representations on temporal context using FiLM modulation.
    
    Example:
        >>> model = TemporalTabularModel(d_in=10, d_out=1, d_time=16)
        >>> x = torch.randn(32, 10)
        >>> time_idx = torch.arange(32)  # Time indices
        >>> out = model(x, time_idx)  # Shape: (32, 1)
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_time: int = 16,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        modulation_type: str = "scale_shift",
        max_time: int = 10000,
        task: str = "regression",
    ):
        """
        Args:
            d_in: Number of input features
            d_out: Number of output dimensions
            d_time: Dimension of temporal encoding
            n_blocks: Number of MLP blocks
            d_block: Hidden dimension
            dropout: Dropout rate
            activation: Activation function
            modulation_type: Type of FiLM modulation
            max_time: Maximum time index supported
            task: "regression" or "classification"
        """
        super().__init__(d_in, d_out, task)
        
        self.d_time = d_time
        
        # Temporal encoder
        self.time_encoder = TemporalEncoder(
            d_time=d_time,
            d_hidden=d_time * 2,
            max_time=max_time,
        )
        
        # Input projection (also modulated)
        self.input_film = FiLMLayer(d_in, d_time, modulation_type)
        
        # Build MLP blocks with temporal modulation
        blocks = []
        current_d = d_in
        
        for _ in range(n_blocks):
            blocks.append(
                TemporalMLPBlock(
                    current_d, d_block, d_time, 
                    dropout, activation, modulation_type
                )
            )
            current_d = d_block
        
        self.blocks = nn.ModuleList(blocks)
        
        # Output layer
        self.output = nn.Linear(d_block, d_out)
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with temporal modulation.
        
        Args:
            x: Input features, shape (batch_size, d_in)
            time_idx: Time indices, shape (batch_size,)
                     If None, uses zeros (no temporal adaptation)
            
        Returns:
            Output, shape (batch_size, d_out)
        """
        batch_size = x.shape[0]
        
        # Handle missing time indices
        if time_idx is None:
            time_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Get temporal encoding
        time_enc = self.time_encoder(time_idx)  # (batch, d_time)
        
        # Modulate input features
        x = self.input_film(x, time_enc)
        
        # Pass through temporally-modulated blocks
        for block in self.blocks:
            x = block(x, time_enc)
        
        # Output
        return self.output(x)
    
    def forward_without_modulation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without temporal modulation (baseline behavior).
        
        Useful for comparing with/without temporal adaptation.
        """
        return self.forward(x, time_idx=None)


class AdaptiveTemporalModel(TemporalTabularModel):
    """
    Extended temporal model with adaptive learning capabilities.
    
    Includes:
    - Feature importance weighting
    - Drift detection mechanism
    - Optional online adaptation
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_time: int = 16,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        task: str = "regression",
    ):
        super().__init__(
            d_in, d_out, d_time, n_blocks, d_block, dropout, task=task
        )
        
        # Feature importance weights (learned)
        self.feature_importance = nn.Parameter(torch.ones(d_in))
        
        # Drift detector: predicts magnitude of temporal shift
        self.drift_detector = nn.Sequential(
            nn.Linear(d_time, d_time // 2),
            nn.ReLU(),
            nn.Linear(d_time // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with feature importance weighting."""
        # Apply learned feature importance
        x = x * F.softmax(self.feature_importance, dim=0)
        return super().forward(x, time_idx)
    
    def estimate_drift(self, time_idx: torch.Tensor) -> torch.Tensor:
        """
        Estimate the magnitude of temporal drift.
        
        Args:
            time_idx: Time indices
            
        Returns:
            Drift magnitude in [0, 1], higher means more drift
        """
        time_enc = self.time_encoder(time_idx)
        return self.drift_detector(time_enc).squeeze(-1)


def create_temporal_model(
    d_in: int,
    d_out: int,
    size: str = "medium",
    task: str = "regression",
) -> TemporalTabularModel:
    """
    Factory function for temporal tabular models.
    
    Args:
        d_in: Number of input features
        d_out: Number of output dimensions  
        size: Model size ("small", "medium", "large")
        task: "regression" or "classification"
    """
    configs = {
        "small": {"n_blocks": 2, "d_block": 128, "d_time": 8, "dropout": 0.1},
        "medium": {"n_blocks": 3, "d_block": 256, "d_time": 16, "dropout": 0.1},
        "large": {"n_blocks": 4, "d_block": 512, "d_time": 32, "dropout": 0.15},
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}")
    
    return TemporalTabularModel(d_in=d_in, d_out=d_out, task=task, **configs[size])
