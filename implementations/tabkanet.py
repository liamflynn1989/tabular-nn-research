"""
TabKANet: Tabular Data Modeling with Kolmogorov-Arnold Network and Transformer

Based on the paper "Revisiting the numerical feature embeddings structure in
neural network-based tabular modelling" (Gao et al., 2025)

Paper: https://arxiv.org/abs/2409.08806
GitHub: https://github.com/AI-thpremed/TabKANet

Key insight: Use KAN (Kolmogorov-Arnold Networks) with learnable B-spline
activation functions to embed numerical features, capturing complex non-linear
relationships that simple linear projections miss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# =============================================================================
# B-Spline and KAN Components
# =============================================================================

class BSplineBasis(nn.Module):
    """
    Compute B-spline basis functions for KAN.

    B-splines are piecewise polynomial functions defined by:
    - A set of knots (breakpoints)
    - A degree k (order k+1)

    Uses the Cox-de Boor recursion formula for efficient computation.
    """

    def __init__(
        self,
        num_splines: int = 8,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Args:
            num_splines: Number of spline basis functions (controls expressiveness)
            spline_order: Order of B-splines (3 = cubic, most common)
            grid_range: Range of the input domain
        """
        super().__init__()
        self.num_splines = num_splines
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Create uniform knot vector with padding for boundary conditions
        num_knots = num_splines + spline_order + 1
        knots = torch.linspace(grid_range[0], grid_range[1], num_knots)
        self.register_buffer('knots', knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions at points x.

        Args:
            x: Input tensor of shape (..., n_features)

        Returns:
            Basis values of shape (..., n_features, num_splines)
        """
        x = x.clamp(self.grid_range[0], self.grid_range[1])
        x = x.unsqueeze(-1)

        knots = self.knots
        bases = ((x >= knots[:-1]) & (x < knots[1:])).float()

        for k in range(1, self.spline_order + 1):
            n_basis = self.num_splines + self.spline_order - k

            left_num = x - knots[:n_basis]
            left_denom = knots[k:k+n_basis] - knots[:n_basis]
            left = left_num / (left_denom + 1e-8) * bases[..., :n_basis]

            right_num = knots[k+1:k+1+n_basis] - x
            right_denom = knots[k+1:k+1+n_basis] - knots[1:1+n_basis]
            right = right_num / (right_denom + 1e-8) * bases[..., 1:1+n_basis]

            bases = left + right

        return bases


class KANLinear(nn.Module):
    """
    A single KAN layer that maps from in_features to out_features.

    For each (input, output) pair, we learn a univariate function using B-splines.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_splines: int = 8,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        base_activation: nn.Module = None,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_splines = num_splines

        self.basis = BSplineBasis(num_splines, spline_order, grid_range)

        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_splines) * scale_noise
        )

        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (scale_base / math.sqrt(in_features))
        )

        self.base_activation = base_activation or nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        basis_values = self.basis(x)

        basis_flat = basis_values.reshape(batch_size, -1)
        weight_flat = self.spline_weight.reshape(self.out_features, -1)

        spline_out = F.linear(basis_flat, weight_flat)
        base_out = F.linear(self.base_activation(x), self.base_weight)

        return spline_out + base_out


class NumericalEmbeddingKAN(nn.Module):
    """
    KAN-based numerical feature embedding module.

    The core innovation of TabKANet: using KAN to embed numerical features
    instead of simple linear projections.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        num_splines: int = 8,
        spline_order: int = 3,
        noise_std: float = 0.0,
    ):
        super().__init__()

        self.num_features = num_features
        self.noise_std = noise_std

        self.batch_norm = nn.BatchNorm1d(num_features)

        self.kan_encoder = KANLinear(
            in_features=num_features,
            out_features=d_model,
            num_splines=num_splines,
            spline_order=spline_order,
        )

        self.feature_embeddings = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        x = self.batch_norm(x)
        global_emb = self.kan_encoder(x)
        feature_emb = self.feature_embeddings.unsqueeze(0) + global_emb.unsqueeze(1)

        return feature_emb


# =============================================================================
# Transformer Components
# =============================================================================

class CategoricalEmbedding(nn.Module):
    """Standard categorical feature embedding with per-feature embedding tables."""

    def __init__(self, num_categories: List[int], d_model: int):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, d_model) for num_cat in num_categories
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model)) for _ in num_categories
        ])

        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = []
        for i, (emb, bias) in enumerate(zip(self.embeddings, self.biases)):
            embedded.append(emb(x[:, i]) + bias)
        return torch.stack(embedded, dim=1)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm for better training stability."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()

        d_ff = d_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# Main TabKANet Model
# =============================================================================

class TabKANet(nn.Module):
    """
    TabKANet: Tabular data modeling with KAN and Transformer.

    Architecture:
    1. Numerical features -> BatchNorm -> KAN Encoder -> Feature Embeddings
    2. Categorical features -> Embedding Tables -> Feature Embeddings
    3. All embeddings -> Transformer Encoder -> Feature Interactions
    4. [CLS] token or pooling -> Classification/Regression Head
    """

    def __init__(
        self,
        num_numerical: int,
        num_categories: Optional[List[int]] = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        n_classes: int = 2,
        task: str = 'classification',
        num_splines: int = 8,
        spline_order: int = 3,
        noise_std: float = 0.0,
        use_cls_token: bool = True,
    ):
        """
        Args:
            num_numerical: Number of numerical features
            num_categories: List of cardinalities for categorical features
            d_model: Embedding dimension for all features
            n_heads: Number of attention heads
            n_layers: Number of Transformer encoder layers
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout rate
            n_classes: Number of output classes (1 for regression)
            task: 'classification' or 'regression'
            num_splines: Number of spline basis functions for KAN
            spline_order: B-spline order (3 = cubic)
            noise_std: Noise to add to numerical features during training
            use_cls_token: Whether to use a [CLS] token for aggregation
        """
        super().__init__()

        self.num_numerical = num_numerical
        self.num_categories = num_categories or []
        self.d_model = d_model
        self.task = task
        self.use_cls_token = use_cls_token

        if num_numerical > 0:
            self.numerical_embedding = NumericalEmbeddingKAN(
                num_features=num_numerical,
                d_model=d_model,
                num_splines=num_splines,
                spline_order=spline_order,
                noise_std=noise_std,
            )
        else:
            self.numerical_embedding = None

        if self.num_categories:
            self.categorical_embedding = CategoricalEmbedding(
                num_categories=self.num_categories,
                d_model=d_model,
            )
        else:
            self.categorical_embedding = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        output_dim = 1 if task == 'regression' or n_classes == 2 else n_classes
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_num: Numerical features of shape (batch, num_numerical)
            x_cat: Categorical features of shape (batch, num_categorical)

        Returns:
            Logits of shape (batch, n_classes) or (batch, 1) for regression
        """
        embeddings = []

        if x_num is not None and self.numerical_embedding is not None:
            num_emb = self.numerical_embedding(x_num)
            embeddings.append(num_emb)

        if x_cat is not None and self.categorical_embedding is not None:
            cat_emb = self.categorical_embedding(x_cat)
            embeddings.append(cat_emb)

        x = torch.cat(embeddings, dim=1)

        if self.use_cls_token:
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.final_norm(x)

        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.head(x)

    def predict(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Make predictions (applies sigmoid/softmax for classification)."""
        logits = self.forward(x_num, x_cat)

        if self.task == 'regression':
            return logits
        elif logits.shape[1] == 1:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
