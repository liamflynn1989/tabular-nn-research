"""
AMFormer: Arithmetic Feature Interaction Is Necessary for Deep Tabular Learning

Paper: https://arxiv.org/abs/2402.02334
Authors: Yi Cheng, Renjun Hu, Haochao Ying, Xing Shi, Jian Wu, Wei Lin
Venue: AAAI 2024

Key Idea:
    AMFormer introduces parallel additive AND multiplicative attention mechanisms
    for tabular data. The key insight is that arithmetic feature interactions
    (both sums and products of features) are crucial for tabular learning, but
    standard transformers only capture additive interactions through weighted sums.
    
    By adding multiplicative attention (weighted products), AMFormer can naturally
    model interactions like x1 * x2, which are common in tabular data (e.g., 
    price * quantity = revenue in financial data).

Architecture:
    1. Numerical/Categorical Embeddings -> Feature tokens
    2. AMFormer Layers with:
       - Additive Attention: Standard weighted sum (captures x1 + x2 patterns)
       - Multiplicative Attention: Weighted product (captures x1 * x2 patterns)
       - Token Descent: Progressive feature reduction across layers
    3. Optional learnable prompt tokens
    4. MLP head for prediction

HFT/MFT Relevance:
    - Financial features often interact multiplicatively (returns, ratios, spreads)
    - Polynomial relationships between price/volume are naturally captured
    - Token descent handles high-dimensional feature sets efficiently
    - Prompt tokens can capture market regime-specific patterns

Reference: https://github.com/aigc-apps/AMFormer
"""

import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TabularModel


class NumericalEmbeddingAM(nn.Module):
    """
    Embed numerical features into token representations for AMFormer.
    
    Each numerical feature is projected to a d_model dimensional embedding,
    creating a sequence of tokens (one per feature).
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # Per-feature linear projection
        self.embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Numerical features (batch_size, n_features)
            
        Returns:
            Token embeddings (batch_size, n_features, d_model)
        """
        batch_size = x.shape[0]
        embeddings = []
        
        for i, embed in enumerate(self.embeddings):
            feat = x[:, i:i+1]  # (batch, 1)
            embeddings.append(embed(feat))  # (batch, d_model)
            
        # Stack to (batch, n_features, d_model)
        out = torch.stack(embeddings, dim=1)
        return self.dropout(out)


class AdditiveAttention(nn.Module):
    """
    Standard multi-head attention (additive combination).
    
    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    This captures additive feature interactions through weighted sums.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens (batch, seq_len, d_model)
            attn_mask: Optional attention mask
            
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values (additive combination)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(out)
        
        return out, attn_weights


class MultiplicativeAttention(nn.Module):
    """
    Multiplicative attention for capturing product-based feature interactions.
    
    Instead of computing a weighted SUM of values:
        output = Σ (attention_weight_i * value_i)
        
    We compute a weighted PRODUCT:
        output = Π (value_i ^ attention_weight_i)
        
    In log space: log(output) = Σ (attention_weight_i * log(value_i))
    
    This naturally captures multiplicative relationships like:
        - x1 * x2 (product terms)
        - x1 / x2 (ratio terms, with negative weights)
        - x1^a * x2^b (polynomial terms)
        
    Critical for financial data where ratios and products are common.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multiplicative attention via log-space computation.
        
        Args:
            x: Input tokens (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores (same as additive)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # For multiplicative: work in log space
        # First, make values positive and take log
        v_abs = torch.abs(v) + self.eps
        v_sign = torch.sign(v)
        v_log = torch.log(v_abs)
        
        # Weighted sum in log space = log of weighted product
        log_out = torch.matmul(attn_weights, v_log)
        
        # Back to normal space
        out = torch.exp(log_out)
        
        # Handle signs: majority vote weighted by attention
        # This is an approximation; for simplicity, we use the mean sign
        sign_weighted = torch.matmul(attn_weights, v_sign)
        out = out * torch.sign(sign_weighted + self.eps)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(out)
        
        return out, attn_weights


class AMFormerLayer(nn.Module):
    """
    Single AMFormer layer with parallel additive and multiplicative attention.
    
    Architecture:
        x -> LayerNorm -> [Additive Attn || Multiplicative Attn] -> Combine -> Residual
        x -> LayerNorm -> FFN -> Residual
        
    The additive and multiplicative branches run in parallel and are combined
    via learnable mixing weights.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_multiplicative: bool = True,
    ):
        super().__init__()
        
        self.use_multiplicative = use_multiplicative
        
        # Additive attention (standard)
        self.add_attn = AdditiveAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Multiplicative attention (novel)
        if use_multiplicative:
            self.mul_attn = MultiplicativeAttention(d_model, n_heads, dropout)
            # Learnable mixing weight between additive and multiplicative
            self.mix_weight = nn.Parameter(torch.tensor(0.5))
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens (batch, seq_len, d_model)
            
        Returns:
            Output tokens (batch, seq_len, d_model)
        """
        # Attention block with parallel additive + multiplicative
        residual = x
        x_norm = self.norm1(x)
        
        add_out, _ = self.add_attn(x_norm)
        
        if self.use_multiplicative:
            mul_out, _ = self.mul_attn(x_norm)
            # Combine with learnable mixing (sigmoid to keep in [0, 1])
            alpha = torch.sigmoid(self.mix_weight)
            attn_out = alpha * add_out + (1 - alpha) * mul_out
        else:
            attn_out = add_out
            
        x = residual + self.dropout(attn_out)
        
        # FFN block
        residual = x
        x = residual + self.ffn(self.norm2(x))
        
        return x


class TokenDescent(nn.Module):
    """
    Token descent module for progressive feature reduction.
    
    Reduces the number of tokens (features) across layers, allowing the
    model to learn hierarchical feature combinations. This is similar to
    pooling in CNNs but done through learned attention-based aggregation.
    
    Args:
        d_model: Token dimension
        n_tokens_in: Number of input tokens
        n_tokens_out: Number of output tokens (< n_tokens_in)
    """
    
    def __init__(
        self,
        d_model: int,
        n_tokens_in: int,
        n_tokens_out: int,
    ):
        super().__init__()
        
        assert n_tokens_out <= n_tokens_in, "Output tokens must be <= input tokens"
        
        self.n_tokens_in = n_tokens_in
        self.n_tokens_out = n_tokens_out
        
        # Learnable query tokens for aggregation
        self.queries = nn.Parameter(torch.randn(n_tokens_out, d_model) * 0.02)
        
        # Attention for aggregation
        self.attention = nn.MultiheadAttention(
            d_model, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens (batch, n_tokens_in, d_model)
            
        Returns:
            Reduced tokens (batch, n_tokens_out, d_model)
        """
        batch_size = x.shape[0]
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: queries attend to input tokens
        out, _ = self.attention(queries, x, x)
        out = self.norm(out + queries)
        
        return out


class AMFormer(TabularModel):
    """
    AMFormer: Arithmetic Feature Interaction Transformer for Tabular Data
    
    Key innovations:
    1. Parallel additive AND multiplicative attention
    2. Token descent for progressive feature reduction  
    3. Optional prompt tokens for task conditioning
    
    The multiplicative attention is critical for capturing interactions like:
    - price * quantity = revenue
    - bid / ask = spread
    - feature1^2 (polynomial terms)
    
    Example:
        >>> model = AMFormer(d_in=50, d_out=1, d_model=64, n_layers=3)
        >>> x = torch.randn(32, 50)
        >>> out = model(x)  # Shape: (32, 1)
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff_mult: float = 4.0,
        dropout: float = 0.1,
        use_multiplicative: bool = True,
        use_token_descent: bool = True,
        n_prompts: int = 4,
        task: str = "regression",
    ):
        """
        Args:
            d_in: Number of input features
            d_out: Number of output dimensions
            d_model: Model/embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of AMFormer layers
            d_ff_mult: FFN hidden dimension multiplier
            dropout: Dropout rate
            use_multiplicative: Whether to use multiplicative attention
            use_token_descent: Whether to reduce tokens across layers
            n_prompts: Number of learnable prompt tokens
            task: "regression" or "classification"
        """
        super().__init__(d_in, d_out, task)
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_token_descent = use_token_descent
        self.n_prompts = n_prompts
        
        # Numerical embedding
        self.embedding = NumericalEmbeddingAM(d_in, d_model, dropout)
        
        # Learnable prompt tokens (for task conditioning)
        if n_prompts > 0:
            self.prompts = nn.Parameter(torch.randn(n_prompts, d_model) * 0.02)
        else:
            self.prompts = None
            
        # Calculate token counts for each layer if using descent
        n_tokens = d_in + n_prompts
        if use_token_descent:
            # Gradually reduce tokens: e.g., 50 -> 32 -> 16 -> 8
            token_schedule = []
            for i in range(n_layers):
                if i < n_layers - 1:
                    # Reduce by ~half each layer, but keep minimum 4
                    next_tokens = max(n_tokens // 2, 4)
                    token_schedule.append((n_tokens, next_tokens))
                    n_tokens = next_tokens
                else:
                    token_schedule.append((n_tokens, n_tokens))
            self.token_schedule = token_schedule
        else:
            self.token_schedule = [(n_tokens, n_tokens)] * n_layers
            
        # AMFormer layers with optional token descent
        self.layers = nn.ModuleList()
        self.descents = nn.ModuleList()
        
        d_ff = int(d_model * d_ff_mult)
        
        for i, (n_in, n_out) in enumerate(self.token_schedule):
            self.layers.append(
                AMFormerLayer(d_model, n_heads, d_ff, dropout, use_multiplicative)
            )
            if n_out < n_in:
                self.descents.append(TokenDescent(d_model, n_in, n_out))
            else:
                self.descents.append(None)
                
        # Final pooling and head
        final_n_tokens = self.token_schedule[-1][1]
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        
        self.head = nn.Sequential(
            nn.Linear(d_model * final_n_tokens, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_out),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, d_in)
            
        Returns:
            Predictions (batch_size, d_out)
        """
        batch_size = x.shape[0]
        
        # Embed numerical features to tokens
        tokens = self.embedding(x)  # (batch, d_in, d_model)
        
        # Add prompt tokens if using
        if self.prompts is not None:
            prompts = self.prompts.unsqueeze(0).expand(batch_size, -1, -1)
            tokens = torch.cat([prompts, tokens], dim=1)
            
        # Apply AMFormer layers with optional token descent
        for layer, descent in zip(self.layers, self.descents):
            tokens = layer(tokens)
            if descent is not None:
                tokens = descent(tokens)
                
        # Pool and predict
        tokens = self.pool(tokens)  # (batch, n_tokens, d_model)
        tokens_flat = tokens.flatten(1)  # (batch, n_tokens * d_model)
        out = self.head(tokens_flat)  # (batch, d_out)
        
        return out
    
    def get_attention_weights(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get attention weights from all layers for interpretability.
        
        Returns list of (additive_weights, multiplicative_weights) per layer.
        """
        batch_size = x.shape[0]
        tokens = self.embedding(x)
        
        if self.prompts is not None:
            prompts = self.prompts.unsqueeze(0).expand(batch_size, -1, -1)
            tokens = torch.cat([prompts, tokens], dim=1)
            
        all_weights = []
        
        for layer, descent in zip(self.layers, self.descents):
            # Get attention weights from layer
            x_norm = layer.norm1(tokens)
            _, add_weights = layer.add_attn(x_norm)
            
            if layer.use_multiplicative:
                _, mul_weights = layer.mul_attn(x_norm)
            else:
                mul_weights = None
                
            all_weights.append((add_weights, mul_weights))
            
            # Apply layer and descent
            tokens = layer(tokens)
            if descent is not None:
                tokens = descent(tokens)
                
        return all_weights


def create_amformer(
    d_in: int,
    d_out: int,
    size: str = "medium",
    task: str = "regression",
) -> AMFormer:
    """
    Factory function to create AMFormer with preset configurations.
    
    Args:
        d_in: Number of input features
        d_out: Number of output dimensions
        size: Model size ("small", "medium", "large")
        task: "regression" or "classification"
        
    Returns:
        Configured AMFormer model
    """
    configs = {
        "small": {
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 2,
            "d_ff_mult": 2.0,
            "dropout": 0.1,
            "n_prompts": 2,
        },
        "medium": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff_mult": 4.0,
            "dropout": 0.1,
            "n_prompts": 4,
        },
        "large": {
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "d_ff_mult": 4.0,
            "dropout": 0.15,
            "n_prompts": 8,
        },
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    return AMFormer(
        d_in=d_in,
        d_out=d_out,
        task=task,
        use_multiplicative=True,
        use_token_descent=True,
        **configs[size],
    )
