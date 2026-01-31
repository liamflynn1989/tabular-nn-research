"""
TabR: Tabular Deep Learning Meets Nearest Neighbors

Paper: https://arxiv.org/abs/2307.14338
Authors: Yury Gorishniy et al. (Yandex Research)
Venue: ICLR 2024

Key Idea:
    TabR is a retrieval-augmented model that combines a feed-forward network
    with a k-Nearest-Neighbors-like component. For each input, it retrieves
    similar examples from the training set and uses an attention mechanism
    to aggregate valuable signal from their features and labels.

Architecture:
    1. Input Encoder: Maps features to embeddings
    2. Retrieval: Find k nearest neighbors in embedding space
    3. Context Aggregation: Attention over retrieved neighbors
    4. Prediction: MLP on combined query + context

HFT/MFT Relevance:
    - Finds similar historical market patterns (regime detection)
    - Retrieval provides implicit ensemble-like benefits (noise handling)
    - Can adapt to non-stationarity by weighting recent similar examples
    - Explainable via retrieved neighbors (regulatory compliance)

Reference implementation: https://github.com/yandex-research/tabular-dl-tabr
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TabularModel


class NumericalEmbeddings(nn.Module):
    """
    Embed numerical features using piecewise linear encoding.
    
    Each numerical feature is embedded into a d_embedding dimensional space.
    Uses a simple linear projection followed by ReLU.
    """
    
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        
        # Per-feature linear projection
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize with small values for stability
        d = self.d_embedding
        std = 1.0 / math.sqrt(d)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features)
            
        Returns:
            (batch_size, n_features, d_embedding)
        """
        # x: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        
        # Linear projection per feature
        # weight: (features, d_embedding) -> (1, features, d_embedding)
        # Result: (batch, features, d_embedding)
        out = x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        
        return F.relu(out)


class RetrievalModule(nn.Module):
    """
    Retrieval component that finds k nearest neighbors and aggregates their information.
    
    Key insight from TabR paper: Use attention-like mechanism for retrieval
    instead of hard k-NN, which allows end-to-end gradient flow.
    """
    
    def __init__(
        self,
        d_embedding: int,
        n_features: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        k_neighbors: int = 96,
    ):
        super().__init__()
        self.d_embedding = d_embedding
        self.n_features = n_features
        self.n_heads = n_heads
        self.k_neighbors = k_neighbors
        
        self.d_total = d_embedding * n_features
        self.d_head = self.d_total // n_heads
        
        # Query, Key, Value projections for attention
        self.W_q = nn.Linear(self.d_total, self.d_total, bias=False)
        self.W_k = nn.Linear(self.d_total, self.d_total, bias=False)
        self.W_v = nn.Linear(self.d_total + 1, self.d_total, bias=False)  # +1 for label
        
        # Output projection
        self.W_o = nn.Linear(self.d_total, self.d_total, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
        # Label embedding for regression
        self.label_embedding = nn.Linear(1, d_embedding)
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve and aggregate information from candidates.
        
        Args:
            query_embeddings: (batch_size, n_features, d_embedding)
            candidate_embeddings: (n_candidates, n_features, d_embedding)
            candidate_labels: (n_candidates,)
            
        Returns:
            context: (batch_size, d_total)
        """
        batch_size = query_embeddings.shape[0]
        n_candidates = candidate_embeddings.shape[0]
        
        # Flatten embeddings
        q = query_embeddings.view(batch_size, -1)  # (batch, d_total)
        k = candidate_embeddings.view(n_candidates, -1)  # (n_candidates, d_total)
        
        # Embed labels and concatenate with features for values
        label_emb = self.label_embedding(candidate_labels.unsqueeze(-1))  # (n_cand, d_emb)
        v_input = torch.cat([
            candidate_embeddings.view(n_candidates, -1),
            candidate_labels.unsqueeze(-1)
        ], dim=-1)  # (n_candidates, d_total + 1)
        
        # Project Q, K, V
        q = self.W_q(q)  # (batch, d_total)
        k = self.W_k(k)  # (n_candidates, d_total)
        v = self.W_v(v_input)  # (n_candidates, d_total)
        
        # Compute attention scores
        # (batch, d_total) @ (d_total, n_candidates) -> (batch, n_candidates)
        scores = torch.matmul(q, k.T) / self.scale
        
        # Top-k selection for efficiency (soft attention on top-k)
        if self.k_neighbors < n_candidates:
            topk_scores, topk_indices = torch.topk(scores, self.k_neighbors, dim=-1)
            
            # Gather top-k values
            v_expanded = v.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_cand, d_total)
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.d_total)
            v_topk = torch.gather(v_expanded, 1, topk_indices_expanded)  # (batch, k, d_total)
            
            # Softmax over top-k
            attn_weights = F.softmax(topk_scores, dim=-1)  # (batch, k)
            attn_weights = self.dropout(attn_weights)
            
            # Weighted sum
            context = torch.einsum('bk,bkd->bd', attn_weights, v_topk)
        else:
            # Full attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.matmul(attn_weights, v)
        
        # Output projection
        context = self.W_o(context)
        
        return context


class TabRBlock(nn.Module):
    """
    MLP block with LayerNorm and residual connection.
    """
    
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_in)
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions differ
        self.residual = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        x = self.norm(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x + residual


class TabR(TabularModel):
    """
    TabR: Retrieval-Augmented Tabular Deep Learning
    
    Combines a feed-forward network with k-NN-like retrieval from training data.
    Achieves state-of-the-art performance by leveraging similar training examples.
    
    Key features:
    1. Soft attention-based retrieval (differentiable)
    2. Label information from retrieved neighbors
    3. End-to-end trainable
    4. Online candidate accumulation during training
    
    Usage:
        TabR automatically accumulates candidates during training:
        
        >>> model = TabR(d_in=10, d_out=1)
        >>> # Training: candidates are accumulated from batches
        >>> for x, y in dataloader:
        ...     pred = model(x, y_for_candidates=y)  # Stores candidates
        ...     loss = criterion(pred, y)
        ...     loss.backward()
        
        >>> # Inference: uses accumulated candidates
        >>> model.eval()
        >>> pred = model(x_test)
        
    Example:
        >>> model = TabR(d_in=10, d_out=1, d_embedding=32, n_blocks=2)
        >>> x = torch.randn(32, 10)
        >>> y = torch.randn(32)
        >>> out = model(x, y_for_candidates=y)  # Shape: (32, 1)
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_embedding: int = 32,
        d_block: int = 256,
        n_blocks: int = 2,
        n_heads: int = 8,
        k_neighbors: int = 96,
        dropout: float = 0.1,
        max_candidates: int = 5000,
        task: str = "regression",
    ):
        """
        Args:
            d_in: Number of input features
            d_out: Number of output dimensions
            d_embedding: Embedding dimension per feature
            d_block: Hidden dimension in MLP blocks
            n_blocks: Number of MLP blocks after retrieval
            n_heads: Number of attention heads in retrieval
            k_neighbors: Number of neighbors to retrieve
            dropout: Dropout rate
            max_candidates: Maximum candidates to store (reservoir sampling)
            task: "regression" or "classification"
        """
        super().__init__(d_in, d_out, task)
        
        self.d_embedding = d_embedding
        self.k_neighbors = k_neighbors
        self.d_total = d_embedding * d_in
        self.max_candidates = max_candidates
        
        # Feature embeddings
        self.embeddings = NumericalEmbeddings(d_in, d_embedding)
        
        # Retrieval module
        self.retrieval = RetrievalModule(
            d_embedding=d_embedding,
            n_features=d_in,
            n_heads=n_heads,
            dropout=dropout,
            k_neighbors=k_neighbors,
        )
        
        # MLP blocks for final prediction
        # Input: query features + retrieved context
        mlp_input_dim = self.d_total * 2
        
        blocks = []
        current_dim = mlp_input_dim
        for i in range(n_blocks):
            out_dim = d_block if i < n_blocks - 1 else d_block
            blocks.append(TabRBlock(current_dim, d_block, out_dim, dropout))
            current_dim = out_dim
        
        self.blocks = nn.ModuleList(blocks)
        
        # Output head
        self.output_norm = nn.LayerNorm(d_block)
        self.output_linear = nn.Linear(d_block, d_out)
        
        # Candidate storage - use lists for dynamic accumulation
        self._candidate_x_list: List[torch.Tensor] = []
        self._candidate_y_list: List[torch.Tensor] = []
        self._n_seen = 0
        
        # Cached tensors for inference
        self.register_buffer('_candidate_embeddings', None)
        self.register_buffer('_candidate_y', None)
    
    def set_candidates(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """
        Set candidate training data for retrieval (replaces accumulated candidates).
        
        Args:
            x: Training features (n_samples, d_in)
            y: Training labels (n_samples,) or (n_samples, d_out)
        """
        # Clear accumulated candidates
        self._candidate_x_list = []
        self._candidate_y_list = []
        self._n_seen = 0
        
        # Store new candidates
        y_flat = y.flatten() if y.dim() > 1 else y
        self._add_candidates(x, y_flat)
        self._compile_candidates()
    
    def _add_candidates(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add candidates using reservoir sampling."""
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            self._n_seen += 1
            
            if len(self._candidate_x_list) < self.max_candidates:
                # Haven't filled reservoir yet
                self._candidate_x_list.append(x[i:i+1].detach())
                self._candidate_y_list.append(y[i:i+1].detach())
            else:
                # Reservoir sampling: replace with probability max_candidates/n_seen
                j = torch.randint(0, self._n_seen, (1,)).item()
                if j < self.max_candidates:
                    self._candidate_x_list[j] = x[i:i+1].detach()
                    self._candidate_y_list[j] = y[i:i+1].detach()
    
    def _compile_candidates(self) -> None:
        """Compile candidate lists into tensors and compute embeddings."""
        if len(self._candidate_x_list) == 0:
            return
        
        candidate_x = torch.cat(self._candidate_x_list, dim=0)
        self._candidate_y = torch.cat(self._candidate_y_list, dim=0).flatten()
        
        # Pre-compute embeddings for efficiency
        with torch.no_grad():
            self._candidate_embeddings = self.embeddings(candidate_x)
    
    def clear_candidates(self) -> None:
        """Clear all accumulated candidates."""
        self._candidate_x_list = []
        self._candidate_y_list = []
        self._n_seen = 0
        self._candidate_embeddings = None
        self._candidate_y = None
    
    def forward(
        self,
        x: torch.Tensor,
        y_for_candidates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with retrieval.
        
        During training, pass y_for_candidates to accumulate training samples
        for retrieval. During inference, candidates are used automatically.
        
        Args:
            x: Input tensor (batch_size, d_in)
            y_for_candidates: Optional labels to store for retrieval (training mode)
            
        Returns:
            Output tensor (batch_size, d_out)
        """
        batch_size = x.shape[0]
        
        # During training: accumulate candidates from this batch
        if self.training and y_for_candidates is not None:
            y_flat = y_for_candidates.flatten() if y_for_candidates.dim() > 1 else y_for_candidates
            self._add_candidates(x, y_flat)
            # Periodically compile for retrieval
            if self._n_seen % 500 == 0:
                self._compile_candidates()
        
        # Compute query embeddings
        query_emb = self.embeddings(x)  # (batch, n_features, d_embedding)
        query_flat = query_emb.view(batch_size, -1)  # (batch, d_total)
        
        # Retrieval (if candidates available)
        if self._candidate_embeddings is not None and self._candidate_y is not None:
            # Ensure candidates are on same device
            if self._candidate_embeddings.device != x.device:
                self._candidate_embeddings = self._candidate_embeddings.to(x.device)
                self._candidate_y = self._candidate_y.to(x.device)
            
            context = self.retrieval(
                query_emb,
                self._candidate_embeddings,
                self._candidate_y,
            )  # (batch, d_total)
        else:
            # No candidates: use zero context (falls back to MLP-like behavior)
            context = torch.zeros_like(query_flat)
        
        # Concatenate query and context
        combined = torch.cat([query_flat, context], dim=-1)  # (batch, d_total * 2)
        
        # MLP blocks
        h = combined
        for block in self.blocks:
            h = block(h)
        
        # Output
        h = self.output_norm(h)
        out = self.output_linear(h)
        
        return out
    
    def train(self, mode: bool = True):
        """Override train to compile candidates when switching to eval."""
        if not mode and self.training:  # Switching from train to eval
            self._compile_candidates()
        return super().train(mode)
    
    def get_nearest_neighbors(
        self,
        x: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get k nearest neighbors for interpretability.
        
        Args:
            x: Input tensor (batch_size, d_in)
            k: Number of neighbors to return
            
        Returns:
            indices: Indices of nearest neighbors (batch_size, k)
            distances: Distances to neighbors (batch_size, k)
            labels: Labels of neighbors (batch_size, k)
        """
        if self._candidate_embeddings is None:
            raise ValueError("No candidates set. Call set_candidates first.")
        
        with torch.no_grad():
            query_emb = self.embeddings(x)
            query_flat = query_emb.view(x.shape[0], -1)
            cand_flat = self._candidate_embeddings.view(
                self._candidate_embeddings.shape[0], -1
            )
            
            # Compute distances
            distances = torch.cdist(query_flat, cand_flat)
            
            # Get top-k
            topk_dist, topk_idx = torch.topk(distances, k, dim=-1, largest=False)
            topk_labels = self._candidate_y[topk_idx]
            
            return topk_idx, topk_dist, topk_labels


def create_tabr(
    d_in: int,
    d_out: int,
    size: str = "medium",
    task: str = "regression",
) -> TabR:
    """
    Factory function to create TabR with preset configurations.
    
    Args:
        d_in: Number of input features
        d_out: Number of output dimensions
        size: Model size ("small", "medium", "large")
        task: "regression" or "classification"
        
    Returns:
        Configured TabR model
    """
    configs = {
        "small": {
            "d_embedding": 24,
            "d_block": 128,
            "n_blocks": 2,
            "n_heads": 4,
            "k_neighbors": 64,
            "dropout": 0.1,
        },
        "medium": {
            "d_embedding": 32,
            "d_block": 256,
            "n_blocks": 2,
            "n_heads": 8,
            "k_neighbors": 96,
            "dropout": 0.1,
        },
        "large": {
            "d_embedding": 48,
            "d_block": 512,
            "n_blocks": 3,
            "n_heads": 8,
            "k_neighbors": 128,
            "dropout": 0.15,
        },
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    return TabR(d_in=d_in, d_out=d_out, task=task, **configs[size])
