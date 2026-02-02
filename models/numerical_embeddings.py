"""
On Embeddings for Numerical Features in Tabular Deep Learning

Paper: https://arxiv.org/abs/2203.05556
Authors: Yury Gorishniy, Ivan Rubachev, Artem Babenko (Yandex Research)
Venue: NeurIPS 2022

Key Idea:
    Transforming scalar numerical features to high-dimensional embeddings before
    mixing in the backbone significantly improves tabular neural network performance.
    Two main approaches:
    1. Piecewise Linear Encoding (PLE) - encodes scalars using learnable bins
    2. Periodic Embeddings - uses sin/cos with learnable frequencies
    
Why it works:
    Numerical features often have irregular distributions and complex relationships
    with targets. Embeddings help neural networks capture these patterns, similar to
    how Fourier features help networks learn high-frequency functions.
    
HFT/MFT Relevance:
    - Price/volume data often has multi-modal distributions
    - Technical indicators have non-linear relationships with future returns
    - Embeddings can capture regime-specific patterns (trending vs mean-reverting)
    - Piecewise linear bins can adapt to different price/volume ranges
    
Reference implementation: https://github.com/yandex-research/rtdl-num-embeddings
"""

import math
from typing import Optional, List, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import TabularModel


def compute_bins(
    X: Union[torch.Tensor, np.ndarray],
    n_bins: int = 64,
    tree_kwargs: Optional[dict] = None,
    y: Optional[Union[torch.Tensor, np.ndarray]] = None,
    regression: bool = True,
) -> List[torch.Tensor]:
    """
    Compute bin boundaries for piecewise linear encoding.
    
    Two modes:
    1. Quantile-based (default): Bins based on data distribution
    2. Tree-based: Target-aware bins using decision tree splits (requires y)
    
    Args:
        X: Input data of shape (n_samples, n_features)
        n_bins: Number of bins per feature
        tree_kwargs: If provided, use decision tree to find bins
                    Example: {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
        y: Target values (required for tree-based bins)
        regression: Whether this is a regression task
        
    Returns:
        List of tensors, one per feature, containing bin boundaries
        Each tensor has shape (n_bins + 1,) defining n_bins intervals
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if y is not None and isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    n_features = X.shape[1]
    bins = []
    
    for i in range(n_features):
        feature = X[:, i]
        
        if tree_kwargs is not None and y is not None:
            # Target-aware tree-based bins
            try:
                from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
                
                TreeClass = DecisionTreeRegressor if regression else DecisionTreeClassifier
                tree = TreeClass(max_leaf_nodes=n_bins, **tree_kwargs)
                tree.fit(feature.reshape(-1, 1), y)
                
                # Extract split thresholds
                thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
                thresholds = np.sort(thresholds)
                
                # Add min and max boundaries
                feature_min = feature.min()
                feature_max = feature.max()
                thresholds = np.concatenate([
                    [feature_min - 1e-6],
                    thresholds,
                    [feature_max + 1e-6]
                ])
                bins.append(torch.tensor(thresholds, dtype=torch.float32))
                
            except ImportError:
                # Fall back to quantile-based
                print("Warning: scikit-learn not available, using quantile bins")
                quantiles = np.linspace(0, 1, n_bins + 1)
                bin_edges = np.quantile(feature, quantiles)
                # Ensure strictly increasing
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) < 2:
                    bin_edges = np.array([feature.min() - 1e-6, feature.max() + 1e-6])
                bins.append(torch.tensor(bin_edges, dtype=torch.float32))
        else:
            # Quantile-based bins
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.quantile(feature, quantiles)
            # Ensure strictly increasing
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([feature.min() - 1e-6, feature.max() + 1e-6])
            bins.append(torch.tensor(bin_edges, dtype=torch.float32))
    
    return bins


class PiecewiseLinearEncoding(nn.Module):
    """
    Piecewise Linear Encoding (PLE) for numerical features.
    
    For each feature, the scalar value x is encoded into a vector where each
    component corresponds to a bin. The encoding is computed as:
    
        For bin [b_i, b_{i+1}]:
            - If x <= b_i: 0
            - If x >= b_{i+1}: 1
            - Otherwise: (x - b_i) / (b_{i+1} - b_i)
    
    This creates a sparse, interpretable encoding where the value "activates"
    bins proportionally to where it falls in the input range.
    
    Args:
        bins: List of tensors containing bin boundaries for each feature
              Each tensor has shape (n_bins + 1,) for n_bins bins
    
    Example:
        >>> bins = compute_bins(X_train, n_bins=32)
        >>> encoding = PiecewiseLinearEncoding(bins)
        >>> x = torch.randn(batch_size, n_features)
        >>> encoded = encoding(x)  # Shape: (batch_size, total_bins)
    """
    
    def __init__(self, bins: List[torch.Tensor]):
        super().__init__()
        
        self.n_features = len(bins)
        self.n_bins_per_feature = [len(b) - 1 for b in bins]
        self.total_bins = sum(self.n_bins_per_feature)
        
        # Register bins as buffers (not trainable)
        for i, b in enumerate(bins):
            self.register_buffer(f"bins_{i}", b)
    
    def get_output_size(self) -> int:
        """Return total encoding dimension."""
        return self.total_bins
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode numerical features using piecewise linear encoding.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Encoded tensor of shape (batch_size, total_bins)
        """
        batch_size = x.shape[0]
        encodings = []
        
        for i in range(self.n_features):
            bins = getattr(self, f"bins_{i}")
            feature = x[:, i:i+1]  # (batch, 1)
            
            # Compute bin boundaries
            left = bins[:-1]   # (n_bins,)
            right = bins[1:]   # (n_bins,)
            
            # Expand for broadcasting: (1, n_bins)
            left = left.unsqueeze(0)
            right = right.unsqueeze(0)
            
            # Piecewise linear encoding
            # Where feature falls in each bin
            widths = right - left
            widths = torch.clamp(widths, min=1e-8)  # Avoid division by zero
            
            # Compute encoding: proportion of bin that feature "fills"
            encoding = (feature - left) / widths
            encoding = torch.clamp(encoding, 0.0, 1.0)  # (batch, n_bins)
            
            encodings.append(encoding)
        
        return torch.cat(encodings, dim=1)


class PiecewiseLinearEmbeddings(nn.Module):
    """
    Piecewise Linear Embeddings (PLE + Linear projection).
    
    Combines piecewise linear encoding with learnable linear projections.
    Each bin gets its own embedding, and the final feature embedding is
    the weighted sum of bin embeddings based on the PLE encoding.
    
    Two versions:
    - Version "A": Original paper implementation
    - Version "B": Improved version from TabM paper (recommended)
    
    Args:
        bins: List of tensors containing bin boundaries for each feature
        d_embedding: Output embedding dimension per feature
        activation: Whether to apply ReLU after linear projection
        version: "A" (original) or "B" (improved, default)
    
    Example:
        >>> bins = compute_bins(X_train, n_bins=32)
        >>> embeddings = PiecewiseLinearEmbeddings(bins, d_embedding=24)
        >>> x = torch.randn(batch_size, n_features)
        >>> embedded = embeddings(x)  # Shape: (batch_size, n_features, d_embedding)
    """
    
    def __init__(
        self,
        bins: List[torch.Tensor],
        d_embedding: int,
        activation: bool = False,
        version: Literal["A", "B"] = "B",
    ):
        super().__init__()
        
        self.n_features = len(bins)
        self.d_embedding = d_embedding
        self.activation = activation
        self.version = version
        
        # Encoding layer
        self.encoding = PiecewiseLinearEncoding(bins)
        
        # Per-feature linear projections
        # Each feature has n_bins bins, each bin gets a d_embedding vector
        self.embeddings = nn.ModuleList()
        for n_bins in self.encoding.n_bins_per_feature:
            if version == "A":
                # Original: each bin has its own embedding vector
                self.embeddings.append(nn.Linear(n_bins, d_embedding))
            else:
                # Version B: parameterize as weight matrix per feature
                self.embeddings.append(
                    nn.Linear(n_bins, d_embedding, bias=True)
                )
        
        self._init_weights()
    
    def _init_weights(self):
        for linear in self.embeddings:
            if self.version == "B":
                # Version B initialization (from TabM)
                nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
                if linear.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(linear.bias, -bound, bound)
    
    def get_output_shape(self) -> torch.Size:
        """Return output shape without batch dimension."""
        return torch.Size([self.n_features, self.d_embedding])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed numerical features using piecewise linear embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Embedded tensor of shape (batch_size, n_features, d_embedding)
        """
        batch_size = x.shape[0]
        
        # Get piecewise linear encoding
        encoded = self.encoding(x)  # (batch, total_bins)
        
        # Split encoding by feature and apply per-feature projections
        embeddings = []
        start = 0
        for i, n_bins in enumerate(self.encoding.n_bins_per_feature):
            feature_encoding = encoded[:, start:start + n_bins]  # (batch, n_bins)
            embedding = self.embeddings[i](feature_encoding)     # (batch, d_embedding)
            embeddings.append(embedding)
            start += n_bins
        
        # Stack: (batch, n_features, d_embedding)
        output = torch.stack(embeddings, dim=1)
        
        if self.activation:
            output = F.relu(output)
        
        return output


class PeriodicEmbeddings(nn.Module):
    """
    Periodic Embeddings for numerical features.
    
    Uses sinusoidal functions with learnable frequencies:
        embed(x) = ReLU(Linear(concat(sin(2π * freq * x), cos(2π * freq * x))))
    
    Where freq is a learnable frequency vector. This is inspired by Fourier
    features and positional encodings in Transformers.
    
    Args:
        n_features: Number of input features
        d_embedding: Output embedding dimension per feature
        n_frequencies: Number of frequency components (default: 48)
        frequency_init_scale: Scale for initializing frequencies (default: 0.01)
        lite: If True, share the final linear layer across features (default: False)
    
    Example:
        >>> embeddings = PeriodicEmbeddings(n_features=10, d_embedding=24)
        >>> x = torch.randn(batch_size, n_features)
        >>> embedded = embeddings(x)  # Shape: (batch_size, n_features, d_embedding)
        
    HFT/MFT Note:
        The learnable frequencies can adapt to capture:
        - Intraday patterns (opening/closing effects)
        - Price level sensitivities (round number effects)
        - Volume regime transitions
    """
    
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        lite: bool = False,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.n_frequencies = n_frequencies
        self.lite = lite
        
        # Learnable frequencies: (n_features, n_frequencies)
        # Initialized with small values for training stability
        self.frequencies = nn.Parameter(
            torch.randn(n_features, n_frequencies) * frequency_init_scale
        )
        
        # Linear projection from sin/cos features to embedding
        # Input: n_frequencies * 2 (sin + cos)
        if lite:
            # Shared linear across features (parameter efficient)
            self.linear = nn.Linear(n_frequencies * 2, d_embedding)
        else:
            # Per-feature linear (more expressive)
            self.linear = nn.ModuleList([
                nn.Linear(n_frequencies * 2, d_embedding)
                for _ in range(n_features)
            ])
    
    def get_output_shape(self) -> torch.Size:
        """Return output shape without batch dimension."""
        return torch.Size([self.n_features, self.d_embedding])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed numerical features using periodic functions.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Embedded tensor of shape (batch_size, n_features, d_embedding)
        """
        batch_size = x.shape[0]
        
        # x: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)
        
        # frequencies: (n_features, n_freq) -> (1, n_features, n_freq)
        freqs = self.frequencies.unsqueeze(0)
        
        # Compute angles: 2π * freq * x
        angles = 2 * math.pi * freqs * x  # (batch, n_features, n_freq)
        
        # Compute sin and cos features
        periodic = torch.cat([
            torch.sin(angles),
            torch.cos(angles)
        ], dim=-1)  # (batch, n_features, n_freq * 2)
        
        # Apply linear projection
        if self.lite:
            # Shared linear: (batch, n_features, n_freq*2) -> (batch, n_features, d_embed)
            output = self.linear(periodic)
        else:
            # Per-feature linear
            outputs = []
            for i in range(self.n_features):
                out = self.linear[i](periodic[:, i, :])  # (batch, d_embed)
                outputs.append(out)
            output = torch.stack(outputs, dim=1)  # (batch, n_features, d_embed)
        
        # Apply ReLU activation
        output = F.relu(output)
        
        return output


class MLPPLR(TabularModel):
    """
    MLP with Numerical Embeddings (MLP-PLR from the paper).
    
    This combines a simple MLP backbone with either:
    - Piecewise Linear Embeddings (PLE)
    - Periodic Embeddings
    - Or both!
    
    The key finding of the paper is that these embeddings can make simple MLPs
    competitive with complex architectures like Transformers on tabular data.
    
    Architecture:
        Input -> Embedding -> Flatten -> MLP -> Output
    
    Args:
        d_in: Number of input features
        d_out: Number of output dimensions
        d_embedding: Embedding dimension per feature
        embedding_type: "periodic", "ple", or "both"
        n_blocks: Number of MLP blocks
        d_block: Hidden dimension in MLP
        dropout: Dropout rate
        n_frequencies: Number of frequencies for periodic embedding
        frequency_init_scale: Initialization scale for frequencies
        bins: Precomputed bins for PLE (optional, will compute from init data)
        n_bins: Number of bins if computing automatically
        task: "regression" or "classification"
    
    Example:
        >>> # With periodic embeddings
        >>> model = MLPPLR(d_in=10, d_out=1, embedding_type="periodic")
        >>> 
        >>> # With piecewise linear embeddings (need to compute bins first)
        >>> bins = compute_bins(X_train, n_bins=32)
        >>> model = MLPPLR(d_in=10, d_out=1, embedding_type="ple", bins=bins)
        >>> 
        >>> out = model(x)
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_embedding: int = 24,
        embedding_type: Literal["periodic", "ple", "both"] = "periodic",
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        bins: Optional[List[torch.Tensor]] = None,
        n_bins: int = 64,
        lite: bool = True,
        task: str = "regression",
    ):
        super().__init__(d_in, d_out, task)
        
        self.embedding_type = embedding_type
        self.d_embedding = d_embedding
        self._bins = bins
        self._n_bins = n_bins
        
        # Build embedding layers
        if embedding_type == "periodic":
            self.embedding = PeriodicEmbeddings(
                n_features=d_in,
                d_embedding=d_embedding,
                n_frequencies=n_frequencies,
                frequency_init_scale=frequency_init_scale,
                lite=lite,
            )
            mlp_in = d_in * d_embedding
            
        elif embedding_type == "ple":
            if bins is None:
                # Create placeholder - needs to be set via set_bins() before forward
                self.embedding = None
                mlp_in = d_in * d_embedding
            else:
                self.embedding = PiecewiseLinearEmbeddings(
                    bins=bins,
                    d_embedding=d_embedding,
                    activation=True,
                )
                mlp_in = d_in * d_embedding
                
        elif embedding_type == "both":
            # Combine both embedding types
            self.periodic = PeriodicEmbeddings(
                n_features=d_in,
                d_embedding=d_embedding // 2,
                n_frequencies=n_frequencies,
                frequency_init_scale=frequency_init_scale,
                lite=lite,
            )
            if bins is None:
                self.ple = None
            else:
                self.ple = PiecewiseLinearEmbeddings(
                    bins=bins,
                    d_embedding=d_embedding - d_embedding // 2,
                    activation=True,
                )
            self.embedding = None  # Will concatenate periodic and ple
            mlp_in = d_in * d_embedding
            
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        # Build MLP backbone
        layers = []
        current_d = mlp_in
        
        for _ in range(n_blocks):
            layers.extend([
                nn.Linear(current_d, d_block),
                nn.BatchNorm1d(d_block),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_d = d_block
        
        layers.append(nn.Linear(d_block, d_out))
        
        self.mlp = nn.Sequential(*layers)
    
    def set_bins(self, bins: List[torch.Tensor]):
        """
        Set bins for PLE embedding after initialization.
        
        Call this before training if you didn't provide bins at init.
        
        Args:
            bins: List of bin boundary tensors from compute_bins()
        """
        if self.embedding_type == "ple":
            self.embedding = PiecewiseLinearEmbeddings(
                bins=bins,
                d_embedding=self.d_embedding,
                activation=True,
            )
        elif self.embedding_type == "both":
            d_ple = self.d_embedding - self.d_embedding // 2
            self.ple = PiecewiseLinearEmbeddings(
                bins=bins,
                d_embedding=d_ple,
                activation=True,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
            
        Returns:
            Output tensor of shape (batch_size, d_out)
        """
        batch_size = x.shape[0]
        
        # Apply embeddings
        if self.embedding_type == "periodic":
            embedded = self.embedding(x)  # (batch, n_features, d_embedding)
            
        elif self.embedding_type == "ple":
            if self.embedding is None:
                raise RuntimeError(
                    "PLE bins not set. Call model.set_bins(compute_bins(X_train)) first."
                )
            embedded = self.embedding(x)  # (batch, n_features, d_embedding)
            
        elif self.embedding_type == "both":
            periodic = self.periodic(x)
            if self.ple is None:
                raise RuntimeError(
                    "PLE bins not set. Call model.set_bins(compute_bins(X_train)) first."
                )
            ple = self.ple(x)
            embedded = torch.cat([periodic, ple], dim=-1)
        
        # Flatten: (batch, n_features, d_embedding) -> (batch, n_features * d_embedding)
        embedded = embedded.reshape(batch_size, -1)
        
        # MLP
        return self.mlp(embedded)
    
    def get_frequency_stats(self) -> dict:
        """
        Get statistics about learned frequencies (for periodic embeddings).
        
        Useful for analyzing what patterns the model learned.
        
        Returns:
            Dict with frequency statistics per feature
        """
        if self.embedding_type == "periodic":
            freqs = self.embedding.frequencies.detach()
        elif self.embedding_type == "both":
            freqs = self.periodic.frequencies.detach()
        else:
            return {}
        
        return {
            "mean": freqs.mean().item(),
            "std": freqs.std().item(),
            "min": freqs.min().item(),
            "max": freqs.max().item(),
            "per_feature_mean": freqs.mean(dim=1).tolist(),
        }


def create_mlpplr(
    d_in: int,
    d_out: int,
    embedding_type: str = "periodic",
    size: str = "medium",
    task: str = "regression",
    bins: Optional[List[torch.Tensor]] = None,
) -> MLPPLR:
    """
    Factory function to create MLPPLR with preset configurations.
    
    Args:
        d_in: Number of input features
        d_out: Number of output dimensions
        embedding_type: "periodic", "ple", or "both"
        size: Model size ("small", "medium", "large")
        task: "regression" or "classification"
        bins: Precomputed bins for PLE (optional)
        
    Returns:
        Configured MLPPLR model
    """
    configs = {
        "small": {
            "d_embedding": 16,
            "n_blocks": 2,
            "d_block": 128,
            "dropout": 0.1,
            "n_frequencies": 32,
        },
        "medium": {
            "d_embedding": 24,
            "n_blocks": 3,
            "d_block": 256,
            "dropout": 0.1,
            "n_frequencies": 48,
        },
        "large": {
            "d_embedding": 32,
            "n_blocks": 4,
            "d_block": 512,
            "dropout": 0.15,
            "n_frequencies": 64,
        },
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    return MLPPLR(
        d_in=d_in,
        d_out=d_out,
        embedding_type=embedding_type,
        bins=bins,
        task=task,
        **configs[size],
    )
