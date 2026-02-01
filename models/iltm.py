"""
iLTM: Integrated Large Tabular Model (Simplified Implementation)

Paper: https://arxiv.org/abs/2511.15941
Authors: David Bonet, Marçal Comajoan Cara, Alvaro Calafell,
         Daniel Mas Montserrat, Alexander G. Ioannidis
Venue: arXiv 2025 (Stanford & UC Santa Cruz)

Key Idea:
    iLTM is an integrated tabular foundation model that unifies:
    1. Tree-derived embeddings (GBDT leaf encodings)
    2. Dimensionality-agnostic representations (random features + PCA)
    3. Meta-trained hypernetwork (generates MLP weights from data)
    4. Retrieval-augmented predictions (soft k-NN)
    
    The original paper pretrains on 1800+ classification datasets and
    achieves state-of-the-art performance by combining the strengths of
    GBDTs and neural networks.

Simplified Implementation:
    This implementation captures the key architectural ideas but does NOT
    include the meta-trained hypernetwork weights from the paper. Instead:
    
    - Tree embeddings: Uses sklearn's GradientBoostingClassifier/Regressor
    - Hypernetwork: Simplified to a conditioning network that modulates
      the main network (not meta-trained)
    - Retrieval: Full implementation of soft k-NN module
    
    For the full pretrained model, see: https://github.com/AI-sandbox/iLTM

HFT/MFT Relevance:
    - Tree embeddings capture regime-like patterns in market data
    - Retrieval finds similar historical market conditions
    - Dimensionality-agnostic representation handles varying feature sets
    - Robust to distribution shift via GBDT inductive biases
    - Ensemble predictions reduce noise impact

Reference: https://github.com/AI-sandbox/iLTM
"""

import math
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from .base import TabularModel


class TreeEmbedding:
    """
    GBDT-based embedding using leaf index one-hot encoding.
    
    From the paper: "We construct a GBDT-based embedding by fitting a GBDT
    on a labeled set. A point falls into exactly one leaf for each tree,
    producing a leaf index. We then define the GBDT embedding function by
    concatenating the one-hot encodings across all trees."
    
    Note: This is a sklearn-based utility, not a PyTorch module.
    Call fit() first, then transform() to get embeddings.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        task: str = "regression",
    ):
        """
        Args:
            n_estimators: Number of boosting rounds (trees)
            max_depth: Maximum depth of each tree
            learning_rate: GBDT learning rate
            task: "regression" or "classification"
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.task = task
        self.gbdt = None
        self.n_leaves_per_tree = None
        self.leaf_offsets = None
        self.embedding_dim = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeEmbedding":
        """
        Fit the GBDT model on training data.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            self
        """
        if self.task == "classification":
            self.gbdt = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            )
        else:
            self.gbdt = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            )
        
        self.gbdt.fit(X, y)
        
        # Compute number of leaves per tree and total embedding dimension
        self.n_leaves_per_tree = []
        for estimator in self.gbdt.estimators_.flatten():
            n_leaves = (estimator.tree_.feature == -2).sum()  # -2 = leaf node
            self.n_leaves_per_tree.append(n_leaves)
        
        # Compute offsets for one-hot encoding
        self.leaf_offsets = np.cumsum([0] + self.n_leaves_per_tree[:-1])
        self.embedding_dim = sum(self.n_leaves_per_tree)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to GBDT leaf embeddings.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Leaf embeddings as sparse binary matrix (n_samples, embedding_dim)
        """
        if self.gbdt is None:
            raise ValueError("TreeEmbedding not fitted. Call fit() first.")
        
        # Get leaf indices for each tree
        # apply() returns (n_samples, n_estimators) with leaf indices
        leaf_indices = self.gbdt.apply(X)  # (n_samples, n_estimators)
        
        # Handle multi-class case (3D array)
        if leaf_indices.ndim == 3:
            # For multi-class, shape is (n_samples, n_estimators, n_classes)
            # Use only first class's tree structure
            leaf_indices = leaf_indices[:, :, 0]
        
        n_samples = X.shape[0]
        
        # Create sparse one-hot encoding
        embeddings = np.zeros((n_samples, self.embedding_dim), dtype=np.float32)
        
        for tree_idx, (offset, n_leaves) in enumerate(
            zip(self.leaf_offsets, self.n_leaves_per_tree)
        ):
            # Convert leaf indices to one-hot
            leaf_idx = leaf_indices[:, tree_idx]
            # Normalize leaf indices to 0-based within each tree
            unique_leaves = np.unique(leaf_idx)
            leaf_map = {old: new for new, old in enumerate(unique_leaves)}
            normalized_idx = np.array([leaf_map.get(l, 0) for l in leaf_idx])
            normalized_idx = np.clip(normalized_idx, 0, n_leaves - 1)
            embeddings[np.arange(n_samples), offset + normalized_idx] = 1.0
        
        return embeddings


class RandomFeatureProjection(nn.Module):
    """
    Random feature expansion followed by PCA-like dimensionality reduction.
    
    From the paper: "We project features into R^r via random features
    approximating the arc-cosine kernel, then reduce to d_main by applying
    PCA to the randomized features."
    
    This creates a dimensionality-agnostic representation regardless of
    the original input dimension.
    """
    
    def __init__(
        self,
        d_out: int = 512,
        n_random_features: int = 4096,
    ):
        """
        Args:
            d_out: Output embedding dimension (d_main in paper)
            n_random_features: Number of random feature projections (r in paper)
        """
        super().__init__()
        self.d_out = d_out
        self.n_random_features = n_random_features
        
        # Random projection matrix - initialized during first forward
        self.register_buffer('omega', None)
        self.register_buffer('mean', None)
        self.register_buffer('components', None)
        self.fitted = False
    
    def fit(self, X: torch.Tensor) -> None:
        """
        Fit the random projection and PCA on training data.
        
        Args:
            X: Training features (n_samples, d_in)
        """
        n_samples, d_in = X.shape
        device = X.device
        
        # Ensure we have enough random features
        n_random = max(self.n_random_features, self.d_out * 2)
        
        # Initialize random projection matrix
        # Variance = 2/r for arc-cosine kernel approximation
        std = math.sqrt(2.0 / n_random)
        omega = torch.randn(d_in, n_random, device=device) * std
        
        # Apply random feature expansion
        random_features = F.relu(X @ omega)  # Arc-cosine kernel approximation
        
        # Compute PCA components
        mean = random_features.mean(dim=0)
        centered = random_features - mean
        
        # SVD for PCA - get min(n_samples, n_random) components
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        
        # Keep top d_out components (or fewer if not enough samples)
        n_components = min(self.d_out, Vh.shape[0])
        components = Vh[:n_components].T  # (n_random, n_components)
        
        # If we have fewer components than d_out, pad with zeros
        if n_components < self.d_out:
            padding = torch.zeros(n_random, self.d_out - n_components, device=device)
            components = torch.cat([components, padding], dim=-1)
        
        # Store fitted parameters
        self.omega = omega
        self.mean = mean
        self.components = components
        self.fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input to fixed-size embedding.
        
        Args:
            x: Input features (batch_size, d_in)
            
        Returns:
            Fixed-size embeddings (batch_size, d_out)
        """
        if not self.fitted:
            raise ValueError("RandomFeatureProjection not fitted. Call fit() first.")
        
        # Ensure omega is on correct device
        if self.omega.device != x.device:
            self.omega = self.omega.to(x.device)
            self.mean = self.mean.to(x.device)
            self.components = self.components.to(x.device)
        
        # Random feature expansion with ReLU (arc-cosine kernel)
        random_features = F.relu(x @ self.omega)
        
        # PCA projection
        centered = random_features - self.mean
        projected = centered @ self.components
        
        # Normalize (as in paper)
        projected = F.layer_norm(projected, [self.d_out])
        
        return projected


class SoftRetrievalModule(nn.Module):
    """
    Retrieval-augmented prediction using soft k-NN.
    
    From the paper: "The main network incorporates a parameter-free retrieval
    mechanism inspired by ModernNCA, which operates as a soft k-nearest
    neighbors in the learned representation space."
    
    Uses cosine similarity for retrieval and weighted label aggregation.
    """
    
    def __init__(
        self,
        d_embedding: int,
        temperature: float = 1.0,
        k_neighbors: int = 64,
    ):
        """
        Args:
            d_embedding: Embedding dimension
            temperature: Temperature for softmax (τ in paper)
            k_neighbors: Number of neighbors for retrieval
        """
        super().__init__()
        self.d_embedding = d_embedding
        self.temperature = temperature
        self.k_neighbors = k_neighbors
    
    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_labels: torch.Tensor,
        n_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute retrieval-based logits.
        
        For classification:
            Returns class logits based on weighted label aggregation
        For regression:
            Returns weighted average of neighbor labels
        
        Args:
            query: Query embeddings (batch_size, d_embedding)
            candidates: Candidate embeddings (n_candidates, d_embedding)
            candidate_labels: Labels for candidates
                - Classification: (n_candidates,) integer labels
                - Regression: (n_candidates,) or (n_candidates, d_out) values
            n_classes: Number of classes (for classification)
            
        Returns:
            Retrieval logits (batch_size, n_classes) or (batch_size, d_out)
        """
        batch_size = query.shape[0]
        n_candidates = candidates.shape[0]
        
        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)
        cand_norm = F.normalize(candidates, dim=-1)
        
        # Compute similarities
        similarities = query_norm @ cand_norm.T  # (batch, n_candidates)
        
        # Top-k selection
        k = min(self.k_neighbors, n_candidates)
        topk_sim, topk_idx = torch.topk(similarities, k, dim=-1)
        
        # Temperature-scaled softmax
        weights = F.softmax(topk_sim / self.temperature, dim=-1)  # (batch, k)
        
        # Aggregate labels
        if n_classes is not None:
            # Classification: weighted voting
            topk_labels = candidate_labels[topk_idx]  # (batch, k)
            
            # One-hot encode labels
            one_hot = F.one_hot(topk_labels, n_classes).float()  # (batch, k, n_classes)
            
            # Weighted aggregation
            logits = torch.einsum('bk,bkc->bc', weights, one_hot)  # (batch, n_classes)
        else:
            # Regression: weighted average
            if candidate_labels.dim() == 1:
                candidate_labels = candidate_labels.unsqueeze(-1)
            
            topk_values = candidate_labels[topk_idx]  # (batch, k, d_out)
            logits = torch.einsum('bk,bkd->bd', weights, topk_values)
        
        return logits


class HyperConditioner(nn.Module):
    """
    Simplified hypernetwork-inspired conditioning module.
    
    The full iLTM uses a meta-trained hypernetwork that generates MLP weights
    from the training data. This simplified version uses FiLM-style modulation
    based on data statistics, providing some of the adaptivity benefits.
    
    Note: For full hypernetwork functionality, load pretrained weights from
    https://github.com/AI-sandbox/iLTM
    """
    
    def __init__(
        self,
        d_embedding: int,
        d_hidden: int,
    ):
        """
        Args:
            d_embedding: Input embedding dimension
            d_hidden: Hidden dimension for modulation
        """
        super().__init__()
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        
        # Conditioning network: computes modulation parameters from data statistics
        self.conditioner = nn.Sequential(
            nn.Linear(d_embedding * 2, d_hidden),  # mean + std
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        
        # Generate scale and shift for FiLM modulation
        self.gamma_net = nn.Linear(d_hidden, d_hidden)
        self.beta_net = nn.Linear(d_hidden, d_hidden)
    
    def compute_conditioning(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditioning parameters from batch statistics.
        
        Args:
            x: Input embeddings (batch_size, d_embedding)
            
        Returns:
            gamma, beta: Scale and shift parameters (d_hidden,)
        """
        # Compute batch statistics
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        
        # Concatenate statistics
        stats = torch.cat([mean, std], dim=-1)  # (d_embedding * 2,)
        
        # Compute modulation
        cond = self.conditioner(stats)
        gamma = self.gamma_net(cond)
        beta = self.beta_net(cond)
        
        return gamma, beta
    
    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM-style modulation.
        
        Args:
            x: Features to modulate (batch_size, d_hidden)
            gamma: Scale parameter (d_hidden,)
            beta: Shift parameter (d_hidden,)
            
        Returns:
            Modulated features (batch_size, d_hidden)
        """
        return x * (1 + gamma) + beta


class iLTM(TabularModel):
    """
    iLTM: Integrated Large Tabular Model (Simplified)
    
    Combines tree-derived embeddings, dimensionality-agnostic representations,
    hypernetwork-inspired conditioning, and retrieval-augmented predictions.
    
    IMPORTANT: This is a simplified implementation that does NOT include the
    meta-trained hypernetwork weights from the paper. For full pretrained model,
    see: https://github.com/AI-sandbox/iLTM
    
    Architecture:
        1. Embedding Stage:
           - Optional GBDT leaf embeddings (tree inductive biases)
           - Robust preprocessing (scaling, one-hot encoding)
           - Random feature expansion + PCA (dimensionality-agnostic)
           
        2. Main Network:
           - MLP with FiLM-style conditioning (simplified hypernetwork)
           - 3 blocks with 512 hidden units (as in paper)
           
        3. Retrieval Module:
           - Soft k-NN using cosine similarity
           - Weighted label aggregation
           - Blended with MLP output via α parameter
    
    Usage:
        # Create model
        >>> model = iLTM(d_in=20, d_out=1)
        
        # Set up with training data (fits GBDT and projection)
        >>> model.setup(X_train_tensor, y_train_tensor)
        
        # Training
        >>> model.train()
        >>> pred = model(x_batch)
        
        # Inference with retrieval
        >>> model.eval()
        >>> model.set_candidates(X_train_tensor, y_train_tensor)
        >>> pred = model(x_test)
    
    Example:
        >>> import torch
        >>> model = iLTM(d_in=20, d_out=1, use_tree_embedding=True)
        >>> 
        >>> # Training data
        >>> X_train = torch.randn(1000, 20)
        >>> y_train = torch.randn(1000)
        >>> 
        >>> # Setup (fits GBDT and random projection)
        >>> model.setup(X_train, y_train)
        >>> 
        >>> # Inference with retrieval
        >>> model.eval()
        >>> model.set_candidates(X_train, y_train)
        >>> X_test = torch.randn(100, 20)
        >>> predictions = model(X_test)  # (100, 1)
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_main: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.1,
        use_tree_embedding: bool = True,
        n_estimators: int = 100,
        retrieval_alpha: float = 0.3,
        retrieval_temperature: float = 1.0,
        k_neighbors: int = 64,
        max_candidates: int = 5000,
        task: str = "regression",
        n_classes: Optional[int] = None,
    ):
        """
        Args:
            d_in: Number of input features
            d_out: Number of output dimensions
            d_main: Main hidden dimension (512 in paper)
            n_blocks: Number of MLP blocks (3 in paper)
            dropout: Dropout rate
            use_tree_embedding: Whether to use GBDT leaf embeddings
            n_estimators: Number of trees for GBDT embedding
            retrieval_alpha: Weight for retrieval predictions (0=MLP only, 1=retrieval only)
            retrieval_temperature: Temperature for retrieval softmax
            k_neighbors: Number of neighbors for retrieval
            max_candidates: Maximum candidates to store
            task: "regression" or "classification"
            n_classes: Number of classes (required for classification)
        """
        super().__init__(d_in, d_out, task)
        
        self.d_main = d_main
        self.n_blocks = n_blocks
        self.use_tree_embedding = use_tree_embedding
        self.n_estimators = n_estimators
        self.retrieval_alpha = retrieval_alpha
        self.k_neighbors = k_neighbors
        self.max_candidates = max_candidates
        self.n_classes = n_classes if task == "classification" else None
        
        # Tree embedding (fitted externally)
        self.tree_embedding = None
        
        # Random feature projection (fitted externally)
        self.projection = RandomFeatureProjection(
            d_out=d_main,
            n_random_features=min(4096, d_in * 100),
        )
        
        # Robust scaler for preprocessing
        self._scaler = None
        
        # Hyper-conditioner (simplified hypernetwork)
        self.conditioner = HyperConditioner(d_main, d_main)
        
        # Main MLP network
        self.input_norm = nn.LayerNorm(d_main)
        
        blocks = []
        for i in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.Linear(d_main, d_main),
                nn.LayerNorm(d_main),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_main)
        self.output = nn.Linear(d_main, d_out)
        
        # Retrieval module
        self.retrieval = SoftRetrievalModule(
            d_embedding=d_main,
            temperature=retrieval_temperature,
            k_neighbors=k_neighbors,
        )
        
        # Candidate storage
        self._candidate_x_list: List[torch.Tensor] = []
        self._candidate_y_list: List[torch.Tensor] = []
        self._n_seen = 0
        self.register_buffer('_candidate_embeddings', None)
        self.register_buffer('_candidate_labels', None)
        
        # Track setup status
        self._is_setup = False
    
    def setup(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """
        Setup the model with training data.
        
        This fits:
        1. GBDT for tree embeddings (if enabled)
        2. Random feature projection + PCA
        3. Robust scaler for preprocessing
        
        Must be called before training or inference.
        
        Args:
            X: Training features (n_samples, d_in)
            y: Training labels (n_samples,) or (n_samples, d_out)
        """
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().flatten()
        
        # Fit robust scaler
        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X_np)
        X_scaled = torch.tensor(X_scaled, dtype=torch.float32, device=X.device)
        
        # Fit tree embedding if enabled
        if self.use_tree_embedding:
            task = "classification" if self.task == "classification" else "regression"
            self.tree_embedding = TreeEmbedding(
                n_estimators=self.n_estimators,
                max_depth=4,
                task=task,
            )
            self.tree_embedding.fit(X_np, y_np)
            
            # Get tree embeddings
            tree_emb = self.tree_embedding.transform(X_np)
            tree_emb = torch.tensor(tree_emb, dtype=torch.float32, device=X.device)
            
            # Concatenate scaled features with tree embeddings
            combined = torch.cat([X_scaled, tree_emb], dim=-1)
        else:
            combined = X_scaled
        
        # Fit random feature projection
        self.projection.fit(combined)
        
        self._is_setup = True
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to input features."""
        if self._scaler is None:
            return x
        
        x_np = x.detach().cpu().numpy()
        x_scaled = self._scaler.transform(x_np)
        return torch.tensor(x_scaled, dtype=torch.float32, device=x.device)
    
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get fixed-size embeddings for input features.
        
        Args:
            x: Input features (batch_size, d_in)
            
        Returns:
            Embeddings (batch_size, d_main)
        """
        # Preprocess
        x_scaled = self._preprocess(x)
        
        # Add tree embeddings if enabled
        if self.use_tree_embedding and self.tree_embedding is not None:
            x_np = x.detach().cpu().numpy()
            tree_emb = self.tree_embedding.transform(x_np)
            tree_emb = torch.tensor(tree_emb, dtype=torch.float32, device=x.device)
            combined = torch.cat([x_scaled, tree_emb], dim=-1)
        else:
            combined = x_scaled
        
        # Project to fixed dimension
        embeddings = self.projection(combined)
        
        return embeddings
    
    def set_candidates(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """
        Set candidate data for retrieval (replaces any accumulated candidates).
        
        Args:
            X: Training features (n_samples, d_in)
            y: Training labels (n_samples,) or (n_samples, d_out)
        """
        # Clear accumulated candidates
        self._candidate_x_list = []
        self._candidate_y_list = []
        self._n_seen = 0
        
        # Compute embeddings
        with torch.no_grad():
            self._candidate_embeddings = self._get_embeddings(X)
            self._candidate_labels = y.flatten() if y.dim() > 1 else y
    
    def clear_candidates(self) -> None:
        """Clear all candidate data."""
        self._candidate_x_list = []
        self._candidate_y_list = []
        self._n_seen = 0
        self._candidate_embeddings = None
        self._candidate_labels = None
    
    def forward(
        self,
        x: torch.Tensor,
        y_for_candidates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional retrieval augmentation.
        
        Args:
            x: Input features (batch_size, d_in)
            y_for_candidates: Labels to accumulate for retrieval (training)
            
        Returns:
            Predictions (batch_size, d_out)
        """
        if not self._is_setup:
            raise ValueError("Model not set up. Call setup(X_train, y_train) first.")
        
        batch_size = x.shape[0]
        
        # Get embeddings
        embeddings = self._get_embeddings(x)  # (batch, d_main)
        
        # Compute conditioning from batch statistics
        gamma, beta = self.conditioner.compute_conditioning(embeddings)
        
        # Main network forward pass
        h = self.input_norm(embeddings)
        
        for i, block in enumerate(self.blocks):
            h = block(h)
            # Apply FiLM modulation after each block
            h = self.conditioner(h, gamma, beta)
        
        h = self.output_norm(h)
        mlp_output = self.output(h)  # (batch, d_out)
        
        # Retrieval augmentation (if candidates available)
        if (self._candidate_embeddings is not None and 
            self._candidate_labels is not None and 
            self.retrieval_alpha > 0):
            
            # Ensure candidates on same device
            if self._candidate_embeddings.device != x.device:
                self._candidate_embeddings = self._candidate_embeddings.to(x.device)
                self._candidate_labels = self._candidate_labels.to(x.device)
            
            # Get penultimate layer embeddings for retrieval
            retrieval_query = h  # Use pre-output embeddings
            
            retrieval_output = self.retrieval(
                retrieval_query,
                self._candidate_embeddings,
                self._candidate_labels,
                n_classes=self.n_classes,
            )
            
            # For regression, retrieval returns values; need to ensure same shape
            if self.task == "regression" and retrieval_output.shape[-1] == 1:
                retrieval_output = retrieval_output.view(batch_size, self.d_out)
            
            # Blend MLP and retrieval outputs
            output = (1 - self.retrieval_alpha) * mlp_output + self.retrieval_alpha * retrieval_output
        else:
            output = mlp_output
        
        return output
    
    def get_nearest_neighbors(
        self,
        x: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get k nearest neighbors for interpretability.
        
        Args:
            x: Input features (batch_size, d_in)
            k: Number of neighbors
            
        Returns:
            indices: Neighbor indices (batch_size, k)
            similarities: Similarity scores (batch_size, k)
            labels: Neighbor labels (batch_size, k)
        """
        if self._candidate_embeddings is None:
            raise ValueError("No candidates set. Call set_candidates() first.")
        
        with torch.no_grad():
            embeddings = self._get_embeddings(x)
            
            # Normalize for cosine similarity
            query_norm = F.normalize(embeddings, dim=-1)
            cand_norm = F.normalize(self._candidate_embeddings, dim=-1)
            
            # Compute similarities
            similarities = query_norm @ cand_norm.T
            
            # Top-k
            topk_sim, topk_idx = torch.topk(similarities, k, dim=-1)
            topk_labels = self._candidate_labels[topk_idx]
            
            return topk_idx, topk_sim, topk_labels


def create_iltm(
    d_in: int,
    d_out: int,
    size: str = "medium",
    task: str = "regression",
    use_tree_embedding: bool = True,
) -> iLTM:
    """
    Factory function to create iLTM with preset configurations.
    
    Args:
        d_in: Number of input features
        d_out: Number of output dimensions
        size: Model size ("small", "medium", "large")
        task: "regression" or "classification"
        use_tree_embedding: Whether to use GBDT embeddings
        
    Returns:
        Configured iLTM model
    """
    configs = {
        "small": {
            "d_main": 256,
            "n_blocks": 2,
            "n_estimators": 50,
            "k_neighbors": 32,
            "retrieval_alpha": 0.2,
            "dropout": 0.1,
        },
        "medium": {
            "d_main": 512,
            "n_blocks": 3,
            "n_estimators": 100,
            "k_neighbors": 64,
            "retrieval_alpha": 0.3,
            "dropout": 0.1,
        },
        "large": {
            "d_main": 512,
            "n_blocks": 3,
            "n_estimators": 200,
            "k_neighbors": 96,
            "retrieval_alpha": 0.4,
            "dropout": 0.15,
        },
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    return iLTM(
        d_in=d_in,
        d_out=d_out,
        task=task,
        use_tree_embedding=use_tree_embedding,
        **configs[size],
    )
