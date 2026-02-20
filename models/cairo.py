"""
CAIRO: Calibrate After Initial Rank Ordering

Implementation of "CAIRO: Decoupling Order from Scale in Regression" (arXiv:2602.14440)

Paper: https://arxiv.org/abs/2602.14440
Authors: Harri Vanhems, Yue Zhao, Peng Shi, Archer Y. Yang (2026)

Key Ideas:
-----------
Standard regression (MSE) conflates learning ordering with learning scale, making models
vulnerable to outliers and heavy-tailed noise. CAIRO decouples these into two stages:

Stage 1: Learn a scoring function by minimizing a scale-invariant ranking loss
  - RankNet: Pairwise loss with uniform weights (maximizes Kendall's τ)
  - RankNet-GiniW: Pairwise loss with |yi - yj| weights (maximizes Gini covariance)
  - GiniNet-SoftRank: Pointwise loss using differentiable softrank (O(n log n))

Stage 2: Recover target scale via isotonic regression
  - Learns a monotone transformation to map scores back to target scale
  - Guaranteed to recover true regression function when Stage 1 is "Optimal in Rank Order"

HFT/MFT Relevance:
------------------
- Financial returns have heavy-tailed distributions (fat tails, outliers)
- Heteroskedastic noise is common (volatility clustering)
- Ranking accuracy often matters more than exact value prediction
- Robust to regime changes where scale shifts but ordering remains stable
- Log-sigmoid saturates for large errors, bounding outlier influence
"""

from typing import Optional, Literal, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from .base import TabularModel, MLPBlock


def softrank(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Differentiable soft rank approximation.
    
    For each element x[i], computes the expected rank by comparing against
    all other elements using a soft comparison via sigmoid.
    
    Complexity: O(n²) naive, but can be optimized to O(n log n) with sorting.
    For simplicity, we use the differentiable pairwise formulation.
    
    Args:
        x: Tensor of shape (batch_size,) or (batch_size, 1)
        tau: Temperature parameter (lower = sharper approximation)
        
    Returns:
        Soft ranks of shape (batch_size,)
    """
    if x.dim() == 2:
        x = x.squeeze(-1)
    
    # Pairwise differences: (n, 1) - (1, n) = (n, n)
    diffs = x.unsqueeze(1) - x.unsqueeze(0)
    
    # Soft comparison: probability that x[j] < x[i]
    probs = torch.sigmoid(-diffs / tau)
    
    # Sum to get expected rank (1-indexed)
    ranks = probs.sum(dim=1) + 0.5  # Add 0.5 for mid-rank
    
    return ranks


def ranknet_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    sigma: float = 1.0,
    weight_type: Literal["uniform", "gini"] = "uniform",
) -> torch.Tensor:
    """
    RankNet pairwise ranking loss with log-sigmoid smoothing.
    
    Minimizes the probability of misordering pairs, upper-bounded by:
    L = sum_{i,j: yi > yj} w_ij * log(1 + exp(-sigma * (s_i - s_j)))
    
    Args:
        scores: Predicted scores of shape (batch_size,)
        targets: Target values of shape (batch_size,)
        sigma: Sigmoid steepness (higher = sharper)
        weight_type: "uniform" (Kendall's τ) or "gini" (Gini covariance)
        
    Returns:
        Scalar loss value
    """
    if scores.dim() == 2:
        scores = scores.squeeze(-1)
    if targets.dim() == 2:
        targets = targets.squeeze(-1)
    
    n = scores.size(0)
    if n < 2:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    
    # Create pairwise matrices
    # s_i - s_j for all pairs
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (n, n)
    
    # Indicator: y_i > y_j
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # (n, n)
    indicator = (target_diff > 0).float()
    
    # Compute weights
    if weight_type == "uniform":
        weights = torch.ones_like(target_diff)
    elif weight_type == "gini":
        # Weight by |y_i - y_j| - larger gaps are more important
        weights = torch.abs(target_diff)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    # Log-sigmoid loss for misordered pairs
    # log(1 + exp(-sigma * (s_i - s_j))) when y_i > y_j
    loss_matrix = F.softplus(-sigma * score_diff)
    
    # Apply indicator and weights
    weighted_loss = indicator * weights * loss_matrix
    
    # Normalize by number of pairs
    n_pairs = n * (n - 1)
    if n_pairs > 0:
        loss = weighted_loss.sum() / n_pairs
    else:
        loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    
    return loss


def gini_softrank_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Gini covariance loss using differentiable soft ranks.
    
    Maximizes Cov(Y, rank(g(X))), equivalent to:
    L = -sum_i y_i * softrank(g(x_i))
    
    This is O(n²) for the softrank computation but can be batched efficiently.
    
    Args:
        scores: Predicted scores of shape (batch_size,)
        targets: Target values of shape (batch_size,)
        tau: Softrank temperature
        
    Returns:
        Scalar loss value (negative Gini covariance)
    """
    if scores.dim() == 2:
        scores = scores.squeeze(-1)
    if targets.dim() == 2:
        targets = targets.squeeze(-1)
    
    n = scores.size(0)
    if n < 2:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    
    # Compute soft ranks of scores
    ranks = softrank(scores, tau=tau)
    
    # Normalize targets and ranks to zero mean
    targets_centered = targets - targets.mean()
    ranks_centered = ranks - ranks.mean()
    
    # Negative covariance (we minimize this)
    cov = (targets_centered * ranks_centered).mean()
    
    return -cov


class CAIROScorer(nn.Module):
    """
    Stage 1 scoring network for CAIRO.
    
    A simple MLP that learns to produce scores that correctly order examples.
    The scores are trained via ranking loss, so their magnitude is arbitrary -
    only their relative ordering matters.
    """
    
    def __init__(
        self,
        d_in: int,
        n_blocks: int = 2,
        d_block: int = 32,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """
        Args:
            d_in: Number of input features
            n_blocks: Number of MLP blocks
            d_block: Hidden dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        layers = []
        current_d = d_in
        
        for _ in range(n_blocks):
            layers.append(nn.Linear(current_d, d_block))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_d = d_block
        
        # Output single score per example
        layers.append(nn.Linear(d_block, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking scores.
        
        Args:
            x: Input features of shape (batch_size, d_in)
            
        Returns:
            Scores of shape (batch_size,)
        """
        return self.network(x).squeeze(-1)


class CAIRO(TabularModel):
    """
    CAIRO: Calibrate After Initial Rank Ordering
    
    A two-stage regression framework that:
    1. Learns to rank examples using scale-invariant loss (robust to outliers)
    2. Calibrates scores to target scale via isotonic regression
    
    This is particularly effective for:
    - Heavy-tailed target distributions
    - Heteroskedastic noise
    - Data with outliers
    - Applications where ranking matters (e.g., HFT signal ranking)
    
    Example:
        model = CAIRO(d_in=10, d_out=1, loss_type="ranknet")
        
        # Training loop (Stage 1)
        for x, y in dataloader:
            scores = model.get_scores(x)
            loss = model.compute_ranking_loss(scores, y)
            loss.backward()
            optimizer.step()
        
        # After training, fit the calibrator (Stage 2)
        model.fit_calibrator(X_train, y_train)
        
        # Inference
        predictions = model(X_test)
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int = 1,
        n_blocks: int = 2,
        d_block: int = 64,
        dropout: float = 0.0,
        loss_type: Literal["ranknet", "ranknet_gini", "gini_softrank"] = "ranknet",
        sigma: float = 1.0,
        tau: float = 1.0,
        task: str = "regression",
    ):
        """
        Args:
            d_in: Number of input features
            d_out: Output dimension (typically 1 for regression)
            n_blocks: Number of hidden layers in scorer
            d_block: Hidden dimension
            dropout: Dropout rate
            loss_type: Ranking loss variant
                - "ranknet": Pairwise loss, uniform weights (Kendall's τ)
                - "ranknet_gini": Pairwise loss, |y_i - y_j| weights (Gini)
                - "gini_softrank": Pointwise loss with softrank (Gini, faster)
            sigma: Log-sigmoid steepness for pairwise losses
            tau: Softrank temperature for gini_softrank
            task: Task type (only "regression" supported)
        """
        super().__init__(d_in, d_out, task)
        
        if task != "regression":
            raise ValueError("CAIRO only supports regression tasks")
        
        self.loss_type = loss_type
        self.sigma = sigma
        self.tau = tau
        
        # Stage 1: Scoring network
        self.scorer = CAIROScorer(
            d_in=d_in,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
        )
        
        # Stage 2: Isotonic calibrator (fitted after Stage 1 training)
        self.calibrator: Optional[IsotonicRegression] = None
        self._is_calibrated = False
        
        # Cache for training set scores/targets (needed for calibration)
        self._train_scores: Optional[np.ndarray] = None
        self._train_targets: Optional[np.ndarray] = None
    
    def get_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw ranking scores (Stage 1 output).
        
        Args:
            x: Input features of shape (batch_size, d_in)
            
        Returns:
            Ranking scores of shape (batch_size,)
        """
        return self.scorer(x)
    
    def compute_ranking_loss(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the ranking loss for Stage 1 training.
        
        Args:
            scores: Predicted scores from get_scores()
            targets: Target values
            
        Returns:
            Scalar loss value
        """
        if self.loss_type == "ranknet":
            return ranknet_loss(scores, targets, sigma=self.sigma, weight_type="uniform")
        elif self.loss_type == "ranknet_gini":
            return ranknet_loss(scores, targets, sigma=self.sigma, weight_type="gini")
        elif self.loss_type == "gini_softrank":
            return gini_softrank_loss(scores, targets, tau=self.tau)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    @torch.no_grad()
    def fit_calibrator(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
    ) -> "CAIRO":
        """
        Fit the isotonic regression calibrator (Stage 2).
        
        This should be called after Stage 1 training is complete.
        
        Args:
            x: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        # Convert to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        
        # Get scores
        self.eval()
        scores = self.get_scores(x).cpu().numpy()
        targets = y.cpu().numpy().ravel()
        
        # Fit isotonic regression
        self.calibrator = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )
        self.calibrator.fit(scores, targets)
        
        self._is_calibrated = True
        self._train_scores = scores
        self._train_targets = targets
        
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with calibration.
        
        If calibrator is fitted, returns calibrated predictions.
        Otherwise, returns raw scores (useful during Stage 1 training).
        
        Args:
            x: Input features of shape (batch_size, d_in)
            
        Returns:
            Predictions of shape (batch_size, d_out)
        """
        scores = self.get_scores(x)
        
        if self._is_calibrated and self.calibrator is not None:
            # Apply isotonic calibration
            scores_np = scores.detach().cpu().numpy()
            calibrated = self.calibrator.predict(scores_np)
            predictions = torch.tensor(
                calibrated,
                dtype=x.dtype,
                device=x.device,
            )
            return predictions.unsqueeze(-1) if self.d_out == 1 else predictions
        else:
            # Return raw scores during training
            return scores.unsqueeze(-1) if self.d_out == 1 else scores
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method for a single training step.
        
        Args:
            x: Input features
            y: Target values
            
        Returns:
            Loss value
        """
        scores = self.get_scores(x)
        return self.compute_ranking_loss(scores, y)
    
    def get_ranking_metrics(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        """
        Compute ranking metrics (Kendall's τ and Spearman's ρ).
        
        Args:
            x: Input features
            y: Target values
            
        Returns:
            Dict with "kendall_tau" and "spearman_rho"
        """
        from scipy.stats import kendalltau, spearmanr
        
        self.eval()
        with torch.no_grad():
            scores = self.get_scores(x).cpu().numpy()
        targets = y.cpu().numpy().ravel()
        
        tau, _ = kendalltau(scores, targets)
        rho, _ = spearmanr(scores, targets)
        
        return {
            "kendall_tau": tau,
            "spearman_rho": rho,
        }


def create_cairo(
    d_in: int,
    d_out: int = 1,
    variant: Literal["ranknet", "ranknet_gini", "gini_softrank"] = "ranknet",
    **kwargs,
) -> CAIRO:
    """
    Factory function to create a CAIRO model.
    
    Args:
        d_in: Number of input features
        d_out: Output dimension
        variant: Which loss variant to use
        **kwargs: Additional arguments passed to CAIRO
        
    Returns:
        Configured CAIRO model
    """
    return CAIRO(
        d_in=d_in,
        d_out=d_out,
        loss_type=variant,
        **kwargs,
    )
