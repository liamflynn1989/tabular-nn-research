"""
Optimizer factory functions for tabular neural networks.

This module provides factory functions for various optimizers that are
particularly relevant for tabular deep learning, including state-of-the-art
and specialized optimizers.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
from muon import Muon
import torch_optimizer as optim


def get_adamw(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs
) -> torch.optim.AdamW:
    """
    AdamW optimizer - the baseline for most deep learning tasks.
    
    AdamW is Adam with decoupled weight decay. It's widely used and robust
    for tabular data due to its adaptive learning rates and effective
    regularization through weight decay.
    
    Key advantages for tabular data:
    - Robust to different feature scales (adaptive learning rates)
    - Good default choice for most tabular architectures
    - Effective weight decay for regularization
    
    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Coefficients for computing momentum and squared gradient averages
        eps: Small constant for numerical stability
        
    Returns:
        AdamW optimizer instance
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        **kwargs
    )


def get_muon(
    model: nn.Module,
    lr: float = 1e-2,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    **kwargs
) -> Muon:
    """
    Muon optimizer - momentum with orthogonalization.
    
    Muon uses Newton-Schulz iteration to orthogonalize momentum, which can
    provide more stable and faster convergence than standard momentum methods.
    
    Key advantages for tabular data:
    - More stable convergence than standard momentum
    - Can handle ill-conditioned problems better
    - Good for models with many parameters relative to data size
    
    Args:
        model: PyTorch model
        lr: Learning rate
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        
    Returns:
        Muon optimizer instance
    """
    return Muon(
        list(model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs
    )


def get_shampoo(
    model: nn.Module,
    lr: float = 1e-3,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    epsilon: float = 1e-4,
    update_freq: int = 1,
    **kwargs
) -> optim.Shampoo:
    """
    Shampoo optimizer - second-order method with efficient preconditioning.
    
    Shampoo uses second-order information (similar to SOAP) to adapt the
    learning rate for each parameter based on the curvature of the loss.
    This is particularly useful for tabular data with features at different scales.
    
    Key advantages for tabular data:
    - Adapts to different feature scales automatically
    - Memory efficient compared to full second-order methods
    - Good for high-dimensional sparse features
    - Can converge faster on well-conditioned problems
    
    Args:
        model: PyTorch model
        lr: Learning rate
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        epsilon: Small constant for numerical stability
        update_freq: Frequency of preconditioner updates
        
    Returns:
        Shampoo optimizer instance
    """
    return optim.Shampoo(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        epsilon=epsilon,
        update_freq=update_freq,
        **kwargs
    )


def get_novograd(
    model: nn.Module,
    lr: float = 1e-3,
    betas: tuple = (0.95, 0.98),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    grad_averaging: bool = True,
    amsgrad: bool = False,
    **kwargs
) -> optim.NovoGrad:
    """
    NovoGrad optimizer - layer-wise adaptive learning rates.
    
    NovoGrad computes layer-wise adaptive learning rates using the norm of
    gradients. Originally designed for speech and NLP, it can be effective
    for tabular data with many layers or complex architectures.
    
    Key advantages for tabular data:
    - Layer-wise adaptation (good for deep architectures)
    - More stable than Adam on some problems
    - Good for models with embedding layers
    - Effective for mixed numerical/categorical features
    
    Args:
        model: PyTorch model
        lr: Learning rate
        betas: Coefficients for computing momentum and exponential moving averages
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient
        grad_averaging: Whether to use gradient averaging
        amsgrad: Whether to use AMSGrad variant
        
    Returns:
        NovoGrad optimizer instance
    """
    return optim.NovoGrad(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        grad_averaging=grad_averaging,
        amsgrad=amsgrad,
        **kwargs
    )


# Registry of available optimizers
OPTIMIZER_REGISTRY = {
    "adamw": get_adamw,
    "muon": get_muon,
    "shampoo": get_shampoo,
    "novograd": get_novograd,
}


def get_optimizer(
    name: str,
    model: nn.Module,
    lr: float = 1e-3,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Factory function to get an optimizer by name.
    
    Args:
        name: Optimizer name ("adamw", "muon", "shampoo", "novograd")
        model: PyTorch model
        lr: Learning rate
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If optimizer name is not recognized
    """
    if name.lower() not in OPTIMIZER_REGISTRY:
        available = list(OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer: {name}. Available: {available}")
    
    return OPTIMIZER_REGISTRY[name.lower()](model, lr=lr, **kwargs)


def get_optimizer_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about available optimizers.
    
    Returns:
        Dictionary with optimizer info including description and use cases
    """
    return {
        "adamw": {
            "description": "Adam with decoupled weight decay - robust baseline",
            "best_for": "General purpose, most tabular architectures",
            "key_features": "Adaptive learning rates, effective regularization"
        },
        "muon": {
            "description": "Momentum with orthogonalization for stable convergence",
            "best_for": "Ill-conditioned problems, many parameters vs data",
            "key_features": "Orthogonalized momentum, stable convergence"
        },
        "shampoo": {
            "description": "Second-order method with efficient preconditioning",
            "best_for": "Different feature scales, high-dimensional sparse data",
            "key_features": "Curvature adaptation, memory efficient"
        },
        "novograd": {
            "description": "Layer-wise adaptive learning rates",
            "best_for": "Deep architectures, mixed feature types",
            "key_features": "Layer-wise adaptation, gradient averaging"
        }
    }


def recommend_optimizer(
    model_type: str = "mlp",
    dataset_size: str = "medium",
    feature_types: str = "mixed",
    architecture_depth: str = "medium"
) -> List[str]:
    """
    Recommend optimizers based on model and data characteristics.
    
    Args:
        model_type: Type of model ("mlp", "attention", "embedding")
        dataset_size: Dataset size ("small", "medium", "large") 
        feature_types: Feature types ("numerical", "categorical", "mixed")
        architecture_depth: Model depth ("shallow", "medium", "deep")
        
    Returns:
        List of recommended optimizer names in order of preference
    """
    recommendations = []
    
    # Default recommendation
    recommendations.append("adamw")
    
    # Model-specific recommendations
    if model_type == "attention" or architecture_depth == "deep":
        recommendations.insert(0, "novograd")
    
    if dataset_size == "small" and model_type != "embedding":
        recommendations.insert(0, "muon")
    
    if feature_types == "mixed" or feature_types == "categorical":
        if "novograd" not in recommendations:
            recommendations.insert(-1, "novograd")
    
    if dataset_size == "large" and feature_types != "categorical":
        recommendations.insert(-1, "shampoo")
    
    return recommendations[:3]  # Return top 3 recommendations