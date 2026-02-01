"""
Tabular Neural Network Implementations

Clean PyTorch implementations of state-of-the-art tabular deep learning papers.
"""

from .base import TabularModel
from .tabm import TabM
from .temporal_modulation import (
    TemporalTabularModel,
    FiLMLayer,
    TemporalEncoder,
    TemporalModulationLayer,
    yeo_johnson_transform,
)
from .tabkanet import TabKANet, KANLinear, NumericalEmbeddingKAN
from .tabr import TabR, NumericalEmbeddings, RetrievalModule
from .numerical_embeddings import (
    MLPPLR,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings,
    PeriodicEmbeddings,
    compute_bins,
    create_mlpplr,
)
from .iltm import (
    iLTM,
    TreeEmbedding,
    RandomFeatureProjection,
    SoftRetrievalModule,
    create_iltm,
)

__all__ = [
    "TabularModel",
    "TabM",
    "TemporalTabularModel",
    "FiLMLayer",
    "TemporalEncoder",
    "TemporalModulationLayer",
    "yeo_johnson_transform",
    "TabKANet",
    "KANLinear",
    "NumericalEmbeddingKAN",
    "TabR",
    "NumericalEmbeddings",
    "RetrievalModule",
    # Numerical Embeddings (NeurIPS 2022)
    "MLPPLR",
    "PiecewiseLinearEncoding",
    "PiecewiseLinearEmbeddings",
    "PeriodicEmbeddings",
    "compute_bins",
    "create_mlpplr",
    # iLTM (arXiv 2511.15941)
    "iLTM",
    "TreeEmbedding",
    "RandomFeatureProjection",
    "SoftRetrievalModule",
    "create_iltm",
]
