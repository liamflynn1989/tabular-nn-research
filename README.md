# Tabular Neural Networks Research

A collection of implementations and experiments with state-of-the-art neural network architectures for tabular data.

## Benchmark Results

Performance comparison on synthetic regression datasets (Test RMSE, lower is better):

| Model | friedman | nonlinear | high_dim | temporal | mixed | Avg |
|-------|----------|-----------|----------|----------|-------|------|
| MLP | 1.2291 | 0.8576 | 1.3678 | 1.2371 | 1.9788 | 1.3341 |
| TabM | 1.1466 | 0.8396 | 1.4507 | 1.2395 | **1.9651** | 1.3283 |
| TabKANet | 1.2948 | 1.1329 | 2.6319 | 1.3976 | 2.2801 | 1.7475 |
| TabR | **1.1165** | 0.7708 | 6.6134 | 1.2450 | 2.0553 | 2.3602 |
| Temporal | 1.1297 | 0.8741 | **1.1896** | **0.6139** | 2.0644 | **1.1743** |
| MLPPLR | 1.6656 | **0.7486** | 1.1740 | 1.2236 | 1.9743 | 1.3572 |
| iLTM | 1.4536 | 1.0963 | 2.0272 | 1.4341 | 2.1780 | 1.6379 |

### Leaderboard

| Rank | Model | Avg RMSE | Notes |
|------|-------|----------|-------|
| 🥇 | Temporal | **1.1743** | Best for temporal data |
| 🥈 | TabM | 1.3283 | Consistent across all datasets |
| 🥉 | MLP | 1.3341 | Strong baseline |
| 4 | MLPPLR | 1.3572 | Best on nonlinear & high-dim data |
| 5 | iLTM | 1.6379 | Tree embeddings + retrieval (simplified) |
| 6 | TabKANet | 1.7475 | Needs tuning |
| 7 | TabR | 2.3602 | Best on friedman, struggles with high-dim |

**Key findings:**
- **Temporal** model excels on datasets with temporal structure and high-dimensional data
- **TabM** performs consistently well across all datasets
- **MLPPLR** (Periodic Embeddings) achieves **best performance on nonlinear_interaction and high_dimensional** - validates the paper's claim that numerical embeddings help with complex feature relationships
- **TabR** achieves best results on friedman, but struggles with very high-dimensional data (curse of dimensionality affects retrieval)
- **MLP** baseline remains competitive, especially on simpler datasets
- **iLTM** (simplified) shows the benefit of tree embeddings for structured data - full pretrained version available at https://github.com/AI-sandbox/iLTM
- **TabKANet** underperforms on these synthetic benchmarks (may need hyperparameter tuning)

## Implemented Models

### 1. TabR: Retrieval-Augmented Tabular Learning (ICLR 2024)
**Paper:** [arXiv:2307.14338](https://arxiv.org/abs/2307.14338)

Combines deep learning with k-NN-style retrieval from training data. For each prediction, retrieves similar training examples and uses attention to aggregate their information.

**Key Features:**
- Soft attention-based retrieval (differentiable end-to-end)
- Uses both features and labels from retrieved neighbors
- Excellent for finding similar historical patterns (regime detection)
- Best performance on moderate-dimensional data

**HFT/MFT Relevance:**
- Finds similar historical market patterns
- Explainable via retrieved neighbors (regulatory compliance)
- Implicit ensemble benefits for noise handling

⚠️ **Note:** TabR struggles with very high-dimensional data due to the curse of dimensionality affecting k-NN retrieval. Consider dimensionality reduction for datasets with 50+ features.

### 2. TabKANet: KAN-based Numerical Embeddings (Knowledge-Based Systems 2025)
**Paper:** [arXiv:2409.08806](https://arxiv.org/abs/2409.08806)

Uses Kolmogorov-Arnold Networks (KAN) with learnable B-spline activation functions to embed numerical features.

**Key Features:**
- B-spline based learnable activation functions
- Transformer encoder for feature interactions
- Supports both numerical and categorical features

### 3. TabM: Parameter-Efficient Ensembling (ICLR 2025)
**Paper:** [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)

Efficiently imitates an ensemble of MLPs using batch-like computation with weight sharing.

**Key Features:**
- Parameter-efficient ensembling via weight sharing
- Simple MLP backbone with BatchEnsemble-style computation
- State-of-the-art performance among tabular DL models

### 4. Feature-aware Temporal Modulation (NeurIPS 2025)
**Paper:** [arXiv:2512.03678](https://arxiv.org/abs/2512.03678)

Addresses temporal distribution shifts by conditioning feature representations on temporal context.

**Key Features:**
- Handles concept drift via feature-aware modulation
- FiLM-style modulation for dynamic adaptation
- Balances generalizability and adaptability

### 5. iLTM: Integrated Large Tabular Model (arXiv 2025)
**Paper:** [arXiv:2511.15941](https://arxiv.org/abs/2511.15941)

An integrated tabular foundation model that unifies tree-derived embeddings, dimensionality-agnostic representations, a meta-trained hypernetwork, MLPs, and retrieval into a single architecture. Pretrained on 1800+ heterogeneous classification datasets.

**Key Features:**
- **GBDT Leaf Embeddings:** Uses gradient-boosted decision tree leaf indices as one-hot encoded features
- **Dimensionality-Agnostic Representation:** Random feature expansion + PCA for consistent embedding sizes
- **Hypernetwork:** Meta-trained network that generates MLP weights from training data statistics
- **Soft Retrieval:** Cosine-similarity based k-NN that blends with MLP predictions
- **Transfer Learning:** Classification pretraining transfers to regression tasks

**HFT/MFT Relevance:**
- Tree embeddings capture regime-like patterns in market data
- Retrieval finds similar historical market conditions
- Robust to distribution shift via GBDT inductive biases
- Works across varying feature dimensions (useful for multi-asset strategies)

⚠️ **Note:** This implementation is a simplified version without the meta-trained hypernetwork weights. For the full pretrained model achieving state-of-the-art results, see: https://github.com/AI-sandbox/iLTM

### 6. On Embeddings for Numerical Features (NeurIPS 2022)
**Paper:** [arXiv:2203.05556](https://arxiv.org/abs/2203.05556)

Demonstrates that transforming scalar numerical features into high-dimensional embeddings before mixing in the backbone significantly improves tabular neural network performance. Introduces two key embedding approaches.

**Key Features:**
- **Piecewise Linear Encoding (PLE):** Encodes scalars using learnable bin boundaries, creating sparse interpretable representations
- **Periodic Embeddings:** Uses sin/cos functions with learnable frequencies (similar to Fourier features)
- Simple MLPs with embeddings can match complex Transformer-based architectures
- Helps networks learn complex, non-linear feature-target relationships

**HFT/MFT Relevance:**
- Price/volume data often has multi-modal distributions that benefit from embeddings
- Periodic embeddings can capture cyclical patterns (intraday effects, round numbers)
- Piecewise linear bins adapt to different price/volume regimes automatically
- Low computational overhead - suitable for real-time inference

## Project Structure

```
tabular-nn-research/
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── base.py                 # Shared base classes (MLP, etc.)
│   ├── iltm.py                 # iLTM (tree embeddings + hypernetwork + retrieval)
│   ├── numerical_embeddings.py # MLPPLR (Periodic & PLE embeddings)
│   ├── tabkanet.py             # TabKANet (KAN + Transformer)
│   ├── tabm.py                 # TabM (parameter-efficient ensembling)
│   ├── tabr.py                 # TabR (retrieval-augmented)
│   └── temporal_modulation.py  # Feature-aware temporal modulation
├── data/                       # Synthetic datasets for benchmarking
│   ├── __init__.py
│   └── datasets.py             # Dataset implementations
├── benchmarks/                 # Benchmark code and results
│   ├── __init__.py
│   ├── runner.py               # Benchmark runner
│   ├── utils.py                # Result formatting utilities
│   ├── run_benchmarks.py       # Main benchmark script
│   └── results.json            # Benchmark results (generated)
├── tutorials/                  # Jupyter notebooks explaining models
│   ├── 01_temporal_modulation.ipynb
│   └── 02_tabr_retrieval.ipynb
├── examples/                   # Usage examples
│   ├── tabkanet_example.py
│   ├── tabm_example.py
│   └── temporal_example.py
├── papers/
│   └── summaries/              # Paper summaries and notes
└── requirements.txt
```

## Datasets

The benchmark uses synthetic regression datasets designed to test different model capabilities:

| Dataset | Features | Description |
|---------|----------|-------------|
| `friedman` | 10 numerical | Classic Friedman #1 with non-linear interactions |
| `nonlinear_interaction` | 15 numerical | Complex higher-order feature interactions |
| `high_dimensional` | 100 numerical | Sparse signal in high dimensions |
| `temporal_drift` | 10 numerical + time | Distribution shift over time periods |
| `mixed_type` | 10 num + 5 cat | Mixed numerical and categorical features |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

### Basic Usage

```python
from models import TabKANet, TabM, TemporalTabularModel, TabR, MLPPLR, iLTM, compute_bins
from data import load_dataset
import torch

# Load a dataset
dataset = load_dataset("friedman", n_samples=5000)
print(dataset.info)

# iLTM - Tree embeddings + Hypernetwork + Retrieval
iltm = iLTM(
    d_in=10,
    d_out=1,
    d_main=256,
    use_tree_embedding=True,  # Use GBDT leaf embeddings
    retrieval_alpha=0.3,  # Blend MLP with retrieval (0=MLP only, 1=retrieval only)
    k_neighbors=32,
)

x_num = torch.randn(1000, 10)
y = torch.randn(1000)

# Setup (fits GBDT and random projection)
iltm.setup(x_num, y)

# Set candidates for retrieval
iltm.set_candidates(x_num, y)

# Inference
iltm.eval()
out = iltm(x_num[:32])  # Shape: (32, 1)

# Get nearest neighbors for interpretability
indices, similarities, labels = iltm.get_nearest_neighbors(x_num[:32], k=5)

# TabR - Retrieval-augmented model
tabr = TabR(
    d_in=10,
    d_out=1,
    d_embedding=32,
    d_block=256,
    n_blocks=2,
    k_neighbors=96,
)

x_num = torch.randn(32, 10)
y = torch.randn(32)

# Training: pass y to accumulate candidates
tabr.train()
out = tabr(x_num, y_for_candidates=y)

# Inference: uses accumulated candidates automatically
tabr.eval()
out = tabr(x_num)  # Shape: (32, 1)

# Get nearest neighbors for interpretability
indices, distances, labels = tabr.get_nearest_neighbors(x_num, k=5)

# TabKANet - KAN-based embeddings with Transformer
tabkanet = TabKANet(
    num_numerical=10,
    d_model=64,
    n_heads=4,
    n_layers=2,
)
out = tabkanet(x_num)  # Shape: (32, 1)

# TabM - Parameter-efficient ensemble
tabm = TabM(
    d_in=10,
    d_out=1,
    n_blocks=3,
    d_block=256,
    n_heads=16,
)
out = tabm(x_num)  # Shape: (32, 1)

# Temporal Modulation - For time-varying data
temporal_model = TemporalTabularModel(
    d_in=10,
    d_out=1,
    d_time=8,
)
time_idx = torch.arange(32)
out = temporal_model(x_num, time_idx)

# MLPPLR - MLP with Periodic Embeddings (best for nonlinear data)
mlpplr = MLPPLR(
    d_in=10,
    d_out=1,
    d_embedding=24,
    embedding_type="periodic",  # or "ple" for piecewise linear
    n_blocks=3,
    d_block=256,
)
out = mlpplr(x_num)  # Shape: (32, 1)

# For PLE embeddings, compute bins from training data first:
# bins = compute_bins(X_train, n_bins=64)
# mlpplr_ple = MLPPLR(d_in=10, d_out=1, embedding_type="ple", bins=bins)
```

## Adding New Models

When implementing a new paper/model, follow these steps:

1. **Create implementation** in `models/`
   - Follow the base interface in `models/base.py`
   - Export the model in `models/__init__.py`

2. **Add to benchmarks** in `benchmarks/run_benchmarks.py`
   - Import the new model
   - Add a factory function (e.g., `make_newmodel`)
   - Add to the factories dict

3. **Run benchmarks** to generate results:
   ```bash
   python benchmarks/run_benchmarks.py
   ```
   (Only the new model will run - existing results are cached)

4. **Update README.md**
   - Add the new model to the Benchmark Results table
   - Add the new model to the Leaderboard
   - Add a description in the Implemented Models section
   - Add the citation in References

5. **Create tutorial** (optional) in `tutorials/`
   - Explain how the model works
   - Include visualizations and examples

## References

```bibtex
@article{bonet2025iltm,
  title={iLTM: Integrated Large Tabular Model},
  author={Bonet, David and Comajoan Cara, Mar{\c{c}}al and Calafell, Alvaro and 
          Mas Montserrat, Daniel and Ioannidis, Alexander G.},
  journal={arXiv preprint arXiv:2511.15941},
  year={2025}
}

@inproceedings{gorishniy2024tabr,
  title={TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023},
  author={Gorishniy, Yury and Rubachev, Ivan and Kartashev, Nikolay and Shlenskii, Daniil and Babenko, Artem},
  booktitle={ICLR},
  year={2024}
}

@article{GAO2025114697,
  title = {Revisiting the numerical feature embeddings structure in neural network-based tabular modelling},
  journal = {Knowledge-Based Systems},
  volume = {330},
  pages = {114697},
  year = {2025},
  author = {Weihao Gao and Zheng Gong and Zhuo Deng and Lan Ma}
}

@inproceedings{gorishniy2025tabm,
  title={TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling},
  author={Gorishniy, Yury and others},
  booktitle={ICLR},
  year={2025}
}

@inproceedings{cai2025feature,
  title={Feature-aware Modulation for Learning from Temporal Tabular Data},
  author={Cai, Hao-Run and Ye, Han-Jia},
  booktitle={NeurIPS},
  year={2025}
}

@inproceedings{gorishniy2022embeddings,
  title={On Embeddings for Numerical Features in Tabular Deep Learning},
  author={Yury Gorishniy and Ivan Rubachev and Artem Babenko},
  booktitle={NeurIPS},
  year={2022}
}
```

## License

MIT License - see implementations for paper-specific licenses.
