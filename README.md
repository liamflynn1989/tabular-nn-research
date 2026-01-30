# Tabular Neural Networks Research

A collection of implementations and experiments with state-of-the-art neural network architectures for tabular data.

## Purpose

This repository provides:
- Clean, readable PyTorch implementations of recent tabular deep learning papers
- Unified interfaces for easy comparison and experimentation
- Example scripts demonstrating usage on real datasets

## Implemented Papers

### 1. TabKANet: KAN-based Numerical Embeddings (Knowledge-Based Systems 2025)
**Paper:** [arXiv:2409.08806](https://arxiv.org/abs/2409.08806)
**Authors:** Weihao Gao, Zheng Gong, Zhuo Deng, Lan Ma

TabKANet uses Kolmogorov-Arnold Networks (KAN) with learnable B-spline activation functions to embed numerical features. The key insight is that numerical features often have complex non-linear relationships that simple linear projections fail to capture.

**Key Features:**
- B-spline based learnable activation functions
- Transformer encoder for feature interactions
- Supports both numerical and categorical features
- Noise injection for regularization

### 2. TabM: Parameter-Efficient Ensembling (ICLR 2025)
**Paper:** [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
**Authors:** Yury Gorishniy et al. (Yandex Research)

TabM efficiently imitates an ensemble of MLPs using a batch-like computation pattern. Key insight: multiple "virtual" MLPs share most parameters but produce diverse predictions, achieving ensemble-like performance with much lower computational cost.

**Key Features:**
- Parameter-efficient ensembling via weight sharing
- Simple MLP backbone with BatchEnsemble-style computation
- State-of-the-art performance among tabular DL models

### 3. Feature-aware Temporal Modulation (NeurIPS 2025)
**Paper:** [arXiv:2512.03678](https://arxiv.org/abs/2512.03678)
**Authors:** Hao-Run Cai, Han-Jia Ye

Addresses temporal distribution shifts in tabular data by conditioning feature representations on temporal context. Uses FiLM-style modulation to adapt feature statistics across time.

**Key Features:**
- Handles concept drift via feature-aware modulation
- Conditions on temporal context for dynamic adaptation
- Balances generalizability and adaptability

## Project Structure

```
tabular-nn-research/
├── implementations/
│   ├── __init__.py
│   ├── base.py               # Shared base classes
│   ├── tabkanet.py           # TabKANet (KAN + Transformer)
│   ├── tabm.py               # TabM (parameter-efficient ensembling)
│   └── temporal_modulation.py # Feature-aware temporal modulation
├── examples/
│   ├── tabkanet_example.py   # TabKANet usage example
│   ├── tabm_example.py       # TabM usage example
│   └── temporal_example.py   # Temporal modulation example
├── papers/
│   └── summaries/            # Paper summaries and notes
├── data/                     # Downloaded datasets (gitignored)
└── requirements.txt
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from implementations import TabKANet, TabM, TemporalTabularModel
import torch

# TabKANet - KAN-based numerical embeddings with Transformer
tabkanet = TabKANet(
    num_numerical=10,       # Number of numerical features
    num_categories=[5, 10], # Cardinalities for categorical features
    d_model=64,             # Embedding dimension
    n_heads=4,              # Attention heads
    n_layers=2,             # Transformer layers
    num_splines=8,          # B-spline basis functions
)

x_num = torch.randn(32, 10)        # Numerical features
x_cat = torch.randint(0, 5, (32, 2))  # Categorical features
out = tabkanet(x_num, x_cat)       # Shape: (32, 1)

# TabM - Parameter-efficient ensemble
tabm = TabM(
    d_in=10,           # Input features
    d_out=1,           # Output dimension
    n_blocks=3,        # MLP depth
    d_block=256,       # Hidden dimension
    n_heads=16,        # Number of ensemble "heads"
)

x = torch.randn(32, 10)
out = tabm(x)          # Shape: (32, 1)

# Temporal Modulation - For time-varying data
temporal_model = TemporalTabularModel(
    d_in=10,
    d_out=1,
    d_time=8,          # Temporal embedding dimension
)

time_idx = torch.arange(32)
out = temporal_model(x, time_idx)
```

## Benchmarks

See `examples/` for benchmark scripts on standard tabular datasets.

## Adding New Papers

1. Create implementation in `implementations/`
2. Follow the base interface in `base.py`
3. Add example script in `examples/`
4. Add paper summary in `papers/summaries/`
5. Update this README

## References

```bibtex
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
```

## License

MIT License - see implementations for paper-specific licenses.
