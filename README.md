# Tabular Neural Networks Research

A collection of implementations and experiments with state-of-the-art neural network architectures for tabular data.

## Benchmark Results

Performance comparison on synthetic regression datasets (Test RMSE, lower is better):

| Model | friedman | nonlinear_interaction | high_dimensional | temporal_drift | mixed_type | Avg |
|-------|----------|----------------------|------------------|----------------|------------|-----|
| MLP | - | - | - | - | - | - |
| TabM | - | - | - | - | - | - |
| TabKANet | - | - | - | - | - | - |
| Temporal | - | - | - | - | - | - |

*Run `python benchmarks/run_benchmarks.py` to generate results.*

### Leaderboard

| Rank | Model | Avg RMSE |
|------|-------|----------|
| 1 | - | - |
| 2 | - | - |
| 3 | - | - |
| 4 | - | - |

## Implemented Models

### 1. TabKANet: KAN-based Numerical Embeddings (Knowledge-Based Systems 2025)
**Paper:** [arXiv:2409.08806](https://arxiv.org/abs/2409.08806)

Uses Kolmogorov-Arnold Networks (KAN) with learnable B-spline activation functions to embed numerical features.

**Key Features:**
- B-spline based learnable activation functions
- Transformer encoder for feature interactions
- Supports both numerical and categorical features

### 2. TabM: Parameter-Efficient Ensembling (ICLR 2025)
**Paper:** [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)

Efficiently imitates an ensemble of MLPs using batch-like computation with weight sharing.

**Key Features:**
- Parameter-efficient ensembling via weight sharing
- Simple MLP backbone with BatchEnsemble-style computation
- State-of-the-art performance among tabular DL models

### 3. Feature-aware Temporal Modulation (NeurIPS 2025)
**Paper:** [arXiv:2512.03678](https://arxiv.org/abs/2512.03678)

Addresses temporal distribution shifts by conditioning feature representations on temporal context.

**Key Features:**
- Handles concept drift via feature-aware modulation
- FiLM-style modulation for dynamic adaptation
- Balances generalizability and adaptability

## Project Structure

```
tabular-nn-research/
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── base.py                 # Shared base classes (MLP, etc.)
│   ├── tabkanet.py             # TabKANet (KAN + Transformer)
│   ├── tabm.py                 # TabM (parameter-efficient ensembling)
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
from models import TabKANet, TabM, TemporalTabularModel
from data import load_dataset
import torch

# Load a dataset
dataset = load_dataset("friedman", n_samples=5000)
print(dataset.info)

# TabKANet - KAN-based embeddings with Transformer
tabkanet = TabKANet(
    num_numerical=10,
    d_model=64,
    n_heads=4,
    n_layers=2,
)

x_num = torch.randn(32, 10)
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
```

## Adding New Models

1. Create implementation in `models/`
2. Follow the base interface in `models/base.py`
3. Add factory function in `benchmarks/run_benchmarks.py`
4. Run benchmarks to compare performance
5. Update this README with results

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
