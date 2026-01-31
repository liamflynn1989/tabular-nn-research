# Claude Instructions for tabular-nn-research

This file contains instructions for Claude when working on this repository.

## Project Overview

This is a research repository for tabular neural network architectures. It contains:

- Model implementations in `models/`
- Synthetic benchmark datasets in `data/`
- Benchmark runner with caching in `benchmarks/`
- Tutorials explaining models in `tutorials/`

## Research Focus

**Primary application:** High-Frequency Trading (HFT) and Medium-Frequency Trading (MFT) stock price prediction using tabular data.

When selecting papers to implement, prioritize architectures that address these challenges:

| Challenge                     | Description                                           | Relevant Techniques                                      |
| ----------------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| **Low signal-to-noise ratio** | Financial signals are weak relative to noise          | Regularization, ensembling, robust loss functions        |
| **Non-stationarity**          | Data distribution changes over time (concept drift)   | Temporal modulation, online learning, adaptive methods   |
| **High dimensionality**       | Many features (technical indicators, order book data) | Feature selection, attention mechanisms, sparse methods  |
| **Correlated features**       | Features are often highly correlated                  | Decorrelation, PCA-like embeddings, feature interactions |
| **Non-linear relationships**  | Complex, non-linear feature-target relationships      | KAN, polynomial features, deep networks                  |
| **Temporal dependencies**     | Past values influence future predictions              | Recurrent components, temporal encodings                 |
| **Class imbalance**           | Rare but important events (large moves)               | Focal loss, oversampling, cost-sensitive learning        |

### Paper Selection Criteria

When asked to find or implement new papers, look for:

1. **Tabular-specific architectures** - Not just adapted vision/NLP models
2. **State-of-the-art on tabular benchmarks** - Check performance on standard datasets
3. **Addresses our challenges** - Particularly non-stationarity, noise, and high dimensionality
4. **Regression focus** - We care about regression more than classification
5. **Computational efficiency** - Must be practical for real-time inference
6. **Recent publications** - Prefer papers from 2023-2025, top venues (NeurIPS, ICML, ICLR, KDD)

### Example Good Papers

- Temporal modulation methods (handle non-stationarity)
- Ensemble methods like TabM (handle noise via diversity)
- KAN-based embeddings (capture non-linear relationships)
- Attention mechanisms for feature selection (handle high dimensionality)

## When Adding a New Paper/Model

**IMPORTANT:** When implementing a new paper, you MUST complete ALL of these steps:

### 1. Implementation (`models/`)

- Create `models/<model_name>.py` with the implementation
- Follow patterns from existing models (TabM, TabKANet, TemporalTabularModel)
- Inherit from `TabularModel` base class when appropriate
- Add to `models/__init__.py` exports

### 2. Benchmarks (`benchmarks/run_benchmarks.py`)

- Import the new model at the top
- Add a factory function: `def make_<model>(info): ...`
- Add to the `factories` dict returned by `create_model_factories()`

### 3. Run Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

The caching system will only run the new model (existing results are cached).

### 4. Update README.md

- Add the new model row to the **Benchmark Results** table
- Update the **Leaderboard** with new rankings
- Add a section under **Implemented Models** with:
  - Paper title and venue
  - arXiv/paper link
  - Brief description
  - Key features (bullet points)
- Add BibTeX citation to **References**

### 5. Tutorial

- Create `tutorials/XX_<model_name>.ipynb`
- Explain the key ideas from the paper
- Include visualizations of how components work
- Show training examples

## Benchmark Caching

Results are cached in `benchmarks/results.json`. When you run benchmarks:

- Cached model/dataset pairs are skipped
- Only new combinations are trained
- Use `--force` to re-run everything (don't do this unless asked)
- Use `--models ModelName` to run specific models

## Code Style

- Use type hints
- Follow existing patterns in the codebase
- Keep implementations clean and well-documented
- Docstrings should explain the paper's key ideas

## Common Commands

```bash
# Run benchmarks (with caching)
python benchmarks/run_benchmarks.py

# Run specific model only
python benchmarks/run_benchmarks.py --models NewModel

# Force re-run all
python benchmarks/run_benchmarks.py --force

# Run tutorial notebook
python -m jupyter notebook tutorials/01_temporal_modulation.ipynb
```
