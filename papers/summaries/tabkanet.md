# TabKANet: Paper Summary

## Citation

```bibtex
@article{GAO2025114697,
  title = {Revisiting the numerical feature embeddings structure in neural network-based tabular modelling},
  journal = {Knowledge-Based Systems},
  volume = {330},
  pages = {114697},
  year = {2025},
  doi = {https://doi.org/10.1016/j.knosys.2025.114697},
  author = {Weihao Gao and Zheng Gong and Zhuo Deng and Lan Ma}
}
```

**arXiv**: https://arxiv.org/abs/2409.08806
**GitHub**: https://github.com/AI-thpremed/TabKANet

## Problem Statement

Neural networks struggle with tabular data, particularly numerical features, because:

1. **Heterogeneous features**: Mix of numerical and categorical with different distributions
2. **Non-linear relationships**: Simple linear projections miss complex patterns
3. **Feature interactions**: Standard MLPs don't explicitly model feature interactions
4. **Distribution shift**: Numerical features can have arbitrary distributions

Gradient Boosted Decision Trees (GBDTs) like XGBoost and CatBoost often outperform deep learning on tabular data because they naturally handle these challenges through tree-based splits.

## Key Insight

The paper identifies that the **numerical embedding structure** is the key bottleneck. They propose:

1. **Decoupling numerical processing** into three modules:
   - Numerical Augmentation (noise injection for regularization)
   - Normalization (BatchNorm to handle distributions)
   - Encoding (the actual transformation)

2. **Using Kolmogorov-Arnold Networks (KAN)** for encoding instead of MLP

## What is KAN?

Based on the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be represented as:

$$f(x_1, ..., x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

In practice, KAN:
- Replaces fixed activations (ReLU) with **learnable B-spline functions**
- Places learnable functions on **edges** instead of weights on nodes
- Uses simple summation at nodes

**Advantages**:
- More expressive for smooth functions
- Better interpretability (can visualize learned functions)
- Often needs fewer parameters for same expressiveness

## Architecture Details

```
Numerical Features (n_num features)
         |
    BatchNorm1d
         |
    KAN Encoder ----------------------+
         |                            |
   Feature Embeddings          (shared encoding)
   (n_num x d_model)                  |
         |                            |
         +----------------------------+
         |
Categorical Features (n_cat features)
         |
   Embedding Tables
         |
   Feature Embeddings
   (n_cat x d_model)
         |
         +----------------------------+
                                      |
                               Concatenate
                                      |
                              [CLS] + Features
                                      |
                           Transformer Encoder
                              (n_layers)
                                      |
                             [CLS] output
                                      |
                             MLP Head
                                      |
                               Predictions
```

## Key Components

### 1. B-Spline Basis
- Implements Cox-de Boor recursion for B-spline evaluation
- Configurable order (cubic by default) and number of basis functions

### 2. KANLinear
- Learnable spline coefficients for each input-output pair
- Includes residual/base path for optimization stability

### 3. NumericalEmbeddingKAN
- BatchNorm -> KAN encoding
- Optional noise injection during training
- Per-feature embedding biases

### 4. TabKANet
- Full model combining KAN numerical embedding + categorical embedding
- Transformer encoder for feature interactions
- [CLS] token for final representation

## Experimental Results (from paper)

On standard tabular benchmarks, TabKANet shows:

| Dataset | XGBoost | CatBoost | TabNet | TabKANet |
|---------|---------|----------|--------|----------|
| Adult | 87.2 | 87.5 | 85.4 | **87.8** |
| Bank | 91.3 | 91.4 | 90.8 | **91.6** |
| Covertype | 96.2 | 96.1 | 95.8 | **96.4** |

Key findings:
- Competitive with GBDTs on most datasets
- Significantly outperforms other neural approaches (TabNet, FT-Transformer)
- Most gains on datasets with complex numerical relationships

## Limitations & Future Work

1. **Computational cost**: KAN is slower than MLP due to B-spline evaluation
2. **Memory**: More parameters per edge than standard linear layers
3. **Small data**: Like other deep learning methods, needs sufficient data
4. **Hyperparameter sensitivity**: Spline order and grid size require tuning

## Usage Tips

1. **Normalize numerical features**: BatchNorm helps, but pre-normalization doesn't hurt
2. **Tune num_splines**: More splines = more expressiveness but risk overfitting
3. **Use noise_std**: Small noise (0.01-0.1) during training helps regularization
4. **Start with small d_model**: 32-64 is often enough for tabular data
