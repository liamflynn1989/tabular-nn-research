# TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling

**Paper:** [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)  
**Authors:** Yury Gorishniy et al. (Yandex Research)  
**Venue:** ICLR 2025  
**Code:** [github.com/yandex-research/tabm](https://github.com/yandex-research/tabm)

## TL;DR

TabM is an MLP-based model that achieves ensemble-like performance with parameter-efficient ensembling. Instead of training K separate MLPs, TabM shares most parameters and uses learned scaling factors to create K "virtual" MLPs.

## Key Insights

1. **Ensembles work well for tabular data**, but are expensive
2. **Parameter sharing + diversity** gives the best of both worlds
3. **MLPs remain competitive** with fancy architectures when properly ensembled

## Method

### BatchEnsemble-style Linear Layer

Standard ensemble of K linear layers:
```
y_k = W_k @ x + b_k  (K separate weight matrices)
```

TabM's efficient version:
```
y_k = (W ⊙ (r_k ⊗ s_k^T)) @ x + b
```

Where:
- `W` is a shared weight matrix
- `r_k` and `s_k` are per-head scaling vectors
- `⊙` is element-wise product
- `⊗` is outer product

This reduces parameters from O(K × d_in × d_out) to O(d_in × d_out + K × (d_in + d_out)).

### Architecture

```
Input → [TabMLinear → BatchNorm → ReLU → Dropout] × N → TabMLinear → Mean
         ↑                                                    ↑
         Multiple virtual heads share these blocks           Average predictions
```

### Training

- All heads trained simultaneously
- Standard cross-entropy/MSE loss on averaged prediction
- Implicit regularization from parameter sharing

## Results

- **Best among DL methods** on OpenML benchmarks
- **Competitive with GBDTs** on many datasets
- **Much faster than retrieval-based** methods (TabR, etc.)
- **Individual heads are weak**, but ensemble is strong

## Key Hyperparameters

| Parameter | Recommended Range |
|-----------|-------------------|
| n_heads | 8-32 (16 default) |
| n_blocks | 2-4 (3 default) |
| d_block | 128-512 (256 default) |
| dropout | 0.0-0.2 (0.1 default) |

## When to Use TabM

✅ Good for:
- Medium-to-large tabular datasets
- When you want DL benefits (GPU, end-to-end learning)
- As a strong baseline before trying complex methods

❌ Consider alternatives when:
- Very small datasets (<1000 samples) → TabPFN
- Need interpretability → GBDTs
- Extreme feature sparsity → specialized methods

## Implementation Notes

1. **Initialization matters**: Scale r and s around 1.0
2. **BatchNorm placement**: After linear, before activation
3. **Ensemble averaging**: Done at prediction time, not logits

## Citation

```bibtex
@inproceedings{gorishniy2025tabm,
  title={TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling},
  author={Gorishniy, Yury and others},
  booktitle={ICLR},
  year={2025}
}
```
