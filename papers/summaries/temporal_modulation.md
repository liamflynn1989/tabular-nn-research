# Feature-aware Modulation for Learning from Temporal Tabular Data

**Paper:** [arXiv:2512.03678](https://arxiv.org/abs/2512.03678)  
**Authors:** Hao-Run Cai, Han-Jia Ye  
**Venue:** NeurIPS 2025

## TL;DR

Temporal tabular data suffers from concept drift — the relationship between features and labels changes over time. This paper proposes using FiLM-style modulation to adapt feature representations based on temporal context, balancing generalization and adaptation.

## The Problem: Temporal Distribution Shift

In many real-world applications (finance, healthcare, e-commerce), data distributions change over time:

- **Covariate shift**: P(X) changes, P(Y|X) stays same
- **Label shift**: P(Y) changes, P(X|Y) stays same  
- **Concept drift**: P(Y|X) changes ← Most challenging!

Traditional approaches:
- **Static models**: Assume fixed mapping, good generalization but miss temporal patterns
- **Adaptive models**: Fit recent data, capture trends but risk overfitting to transient patterns

## Key Insight: Feature Semantics Evolve

The paper identifies that feature *meanings* change over time:

1. **Objective meaning changes**: What "high income" means in 2020 vs 2024
2. **Subjective meaning changes**: How users interpret features differently

Rather than completely relearning, we can *modulate* feature representations.

## Method: Feature-aware Temporal Modulation

### Core Idea

Use FiLM (Feature-wise Linear Modulation) to adapt features based on time:

```
output = γ(t) * features + β(t)
```

Where γ and β are learned functions of temporal context.

### Architecture

```
Time Index → Temporal Encoder → Conditioning Signal
                                       ↓
Input Features → Linear → FiLM Modulation → Norm → Activation → ...
```

### Components

1. **Temporal Encoder**: 
   - Sinusoidal positional encoding (captures periodicity)
   - MLP transformation (captures complex patterns)

2. **FiLM Layer**:
   - Predicts scale (γ) and shift (β) from temporal encoding
   - Initialized to identity (γ=1, β=0)
   - Learns time-dependent transformations

3. **Modulated MLP**:
   - Standard MLP backbone with FiLM layers at each block
   - Shared weights + temporal adaptation

### Why This Works

- **Shared backbone**: Captures time-invariant patterns → generalization
- **FiLM modulation**: Captures time-varying patterns → adaptation
- **Lightweight**: Only adds O(d_time × d_feature) parameters per layer

## Results

The paper shows improvements on temporal tabular benchmarks where:
- Periodic patterns exist (seasonal effects)
- Gradual drift occurs (changing user behavior)
- Sudden shifts happen (market changes)

## Key Hyperparameters

| Parameter | Recommended |
|-----------|-------------|
| d_time | 8-32 (16 default) |
| modulation_type | "scale_shift" (full FiLM) |
| max_time | Depends on data granularity |

## When to Use

✅ Good for:
- Time series classification/regression on tabular features
- Data with known temporal drift
- Combining historical and recent data

❌ Consider alternatives when:
- No temporal information available
- Static/IID data assumption is valid
- Need real-time online adaptation

## Comparison with Alternatives

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| Periodic retraining | Retrain on recent data | Simple | Expensive, loses history |
| Domain adaptation | Align distributions | Principled | Assumes fixed target |
| **Temporal modulation** | Condition on time | Lightweight, uses all data | Needs time indices |

## Implementation Notes

1. **Time representation**: Can use raw indices, normalized [0,1], or date features
2. **Initialization**: FiLM should start as identity transform
3. **Regularization**: May need to constrain modulation magnitude

## Citation

```bibtex
@inproceedings{cai2025feature,
  title={Feature-aware Modulation for Learning from Temporal Tabular Data},
  author={Cai, Hao-Run and Ye, Han-Jia},
  booktitle={NeurIPS},
  year={2025}
}
```
