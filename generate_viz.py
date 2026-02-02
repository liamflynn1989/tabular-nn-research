#!/usr/bin/env python3
"""Generate benchmark visualizations."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('benchmarks/results.json') as f:
    results = json.load(f)

# Organize by model and dataset
models = {}
datasets = set()
for r in results:
    model = r['model_name']
    dataset = r['dataset_name']
    datasets.add(dataset)
    if model not in models:
        models[model] = {}
    models[model][dataset] = r['test_rmse']

datasets = sorted(datasets)
model_names = sorted(models.keys())

# Create comparison chart
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))
width = 0.15
multiplier = 0

colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

for i, model in enumerate(model_names):
    rmses = [models[model].get(d, 0) for d in datasets]
    offset = width * multiplier
    bars = ax.bar(x + offset, rmses, width, label=model, color=colors[i])
    multiplier += 1

ax.set_ylabel('Test RMSE (lower is better)', fontsize=12)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_title('Tabular NN Benchmark Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * (len(model_names) - 1) / 2)
ax.set_xticklabels(datasets, rotation=15, ha='right')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
print("Saved benchmark_comparison.png")

# Create average RMSE leaderboard
fig2, ax2 = plt.subplots(figsize=(8, 5))

avg_rmses = []
for model in model_names:
    rmses = [models[model].get(d, 0) for d in datasets]
    avg_rmses.append((model, np.mean(rmses)))

avg_rmses.sort(key=lambda x: x[1])
names, values = zip(*avg_rmses)

colors2 = ['#2ecc71' if i == 0 else '#3498db' if i < 3 else '#95a5a6' 
           for i in range(len(names))]
bars = ax2.barh(names, values, color=colors2)
ax2.set_xlabel('Average RMSE (lower is better)', fontsize=12)
ax2.set_title('Model Leaderboard (Average across datasets)', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, values):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('leaderboard.png', dpi=150, bbox_inches='tight')
print("Saved leaderboard.png")
