#!/usr/bin/env python3
"""
Run benchmarks for all models on all datasets.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --n-samples 10000 --epochs 200
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch

from benchmarks.runner import BenchmarkRunner
from benchmarks.utils import format_results_table, format_leaderboard, save_results

from models import TabM, TabKANet, TemporalTabularModel
from models.base import MLP


def create_model_factories():
    """
    Create factory functions for each model.

    Each factory takes a DatasetInfo and returns a configured model.
    """

    def make_mlp(info):
        return MLP(
            d_in=info.n_numerical,
            d_out=1,
            n_blocks=3,
            d_block=128,
            dropout=0.1,
        )

    def make_tabm(info):
        return TabM(
            d_in=info.n_numerical,
            d_out=1,
            n_blocks=3,
            d_block=128,
            n_heads=8,
            dropout=0.1,
        )

    def make_tabkanet(info):
        return TabKANet(
            num_numerical=info.n_numerical,
            num_categories=info.num_categories if info.n_categorical > 0 else None,
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_classes=1,
            task='regression',
        )

    def make_temporal(info):
        return TemporalTabularModel(
            d_in=info.n_numerical,
            d_out=1,
            d_time=16,
            n_blocks=3,
            d_block=128,
            dropout=0.1,
        )

    return {
        "MLP": make_mlp,
        "TabM": make_tabm,
        "TabKANet": make_tabkanet,
        "Temporal": make_temporal,
    }


def main():
    parser = argparse.ArgumentParser(description='Run tabular NN benchmarks')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of samples per dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--output', type=str, default='benchmarks/results.json',
                        help='Output file for results')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all)')
    args = parser.parse_args()

    print("="*60)
    print("Tabular Neural Network Benchmarks")
    print("="*60)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Samples per dataset: {args.n_samples}")
    print(f"Max epochs: {args.epochs}")

    runner = BenchmarkRunner(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=True,
    )

    factories = create_model_factories()

    # Run all benchmarks
    results = runner.run_all(
        model_factories=factories,
        dataset_names=args.datasets,
        n_samples=args.n_samples,
    )

    # Save results
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")

    # Print tables
    print("\n" + "="*60)
    print("Results Table (Test RMSE, lower is better)")
    print("="*60)
    print(format_results_table(results))

    print("\n" + "="*60)
    print("Leaderboard")
    print("="*60)
    print(format_leaderboard(results))


if __name__ == "__main__":
    main()
