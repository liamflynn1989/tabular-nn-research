#!/usr/bin/env python3
"""
Run benchmarks for all models on all datasets.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --n-samples 10000 --epochs 200
    python benchmarks/run_benchmarks.py --force  # Re-run all, ignore cache
    python benchmarks/run_benchmarks.py --models MLP TabM  # Run specific models
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch

from benchmarks.runner import BenchmarkRunner
from benchmarks.utils import format_results_table, format_leaderboard, save_results

from models import TabM, TabKANet, TemporalTabularModel, TabR, MLPPLR, compute_bins, iLTM, AMFormer, CAIRO
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

    def make_tabr(info):
        return TabR(
            d_in=info.n_numerical,
            d_out=1,
            d_embedding=24,
            d_block=128,
            n_blocks=2,
            n_heads=4,
            k_neighbors=64,
            dropout=0.1,
            max_candidates=3000,
        )

    def make_mlpplr(info):
        """MLP with Periodic Embeddings (MLP-PLR from NeurIPS 2022)."""
        return MLPPLR(
            d_in=info.n_numerical,
            d_out=1,
            d_embedding=24,
            embedding_type="periodic",
            n_blocks=3,
            d_block=128,
            dropout=0.1,
            n_frequencies=48,
            frequency_init_scale=0.01,
            lite=True,
        )

    def make_iltm(info):
        """iLTM: Integrated Large Tabular Model (arXiv 2511.15941).
        
        Combines tree embeddings, hypernetwork conditioning, and retrieval.
        Note: This is a simplified version without pretrained weights.
        """
        return iLTM(
            d_in=info.n_numerical,
            d_out=1,
            d_main=256,  # Smaller for benchmarks
            n_blocks=2,
            n_estimators=50,  # Fewer trees for speed
            k_neighbors=32,
            retrieval_alpha=0.3,
            dropout=0.1,
            use_tree_embedding=True,
            task="regression",
        )

    def make_amformer(info):
        """AMFormer: Arithmetic Feature Interaction Transformer (AAAI 2024).
        
        Uses parallel additive and multiplicative attention to capture
        both sum and product feature interactions. Excellent for data
        with polynomial relationships (like financial ratios/products).
        """
        return AMFormer(
            d_in=info.n_numerical,
            d_out=1,
            d_model=64,
            n_heads=4,
            n_layers=3,
            d_ff_mult=4.0,
            dropout=0.1,
            use_multiplicative=True,
            use_token_descent=True,
            n_prompts=4,
            task="regression",
        )

    def make_cairo(info):
        """CAIRO: Calibrate After Initial Rank Ordering (arXiv 2602.14440).
        
        Two-stage regression: learns ranking via scale-invariant loss,
        then calibrates with isotonic regression. Robust to outliers
        and heavy-tailed noise.
        """
        return CAIRO(
            d_in=info.n_numerical,
            d_out=1,
            n_blocks=2,
            d_block=64,
            dropout=0.0,
            loss_type="ranknet",  # Most robust variant
            sigma=1.0,
            task="regression",
        )

    return {
        "MLP": make_mlp,
        "TabM": make_tabm,
        "TabKANet": make_tabkanet,
        "Temporal": make_temporal,
        "TabR": make_tabr,
        "MLPPLR": make_mlpplr,
        "iLTM": make_iltm,
        "AMFormer": make_amformer,
        "CAIRO": make_cairo,
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
                        help='Output file for results (also used as cache)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to run (default: all)')
    parser.add_argument('--force', action='store_true',
                        help='Ignore cache and re-run all benchmarks')
    args = parser.parse_args()

    print("="*60)
    print("Tabular Neural Network Benchmarks")
    print("="*60)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Samples per dataset: {args.n_samples}")
    print(f"Max epochs: {args.epochs}")
    if args.force:
        print("Force mode: ignoring cache")

    runner = BenchmarkRunner(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=True,
        cache_file=args.output,
        force=args.force,
    )

    factories = create_model_factories()

    # Filter models if specified
    if args.models:
        available = set(factories.keys())
        requested = set(args.models)
        invalid = requested - available
        if invalid:
            print(f"Warning: Unknown models {invalid}, available: {available}")
        factories = {k: v for k, v in factories.items() if k in requested}

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
