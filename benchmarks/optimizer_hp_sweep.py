#!/usr/bin/env python3
"""
Comprehensive hyperparameter sweep for optimizers in tabular neural networks.

This script performs a grid search to find the optimal hyperparameters for each
optimizer, then compares their best performance across multiple datasets.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import time
import json
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Set up paths
sys.path.append('.')

try:
    from models.optimizers import get_optimizer, get_optimizer_info
    from models.base import MLP
    from data.datasets import load_dataset
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Set device and seeds for reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# Hyperparameter grids for each optimizer
HYPERPARAMETER_GRIDS = {
    'adamw': {
        'lr': [1e-4, 3e-4, 1e-3, 3e-3],
        'weight_decay': [0, 1e-3, 1e-2]
    },
    'shampoo': {
        'lr': [1e-4, 3e-4, 1e-3, 3e-3],
        'epsilon': [1e-6, 1e-4, 1e-2]
    },
    'novograd': {
        'lr': [1e-4, 3e-4, 1e-3, 3e-3],
        'betas': [(0.9, 0.99), (0.95, 0.98)]
    }
}

def create_model(n_features: int) -> nn.Module:
    """Create a simple MLP model for comparison."""
    return MLP(
        d_in=n_features,
        d_out=1,
        n_blocks=3,
        d_block=128,
        dropout=0.1,
        task="regression"
    ).to(device)

def prepare_data(dataset):
    """Prepare data for training."""
    X = dataset.X_num.numpy()
    y = dataset.y.numpy().reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_with_config(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 50,
    patience: int = 10
) -> Dict[str, Any]:
    """Train model with given configuration."""
    
    model.train()
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    try:
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            train_pred = model(X_train)
            train_loss = criterion(train_pred, y_train)
            
            # Check for NaN/inf loss (divergence)
            if torch.isnan(train_loss) or torch.isinf(train_loss):
                return {
                    'converged': False,
                    'diverged': True,
                    'best_val_loss': float('inf'),
                    'val_r2': -float('inf'),
                    'total_time': time.time() - start_time,
                    'epochs_trained': epoch,
                    'error': 'Training diverged (NaN/inf loss)'
                }
            
            train_loss.backward()
            
            # Check for NaN/inf gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                return {
                    'converged': False,
                    'diverged': True,
                    'best_val_loss': float('inf'),
                    'val_r2': -float('inf'),
                    'total_time': time.time() - start_time,
                    'epochs_trained': epoch,
                    'error': 'Training diverged (NaN/inf gradients)'
                }
            
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)
                
                # Check for divergence in validation too
                if torch.isnan(val_loss) or torch.isinf(val_loss):
                    return {
                        'converged': False,
                        'diverged': True,
                        'best_val_loss': float('inf'),
                        'val_r2': -float('inf'),
                        'total_time': time.time() - start_time,
                        'epochs_trained': epoch,
                        'error': 'Validation loss diverged (NaN/inf)'
                    }
            
            # Record history
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['epoch'].append(epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save best model state
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        # Restore best model for final evaluation
        if 'best_state' in locals():
            model.load_state_dict(best_state)
        
        # Final validation evaluation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_r2 = r2_score(y_val.cpu().numpy(), val_pred.cpu().numpy())
        
        total_time = time.time() - start_time
        
        return {
            'converged': True,
            'diverged': False,
            'best_val_loss': best_val_loss,
            'val_r2': val_r2,
            'total_time': total_time,
            'epochs_trained': len(history['train_loss']),
            'history': history
        }
        
    except Exception as e:
        return {
            'converged': False,
            'diverged': True,
            'best_val_loss': float('inf'),
            'val_r2': -float('inf'),
            'total_time': time.time() - start_time,
            'epochs_trained': 0,
            'error': str(e)
        }

def grid_search_optimizer(
    optimizer_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_features: int
) -> Dict[str, Any]:
    """Perform grid search for a single optimizer."""
    
    print(f"  🔍 Grid search for {optimizer_name.upper()}...")
    
    grid = HYPERPARAMETER_GRIDS[optimizer_name]
    param_names = list(grid.keys())
    param_values = list(grid.values())
    
    best_config = None
    best_result = None
    best_val_r2 = -float('inf')
    
    # Generate all parameter combinations
    param_combinations = list(product(*param_values))
    total_configs = len(param_combinations)
    
    print(f"    Testing {total_configs} configurations...")
    
    converged_configs = 0
    diverged_configs = 0
    
    for i, param_combo in enumerate(param_combinations):
        # Create parameter dictionary
        config = dict(zip(param_names, param_combo))
        
        # Create fresh model for each configuration
        model = create_model(n_features)
        
        try:
            # Create optimizer with current config
            optimizer = get_optimizer(optimizer_name, model, **config)
            
            # Train with this configuration
            result = train_with_config(
                model, optimizer, X_train, y_train, X_val, y_val
            )
            
            result['config'] = config
            
            if result['converged']:
                converged_configs += 1
                if result['val_r2'] > best_val_r2:
                    best_val_r2 = result['val_r2']
                    best_config = config
                    best_result = result
            else:
                diverged_configs += 1
            
            # Print progress every 5 configs or if it's the last one
            if (i + 1) % 5 == 0 or (i + 1) == total_configs:
                print(f"    Progress: {i + 1}/{total_configs} configs tested")
                
        except Exception as e:
            diverged_configs += 1
            print(f"    Config {config} failed: {e}")
    
    print(f"    ✅ Completed: {converged_configs} converged, {diverged_configs} diverged")
    
    return {
        'best_config': best_config,
        'best_result': best_result,
        'total_configs_tested': total_configs,
        'converged_configs': converged_configs,
        'diverged_configs': diverged_configs
    }

def run_final_comparison(
    best_configs: Dict[str, Dict],
    dataset_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Run final comparison using best hyperparameters for each optimizer."""
    
    print("\n🏆 Running final comparison with best hyperparameters...")
    
    final_results = {}
    
    for dataset_name in dataset_names:
        print(f"\n📊 Dataset: {dataset_name.upper()}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, n_samples=1500, random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(dataset)
        n_features = X_train.shape[1]
        
        final_results[dataset_name] = {}
        
        for optimizer_name, best_data in best_configs.items():
            if best_data['best_config'] is None:
                print(f"  ❌ {optimizer_name.upper()}: No valid configuration found")
                continue
                
            print(f"  🔧 {optimizer_name.upper()}: ", end="")
            
            try:
                # Create model and optimizer with best config
                model = create_model(n_features)
                config = best_data['best_config']
                optimizer = get_optimizer(optimizer_name, model, **config)
                
                # Train with best configuration (more epochs for final evaluation)
                result = train_with_config(
                    model, optimizer, X_train, y_train, X_val, y_val,
                    epochs=100, patience=20
                )
                
                if result['converged']:
                    # Test performance
                    model.eval()
                    with torch.no_grad():
                        test_pred = model(X_test)
                        test_mse = F.mse_loss(test_pred, y_test).item()
                        test_r2 = r2_score(y_test.cpu().numpy(), test_pred.cpu().numpy())
                    
                    final_results[dataset_name][optimizer_name] = {
                        'test_mse': test_mse,
                        'test_r2': test_r2,
                        'val_r2': result['val_r2'],
                        'best_config': config,
                        'total_time': result['total_time'],
                        'epochs_trained': result['epochs_trained'],
                        'history': result['history']
                    }
                    
                    print(f"R²={test_r2:.3f}, MSE={test_mse:.4f}")
                    
                else:
                    print(f"Failed to converge: {result.get('error', 'Unknown error')}")
                    final_results[dataset_name][optimizer_name] = {
                        'test_mse': float('inf'),
                        'test_r2': -float('inf'),
                        'error': result.get('error', 'Failed to converge')
                    }
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                final_results[dataset_name][optimizer_name] = {
                    'test_mse': float('inf'),
                    'test_r2': -float('inf'),
                    'error': str(e)
                }
    
    return final_results

def generate_visualizations(
    hp_sweep_results: Dict[str, Dict],
    final_results: Dict[str, Dict[str, Any]]
):
    """Generate visualization plots for the hyperparameter sweep results."""
    
    print("\n🎨 Generating visualizations...")
    
    # Colors for each optimizer
    colors = {'adamw': '#1f77b4', 'shampoo': '#2ca02c', 'novograd': '#d62728'}
    
    # 1. Training curves with best hyperparameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Optimizer HP Sweep: Training Curves with Best Hyperparameters', 
                 fontsize=16, fontweight='bold')
    
    dataset_names = list(final_results.keys())
    
    for i, dataset_name in enumerate(dataset_names):
        ax1 = axes[0, i] if i < 3 else axes[1, i-3]
        ax2 = axes[1, i] if i < 3 else None  # Only use second row for bar charts
        
        # Training curves
        for optimizer_name, result in final_results[dataset_name].items():
            if 'history' in result:
                history = result['history']
                epochs = range(len(history['train_loss']))
                
                ax1.plot(epochs, history['train_loss'], 
                        color=colors[optimizer_name], alpha=0.7, 
                        label=f"{optimizer_name.upper()} (Train)")
                ax1.plot(epochs, history['val_loss'], 
                        color=colors[optimizer_name], linestyle='--', 
                        label=f"{optimizer_name.upper()} (Val)")
        
        ax1.set_title(f'{dataset_name.replace("_", " ").title()} - Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_yscale('log')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for i in range(len(dataset_names), 6):
        row, col = (0, i) if i < 3 else (1, i-3)
        axes[row, col].remove()
    
    plt.tight_layout()
    plt.savefig('optimizers_hp_sweep_results.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: optimizers_hp_sweep_results.png")
    
    # 2. Bar chart comparison of best R² scores
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x_pos = np.arange(len(dataset_names))
    width = 0.2
    
    optimizer_names = list(colors.keys())
    
    for i, optimizer_name in enumerate(optimizer_names):
        r2_scores = []
        for dataset_name in dataset_names:
            if (optimizer_name in final_results[dataset_name] and 
                'test_r2' in final_results[dataset_name][optimizer_name]):
                r2 = final_results[dataset_name][optimizer_name]['test_r2']
                r2_scores.append(max(0, r2))  # Clip negative R² to 0 for visualization
            else:
                r2_scores.append(0)
        
        ax.bar(x_pos + i*width, r2_scores, width, 
               label=optimizer_name.upper(), color=colors[optimizer_name], alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Test R² Score')
    ax.set_title('Optimizer Comparison: Best R² Score by Dataset (After HP Sweep)')
    ax.set_xticks(x_pos + width*1.5)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in dataset_names])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('optimizers_hp_sweep_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: optimizers_hp_sweep_comparison.png")
    
    plt.close('all')  # Close plots to free memory

def save_results(hp_sweep_results: Dict, final_results: Dict):
    """Save all results to JSON file."""
    
    print("\n💾 Saving results...")
    
    # Prepare results for JSON serialization
    results_for_json = {
        'hp_sweep_results': {},
        'final_results': {}
    }
    
    # HP sweep results
    for optimizer_name, result in hp_sweep_results.items():
        json_result = {
            'best_config': result['best_config'],
            'best_val_r2': result['best_result']['val_r2'] if result['best_result'] else None,
            'total_configs_tested': result['total_configs_tested'],
            'converged_configs': result['converged_configs'],
            'diverged_configs': result['diverged_configs']
        }
        results_for_json['hp_sweep_results'][optimizer_name] = json_result
    
    # Final comparison results
    for dataset_name, dataset_results in final_results.items():
        results_for_json['final_results'][dataset_name] = {}
        for optimizer_name, result in dataset_results.items():
            # Convert history to lists for JSON serialization
            json_result = result.copy()
            if 'history' in json_result:
                json_result['history'] = {
                    k: v if isinstance(v, list) else [v] 
                    for k, v in json_result['history'].items()
                }
            results_for_json['final_results'][dataset_name][optimizer_name] = json_result
    
    with open('optimizer_hp_sweep_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print("  ✅ Saved: optimizer_hp_sweep_results.json")

def main():
    """Main function to run the hyperparameter sweep."""
    
    print("🚀 Starting Comprehensive Hyperparameter Sweep for Optimizers")
    print("=" * 70)
    
    # Phase 1: Hyperparameter sweep on friedman dataset
    print("\n📊 Phase 1: Hyperparameter sweep on Friedman dataset")
    print("-" * 50)
    
    # Load friedman dataset for HP sweep
    dataset = load_dataset('friedman', n_samples=1500, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(dataset)
    n_features = X_train.shape[1]
    
    print(f"Dataset: Friedman with {n_features} features, {len(X_train)} train samples")
    
    # Run grid search for each optimizer
    hp_sweep_results = {}
    
    for optimizer_name in HYPERPARAMETER_GRIDS.keys():
        print(f"\n🔧 {optimizer_name.upper()} hyperparameter search:")
        
        result = grid_search_optimizer(
            optimizer_name, X_train, y_train, X_val, y_val, n_features
        )
        
        hp_sweep_results[optimizer_name] = result
        
        if result['best_config']:
            print(f"  🏆 Best config: {result['best_config']}")
            print(f"  📊 Best val R²: {result['best_result']['val_r2']:.4f}")
        else:
            print(f"  ❌ No valid configuration found!")
    
    # Phase 2: Final comparison with best hyperparameters
    print("\n" + "=" * 70)
    print("📈 Phase 2: Final comparison with best hyperparameters")
    print("-" * 50)
    
    # Filter out optimizers with no valid configs
    valid_configs = {k: v for k, v in hp_sweep_results.items() if v['best_config'] is not None}
    
    if not valid_configs:
        print("❌ No valid configurations found for any optimizer!")
        return
    
    print(f"Valid optimizers: {list(valid_configs.keys())}")
    
    # Run final comparison on all datasets
    dataset_names = ['friedman', 'high_dimensional', 'nonlinear_interaction']
    final_results = run_final_comparison(valid_configs, dataset_names)
    
    # Phase 3: Analysis and visualization
    print("\n" + "=" * 70)
    print("📊 Phase 3: Analysis and Results")
    print("-" * 50)
    
    # Print summary
    print("\n🏆 HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 40)
    
    for optimizer_name, result in hp_sweep_results.items():
        print(f"\n{optimizer_name.upper()}:")
        if result['best_config']:
            print(f"  🔧 Best config: {result['best_config']}")
            print(f"  📊 Best val R²: {result['best_result']['val_r2']:.4f}")
            print(f"  ✅ Converged: {result['converged_configs']}/{result['total_configs_tested']}")
        else:
            print(f"  ❌ No valid configuration found")
            print(f"  💥 All {result['total_configs_tested']} configs diverged")
    
    print(f"\n🏆 FINAL COMPARISON (Best R² by Dataset)")
    print("=" * 50)
    
    # Create summary table
    summary_data = []
    for dataset_name in dataset_names:
        row = {'Dataset': dataset_name.replace('_', ' ').title()}
        best_r2 = -float('inf')
        best_optimizer = None
        
        for optimizer_name in valid_configs.keys():
            if (optimizer_name in final_results[dataset_name] and 
                'test_r2' in final_results[dataset_name][optimizer_name]):
                r2 = final_results[dataset_name][optimizer_name]['test_r2']
                row[optimizer_name.upper()] = f"{r2:.3f}"
                if r2 > best_r2:
                    best_r2 = r2
                    best_optimizer = optimizer_name
            else:
                row[optimizer_name.upper()] = "FAIL"
        
        row['Winner'] = best_optimizer.upper() if best_optimizer else "None"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Find overall winner
    print(f"\n🎯 OVERALL ANALYSIS")
    print("=" * 30)
    
    wins = {}
    for dataset_name in dataset_names:
        best_r2 = -float('inf')
        best_optimizer = None
        
        for optimizer_name in valid_configs.keys():
            if (optimizer_name in final_results[dataset_name] and 
                'test_r2' in final_results[dataset_name][optimizer_name]):
                r2 = final_results[dataset_name][optimizer_name]['test_r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_optimizer = optimizer_name
        
        if best_optimizer:
            wins[best_optimizer] = wins.get(best_optimizer, 0) + 1
    
    if wins:
        sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
        print(f"🏆 Overall winner: {sorted_wins[0][0].upper()} ({sorted_wins[0][1]} datasets)")
        for optimizer, win_count in sorted_wins:
            print(f"   {optimizer.upper()}: {win_count} dataset(s) won")
    
    # Surprising findings
    print(f"\n💡 KEY FINDINGS:")
    
    # Check if Muon performed better with lower learning rates
    if 'muon' in hp_sweep_results and hp_sweep_results['muon']['best_config']:
        muon_lr = hp_sweep_results['muon']['best_config']['lr']
        print(f"   - Muon's best LR: {muon_lr} (much lower than typical optimizers)")
    
    # Check convergence rates
    print(f"   - Convergence rates:")
    for optimizer_name, result in hp_sweep_results.items():
        conv_rate = result['converged_configs'] / result['total_configs_tested'] * 100
        print(f"     {optimizer_name.upper()}: {conv_rate:.1f}% of configs converged")
    
    # Generate visualizations
    generate_visualizations(hp_sweep_results, final_results)
    
    # Save results
    save_results(hp_sweep_results, final_results)
    
    print(f"\n✅ Hyperparameter sweep completed successfully!")
    print(f"📁 Generated files:")
    print(f"   - optimizers_hp_sweep_results.png")
    print(f"   - optimizers_hp_sweep_comparison.png") 
    print(f"   - optimizer_hp_sweep_results.json")

if __name__ == "__main__":
    main()