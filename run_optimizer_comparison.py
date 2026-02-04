#!/usr/bin/env python3
"""
Standalone script to run optimizer comparison for tabular neural networks.
This generates the results and visualizations for the optimizer tutorial.
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
from sklearn.metrics import mean_squared_error, r2_score
import time
from typing import Dict, List, Tuple
from collections import Counter

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

# Set device and seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

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

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 200,
    patience: int = 50,
    verbose: bool = False
) -> Dict:
    """Train model and return training history."""
    
    model.train()
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': [],
        'time_per_epoch': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['epoch'].append(epoch)
        history['time_per_epoch'].append(epoch_time)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
            
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    total_time = time.time() - start_time
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'epochs_trained': len(history['train_loss'])
    }

def main():
    """Run the optimizer comparison experiment."""
    
    print("🚀 Starting Optimizer Comparison for Tabular Neural Networks")
    print("=" * 60)
    
    # Load datasets
    print("\n📊 Loading datasets...")
    datasets = {
        'friedman': load_dataset('friedman', n_samples=1500, random_state=42),
        'high_dimensional': load_dataset('high_dimensional', n_samples=1500, n_features=40, 
                                       n_informative=8, random_state=42),
        'nonlinear_interaction': load_dataset('nonlinear_interaction', n_samples=1500, 
                                            n_features=12, random_state=42)
    }
    
    # Display dataset info
    for name, dataset in datasets.items():
        info = dataset.info
        print(f"  ✅ {name}: {info.n_samples} samples, {info.n_numerical} features")
    
    # Optimizers to test
    optimizers_to_test = ['adamw', 'muon', 'shampoo', 'novograd']
    results = {}
    
    print("\n🔧 Running optimizer comparison...")
    
    for dataset_name, dataset in datasets.items():
        print(f"\n📈 Testing on {dataset_name.upper()} dataset")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(dataset)
        n_features = X_train.shape[1]
        
        results[dataset_name] = {}
        
        for opt_name in optimizers_to_test:
            print(f"  🔧 {opt_name.upper()}: ", end="")
            
            try:
                # Create fresh model
                model = create_model(n_features)
                
                # Create optimizer with dataset-specific learning rates
                lr = 0.001  # Base learning rate
                if opt_name == 'muon':
                    lr = 0.005  # Muon typically needs higher lr
                elif opt_name == 'shampoo':
                    lr = 0.001  # Shampoo with reasonable lr
                elif opt_name == 'novograd':
                    lr = 0.001  # NovoGrad with reasonable lr
                
                optimizer = get_optimizer(opt_name, model, lr=lr)
                
                # Train model
                result = train_model(
                    model, optimizer, X_train, y_train, X_val, y_val,
                    epochs=200, patience=30, verbose=False
                )
                
                # Test performance
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test)
                    test_mse = F.mse_loss(test_pred, y_test).item()
                    test_r2 = r2_score(
                        y_test.cpu().numpy(), 
                        test_pred.cpu().numpy()
                    )
                
                results[dataset_name][opt_name] = {
                    'test_mse': test_mse,
                    'test_r2': test_r2,
                    'best_val_loss': result['best_val_loss'],
                    'total_time': result['total_time'],
                    'epochs_trained': result['epochs_trained'],
                    'history': result['history'],
                    'learning_rate': lr
                }
                
                print(f"MSE={test_mse:.4f}, R²={test_r2:.3f}, Time={result['total_time']:.1f}s")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                results[dataset_name][opt_name] = {
                    'test_mse': float('inf'),
                    'test_r2': -float('inf'),
                    'error': str(e)
                }
    
    print("\n📈 RESULTS SUMMARY")
    print("=" * 60)
    
    # Create results summary
    summary_data = []
    for dataset_name, dataset_results in results.items():
        for opt_name, opt_results in dataset_results.items():
            if 'error' not in opt_results:
                summary_data.append({
                    'Dataset': dataset_name.replace('_', ' ').title(),
                    'Optimizer': opt_name.upper(),
                    'Test R²': f"{opt_results['test_r2']:.3f}",
                    'Test MSE': f"{opt_results['test_mse']:.4f}",
                    'Time (s)': f"{opt_results['total_time']:.1f}",
                    'Epochs': opt_results['epochs_trained']
                })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Find winners
    print("\n🏆 BEST OPTIMIZER BY DATASET")
    print("=" * 40)
    
    winners = {}
    for dataset_name, dataset_results in results.items():
        valid_results = {k: v for k, v in dataset_results.items() 
                        if 'test_r2' in v and v['test_r2'] > -float('inf')}
        
        if valid_results:
            best_opt = max(valid_results.items(), key=lambda x: x[1]['test_r2'])
            winners[dataset_name] = best_opt[0]
            
            print(f"\n{dataset_name.replace('_', ' ').title()}:")
            print(f"  🥇 Winner: {best_opt[0].upper()}")
            print(f"  📊 R² Score: {best_opt[1]['test_r2']:.4f}")
            print(f"  ⏱️  Time: {best_opt[1]['total_time']:.1f}s")
    
    # Overall ranking
    print(f"\n🏆 OVERALL RANKING:")
    winner_count = Counter(winners.values())
    for rank, (optimizer, wins) in enumerate(winner_count.most_common(), 1):
        emoji = ["🏆", "🥈", "🥉", "📉"][min(rank-1, 3)]
        print(f"  {emoji} {rank}. {optimizer.upper()} - {wins} dataset(s) won")
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    
    # Training curves plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Optimizer Comparison: Training Curves', fontsize=16, fontweight='bold')
    
    colors = {'adamw': '#1f77b4', 'muon': '#ff7f0e', 'shampoo': '#2ca02c', 'novograd': '#d62728'}
    
    for i, (dataset_name, dataset_results) in enumerate(results.items()):
        ax1 = axes[0, i]
        ax2 = axes[1, i]
        
        # Training curves
        for opt_name, opt_results in dataset_results.items():
            if 'history' in opt_results:
                history = opt_results['history']
                epochs = range(len(history['train_loss']))
                
                ax1.plot(epochs, history['train_loss'], 
                        color=colors[opt_name], alpha=0.7, 
                        label=f"{opt_name.upper()} (Train)")
                ax1.plot(epochs, history['val_loss'], 
                        color=colors[opt_name], linestyle='--', 
                        label=f"{opt_name.upper()} (Val)")
        
        ax1.set_title(f'{dataset_name.replace("_", " ").title()} - Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_yscale('log')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Performance bars
        opt_names = []
        test_r2s = []
        
        for opt_name, opt_results in dataset_results.items():
            if 'test_r2' in opt_results and opt_results['test_r2'] > -float('inf'):
                opt_names.append(opt_name.upper())
                test_r2s.append(opt_results['test_r2'])
        
        bars = ax2.bar(opt_names, test_r2s, color=[colors[name.lower()] for name in opt_names], alpha=0.8)
        ax2.set_title(f'{dataset_name.replace("_", " ").title()} - Test R² Score')
        ax2.set_ylabel('R² Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, test_r2s):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizers_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: optimizers_comparison.png")
    
    # Efficiency plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Optimizer Efficiency: Performance vs Training Time', fontsize=16, fontweight='bold')
    
    for i, (dataset_name, dataset_results) in enumerate(results.items()):
        ax = axes[i]
        
        for opt_name, opt_results in dataset_results.items():
            if 'test_r2' in opt_results and 'total_time' in opt_results:
                if opt_results['test_r2'] > -float('inf'):
                    ax.scatter(opt_results['total_time'], opt_results['test_r2'], 
                              color=colors[opt_name], s=150, alpha=0.8,
                              label=opt_name.upper(), edgecolors='black', linewidth=1)
                    
                    ax.annotate(opt_name.upper(), 
                               (opt_results['total_time'], opt_results['test_r2']),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, fontweight='bold')
        
        ax.set_title(f'{dataset_name.replace("_", " ").title()}')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Test R² Score')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('optimizers_efficiency.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: optimizers_efficiency.png")
    
    plt.close('all')  # Close plots to free memory
    
    # Save results to JSON
    import json
    results_for_json = {}
    for dataset_name, dataset_results in results.items():
        results_for_json[dataset_name] = {}
        for opt_name, opt_results in dataset_results.items():
            # Convert history to lists for JSON serialization
            json_result = opt_results.copy()
            if 'history' in json_result:
                json_result['history'] = {
                    k: v.tolist() if hasattr(v, 'tolist') else v 
                    for k, v in json_result['history'].items()
                }
            results_for_json[dataset_name][opt_name] = json_result
    
    with open('optimizer_comparison_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print("  ✅ Saved: optimizer_comparison_results.json")
    
    print(f"\n✅ Optimizer comparison completed successfully!")
    print(f"📁 Generated files:")
    print(f"   - optimizers_comparison.png")
    print(f"   - optimizers_efficiency.png") 
    print(f"   - optimizer_comparison_results.json")

if __name__ == "__main__":
    main()