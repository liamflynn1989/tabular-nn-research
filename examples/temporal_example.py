"""
Temporal Modulation Example: Handling Concept Drift

This example demonstrates:
1. Creating synthetic data with temporal distribution shift
2. Training with temporal modulation
3. Comparing performance with/without temporal awareness
"""

import sys
sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from implementations import TemporalTabularModel, create_temporal_model
from implementations.base import MLP


def generate_temporal_data(
    n_samples: int = 3000,
    n_features: int = 10,
    n_time_periods: int = 5,
    drift_strength: float = 0.5,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic regression data with temporal drift.
    
    The relationship y = f(x) changes over time to simulate concept drift.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of input features
        n_time_periods: Number of distinct time periods
        drift_strength: How much the relationship changes over time
        random_state: Random seed
        
    Returns:
        X: Features (n_samples, n_features)
        y: Targets (n_samples,)
        time_idx: Time period indices (n_samples,)
    """
    np.random.seed(random_state)
    
    samples_per_period = n_samples // n_time_periods
    
    X_list = []
    y_list = []
    time_list = []
    
    # Base coefficients
    base_coef = np.random.randn(n_features)
    
    for t in range(n_time_periods):
        # Generate features
        X_t = np.random.randn(samples_per_period, n_features)
        
        # Temporal drift: coefficients change over time
        drift = np.sin(2 * np.pi * t / n_time_periods) * drift_strength
        temporal_coef = base_coef + drift * np.random.randn(n_features)
        
        # Also add non-linear interaction that changes with time
        interaction_strength = 0.3 * np.cos(2 * np.pi * t / n_time_periods)
        
        # Generate targets with time-varying relationship
        y_t = (
            X_t @ temporal_coef 
            + interaction_strength * X_t[:, 0] * X_t[:, 1]
            + 0.1 * np.random.randn(samples_per_period)
        )
        
        X_list.append(X_t)
        y_list.append(y_t)
        time_list.append(np.full(samples_per_period, t))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    time_idx = np.concatenate(time_list)
    
    return (
        torch.FloatTensor(X),
        torch.FloatTensor(y).unsqueeze(1),
        torch.LongTensor(time_idx),
    )


def train_epoch(model, dataloader, optimizer, criterion, device, use_time=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        if use_time:
            X_batch, y_batch, time_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            time_batch = time_batch.to(device)
        else:
            X_batch, y_batch = batch[:2]
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            time_batch = None
        
        optimizer.zero_grad()
        
        if use_time and hasattr(model, 'time_encoder'):
            outputs = model(X_batch, time_batch)
        else:
            outputs = model(X_batch)
        
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_time=True):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    
    for batch in dataloader:
        if use_time:
            X_batch, y_batch, time_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            time_batch = time_batch.to(device)
        else:
            X_batch, y_batch = batch[:2]
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            time_batch = None
        
        if use_time and hasattr(model, 'time_encoder'):
            outputs = model(X_batch, time_batch)
        else:
            outputs = model(X_batch)
        
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_per_period(model, X, y, time_idx, criterion, device, use_time=True):
    """Evaluate model performance for each time period."""
    model.eval()
    
    unique_times = torch.unique(time_idx).tolist()
    results = {}
    
    for t in unique_times:
        mask = time_idx == t
        X_t = X[mask].to(device)
        y_t = y[mask].to(device)
        time_t = time_idx[mask].to(device) if use_time else None
        
        if use_time and hasattr(model, 'time_encoder'):
            pred = model(X_t, time_t)
        else:
            pred = model(X_t)
        
        loss = criterion(pred, y_t)
        results[t] = loss.item()
    
    return results


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data with temporal drift
    print("\nGenerating data with temporal drift...")
    X, y, time_idx = generate_temporal_data(
        n_samples=3000,
        n_features=10,
        n_time_periods=5,
        drift_strength=0.8,
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}, time_idx={time_idx.shape}")
    print(f"Time periods: {torch.unique(time_idx).tolist()}")
    
    # Split into train/test (using last time period as test)
    train_mask = time_idx < 4
    test_mask = time_idx >= 4
    
    X_train, y_train, time_train = X[train_mask], y[train_mask], time_idx[train_mask]
    X_test, y_test, time_test = X[test_mask], y[test_mask], time_idx[test_mask]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train, time_train)
    test_dataset = TensorDataset(X_test, y_test, time_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    criterion = nn.MSELoss()
    
    # =====================
    # Model 1: Standard MLP (no temporal awareness)
    # =====================
    print("\n" + "="*50)
    print("Training Standard MLP (no temporal awareness)")
    print("="*50)
    
    mlp = MLP(
        d_in=10,
        d_out=1,
        n_blocks=3,
        d_block=128,
        dropout=0.1,
    ).to(device)
    
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3)
    
    for epoch in range(100):
        train_loss = train_epoch(mlp, train_loader, optimizer, criterion, device, use_time=False)
        if (epoch + 1) % 25 == 0:
            test_loss = evaluate(mlp, test_loader, criterion, device, use_time=False)
            print(f"Epoch {epoch+1:3d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
    
    mlp_test_loss = evaluate(mlp, test_loader, criterion, device, use_time=False)
    
    # =====================
    # Model 2: Temporal Model with FiLM modulation
    # =====================
    print("\n" + "="*50)
    print("Training Temporal Model (with FiLM modulation)")
    print("="*50)
    
    temporal_model = TemporalTabularModel(
        d_in=10,
        d_out=1,
        d_time=16,
        n_blocks=3,
        d_block=128,
        dropout=0.1,
        modulation_type="scale_shift",
    ).to(device)
    
    optimizer = optim.AdamW(temporal_model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        train_loss = train_epoch(
            temporal_model, train_loader, optimizer, criterion, device, use_time=True
        )
        if (epoch + 1) % 25 == 0:
            test_loss = evaluate(
                temporal_model, test_loader, criterion, device, use_time=True
            )
            print(f"Epoch {epoch+1:3d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
    
    temporal_test_loss = evaluate(
        temporal_model, test_loader, criterion, device, use_time=True
    )
    
    # =====================
    # Compare Results
    # =====================
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    print(f"Standard MLP Test MSE: {mlp_test_loss:.4f}")
    print(f"Temporal Model Test MSE: {temporal_test_loss:.4f}")
    print(f"Improvement: {(mlp_test_loss - temporal_test_loss) / mlp_test_loss * 100:.1f}%")
    
    # Evaluate per time period
    print("\n" + "="*50)
    print("Performance by Time Period")
    print("="*50)
    
    mlp_per_period = evaluate_per_period(
        mlp, X, y, time_idx, criterion, device, use_time=False
    )
    temporal_per_period = evaluate_per_period(
        temporal_model, X, y, time_idx, criterion, device, use_time=True
    )
    
    print(f"{'Period':<10} {'MLP MSE':<15} {'Temporal MSE':<15} {'Diff':<10}")
    print("-" * 50)
    for t in sorted(mlp_per_period.keys()):
        mlp_mse = mlp_per_period[t]
        temp_mse = temporal_per_period[t]
        diff = mlp_mse - temp_mse
        marker = "train" if t < 4 else "test"
        print(f"{t} ({marker})<8} {mlp_mse:<15.4f} {temp_mse:<15.4f} {diff:+.4f}")
    
    # =====================
    # Visualize temporal encoding
    # =====================
    print("\n" + "="*50)
    print("Temporal Encoding Analysis")
    print("="*50)
    
    temporal_model.eval()
    time_indices = torch.arange(5).to(device)
    time_encodings = temporal_model.time_encoder(time_indices)
    
    print("Time encoding similarity matrix:")
    sim_matrix = torch.cosine_similarity(
        time_encodings.unsqueeze(0), 
        time_encodings.unsqueeze(1), 
        dim=-1
    )
    
    for i in range(5):
        row = [f"{sim_matrix[i, j]:.2f}" for j in range(5)]
        print(f"  Period {i}: [{', '.join(row)}]")


if __name__ == "__main__":
    main()
