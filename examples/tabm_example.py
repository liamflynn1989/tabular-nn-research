"""
TabM Example: Training on a Tabular Classification Task

This example demonstrates:
1. Creating a TabM model
2. Training on synthetic data
3. Analyzing ensemble diversity
"""

import sys
sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from implementations import TabM, create_tabm


def generate_tabular_data(
    n_samples: int = 5000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = 42,
):
    """Generate synthetic tabular classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_classes=n_classes,
        random_state=random_state,
    )
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_test),
        torch.LongTensor(y_train),
        torch.LongTensor(y_test),
    )


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def analyze_ensemble_diversity(model, X_sample, device):
    """Analyze the diversity of ensemble predictions."""
    model.eval()
    X_sample = X_sample.to(device)
    
    # Get predictions from all heads
    all_preds = model(X_sample, return_all_heads=True)  # (batch, heads, classes)
    all_probs = torch.softmax(all_preds, dim=-1)
    
    # Compute diversity metrics
    mean_pred = all_probs.mean(dim=1)
    std_pred = all_probs.std(dim=1)
    
    # Pairwise disagreement between heads
    n_heads = all_preds.shape[1]
    disagreements = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pred_i = all_probs[:, i].argmax(dim=1)
            pred_j = all_probs[:, j].argmax(dim=1)
            disagreement = (pred_i != pred_j).float().mean()
            disagreements.append(disagreement.item())
    
    print("\n=== Ensemble Diversity Analysis ===")
    print(f"Number of heads: {n_heads}")
    print(f"Mean prediction std across heads: {std_pred.mean():.4f}")
    print(f"Mean pairwise disagreement: {np.mean(disagreements):.4f}")
    print(f"Max pairwise disagreement: {np.max(disagreements):.4f}")


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    print("\nGenerating synthetic data...")
    X_train, X_test, y_train, y_test = generate_tabular_data(
        n_samples=5000,
        n_features=20,
        n_classes=3,
    )
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Create model
    print("\nCreating TabM model...")
    model = TabM(
        d_in=20,
        d_out=3,
        n_blocks=3,
        d_block=128,
        n_heads=16,
        dropout=0.1,
        task="classification",
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
    print("\nTraining...")
    best_acc = 0
    
    for epoch in range(50):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )
    
    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    
    # Analyze ensemble diversity
    analyze_ensemble_diversity(model, X_test[:100], device)
    
    # Compare with averaging disabled (single head behavior)
    print("\n=== Comparing Ensemble vs Single Head ===")
    model.eval()
    X_sample = X_test[:100].to(device)
    y_sample = y_test[:100].to(device)
    
    # Full ensemble prediction
    ensemble_pred = model(X_sample).argmax(dim=1)
    ensemble_acc = (ensemble_pred == y_sample).float().mean()
    
    # Individual head predictions
    all_head_preds = model(X_sample, return_all_heads=True)
    head_accs = []
    for h in range(model.n_heads):
        head_pred = all_head_preds[:, h].argmax(dim=1)
        head_acc = (head_pred == y_sample).float().mean()
        head_accs.append(head_acc.item())
    
    print(f"Ensemble accuracy: {ensemble_acc:.4f}")
    print(f"Mean single head accuracy: {np.mean(head_accs):.4f}")
    print(f"Best single head accuracy: {np.max(head_accs):.4f}")
    print(f"Worst single head accuracy: {np.min(head_accs):.4f}")


if __name__ == "__main__":
    main()
