#!/usr/bin/env python3
"""
Example training script for TabKANet.

Demonstrates how to train TabKANet on a binary classification task
using synthetic tabular data.

TabKANet uses Kolmogorov-Arnold Networks (KAN) with learnable B-spline
functions to embed numerical features, then feeds through a Transformer encoder.
"""

import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import argparse

from implementations.tabkanet import TabKANet


class TabularDataset(Dataset):
    """Simple PyTorch Dataset for tabular data."""

    def __init__(self, X_num, X_cat=None, y=None):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y
        self.n_samples = X_num.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = {'x_num': torch.tensor(self.X_num[idx], dtype=torch.float32)}
        if self.X_cat is not None:
            item['x_cat'] = torch.tensor(self.X_cat[idx], dtype=torch.long)
        if self.y is not None:
            item['y'] = torch.tensor(self.y[idx], dtype=torch.float32)
        return item


def create_synthetic_data(
    n_samples: int = 10000,
    n_numerical: int = 10,
    n_categorical: int = 5,
    random_state: int = 42,
):
    """Create synthetic tabular data for testing."""
    X_num, y = make_classification(
        n_samples=n_samples,
        n_features=n_numerical,
        n_informative=n_numerical // 2,
        n_redundant=n_numerical // 4,
        n_classes=2,
        random_state=random_state,
    )

    rng = np.random.RandomState(random_state)
    X_cat = np.column_stack([
        rng.randint(0, cardinality, n_samples)
        for cardinality in range(3, 3 + n_categorical)
    ])

    num_categories = list(range(3, 3 + n_categorical))

    return X_num.astype(np.float32), X_cat.astype(np.int64), y.astype(np.float32), num_categories


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x_num = batch['x_num'].to(device)
        x_cat = batch.get('x_cat')
        if x_cat is not None:
            x_cat = x_cat.to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = criterion(logits.squeeze(1), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        x_num = batch['x_num'].to(device)
        x_cat = batch.get('x_cat')
        if x_cat is not None:
            x_cat = x_cat.to(device)
        y = batch['y'].to(device)

        logits = model(x_num, x_cat)
        loss = criterion(logits.squeeze(1), y)
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()

        total_loss += loss.item()
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    avg_loss = total_loss / len(dataloader)
    accuracy = (all_preds == all_targets).float().mean().item()

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train TabKANet on tabular data')
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-numerical', type=int, default=10)
    parser.add_argument('--n-categorical', type=int, default=5)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--num-splines', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--noise-std', type=float, default=0.01,
                        help='Noise to add to numerical features during training')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create synthetic data
    print("Creating synthetic dataset...")
    X_num, X_cat, y, num_categories = create_synthetic_data(
        n_samples=args.n_samples,
        n_numerical=args.n_numerical,
        n_categorical=args.n_categorical,
        random_state=args.seed,
    )

    # Split data
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_num, X_cat, y, test_size=0.2, random_state=args.seed
    )
    X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_num_train, X_cat_train, y_train, test_size=0.125, random_state=args.seed
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Create datasets and dataloaders
    train_dataset = TabularDataset(X_num_train, X_cat_train, y_train)
    val_dataset = TabularDataset(X_num_val, X_cat_val, y_val)
    test_dataset = TabularDataset(X_num_test, X_cat_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    print(f"\nCreating TabKANet model...")
    model = TabKANet(
        num_numerical=args.n_numerical,
        num_categories=num_categories,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_classes=2,
        task='classification',
        num_splines=args.num_splines,
        noise_std=args.noise_std,
    )

    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    print("\nTraining...")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
