"""
Benchmark runner for comparing tabular models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
import time
import json
from pathlib import Path

from data.datasets import load_dataset, list_datasets


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    dataset_name: str
    train_mse: float
    val_mse: float
    test_mse: float
    test_rmse: float
    test_mae: float
    train_time: float
    n_parameters: int
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "train_mse": self.train_mse,
            "val_mse": self.val_mse,
            "test_mse": self.test_mse,
            "test_rmse": self.test_rmse,
            "test_mae": self.test_mae,
            "train_time": self.train_time,
            "n_parameters": self.n_parameters,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkResult":
        return cls(
            model_name=d["model_name"],
            dataset_name=d["dataset_name"],
            train_mse=d["train_mse"],
            val_mse=d["val_mse"],
            test_mse=d["test_mse"],
            test_rmse=d["test_rmse"],
            test_mae=d["test_mae"],
            train_time=d["train_time"],
            n_parameters=d["n_parameters"],
            config=d.get("config", {}),
        )


class BenchmarkRunner:
    """
    Run benchmarks comparing different models on different datasets.

    Supports caching: previously computed results are loaded from cache_file
    and only new model/dataset combinations are run.
    """

    def __init__(
        self,
        device: str = "auto",
        n_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        verbose: bool = True,
        cache_file: Optional[str] = None,
        force: bool = False,
    ):
        """
        Args:
            device: Device to use ("auto", "cpu", "cuda")
            n_epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            patience: Early stopping patience
            verbose: Print progress
            cache_file: Path to cache file (JSON). If provided, loads cached results
                       and skips already-computed model/dataset pairs.
            force: If True, ignore cache and re-run all benchmarks
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.verbose = verbose
        self.cache_file = cache_file
        self.force = force
        self.results: List[BenchmarkResult] = []
        self._cached_keys: Set[Tuple[str, str]] = set()

        # Load cached results if available
        if cache_file and not force:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load cached results from file."""
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                for d in data:
                    result = BenchmarkResult.from_dict(d)
                    self.results.append(result)
                    self._cached_keys.add((result.model_name, result.dataset_name))
                if self.verbose:
                    print(f"Loaded {len(self.results)} cached results from {self.cache_file}")
            except (json.JSONDecodeError, KeyError) as e:
                if self.verbose:
                    print(f"Warning: Could not load cache ({e}), starting fresh")

    def is_cached(self, model_name: str, dataset_name: str) -> bool:
        """Check if a model/dataset combination is already cached."""
        return (model_name, dataset_name) in self._cached_keys

    def get_cached_result(self, model_name: str, dataset_name: str) -> Optional[BenchmarkResult]:
        """Get cached result for a model/dataset combination."""
        for r in self.results:
            if r.model_name == model_name and r.dataset_name == dataset_name:
                return r
        return None

    def _create_dataloaders(self, dataset, train_ratio=0.7, val_ratio=0.15):
        """Split dataset and create dataloaders."""
        n = len(dataset)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_set, val_set, test_set = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader

    def _train_epoch(self, model, dataloader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in dataloader:
            x_num = batch['x_num'].to(self.device)
            y = batch['y'].to(self.device)
            x_cat = batch.get('x_cat')
            if x_cat is not None:
                x_cat = x_cat.to(self.device)
            time_idx = batch.get('time_idx')
            if time_idx is not None:
                time_idx = time_idx.to(self.device)

            optimizer.zero_grad()

            # Handle different model interfaces
            if hasattr(model, 'time_encoder') and time_idx is not None:
                pred = model(x_num, time_idx)
            elif x_cat is not None and hasattr(model, 'cat_embeddings'):
                pred = model(x_num, x_cat)
            elif hasattr(model, '_add_candidates'):
                # TabR: pass labels for candidate accumulation
                pred = model(x_num, y_for_candidates=y)
            elif hasattr(model, '_is_setup') and model._is_setup:
                # iLTM and similar models that are already setup
                pred = model(x_num)
            else:
                pred = model(x_num)

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def _evaluate(self, model, dataloader, criterion):
        """Evaluate model and return metrics."""
        model.eval()
        total_mse = 0
        total_mae = 0
        n_samples = 0

        for batch in dataloader:
            x_num = batch['x_num'].to(self.device)
            y = batch['y'].to(self.device)
            x_cat = batch.get('x_cat')
            if x_cat is not None:
                x_cat = x_cat.to(self.device)
            time_idx = batch.get('time_idx')
            if time_idx is not None:
                time_idx = time_idx.to(self.device)

            if hasattr(model, 'time_encoder') and time_idx is not None:
                pred = model(x_num, time_idx)
            elif x_cat is not None and hasattr(model, 'cat_embeddings'):
                pred = model(x_num, x_cat)
            else:
                pred = model(x_num)

            total_mse += criterion(pred, y).item() * y.size(0)
            total_mae += torch.abs(pred - y).sum().item()
            n_samples += y.size(0)

        mse = total_mse / n_samples
        mae = total_mae / n_samples
        rmse = np.sqrt(mse)

        return mse, rmse, mae

    def run_single(
        self,
        model: nn.Module,
        model_name: str,
        dataset_name: str,
        dataset_kwargs: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark for a single model-dataset combination.

        Args:
            model: PyTorch model instance
            model_name: Name for the model
            dataset_name: Name of the dataset to use
            dataset_kwargs: Additional kwargs for dataset loading
            config: Model configuration to store in results

        Returns:
            BenchmarkResult with all metrics
        """
        if dataset_kwargs is None:
            dataset_kwargs = {}

        dataset = load_dataset(dataset_name, **dataset_kwargs)
        train_loader, val_loader, test_loader = self._create_dataloaders(dataset)

        model = model.to(self.device)
        
        # Handle models that need setup (e.g., iLTM)
        if hasattr(model, 'setup') and callable(model.setup):
            # Collect all training data for setup
            all_x = []
            all_y = []
            for batch in train_loader:
                all_x.append(batch['x_num'])
                all_y.append(batch['y'])
            X_train = torch.cat(all_x, dim=0).to(self.device)
            y_train = torch.cat(all_y, dim=0).to(self.device)
            model.setup(X_train, y_train)
            # Also set candidates for retrieval
            if hasattr(model, 'set_candidates'):
                model.set_candidates(X_train, y_train)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        best_val_mse = float('inf')
        best_state = None
        patience_counter = 0
        train_mse = 0

        start_time = time.time()

        for epoch in range(self.n_epochs):
            train_mse = self._train_epoch(model, train_loader, optimizer, criterion)
            val_mse, _, _ = self._evaluate(model, val_loader, criterion)

            scheduler.step(val_mse)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: train_mse={train_mse:.4f}, val_mse={val_mse:.4f}")

        train_time = time.time() - start_time

        # Load best model and evaluate on test set
        model.load_state_dict(best_state)
        test_mse, test_rmse, test_mae = self._evaluate(model, test_loader, criterion)

        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            train_mse=train_mse,
            val_mse=best_val_mse,
            test_mse=test_mse,
            test_rmse=test_rmse,
            test_mae=test_mae,
            train_time=train_time,
            n_parameters=n_params,
            config=config or {},
        )

        self.results.append(result)

        if self.verbose:
            print(f"  {model_name} on {dataset_name}: test_rmse={test_rmse:.4f}")

        return result

    def run_all(
        self,
        model_factories: Dict[str, Callable],
        dataset_names: Optional[List[str]] = None,
        n_samples: int = 5000,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for all model-dataset combinations.

        Skips combinations that are already cached (unless force=True).

        Args:
            model_factories: Dict mapping model names to factory functions
                             Factory signature: (dataset_info) -> model
            dataset_names: List of dataset names to use (default: all)
            n_samples: Number of samples per dataset

        Returns:
            List of all BenchmarkResults (including cached)
        """
        if dataset_names is None:
            dataset_names = list_datasets()

        # Count what needs to be run
        total_pairs = len(dataset_names) * len(model_factories)
        cached_count = sum(
            1 for d in dataset_names for m in model_factories
            if self.is_cached(m, d)
        )
        to_run = total_pairs - cached_count

        if self.verbose:
            print(f"\nBenchmark pairs: {total_pairs} total, {cached_count} cached, {to_run} to run")

        if to_run == 0:
            if self.verbose:
                print("All results cached. Use --force to re-run.")
            return self.results

        for dataset_name in dataset_names:
            # Check if any models need to run for this dataset
            models_to_run = [
                (name, factory) for name, factory in model_factories.items()
                if not self.is_cached(name, dataset_name)
            ]

            if not models_to_run:
                if self.verbose:
                    print(f"\nDataset: {dataset_name} (all cached, skipping)")
                continue

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Dataset: {dataset_name}")
                print('='*50)

            dataset = load_dataset(dataset_name, n_samples=n_samples)

            for model_name, factory in model_factories.items():
                if self.is_cached(model_name, dataset_name):
                    if self.verbose:
                        cached = self.get_cached_result(model_name, dataset_name)
                        print(f"\nModel: {model_name} (cached, test_rmse={cached.test_rmse:.4f})")
                    continue

                if self.verbose:
                    print(f"\nModel: {model_name}")

                model = factory(dataset.info)
                self.run_single(
                    model=model,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    dataset_kwargs={"n_samples": n_samples},
                )

        return self.results

    def get_results_table(self) -> Dict[str, Dict[str, float]]:
        """
        Get results as a nested dict: {dataset: {model: test_rmse}}.
        """
        table = {}
        for r in self.results:
            if r.dataset_name not in table:
                table[r.dataset_name] = {}
            table[r.dataset_name][r.model_name] = r.test_rmse
        return table
