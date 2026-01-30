"""
Utility functions for benchmarking.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from .runner import BenchmarkResult


def format_results_table(
    results: List[BenchmarkResult],
    metric: str = "test_rmse",
) -> str:
    """
    Format benchmark results as a markdown table.

    Args:
        results: List of BenchmarkResult objects
        metric: Which metric to display (test_rmse, test_mse, test_mae)

    Returns:
        Markdown-formatted table string
    """
    # Organize by dataset and model
    datasets = sorted(set(r.dataset_name for r in results))
    models = sorted(set(r.model_name for r in results))

    # Build lookup dict
    lookup = {}
    for r in results:
        key = (r.dataset_name, r.model_name)
        lookup[key] = getattr(r, metric)

    # Find best per dataset for highlighting
    best_per_dataset = {}
    for dataset in datasets:
        scores = [lookup.get((dataset, m), float('inf')) for m in models]
        best_per_dataset[dataset] = min(scores)

    # Build table
    lines = []

    # Header
    header = "| Model | " + " | ".join(datasets) + " | Avg |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(datasets) + 2)) + "|")

    # Rows
    for model in models:
        row = [model]
        scores = []
        for dataset in datasets:
            score = lookup.get((dataset, model))
            if score is not None:
                scores.append(score)
                # Bold if best
                if score == best_per_dataset[dataset]:
                    row.append(f"**{score:.4f}**")
                else:
                    row.append(f"{score:.4f}")
            else:
                row.append("-")

        # Average
        if scores:
            avg = sum(scores) / len(scores)
            row.append(f"{avg:.4f}")
        else:
            row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_leaderboard(
    results: List[BenchmarkResult],
    metric: str = "test_rmse",
) -> str:
    """
    Format results as a leaderboard ranked by average performance.

    Args:
        results: List of BenchmarkResult objects
        metric: Which metric to rank by

    Returns:
        Markdown-formatted leaderboard string
    """
    # Calculate average per model
    model_scores: Dict[str, List[float]] = {}
    for r in results:
        if r.model_name not in model_scores:
            model_scores[r.model_name] = []
        model_scores[r.model_name].append(getattr(r, metric))

    # Compute averages and sort
    model_avgs = [(m, sum(s)/len(s)) for m, s in model_scores.items()]
    model_avgs.sort(key=lambda x: x[1])

    lines = [
        "| Rank | Model | Avg RMSE |",
        "|------|-------|----------|",
    ]

    for rank, (model, avg) in enumerate(model_avgs, 1):
        medal = ""
        if rank == 1:
            medal = " :1st_place_medal:"
        elif rank == 2:
            medal = " :2nd_place_medal:"
        elif rank == 3:
            medal = " :3rd_place_medal:"

        lines.append(f"| {rank} | {model}{medal} | {avg:.4f} |")

    return "\n".join(lines)


def save_results(
    results: List[BenchmarkResult],
    path: str = "benchmarks/results.json",
) -> None:
    """Save benchmark results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_results(path: str = "benchmarks/results.json") -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        data = json.load(f)

    return [
        BenchmarkResult(
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
        for d in data
    ]
