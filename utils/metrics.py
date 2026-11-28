"""Shared distribution and summary metrics for every world.

The helpers here are deliberately world-agnostic: they operate on plain
numbers, NumPy arrays, or a DataFrame with a named value column. A world can
use them for money, berries, wood, health, or any other quantity.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def calculate_gini(values: Iterable[float]) -> float:
    """Return the Gini coefficient for non-negative values.

    The coefficient is 0 for perfect equality and approaches 1 as one holder
    takes everything. An all-zero distribution is treated as equal.
    """

    values_array = np.asarray(list(values), dtype=float)
    if values_array.size == 0:
        raise ValueError("Input array must not be empty")
    if np.any(values_array < 0):
        raise ValueError("Gini is undefined for negative values")

    total = values_array.sum()
    if total == 0:
        return 0.0

    differences = np.abs(values_array[:, None] - values_array[None, :])
    return float(differences.sum() / (2 * values_array.size * total))


def calculate_statistics(df: pd.DataFrame, column: str = "resources") -> dict[str, float]:
    """Return common summary statistics for one numeric column."""

    if column not in df:
        raise ValueError(f"DataFrame must contain a {column} column")
    values = df[column].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("Cannot calculate statistics for an empty DataFrame")
    return {
        "gini": calculate_gini(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "total": float(np.sum(values)),
    }


class MetricsCalculator:
    """Backward-compatible wrapper around the shared metric helpers."""

    @staticmethod
    def calculate_gini(values: Iterable[float]) -> float:
        return calculate_gini(values)

    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> dict[str, float]:
        return calculate_statistics(df, "resources")
