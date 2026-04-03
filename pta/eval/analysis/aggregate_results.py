"""Aggregate evaluation results across seeds and splits."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def aggregate_seeds(
    results_dir: Optional[Path] = None,
) -> "pd.DataFrame":
    """Load per-seed result JSONs and aggregate into a single DataFrame.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing per-seed result files.  Defaults to
        ``results/`` in the project root.

    Returns
    -------
    pandas.DataFrame
        One row per (method, task, split, seed) with columns for every
        metric.  Includes ``mean`` and ``std`` summary rows.
    """
    raise NotImplementedError
