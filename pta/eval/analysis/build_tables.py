"""Build LaTeX / CSV tables for the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def build_main_table(
    results_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Build the main results table (Table 1 in the paper).

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing aggregated results.
    output_path : Path, optional
        If provided, write the table to this file as LaTeX.

    Returns
    -------
    str
        LaTeX-formatted table string.
    """
    raise NotImplementedError


def build_ablation_table(
    results_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Build the ablation study table.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing ablation results.
    output_path : Path, optional
        If provided, write the table to this file as LaTeX.

    Returns
    -------
    str
        LaTeX-formatted table string.
    """
    raise NotImplementedError
