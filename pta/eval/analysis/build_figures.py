"""Build publication-quality figures for the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_ood_figure(
    results_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Build the OOD generalisation bar-chart figure.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing aggregated OOD results.
    output_path : Path, optional
        File path for the saved figure (PDF / PNG).
    """
    raise NotImplementedError


def build_ablation_figure(
    results_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Build the ablation study figure.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing ablation results.
    output_path : Path, optional
        File path for the saved figure (PDF / PNG).
    """
    raise NotImplementedError
