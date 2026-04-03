"""Export result tables as CSV / LaTeX for the paper."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict


def main() -> None:
    """Aggregate results and export tables for the paper.

    Reads evaluation result JSONs, computes summary statistics, and
    writes CSV and LaTeX table files to ``results/tables/``.

    Usage::

        python -m pta.scripts.export_tables --results-dir results/
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
