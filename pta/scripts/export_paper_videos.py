"""Export evaluation rollout videos for the paper."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict


def main() -> None:
    """Export polished rollout videos suitable for the paper / supplement.

    Renders selected rollouts, adds annotations, and creates montage
    grids.

    Usage::

        python -m pta.scripts.export_paper_videos --results-dir results/
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
