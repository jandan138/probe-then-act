"""CLI entry point for student training (behavioural cloning / distillation)."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict


def main() -> None:
    """Parse CLI arguments and launch student training.

    Usage::

        python -m pta.scripts.train_student --config configs/student.yaml
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
