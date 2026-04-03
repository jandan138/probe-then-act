"""Debug script: run probe actions and visualise the resulting belief."""

from __future__ import annotations

import argparse
from typing import Any, Dict


def main(config: Dict[str, Any] | None = None) -> None:
    """Execute probe actions in an environment and visualise outputs.

    Runs the probe policy for a configurable number of steps, then
    displays the resulting latent belief, material predictions, and
    uncertainty estimates.

    Parameters
    ----------
    config : dict, optional
        Debug configuration.  If ``None``, reads from CLI args.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
