"""Visualise a simulation scene (objects, robot, cameras)."""

from __future__ import annotations

import argparse
from typing import Any, Dict


def main(config: Dict[str, Any] | None = None) -> None:
    """Render the simulation scene for visual inspection.

    Opens an interactive viewer or saves a static image of the
    configured scene, including robot, objects, and camera placements.

    Parameters
    ----------
    config : dict, optional
        Scene configuration.  If ``None``, reads from CLI args.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
