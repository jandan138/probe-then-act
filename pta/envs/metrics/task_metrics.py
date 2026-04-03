"""Task-level metrics: success rate and transfer efficiency.

These are the primary metrics reported in the paper tables.
"""

from __future__ import annotations

from typing import List

import numpy as np


def compute_success_rate(
    episode_results: List[dict],
    threshold: float = 0.8,
) -> float:
    """Fraction of episodes where transfer efficiency >= *threshold*.

    Parameters
    ----------
    episode_results:
        List of per-episode result dicts, each containing
        ``"transfer_efficiency"``.
    threshold:
        Minimum transfer efficiency to count as success.

    Returns
    -------
    float
        Success rate in [0, 1].
    """
    raise NotImplementedError


def compute_transfer_efficiency(
    particles_in_target: int,
    total_particles: int,
) -> float:
    """Fraction of particles that ended up in the target container.

    Parameters
    ----------
    particles_in_target:
        Number of MPM particles inside the target AABB.
    total_particles:
        Total number of MPM particles in the scene.

    Returns
    -------
    float
        Transfer efficiency in [0, 1].
    """
    raise NotImplementedError
