"""Contact metrics -- Measures contact-related failures.

Tracks episodes where excessive contact force or unexpected collision
events occurred (e.g. tool collision with container walls, robot
self-collision).
"""

from __future__ import annotations

from typing import List


def compute_contact_failure_rate(
    episode_results: List[dict],
    force_threshold: float = 50.0,
) -> float:
    """Fraction of episodes with at least one unsafe contact event.

    Parameters
    ----------
    episode_results:
        List of per-episode result dicts, each containing
        ``"max_contact_force"`` (float).
    force_threshold:
        Contact force (N) above which an event is considered unsafe.

    Returns
    -------
    float
        Contact failure rate in [0, 1].  Lower is better.
    """
    raise NotImplementedError
