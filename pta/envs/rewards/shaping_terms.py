"""Potential-based reward shaping terms.

Optional dense reward signals that guide learning without changing
the optimal policy (when designed as potential differences).
"""

from __future__ import annotations

from typing import Any, Dict


def compute_shaping(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    config: Dict[str, Any],
) -> float:
    """Compute potential-based shaping reward.

    Parameters
    ----------
    prev_state:
        Task state at the previous timestep.
    curr_state:
        Task state at the current timestep.
    config:
        Shaping sub-config with keys ``"distance_weight"``,
        ``"alignment_weight"``, etc.

    Returns
    -------
    float
        Shaping reward (phi(s') - phi(s)).
    """
    raise NotImplementedError
