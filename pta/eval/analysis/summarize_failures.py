"""Categorise evaluation failures according to a failure taxonomy."""

from __future__ import annotations

from typing import Any, Dict, List


def categorize_failures(
    rollout_infos: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Bin failed episodes into failure categories.

    The taxonomy includes categories such as ``grasp_slip``,
    ``wrong_material_estimate``, ``collision``, ``timeout``, and
    ``other``.

    Parameters
    ----------
    rollout_infos : list[dict[str, Any]]
        Per-episode info dictionaries from evaluation rollouts.

    Returns
    -------
    dict[str, int]
        Mapping from failure category name to count.
    """
    raise NotImplementedError
