"""Evaluate a trained task policy on a given environment split."""

from __future__ import annotations

from typing import Any, Dict


def evaluate_policy(
    policy: Any,
    env: Any,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run a trained policy for multiple episodes and compute metrics.

    Parameters
    ----------
    policy : Any
        Trained task policy (teacher or student).
    env : Any
        Gym-compatible environment instance.
    config : dict
        Evaluation configuration (``n_episodes``, ``max_steps``,
        ``record_video``, ``seed``, ...).

    Returns
    -------
    dict[str, float]
        Evaluation metrics including at minimum ``success_rate``,
        ``mean_return``, ``mean_episode_length``.
    """
    raise NotImplementedError
