"""Task reward -- Primary reward signal for manipulation tasks.

Measures progress toward the task goal (e.g. fraction of material
successfully transferred to the target container).
"""

from __future__ import annotations

from typing import Any, Dict


def compute_task_reward(
    task_state: Dict[str, Any],
    config: Dict[str, Any],
) -> float:
    """Compute the primary task reward.

    Parameters
    ----------
    task_state:
        Current task state dict with keys such as
        ``"particles_in_target"``, ``"total_particles"``,
        ``"fill_level"``, etc.
    config:
        Reward sub-config with keys ``"success_bonus"``,
        ``"efficiency_weight"``, etc.

    Returns
    -------
    float
        Non-negative scalar reward.
    """
    raise NotImplementedError
