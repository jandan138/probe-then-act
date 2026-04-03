"""Risk penalty -- Penalises spill, excessive force, and unsafe states.

Applied as a negative reward component to discourage risky behaviour
during training.
"""

from __future__ import annotations

from typing import Any, Dict


def compute_risk_penalty(
    task_state: Dict[str, Any],
    config: Dict[str, Any],
) -> float:
    """Compute the risk penalty for the current state.

    Parameters
    ----------
    task_state:
        Current task state dict with keys such as
        ``"spilled_particles"``, ``"contact_force_magnitude"``,
        ``"joint_limit_violation"``.
    config:
        Penalty sub-config with keys ``"spill_weight"``,
        ``"force_threshold"``, ``"force_weight"``, etc.

    Returns
    -------
    float
        Non-negative penalty value (will be **subtracted** from reward).
    """
    raise NotImplementedError
