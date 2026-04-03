"""Evaluate a probe policy in isolation."""

from __future__ import annotations

from typing import Any, Dict


def evaluate_probe(
    probe_policy: Any,
    env: Any,
) -> Dict[str, float]:
    """Run the probe policy and measure information-gathering quality.

    Parameters
    ----------
    probe_policy : Any
        A probe policy that selects exploratory actions to identify
        object properties.
    env : Any
        Gym-compatible environment instance.

    Returns
    -------
    dict[str, float]
        Probe-specific metrics such as ``material_accuracy``,
        ``mass_estimation_error``, ``probe_steps_used``.
    """
    raise NotImplementedError
