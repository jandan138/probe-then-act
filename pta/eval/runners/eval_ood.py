"""Evaluate a policy across out-of-distribution splits."""

from __future__ import annotations

from typing import Any, Dict, List


def evaluate_ood(
    policy: Any,
    splits: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Run a policy on multiple OOD evaluation splits.

    Parameters
    ----------
    policy : Any
        Trained task policy.
    splits : list[dict[str, Any]]
        List of OOD split configurations.  Each dictionary must contain
        a ``name`` key and environment parameter overrides.
    config : dict
        Shared evaluation configuration (``n_episodes``, ``seed``, ...).

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from split name to its evaluation metrics dictionary.
    """
    raise NotImplementedError
