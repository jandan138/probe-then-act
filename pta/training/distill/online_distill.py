"""Online distillation — student learns while teacher acts in the environment."""

from __future__ import annotations

from typing import Any, Dict, Optional


def online_distillation(
    teacher: Any,
    student: Any,
    env: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform online distillation from teacher to student.

    The teacher and student interact with the environment simultaneously.
    At each step the student is trained to match the teacher's action
    distribution via a KL-divergence loss, while optionally also
    receiving the environment reward signal.

    Parameters
    ----------
    teacher : Any
        Trained privileged teacher policy.
    student : Any
        Student policy to be trained.
    env : Any
        Gym-compatible environment.
    config : dict, optional
        Distillation hyper-parameters (lr, kl_weight, total_steps, ...).

    Returns
    -------
    dict[str, Any]
        Training summary including loss curves and final metrics.
    """
    raise NotImplementedError
