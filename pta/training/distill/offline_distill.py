"""Offline distillation — student learns from pre-collected teacher data."""

from __future__ import annotations

from typing import Any, Dict, Optional


def offline_distillation(
    teacher_data: Any,
    student: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform offline distillation from pre-collected teacher rollouts.

    The student is trained purely from a static dataset of teacher
    observations and action distributions, without any environment
    interaction.

    Parameters
    ----------
    teacher_data : Any
        Pre-collected dataset of teacher rollouts (observations, actions,
        action distributions).
    student : Any
        Student policy to be trained.
    config : dict, optional
        Distillation hyper-parameters (lr, epochs, batch_size, ...).

    Returns
    -------
    dict[str, Any]
        Training summary including loss curves and final metrics.
    """
    raise NotImplementedError
