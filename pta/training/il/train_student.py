"""Train a student policy via behavioural cloning."""

from __future__ import annotations

from typing import Any, Dict


def train_student_bc(config: Dict[str, Any]) -> None:
    """Train the student policy using behavioural cloning on teacher data.

    The student only has access to sensor observations (no privileged
    state).  It learns to imitate the teacher's actions from a
    pre-collected demonstration dataset.

    Parameters
    ----------
    config : dict
        Training configuration containing at minimum:
        - ``dataset``: path to collected teacher demonstrations.
        - ``policy``: student network architecture.
        - ``bc``: behavioural-cloning hyper-parameters (lr, epochs, batch_size).
        - ``training``: seed, checkpoint frequency.
    """
    raise NotImplementedError
