"""Curriculum scheduler for progressive difficulty scaling."""

from __future__ import annotations

from typing import Any, Dict, Optional


class CurriculumScheduler:
    """Schedule environment difficulty over the course of training.

    The scheduler tracks training progress (e.g. success rate, episode
    count) and adjusts environment parameters to progressively increase
    task difficulty.

    Parameters
    ----------
    stages : list[dict[str, Any]]
        Ordered list of curriculum stages.  Each stage is a dictionary
        specifying environment parameter overrides and the promotion
        criterion (e.g. ``{"friction_range": [0.4, 0.6], "promote_at": 0.8}``).
    metric_key : str
        Name of the metric used to decide stage promotion
        (default ``"success_rate"``).
    """

    def __init__(
        self,
        stages: list[Dict[str, Any]],
        metric_key: str = "success_rate",
    ) -> None:
        self.stages = stages
        self.metric_key = metric_key
        self._current_stage: int = 0
        raise NotImplementedError

    @property
    def current_stage(self) -> int:
        """Return the zero-based index of the active curriculum stage."""
        raise NotImplementedError

    def get_env_params(self) -> Dict[str, Any]:
        """Return the environment parameter overrides for the current stage.

        Returns
        -------
        dict[str, Any]
            Parameter dictionary to pass to the environment constructor
            or ``reset`` call.
        """
        raise NotImplementedError

    def step(self, metrics: Dict[str, float]) -> bool:
        """Update the scheduler with the latest training metrics.

        Parameters
        ----------
        metrics : dict[str, float]
            Current training metrics (must include ``metric_key``).

        Returns
        -------
        bool
            ``True`` if the scheduler promoted to a new stage.
        """
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Serialise scheduler state for checkpointing."""
        raise NotImplementedError

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint.

        Parameters
        ----------
        state : dict[str, Any]
            State dictionary previously returned by ``state_dict``.
        """
        raise NotImplementedError
