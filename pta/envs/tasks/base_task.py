"""BaseTask -- Abstract interface for all Genesis-based manipulation tasks.

Every task owns the episode lifecycle:
  reset -> (step, compute_reward, compute_metrics, is_done)* -> reset ...

Sub-classes must implement all abstract methods.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Tuple

import numpy as np
import torch


class BaseTask(abc.ABC):
    """Abstract base class for a single manipulation task.

    A task operates on a fully-built :class:`SceneComponents` instance
    and is responsible for episode logic (reset, step, reward, done).
    It does **not** own scene construction -- that belongs to the builders.
    """

    def __init__(self, scene_components: Any, config: Dict[str, Any]) -> None:
        """Store scene handles and task-level config.

        Parameters
        ----------
        scene_components:
            A :class:`~pta.envs.builders.scene_builder.SceneComponents`
            dataclass with handles to scene, robot, tool, containers,
            and sensors.
        config:
            Task-level config (horizon, reward weights, thresholds, ...).
        """
        self.scene_components = scene_components
        self.config = config
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the episode and return the initial observation dict.

        Returns
        -------
        dict[str, Tensor]
            Observation dictionary with keys matching the sensor stack.
        """
        ...

    @abc.abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """Execute one environment step.

        Parameters
        ----------
        action:
            Joint-space or end-effector action tensor.

        Returns
        -------
        obs:
            Next observation dict.
        reward:
            Scalar reward for this transition.
        done:
            Whether the episode has terminated.
        info:
            Auxiliary information (metrics, debug data).
        """
        ...

    @abc.abstractmethod
    def compute_reward(self) -> float:
        """Compute the scalar reward for the current state.

        Returns
        -------
        float
            Reward value (positive = good).
        """
        ...

    @abc.abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        """Compute task-level metrics for logging / evaluation.

        Returns
        -------
        dict[str, float]
            Metric name -> value.  Must include at least
            ``"success_rate"`` and ``"transfer_efficiency"``.
        """
        ...

    @abc.abstractmethod
    def is_done(self) -> bool:
        """Return True if the episode should terminate."""
        ...
