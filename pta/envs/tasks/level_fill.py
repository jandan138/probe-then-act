"""LevelFillTask -- Fill a target container to a specified level.

Secondary task for the Probe-Then-Act paper.

Episode flow:
  1. Probe phase  -- identify material properties via diagnostic actions.
  2. Fill phase   -- repeatedly scoop and deposit material until the
     target container reaches the desired fill level.

Success: absolute fill-level error <= ``level_tolerance`` without
exceeding the spill budget.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from pta.envs.tasks.base_task import BaseTask


class LevelFillTask(BaseTask):
    """Level-and-Fill manipulation task."""

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset episode: set target fill level, randomise material.

        Returns
        -------
        dict[str, Tensor]
            Initial observation from the sensor stack.
        """
        raise NotImplementedError

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """Apply *action*, step physics, return (obs, reward, done, info)."""
        raise NotImplementedError

    def compute_reward(self) -> float:
        """Reward based on fill-level progress and spill penalty."""
        raise NotImplementedError

    def compute_metrics(self) -> Dict[str, float]:
        """Return ``fill_level_error``, ``success_rate``, ``spill_ratio``."""
        raise NotImplementedError

    def is_done(self) -> bool:
        """Done when target level reached, horizon exceeded, or spill budget exhausted."""
        raise NotImplementedError

    def _measure_fill_level(self) -> float:
        """Estimate fill level (fraction) in the target container."""
        raise NotImplementedError
