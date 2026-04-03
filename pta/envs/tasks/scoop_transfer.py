"""ScoopTransferTask -- Scoop material from source and transfer to target.

Primary task for the Probe-Then-Act paper.

Episode flow:
  1. Probe phase  -- short diagnostic motions to identify material.
  2. Scoop phase  -- insert tool into source container, acquire material.
  3. Transfer phase -- move tool to target container, deposit material.

Success: >= ``success_threshold`` fraction of material reaches the target
without excessive spill or contact failure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from pta.envs.tasks.base_task import BaseTask


class ScoopTransferTask(BaseTask):
    """Scoop-and-Transfer manipulation task."""

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset episode: randomise material, fill source container.

        Returns
        -------
        dict[str, Tensor]
            Initial observation from the sensor stack.
        """
        raise NotImplementedError

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """Apply *action*, step physics, return (obs, reward, done, info).

        Parameters
        ----------
        action:
            Robot action tensor (joint velocities or EE delta).
        """
        raise NotImplementedError

    def compute_reward(self) -> float:
        """Reward = task_reward - risk_penalty + shaping.

        Combines transfer efficiency reward, spill penalty, and
        optional potential-based shaping terms.
        """
        raise NotImplementedError

    def compute_metrics(self) -> Dict[str, float]:
        """Return ``success_rate``, ``transfer_efficiency``, ``spill_ratio``."""
        raise NotImplementedError

    def is_done(self) -> bool:
        """Done when horizon exceeded or early termination triggered."""
        raise NotImplementedError

    def _count_particles_in_target(self) -> int:
        """Count MPM particles inside the target container AABB."""
        raise NotImplementedError

    def _count_spilled_particles(self) -> int:
        """Count MPM particles outside both containers."""
        raise NotImplementedError
