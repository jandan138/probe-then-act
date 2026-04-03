"""ProprioceptionObservation -- Robot joint and end-effector state.

Reads joint positions, velocities, torques, and derived end-effector
pose from the Genesis robot entity.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


class ProprioceptionObservation:
    """Extract proprioceptive state from the robot entity.

    Usage::

        proprio = ProprioceptionObservation(robot_entity, config)
        obs = proprio.get_observation()
    """

    def __init__(self, robot_entity: Any, config: Dict[str, Any]) -> None:
        """Store robot handle and selection config.

        Parameters
        ----------
        robot_entity:
            Genesis robot entity loaded via URDF.
        config:
            Proprioception config (which joints, whether to include
            velocities/torques, EE pose, etc.).
        """
        self.robot = robot_entity
        self.config = config

    def get_observation(self) -> Dict[str, torch.Tensor]:
        """Return current proprioceptive state.

        Returns
        -------
        dict[str, Tensor]
            Keys may include ``"joint_pos"``, ``"joint_vel"``,
            ``"joint_torque"``, ``"ee_pos"``, ``"ee_quat"``.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state."""
        raise NotImplementedError
