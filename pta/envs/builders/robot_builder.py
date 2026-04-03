"""RobotBuilder -- Loads the Franka Panda robot into a Genesis scene.

The robot is loaded from an MJCF file using ``gs.morphs.MJCF``.  Joint
limits, initial configuration, and control mode are set according to the
provided config dict.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

import genesis as gs


# Default Franka home configuration (7 arm DOFs + 2 gripper DOFs)
_DEFAULT_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]


class RobotBuilder:
    """Build and configure the Franka Panda robot entity."""

    def build_robot(
        self,
        scene: Any,
        config: Dict[str, Any] | None = None,
    ) -> Any:
        """Add a Franka Panda to *scene* and return the entity handle.

        Parameters
        ----------
        scene:
            A ``gs.Scene`` instance (not yet built).
        config:
            Robot sub-config with keys such as ``pos``,
            ``initial_qpos``, ``coup_friction``.

        Returns
        -------
        entity
            The Genesis entity representing the loaded robot.
        """
        cfg = config or {}
        robot_pos = cfg.get("pos", (0.0, 0.0, 0.0))
        coup_friction = cfg.get("coup_friction", 1.0)
        needs_coup = cfg.get("needs_coup", True)

        robot = scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=needs_coup,
                coup_friction=coup_friction,
            ),
            morph=gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=robot_pos,
            ),
        )
        return robot

    def configure_gains(self, robot: Any) -> None:
        """Set PD gains and force ranges. Must be called after scene.build()."""
        robot.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        robot.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def _set_initial_configuration(self, robot: Any, config: Dict[str, Any] | None = None) -> None:
        """Apply the initial joint positions from *config*."""
        cfg = config or {}
        qpos = cfg.get("initial_qpos", _DEFAULT_QPOS)
        qpos_t = torch.tensor(qpos, dtype=torch.float32, device=gs.device)
        robot.set_qpos(qpos_t)
