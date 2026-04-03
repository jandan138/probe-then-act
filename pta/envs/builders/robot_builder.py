"""RobotBuilder -- Loads the Franka Panda robot into a Genesis scene.

The robot is loaded from a URDF file using ``gs.morphs.URDF``.  Joint
limits, initial configuration, and control mode are set according to the
provided config dict.
"""

from __future__ import annotations

from typing import Any, Dict

import genesis as gs


class RobotBuilder:
    """Build and configure the Franka Panda robot entity."""

    def build_robot(self, scene: Any, config: Dict[str, Any]) -> Any:
        """Add a Franka Panda to *scene* and return the entity handle.

        Parameters
        ----------
        scene:
            A ``gs.Scene`` instance (not yet built).
        config:
            Robot sub-config with keys such as ``urdf_path``,
            ``initial_qpos``, ``control_mode``.

        Returns
        -------
        entity
            The Genesis entity representing the loaded robot.
        """
        raise NotImplementedError

    def _set_initial_configuration(self, robot: Any, config: Dict[str, Any]) -> None:
        """Apply the initial joint positions from *config*."""
        raise NotImplementedError
