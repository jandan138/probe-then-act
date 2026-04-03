"""SensorBuilder -- Attaches cameras and tactile sensors to a scene.

Genesis sensors include RGB-D cameras (``gs.sensors.Camera``) and
kinematic contact probes for tactile feedback.
"""

from __future__ import annotations

from typing import Any, Dict

import genesis as gs


class SensorBuilder:
    """Build and register all sensors for a Genesis scene."""

    def build_sensors(
        self,
        scene: Any,
        robot: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create cameras, tactile sensors, and proprioception hooks.

        Parameters
        ----------
        scene:
            A ``gs.Scene`` instance (not yet built).
        robot:
            The robot entity to which sensors are attached.
        config:
            Sensor sub-config describing camera intrinsics/extrinsics,
            tactile probe locations, and proprioception fields.

        Returns
        -------
        dict[str, Any]
            Mapping from sensor name (e.g. ``"wrist_cam"``,
            ``"tactile_left"``) to the sensor handle.
        """
        raise NotImplementedError

    def _add_camera(self, scene: Any, cam_cfg: Dict[str, Any]) -> Any:
        """Add a single camera sensor to *scene*."""
        raise NotImplementedError

    def _add_tactile(self, scene: Any, robot: Any, tactile_cfg: Dict[str, Any]) -> Any:
        """Add a KinematicContactProbe-based tactile sensor."""
        raise NotImplementedError
