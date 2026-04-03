"""SensorBuilder -- Attaches cameras and tactile sensors to a scene.

Genesis sensors include RGB-D cameras (``scene.add_camera``) and
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
        robot: Any = None,
        config: Dict[str, Any] | None = None,
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
            Mapping from sensor name to the sensor handle.
        """
        cfg = config or {}
        sensors: Dict[str, Any] = {}

        # Overhead camera (default)
        cam_cfg = cfg.get("overhead_cam", {
            "res": (128, 128),
            "pos": (1.2, -0.3, 0.8),
            "lookat": (0.5, 0.15, 0.1),
            "fov": 60,
        })
        sensors["overhead_cam"] = self._add_camera(scene, cam_cfg)

        # Optional wrist camera
        if "wrist_cam" in cfg:
            sensors["wrist_cam"] = self._add_camera(scene, cfg["wrist_cam"])

        return sensors

    def _add_camera(self, scene: Any, cam_cfg: Dict[str, Any]) -> Any:
        """Add a single camera sensor to *scene*."""
        return scene.add_camera(
            res=cam_cfg.get("res", (128, 128)),
            pos=cam_cfg.get("pos", (1.2, -0.3, 0.8)),
            lookat=cam_cfg.get("lookat", (0.5, 0.15, 0.1)),
            fov=cam_cfg.get("fov", 60),
            GUI=False,
        )

    def _add_tactile(self, scene: Any, robot: Any, tactile_cfg: Dict[str, Any]) -> Any:
        """Add a KinematicContactProbe-based tactile sensor.

        Reserved for future implementation -- tactile sensing requires
        contact force queries that are not yet wired up in v1.
        """
        # Placeholder: returns None until tactile probes are implemented
        return None
