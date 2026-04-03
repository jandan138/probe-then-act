"""CameraObservation -- RGB-D image acquisition from Genesis cameras.

Wraps a Genesis camera sensor to produce observation tensors
(RGB image, depth map) ready for the vision encoder.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


class CameraObservation:
    """Extract RGB and depth tensors from a Genesis camera sensor.

    Usage::

        cam_obs = CameraObservation(camera_handle, config)
        obs = cam_obs.get_observation()
    """

    def __init__(self, camera_handle: Any, config: Dict[str, Any]) -> None:
        """Store camera handle and image processing config.

        Parameters
        ----------
        camera_handle:
            Genesis camera sensor returned by :class:`SensorBuilder`.
        config:
            Camera observation config (resolution, normalisation, etc.).
        """
        self.camera = camera_handle
        self.config = config

    def get_observation(self) -> Dict[str, torch.Tensor]:
        """Capture and return the current camera observation.

        Returns
        -------
        dict[str, Tensor]
            ``{"rgb": Tensor[C,H,W], "depth": Tensor[1,H,W]}``.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state (e.g. image buffers)."""
        raise NotImplementedError
