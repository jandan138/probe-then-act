"""ObservationStack -- Combines all sensor observations into one dict.

Central aggregation point that collects camera, tactile, and
proprioception observations, applies optional noise, and returns
a single flat dict suitable for the policy's multimodal encoder.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from pta.envs.sensors.camera_obs import CameraObservation
from pta.envs.sensors.tactile_obs import TactileObservation
from pta.envs.sensors.proprio_obs import ProprioceptionObservation


class ObservationStack:
    """Aggregate observations from all sensors.

    Usage::

        stack = ObservationStack(sensors, config)
        obs = stack.get_observation()
    """

    def __init__(
        self,
        cameras: List[CameraObservation],
        tactile: List[TactileObservation],
        proprio: ProprioceptionObservation,
        config: Dict[str, Any],
    ) -> None:
        """Register all observation providers.

        Parameters
        ----------
        cameras:
            List of camera observation providers.
        tactile:
            List of tactile observation providers.
        proprio:
            Proprioception observation provider.
        config:
            Stack-level config (which modalities to include, frame
            stacking depth, etc.).
        """
        self.cameras = cameras
        self.tactile = tactile
        self.proprio = proprio
        self.config = config

    def get_observation(self) -> Dict[str, torch.Tensor]:
        """Collect and merge observations from all providers.

        Returns
        -------
        dict[str, Tensor]
            Combined observation dict with prefixed keys, e.g.
            ``"cam0_rgb"``, ``"tactile0_tactile"``, ``"joint_pos"``.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset all sub-providers."""
        raise NotImplementedError

    def observation_spec(self) -> Dict[str, tuple]:
        """Return the expected shape for each observation key.

        Returns
        -------
        dict[str, tuple[int, ...]]
            Mapping from observation key to its tensor shape.
        """
        raise NotImplementedError
