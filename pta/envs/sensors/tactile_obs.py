"""TactileObservation -- Tactile signal from Genesis contact probes.

Uses ``KinematicContactProbe`` to measure contact forces/locations on
the tool surface.  The raw signal is processed into a fixed-size
tensor for the tactile encoder.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


class TactileObservation:
    """Extract tactile features from Genesis contact probes.

    Usage::

        tact_obs = TactileObservation(probe_handle, config)
        obs = tact_obs.get_observation()
    """

    def __init__(self, probe_handle: Any, config: Dict[str, Any]) -> None:
        """Store probe handle and processing config.

        Parameters
        ----------
        probe_handle:
            Genesis KinematicContactProbe sensor handle.
        config:
            Tactile observation config (num_taxels, normalisation, etc.).
        """
        self.probe = probe_handle
        self.config = config

    def get_observation(self) -> Dict[str, torch.Tensor]:
        """Read current tactile data and return as a tensor.

        Returns
        -------
        dict[str, Tensor]
            ``{"tactile": Tensor[num_taxels, 3]}`` with
            (normal_force, shear_x, shear_y) per taxel.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal buffers (e.g. running statistics)."""
        raise NotImplementedError
