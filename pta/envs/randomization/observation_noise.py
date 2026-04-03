"""ObservationNoise -- Injects noise into sensor observations.

Used for domain randomisation (training) and OOD sensor-perturbation
evaluation.  Supports additive Gaussian, multiplicative scaling,
salt-and-pepper, and dropout masks.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


class ObservationNoise:
    """Apply configurable noise to observation tensors.

    Usage::

        noise = ObservationNoise(config)
        noisy_obs = noise.apply(obs_dict, rng)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Configure noise types and magnitudes.

        Parameters
        ----------
        config:
            Noise sub-config with per-modality settings, e.g.
            ``{"camera": {"std": 0.01}, "tactile": {"dropout": 0.05}}``.
        """
        self.config = config

    def apply(
        self,
        obs: Dict[str, torch.Tensor],
        rng: np.random.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Return a noisy copy of *obs*.

        Parameters
        ----------
        obs:
            Observation dict from the sensor stack.
        rng:
            Numpy random generator for reproducibility.

        Returns
        -------
        dict[str, Tensor]
            Observation dict with noise injected.
        """
        raise NotImplementedError

    def _gaussian_noise(self, tensor: torch.Tensor, std: float, rng: np.random.Generator) -> torch.Tensor:
        """Add zero-mean Gaussian noise with standard deviation *std*."""
        raise NotImplementedError

    def _dropout_noise(self, tensor: torch.Tensor, rate: float, rng: np.random.Generator) -> torch.Tensor:
        """Zero out elements with probability *rate*."""
        raise NotImplementedError
