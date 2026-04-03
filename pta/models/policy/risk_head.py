"""Risk head — predict task-specific risk scores."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RiskHead(nn.Module):
    """Estimate per-category risk from observation, belief, and uncertainty.

    Outputs calibrated probabilities for several risk categories that can
    be used to gate or modify actions at execution time.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of observation features.
    latent_dim : int
        Dimensionality of the latent belief vector.
    uncertainty_dim : int
        Dimensionality of the uncertainty vector.
    hidden_dim : int
        Width of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int = 256,
        latent_dim: int = 64,
        uncertainty_dim: int = 1,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.uncertainty_dim = uncertainty_dim
        self.hidden_dim = hidden_dim
        raise NotImplementedError

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict risk scores.

        Parameters
        ----------
        obs : torch.Tensor
            Observation features of shape ``(B, obs_dim)``.
        z : torch.Tensor
            Latent belief of shape ``(B, latent_dim)``.
        sigma : torch.Tensor
            Uncertainty of shape ``(B, uncertainty_dim)``.

        Returns
        -------
        spill_risk : torch.Tensor
            Probability of spill, shape ``(B, 1)``.
        jam_risk : torch.Tensor
            Probability of mechanical jam, shape ``(B, 1)``.
        instability_risk : torch.Tensor
            Probability of grasp / placement instability, shape ``(B, 1)``.
        """
        raise NotImplementedError
