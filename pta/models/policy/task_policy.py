"""Task policy — maps (observation, belief, uncertainty) to a task action."""

from __future__ import annotations

import torch
import torch.nn as nn


class TaskPolicy(nn.Module):
    """High-level task policy that conditions on latent belief.

    Given the current observation features, latent belief *z*, and
    calibrated uncertainty *sigma*, output a task-level action.

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
    action_dim : int
        Dimensionality of the output action.
    """

    def __init__(
        self,
        obs_dim: int = 256,
        latent_dim: int = 64,
        uncertainty_dim: int = 1,
        hidden_dim: int = 256,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.uncertainty_dim = uncertainty_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        raise NotImplementedError

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute task action.

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
        torch.Tensor
            Action vector of shape ``(B, action_dim)``.
        """
        raise NotImplementedError
