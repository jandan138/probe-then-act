"""Auxiliary prediction heads — material and dynamics self-supervision."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaterialPredictionHead(nn.Module):
    """Predict material properties from the latent belief vector.

    Used as an auxiliary self-supervised loss to encourage the belief
    encoder to capture material-relevant information.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent vector.
    hidden_dim : int
        Width of hidden layers.
    num_classes : int
        Number of discrete material categories.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict material class logits.

        Parameters
        ----------
        z : torch.Tensor
            Latent belief vector of shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Logits over material classes, shape ``(B, num_classes)``.
        """
        raise NotImplementedError


class DynamicsPredictionHead(nn.Module):
    """Predict next-step dynamics from belief and action.

    Auxiliary head for self-supervised learning: given the current belief
    and an action, predict the next observation delta.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent belief vector.
    action_dim : int
        Dimensionality of the action vector.
    hidden_dim : int
        Width of hidden layers.
    obs_delta_dim : int
        Dimensionality of the predicted observation delta.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 7,
        hidden_dim: int = 256,
        obs_delta_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.obs_delta_dim = obs_delta_dim
        raise NotImplementedError

    def forward(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict observation delta.

        Parameters
        ----------
        z : torch.Tensor
            Latent belief vector of shape ``(B, latent_dim)``.
        action : torch.Tensor
            Action vector of shape ``(B, action_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted observation delta of shape ``(B, obs_delta_dim)``.
        """
        raise NotImplementedError
