"""Proprioception encoder — maps joint / end-effector states to a feature vector."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProprioceptionEncoder(nn.Module):
    """Encode proprioceptive (joint-state) information into a feature vector.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the proprioceptive state (joint positions,
        velocities, torques, gripper width, etc.).
    feature_dim : int
        Dimensionality of the output feature vector.
    hidden_dim : int
        Width of hidden layers in the MLP.
    num_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        state_dim: int = 14,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        raise NotImplementedError

    def forward(self, proprio_state: torch.Tensor) -> torch.Tensor:
        """Encode proprioceptive state.

        Parameters
        ----------
        proprio_state : torch.Tensor
            Tensor of shape ``(B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(B, feature_dim)``.
        """
        raise NotImplementedError
