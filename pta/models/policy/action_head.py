"""Action head — convert policy features to delta end-effector pose."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Map policy features to a delta end-effector pose command.

    Outputs a 6-DoF (or 7-DoF with gripper) incremental pose that is
    sent to the low-level robot controller.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of input features.
    hidden_dim : int
        Width of hidden layers.
    action_dim : int
        Dimensionality of the output delta pose (e.g. 7 = 6-DoF + gripper).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        raise NotImplementedError

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict delta end-effector pose.

        Parameters
        ----------
        features : torch.Tensor
            Policy features of shape ``(B, feature_dim)``.

        Returns
        -------
        torch.Tensor
            Delta end-effector pose of shape ``(B, action_dim)``.
        """
        raise NotImplementedError
