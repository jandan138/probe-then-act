"""Privileged teacher — oracle policy with access to hidden sim state."""

from __future__ import annotations

import torch
import torch.nn as nn


class PrivilegedTeacher(nn.Module):
    """Teacher policy that has access to ground-truth material parameters.

    Used in simulation only. The teacher observes the full hidden state
    (friction, stiffness, damping, etc.) and learns an optimal policy,
    which is later distilled into the student.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the standard observation features.
    material_param_dim : int
        Dimensionality of the hidden material parameter vector.
    hidden_dim : int
        Width of hidden layers.
    action_dim : int
        Dimensionality of the output action.
    num_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int = 256,
        material_param_dim: int = 16,
        hidden_dim: int = 256,
        action_dim: int = 7,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.material_param_dim = material_param_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        raise NotImplementedError

    def forward(
        self,
        obs: torch.Tensor,
        hidden_material_params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute privileged action.

        Parameters
        ----------
        obs : torch.Tensor
            Observation features of shape ``(B, obs_dim)``.
        hidden_material_params : torch.Tensor
            Ground-truth material parameters of shape
            ``(B, material_param_dim)``.

        Returns
        -------
        torch.Tensor
            Action vector of shape ``(B, action_dim)``.
        """
        raise NotImplementedError
