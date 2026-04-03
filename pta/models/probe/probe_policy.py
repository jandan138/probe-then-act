"""Probe policy — selects informative probe actions (tap / press / drag)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from pta.models.probe.probe_action_space import ProbeAction


class ProbePolicy(nn.Module):
    """Given an observation, output a probe action to gather tactile information.

    The policy outputs a categorical distribution over probe primitives
    together with continuous parameters for the selected primitive (location,
    force, speed, etc.).

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation / fused feature vector.
    hidden_dim : int
        Width of hidden layers.
    num_primitives : int
        Number of probe primitive types (default follows ``ProbeAction``).
    param_dim : int
        Dimensionality of continuous probe parameters per primitive.
    """

    def __init__(
        self,
        obs_dim: int = 256,
        hidden_dim: int = 256,
        num_primitives: int = len(ProbeAction),
        param_dim: int = 6,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_primitives = num_primitives
        self.param_dim = param_dim
        raise NotImplementedError

    def forward(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select a probe action.

        Parameters
        ----------
        observation : torch.Tensor
            Observation features of shape ``(B, obs_dim)``.

        Returns
        -------
        primitive_logits : torch.Tensor
            Logits over probe primitives, shape ``(B, num_primitives)``.
        params : torch.Tensor
            Continuous probe parameters, shape ``(B, param_dim)``.
        """
        raise NotImplementedError
