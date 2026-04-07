"""Task policy — belief-conditioned policy for the Probe-Then-Act pipeline.

Architecture decision (2026-04-08):
    The task policy is an SB3 PPO MlpPolicy.  Belief conditioning is handled
    at the *observation level* — ProbePhaseWrapper appends the latent z from
    the belief encoder to every observation:

        obs_policy = [base_obs, z]       shape: (base_obs_dim + latent_dim,)

    Because SB3's MlpPolicy processes flat observation vectors through its
    own MLP feature extractor, no custom network is needed.  This module
    provides a thin helper class that:

    1. Documents the obs = [base_obs, z] convention.
    2. Provides ``make_obs()`` for manual inference / evaluation outside SB3.
    3. Stores metadata (dimensions) for logging and checkpoint reload.

    For training, just pass the ProbePhaseWrapper-wrapped env to PPO and
    SB3 handles the rest.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class TaskPolicy(nn.Module):
    """Belief-conditioned task policy (thin wrapper for SB3 PPO).

    This module concatenates base observations with the latent belief
    vector z to form the input to the SB3 MlpPolicy.  It does NOT contain
    trainable parameters itself — the actual policy weights live inside
    PPO's MlpPolicy.

    Parameters
    ----------
    base_obs_dim : int
        Dimensionality of the base observation (from JointResidualWrapper).
    latent_dim : int
        Dimensionality of the latent belief vector z.
    action_dim : int
        Dimensionality of the output action (7 for Franka joint residuals).
    """

    def __init__(
        self,
        base_obs_dim: int = 22,
        latent_dim: int = 16,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.base_obs_dim = base_obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.total_obs_dim = base_obs_dim + latent_dim

    @staticmethod
    def make_obs(
        base_obs: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Concatenate base observation and belief vector.

        Parameters
        ----------
        base_obs : np.ndarray
            Shape ``(base_obs_dim,)`` or ``(B, base_obs_dim)``.
        z : np.ndarray
            Shape ``(latent_dim,)`` or ``(B, latent_dim)``.

        Returns
        -------
        np.ndarray
            Concatenated observation ``[base_obs, z]``.
        """
        return np.concatenate([base_obs, z], axis=-1).astype(np.float32)

    def forward(
        self,
        obs: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the combined observation tensor.

        If *z* is provided separately, concatenates ``[obs, z]``.
        If *z* is None, assumes *obs* already contains z (i.e. came from
        ProbePhaseWrapper) and returns it unchanged.

        Parameters
        ----------
        obs : torch.Tensor
            Base observation ``(B, base_obs_dim)`` or pre-augmented
            observation ``(B, base_obs_dim + latent_dim)``.
        z : torch.Tensor, optional
            Latent belief ``(B, latent_dim)``.  Ignored if None.
        sigma : torch.Tensor, optional
            Uncertainty (reserved for future use, currently ignored).

        Returns
        -------
        torch.Tensor
            Combined observation ``(B, base_obs_dim + latent_dim)``.
        """
        if z is not None:
            return torch.cat([obs, z], dim=-1)
        return obs
