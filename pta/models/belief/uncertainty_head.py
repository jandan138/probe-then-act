"""Uncertainty head — calibrated scalar uncertainty from latent belief."""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyHead(nn.Module):
    """Produce a calibrated uncertainty estimate from the latent belief.

    Takes the latent vector *z* and outputs a scalar (or per-dimension)
    uncertainty measure suitable for risk-aware downstream decision-making.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent vector.
    hidden_dim : int
        Width of hidden layers.
    output_dim : int
        Number of uncertainty outputs (1 for scalar, or per-dimension).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute calibrated uncertainty.

        Parameters
        ----------
        z : torch.Tensor
            Latent belief vector of shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Uncertainty estimate of shape ``(B, output_dim)``.
        """
        raise NotImplementedError
