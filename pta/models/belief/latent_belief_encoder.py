"""Latent belief encoder — aggregate probe traces into a belief state."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LatentBeliefEncoder(nn.Module):
    """Map a set of probe traces to a latent belief distribution.

    The encoder outputs ``(z, sigma)`` — a latent mean and a diagonal
    standard-deviation vector — so that downstream modules can reason
    about epistemic uncertainty.

    Parameters
    ----------
    trace_dim : int
        Dimensionality of each encoded probe-trace feature.
    latent_dim : int
        Dimensionality of the latent belief vector *z*.
    hidden_dim : int
        Width of hidden layers.
    num_layers : int
        Number of hidden layers in the aggregation network.
    """

    def __init__(
        self,
        trace_dim: int = 128,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.trace_dim = trace_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        raise NotImplementedError

    def forward(
        self, probe_traces: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode probe traces into a belief distribution.

        Parameters
        ----------
        probe_traces : torch.Tensor
            Set of encoded probe-trace features, shape
            ``(B, N, trace_dim)`` where *N* is the number of probes.

        Returns
        -------
        z : torch.Tensor
            Latent belief mean of shape ``(B, latent_dim)``.
        sigma : torch.Tensor
            Diagonal standard deviation of shape ``(B, latent_dim)``.
        """
        raise NotImplementedError
