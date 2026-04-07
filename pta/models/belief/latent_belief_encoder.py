"""Latent belief encoder — aggregate probe traces into a belief state."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentBeliefEncoder(nn.Module):
    """Map a set of probe traces to a latent belief distribution.

    The encoder outputs ``(z, sigma)`` — a latent mean and a diagonal
    standard-deviation vector — so that downstream modules can reason
    about epistemic uncertainty.

    Architecture: a per-trace MLP encoder is applied independently to
    each probe trace, the resulting features are mean-pooled across
    probes, and two linear heads produce ``z`` (mean) and ``log_sigma``
    (transformed to ``sigma`` via softplus).

    Parameters
    ----------
    trace_dim : int
        Dimensionality of each probe-trace observation (default 30,
        matching the JointResidualWrapper output: 22 base + 7 q_base
        + 1 step_frac).
    latent_dim : int
        Dimensionality of the latent belief vector *z*.
    hidden_dim : int
        Width of hidden layers.
    num_layers : int
        Number of hidden layers in the per-trace encoder.
    """

    def __init__(
        self,
        trace_dim: int = 30,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.trace_dim = trace_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # -- per-trace encoder (shared across probes) -------------------
        layers: list[nn.Module] = []
        in_dim = trace_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.trace_encoder = nn.Sequential(*layers)

        # -- output heads -----------------------------------------------
        self.z_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

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
        # Encode each trace independently: (B, N, trace_dim) -> (B, N, hidden_dim)
        h = self.trace_encoder(probe_traces)

        # Mean-pool across probes: (B, N, hidden_dim) -> (B, hidden_dim)
        h = h.mean(dim=1)

        # Output heads
        z = self.z_head(h)
        log_sigma = self.log_sigma_head(h)
        sigma = F.softplus(log_sigma) + 1e-6

        return z, sigma

    # -- utility methods ------------------------------------------------

    def sample(
        self, z: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Draw a sample via the reparameterization trick.

        Parameters
        ----------
        z : torch.Tensor
            Latent mean of shape ``(B, latent_dim)``.
        sigma : torch.Tensor
            Standard deviation of shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Sampled latent vector of shape ``(B, latent_dim)``.
        """
        eps = torch.randn_like(sigma)
        return z + sigma * eps

    @staticmethod
    def kl_divergence(
        z: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence from q(z) = N(z, diag(sigma^2)) to N(0, I).

        Parameters
        ----------
        z : torch.Tensor
            Latent mean of shape ``(B, latent_dim)``.
        sigma : torch.Tensor
            Standard deviation of shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Per-sample KL divergence of shape ``(B,)``.
        """
        # KL(N(mu, sigma^2) || N(0, I))
        #   = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        return 0.5 * torch.sum(
            sigma.pow(2) + z.pow(2) - 1.0 - 2.0 * torch.log(sigma), dim=-1
        )
