"""Student policy — full probe-then-act pipeline without privileged info."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class StudentPolicy(nn.Module):
    """End-to-end student that wraps probe, belief, and task policy.

    At inference time the student:
    1. Selects and executes probe actions (``ProbePolicy``).
    2. Aggregates tactile traces into a belief (``LatentBeliefEncoder``).
    3. Conditions the task policy on the inferred belief and uncertainty.

    Parameters
    ----------
    probe_policy : nn.Module
        A ``ProbePolicy`` instance.
    belief_encoder : nn.Module
        A ``LatentBeliefEncoder`` instance.
    task_policy : nn.Module
        A ``TaskPolicy`` instance.
    max_probes : int
        Maximum number of probe interactions before acting.
    """

    def __init__(
        self,
        probe_policy: nn.Module,
        belief_encoder: nn.Module,
        task_policy: nn.Module,
        max_probes: int = 3,
    ) -> None:
        super().__init__()
        self.probe_policy = probe_policy
        self.belief_encoder = belief_encoder
        self.task_policy = task_policy
        self.max_probes = max_probes
        raise NotImplementedError

    def forward(
        self,
        obs: torch.Tensor,
        probe_traces: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the full probe-then-act pipeline.

        Parameters
        ----------
        obs : torch.Tensor
            Observation features of shape ``(B, obs_dim)``.
        probe_traces : torch.Tensor
            Encoded probe traces of shape ``(B, N, trace_dim)``.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``"action"`` : torch.Tensor of shape ``(B, action_dim)``
            - ``"z"``      : torch.Tensor of shape ``(B, latent_dim)``
            - ``"sigma"``  : torch.Tensor of shape ``(B, latent_dim)``
        """
        raise NotImplementedError
