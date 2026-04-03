"""Rollout storage buffer for on-policy PPO training."""

from __future__ import annotations

from typing import Dict, Optional

import torch


class RolloutStorage:
    """Fixed-length buffer that stores transitions for PPO updates.

    Parameters
    ----------
    num_steps : int
        Number of environment steps collected per rollout.
    num_envs : int
        Number of parallel environments.
    obs_shape : tuple[int, ...]
        Shape of a single observation tensor.
    action_dim : int
        Dimensionality of the action space.
    device : torch.device
        Device on which tensors are allocated.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        raise NotImplementedError

    def insert(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Insert a single transition into the buffer.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape ``(num_envs, *obs_shape)``.
        actions : torch.Tensor
            Actions taken, shape ``(num_envs, action_dim)``.
        log_probs : torch.Tensor
            Log-probabilities of the actions, shape ``(num_envs,)``.
        values : torch.Tensor
            Value estimates, shape ``(num_envs,)``.
        rewards : torch.Tensor
            Rewards received, shape ``(num_envs,)``.
        dones : torch.Tensor
            Episode termination flags, shape ``(num_envs,)``.
        """
        raise NotImplementedError

    def compute_returns(
        self,
        next_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE-based returns and advantages in-place.

        Parameters
        ----------
        next_value : torch.Tensor
            Bootstrap value estimate for the state after the last step.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda for bias-variance trade-off.
        """
        raise NotImplementedError

    def batch_generator(
        self,
        num_mini_batches: int,
    ) -> Dict[str, torch.Tensor]:
        """Yield randomised mini-batches for PPO updates.

        Parameters
        ----------
        num_mini_batches : int
            Number of mini-batches to split the data into.

        Yields
        ------
        dict[str, torch.Tensor]
            Dictionary with keys ``obs``, ``actions``, ``log_probs``,
            ``values``, ``returns``, ``advantages``.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the buffer for the next rollout collection."""
        raise NotImplementedError
