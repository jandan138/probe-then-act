"""VectorEnvWrapper -- Batched environment for parallel rollouts.

Leverages Genesis's built-in GPU parallelism to run multiple
environment instances in a single scene (batch dimension).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper


class VectorEnvWrapper:
    """Vectorised environment that runs *num_envs* instances in parallel.

    Unlike CPU-based vec-env implementations, this wrapper exploits
    Genesis's GPU-batched simulation to step all environments in a
    single kernel launch.

    Usage::

        vec_env = VectorEnvWrapper(env_fn, num_envs=64, config=cfg)
        obs = vec_env.reset()
        obs, rewards, dones, infos = vec_env.step(actions)
    """

    def __init__(
        self,
        env_fn: Any,
        num_envs: int,
        config: Dict[str, Any],
    ) -> None:
        """Create *num_envs* parallel environment instances.

        Parameters
        ----------
        env_fn:
            Callable that returns a :class:`GenesisGymWrapper`.
        num_envs:
            Number of parallel instances.
        config:
            Vector-env config (auto-reset, observation stacking, etc.).
        """
        self.num_envs = num_envs
        self.config = config

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset all environments and return batched observations.

        Returns
        -------
        dict[str, Tensor]
            Observation dict with leading batch dimension ``num_envs``.
        """
        raise NotImplementedError

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Step all environments with *actions*.

        Parameters
        ----------
        actions:
            Batched action tensor, shape ``(num_envs, action_dim)``.

        Returns
        -------
        tuple
            ``(obs, rewards, dones, infos)`` with batch dimension.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release all environment resources."""
        raise NotImplementedError
