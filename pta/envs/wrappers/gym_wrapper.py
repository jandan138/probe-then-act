"""GenesisGymWrapper -- Gymnasium-compatible wrapper for Genesis tasks.

Exposes a :class:`BaseTask` as a standard ``gymnasium.Env`` so that
it can be used with off-the-shelf RL libraries (SB3, CleanRL, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
import torch

from pta.envs.tasks.base_task import BaseTask


class GenesisGymWrapper(gymnasium.Env):
    """Gymnasium wrapper around a Genesis-based :class:`BaseTask`.

    Handles observation/action space definition, tensor <-> numpy
    conversion, and the standard Gymnasium ``reset`` / ``step`` API.
    """

    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, task: BaseTask, config: Dict[str, Any]) -> None:
        """Wrap *task* as a Gymnasium environment.

        Parameters
        ----------
        task:
            An instantiated :class:`BaseTask` subclass.
        config:
            Wrapper config with ``observation_space`` and
            ``action_space`` definitions.
        """
        super().__init__()
        self.task = task
        self.config = config
        self.observation_space: gymnasium.spaces.Space = self._build_obs_space(config)
        self.action_space: gymnasium.spaces.Space = self._build_action_space(config)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment and return (obs, info).

        Parameters
        ----------
        seed:
            Optional RNG seed for reproducibility.
        options:
            Optional reset options.

        Returns
        -------
        tuple[dict, dict]
            Gymnasium-style ``(observation, info)`` pair.
        """
        raise NotImplementedError

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step and return (obs, reward, terminated, truncated, info).

        Parameters
        ----------
        action:
            Numpy action array matching ``self.action_space``.
        """
        raise NotImplementedError

    def render(self) -> Optional[np.ndarray]:
        """Render current frame as an RGB array (if supported)."""
        raise NotImplementedError

    def close(self) -> None:
        """Clean up Genesis resources."""
        raise NotImplementedError

    def _build_obs_space(self, config: Dict[str, Any]) -> gymnasium.spaces.Space:
        """Construct the Gymnasium observation space from config."""
        raise NotImplementedError

    def _build_action_space(self, config: Dict[str, Any]) -> gymnasium.spaces.Space:
        """Construct the Gymnasium action space from config."""
        raise NotImplementedError
