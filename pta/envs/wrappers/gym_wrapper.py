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
from pta.envs.tasks.scoop_transfer import ScoopTransferTask


_DEFAULT_WRAPPER_CONFIG: Dict[str, Any] = {
    "obs_dim": 37,    # 9 qpos + 9 qvel + 3 ee_pos + 4 ee_quat + 3 lf + 3 rf + 1 step_frac + ... (padded)
    "action_dim": 7,  # dx, dy, dz, droll, dpitch, dyaw, gripper
}


class GenesisGymWrapper(gymnasium.Env):
    """Gymnasium wrapper around a Genesis-based :class:`BaseTask`.

    Handles observation/action space definition, tensor <-> numpy
    conversion, and the standard Gymnasium ``reset`` / ``step`` API.
    """

    metadata: Dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: BaseTask | None = None,
        config: Dict[str, Any] | None = None,
        task_config: Dict[str, Any] | None = None,
        scene_config: Dict[str, Any] | None = None,
    ) -> None:
        """Wrap *task* as a Gymnasium environment.

        Parameters
        ----------
        task:
            An instantiated :class:`BaseTask` subclass.  If ``None``,
            a :class:`ScoopTransferTask` is created automatically.
        config:
            Wrapper config with ``obs_dim`` and ``action_dim``.
        task_config:
            Passed to :class:`ScoopTransferTask` if *task* is None.
        scene_config:
            Passed to :class:`SceneBuilder` if *task* is None.
        """
        super().__init__()

        cfg = {**_DEFAULT_WRAPPER_CONFIG, **(config or {})}

        if task is None:
            task = ScoopTransferTask(
                config=task_config,
                scene_config=scene_config,
            )

        self.task = task
        self.config = cfg

        # Build spaces
        self.observation_space = self._build_obs_space(cfg)
        self.action_space = self._build_action_space(cfg)

        # Internal state
        self._obs_dim = cfg["obs_dim"]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return (obs, info).

        Parameters
        ----------
        seed:
            Optional RNG seed for reproducibility.
        options:
            Optional reset options.

        Returns
        -------
        tuple[ndarray, dict]
            Gymnasium-style ``(observation, info)`` pair.
        """
        super().reset(seed=seed)

        obs_dict = self.task.reset()
        obs_np = self._obs_dict_to_numpy(obs_dict)
        info = {"raw_obs": {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                            for k, v in obs_dict.items()}}
        return obs_np, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step and return (obs, reward, terminated, truncated, info).

        Parameters
        ----------
        action:
            Numpy action array matching ``self.action_space``.
        """
        action_t = torch.tensor(action, dtype=torch.float32, device="cuda")
        try:
            obs_dict, reward, done, info = self.task.step(action_t)
        except Exception as e:
            if "nan" in str(e).lower():
                # Rigid/MPM solver NaN -- reset and return truncated episode
                obs_np, info = self.reset()
                info["nan_reset"] = True
                return obs_np, 0.0, False, True, info
            raise
        obs_np = self._obs_dict_to_numpy(obs_dict)

        # In Gymnasium API, distinguish terminated vs truncated
        terminated = done and info.get("success_rate", 0.0) >= 0.5
        truncated = done and not terminated

        return obs_np, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render current frame as an RGB array (if supported)."""
        try:
            rgb, _, _, _ = self.task.camera.render(
                rgb=True, depth=False, segmentation=False, normal=False,
            )
            if isinstance(rgb, torch.Tensor):
                img = rgb.cpu().numpy()
            else:
                img = np.asarray(rgb)
            # Handle batched output (n_envs=0 still returns 4D)
            if img.ndim == 4:
                img = img[0]
            # Ensure uint8
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return img
        except Exception:
            return None

    def close(self) -> None:
        """Clean up Genesis resources."""
        pass  # Genesis manages its own cleanup via gs.destroy()

    # ------------------------------------------------------------------
    # Space construction
    # ------------------------------------------------------------------

    def _build_obs_space(self, config: Dict[str, Any]) -> gymnasium.spaces.Space:
        """Construct the Gymnasium observation space from config."""
        obs_dim = config.get("obs_dim", 37)
        return gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _build_action_space(self, config: Dict[str, Any]) -> gymnasium.spaces.Space:
        """Construct the Gymnasium action space from config."""
        act_dim = config.get("action_dim", 7)
        return gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _obs_dict_to_numpy(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten the observation dict into a fixed-size numpy array."""
        parts = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().float()
                if val.dim() == 0:
                    val = val.unsqueeze(0)
                parts.append(val.numpy().flatten())
            elif isinstance(val, (int, float)):
                parts.append(np.array([val], dtype=np.float32))

        flat = np.concatenate(parts).astype(np.float32)

        # Pad or truncate to match obs_dim
        target_dim = self._obs_dim
        if len(flat) < target_dim:
            flat = np.pad(flat, (0, target_dim - len(flat)))
        elif len(flat) > target_dim:
            flat = flat[:target_dim]

        return flat
