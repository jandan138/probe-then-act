"""Fixed-probe wrapper for the Genesis Scoop-and-Transfer env.

Executes a scripted sequence of probe actions (tap, press, drag) at the
start of each episode before handing control to the RL policy.  The
observations gathered during probing are concatenated into the first
observation returned to the agent, giving it access to tactile/force
information from the probing phase.

This is used for the Fixed-Probe+PPO baseline (M4).  Unlike the learned
probe policy (M7 — Probe-Then-Act), the probe sequence here is fixed
and not optimised.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np


# ---------------------------------------------------------------------------
# Default scripted probe sequence
# ---------------------------------------------------------------------------

# Each probe is a dict with:
#   "type":    one of "tap", "press", "drag"
#   "params":  action array matching env action_dim (7-D delta EE)
#   "steps":   how many env steps to hold this action

_DEFAULT_PROBE_SEQUENCE: List[Dict[str, Any]] = [
    {
        # Tap: move down quickly, then retract
        "type": "tap",
        "params": [0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0],
        "steps": 5,
    },
    {
        # Retract after tap
        "type": "tap_retract",
        "params": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        "steps": 5,
    },
    {
        # Press: sustained downward pressure
        "type": "press",
        "params": [0.0, 0.0, -0.15, 0.0, 0.0, 0.0, 0.0],
        "steps": 10,
    },
    {
        # Retract after press
        "type": "press_retract",
        "params": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        "steps": 5,
    },
    {
        # Drag: lateral motion while in contact
        "type": "drag",
        "params": [0.2, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0],
        "steps": 10,
    },
    {
        # Retract after drag
        "type": "drag_retract",
        "params": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        "steps": 5,
    },
]


class FixedProbeWrapper(gymnasium.Wrapper):
    """Gymnasium wrapper that executes a fixed probe sequence at episode start.

    After ``reset()``, the wrapper internally steps the environment
    through the scripted probe actions and records observations.  The
    probe summary (mean and std of observations during probing) is
    appended to the observation space so the downstream policy can
    condition on it.

    Parameters
    ----------
    env:
        The base GenesisGymWrapper to wrap.
    probe_sequence:
        List of probe action dicts.  If None, uses the default
        tap-press-drag sequence.
    summary_dim:
        Dimensionality of the probe summary appended to observations.
        Set to ``2 * obs_dim`` (mean + std of probe observations).
    """

    def __init__(
        self,
        env: gymnasium.Env,
        probe_sequence: Optional[List[Dict[str, Any]]] = None,
        summary_dim: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.probe_sequence = probe_sequence or _DEFAULT_PROBE_SEQUENCE
        self._base_obs_dim = env.observation_space.shape[0]

        # Probe summary = mean + std of all probe observations
        self._summary_dim = summary_dim or (2 * self._base_obs_dim)

        # Expand observation space to include probe summary
        total_dim = self._base_obs_dim + self._summary_dim
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

        self._probe_summary: Optional[np.ndarray] = None

    @property
    def total_probe_steps(self) -> int:
        """Total number of env steps consumed by the probe sequence."""
        return sum(p["steps"] for p in self.probe_sequence)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset env and execute the fixed probe sequence.

        Returns the first post-probe observation with probe summary
        appended.
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Execute probe sequence and collect observations
        probe_obs_list = [obs.copy()]
        total_probe_reward = 0.0

        for probe_action in self.probe_sequence:
            action = np.array(probe_action["params"], dtype=np.float32)
            for _ in range(probe_action["steps"]):
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                probe_obs_list.append(obs.copy())
                total_probe_reward += reward
                if terminated or truncated:
                    # Episode ended during probing — return with summary
                    break
            if terminated or truncated:
                break

        # Compute probe summary statistics
        probe_obs_array = np.stack(probe_obs_list, axis=0)
        probe_mean = probe_obs_array.mean(axis=0)
        probe_std = probe_obs_array.std(axis=0)
        self._probe_summary = np.concatenate([probe_mean, probe_std]).astype(
            np.float32
        )

        # Pad or truncate summary to match expected dim
        if len(self._probe_summary) < self._summary_dim:
            self._probe_summary = np.pad(
                self._probe_summary,
                (0, self._summary_dim - len(self._probe_summary)),
            )
        elif len(self._probe_summary) > self._summary_dim:
            self._probe_summary = self._probe_summary[: self._summary_dim]

        # Augment observation
        aug_obs = np.concatenate([obs, self._probe_summary]).astype(np.float32)

        info["probe_steps"] = sum(p["steps"] for p in self.probe_sequence)
        info["probe_reward"] = total_probe_reward

        return aug_obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the env and augment observation with cached probe summary."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Append the same probe summary from this episode
        if self._probe_summary is not None:
            aug_obs = np.concatenate([obs, self._probe_summary]).astype(np.float32)
        else:
            # Should not happen, but handle gracefully
            aug_obs = np.concatenate(
                [obs, np.zeros(self._summary_dim, dtype=np.float32)]
            ).astype(np.float32)

        return aug_obs, reward, terminated, truncated, info
