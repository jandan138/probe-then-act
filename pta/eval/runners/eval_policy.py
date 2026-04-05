"""Evaluate a trained task policy on a given environment split."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def evaluate_policy(
    model: Any,
    env: Any,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Run a trained SB3 policy for N episodes and compute metrics.

    Parameters
    ----------
    model :
        Trained SB3 model (PPO, RecurrentPPO, etc.).
    env :
        Gymnasium-compatible environment instance.
    n_episodes :
        Number of evaluation episodes.
    deterministic :
        Whether to use deterministic actions.

    Returns
    -------
    dict[str, float]
        Evaluation metrics: success_rate, mean_return, mean_episode_length,
        mean_transfer_efficiency, mean_spill_ratio.
    """
    episode_returns = []
    episode_lengths = []
    successes = []
    transfer_effs = []
    spill_ratios = []
    nan_crashes = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        last_info = {}

        # Handle RecurrentPPO (needs LSTM states)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            try:
                # Try RecurrentPPO-style predict (with states)
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
                episode_starts = np.zeros((1,), dtype=bool)
            except TypeError:
                # Standard PPO — no state argument
                action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            done = terminated or truncated
            last_info = info

            if info.get("nan_crash", False) or info.get("nan_reset", False):
                nan_crashes += 1

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        successes.append(last_info.get("success_rate", 0.0))
        transfer_effs.append(last_info.get("transfer_efficiency", 0.0))
        spill_ratios.append(last_info.get("spill_ratio", 0.0))

    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_transfer_efficiency": float(np.mean(transfer_effs)),
        "mean_spill_ratio": float(np.mean(spill_ratios)),
        "nan_crash_count": nan_crashes,
        "n_episodes": n_episodes,
    }


def load_sb3_model(
    checkpoint_path: str,
    method: str,
    env: Optional[Any] = None,
) -> Any:
    """Load a trained SB3 model from checkpoint.

    Parameters
    ----------
    checkpoint_path :
        Path to the .zip checkpoint file.
    method :
        Method name to determine model class (reactive_ppo, rnn_ppo, etc.).
    env :
        Optional environment for model initialization.

    Returns
    -------
    model
        Loaded SB3 model ready for inference.
    """
    if method == "rnn_ppo":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(checkpoint_path, env=env)
    else:
        from stable_baselines3 import PPO
        return PPO.load(checkpoint_path, env=env)
