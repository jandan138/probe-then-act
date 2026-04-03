"""Train the task policy end-to-end with PPO."""

from __future__ import annotations

from typing import Any, Dict


def train_task_policy(config: Dict[str, Any]) -> None:
    """Train the full Probe-Then-Act task policy with PPO.

    Unlike the teacher, the task policy only receives sensor observations
    (RGB, depth, tactile, proprioception) and must first execute probe
    actions to build a latent belief before acting.

    Parameters
    ----------
    config : dict
        Training configuration containing at minimum:
        - ``env``: environment specification.
        - ``policy``: architecture (probe head, belief encoder, action head).
        - ``ppo``: PPO hyper-parameters.
        - ``training``: total timesteps, checkpoint frequency, seed.
    """
    raise NotImplementedError
