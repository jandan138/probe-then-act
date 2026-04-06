"""ActionRepeatWrapper -- Holds each policy action for N physics steps.

Reduces effective policy frequency. With physics at 500Hz and repeat=25,
the policy runs at 20Hz (80 decisions per 4-second episode).

Rewards are summed across repeats. Episode terminates early if the
underlying env signals done during a repeat.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium
import numpy as np


class ActionRepeatWrapper(gymnasium.Wrapper):
    """Repeat each action for *repeat* physics steps, summing rewards."""

    def __init__(self, env: gymnasium.Env, repeat: int = 25) -> None:
        super().__init__(env)
        self.repeat = repeat

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        total_reward = 0.0
        for i in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
