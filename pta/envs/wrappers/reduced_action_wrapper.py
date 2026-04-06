"""ReducedActionWrapper -- Reduces 7D EE delta to 3D position-only control.

The policy outputs (dx, dy, dz). Orientation is fixed (scoop-down) for the
entire episode. Gripper command is zero (scoop has no gripper).

This is the simplest action space for the tiny-task overfit protocol.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium
import numpy as np


class ReducedActionWrapper(gymnasium.Wrapper):
    """Wrap a ScoopTransferTask to accept 3D position-only actions."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

    def step(self, action_3d: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        # Pad 3D action to 7D: [dx, dy, dz, 0, 0, 0, 0]
        full_action = np.zeros(7, dtype=np.float32)
        full_action[:3] = action_3d
        return self.env.step(full_action)
