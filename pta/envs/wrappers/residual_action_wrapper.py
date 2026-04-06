"""Residual RL wrapper: scripted base + learned correction.

The scripted base is a time-indexed Cartesian-delta policy that
approximates the edge-push motion.  The RL policy outputs small
corrections (residual) which are added to the base action.

This drastically reduces exploration difficulty — the agent only
needs to refine timing and trajectory, not discover the entire
push motion from scratch.
"""
from __future__ import annotations

import gymnasium
import numpy as np


class ResidualActionWrapper(gymnasium.Wrapper):
    """``a_total = a_scripted(obs, t) + scale * a_residual``."""

    def __init__(
        self,
        env: gymnasium.Env,
        residual_scale: float = 0.3,
        episode_len: int = 80,
        ee_obs_idx: int = 18,
    ) -> None:
        super().__init__(env)
        self.residual_scale = residual_scale
        self.episode_len = episode_len
        self.ee_obs_idx = ee_obs_idx  # index of EE x,y,z in obs vector
        self._step = 0
        self._last_obs = None

    def reset(self, **kwargs):
        self._step = 0
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            self._last_obs = result[0]
        else:
            self._last_obs = result
        return result

    def step(self, residual_action):
        base = self._scripted_action(self._step, self._last_obs)
        total = base + self.residual_scale * np.asarray(residual_action, dtype=np.float32)
        total = np.clip(total, -1.0, 1.0)
        self._step += 1
        result = self.env.step(total)
        if isinstance(result, tuple) and len(result) >= 1:
            self._last_obs = result[0]
        return result

    def _scripted_action(self, t: int, obs: np.ndarray) -> np.ndarray:
        """Observation-based scripted policy for edge-push.

        Reads EE position from obs (first 3 elements of proprio) and
        computes proportional control toward waypoints.

        Targets:
          Phase 0: approach → particle region (0.55, -0.03, 0.18)
          Phase 1: push → platform edge (*, 0.10, 0.18)
          Phase 2: retract → behind particles (*, -0.05, 0.25)
          Phase 3: push2 → platform edge again
        """
        # EE position is at obs[ee_obs_idx : ee_obs_idx+3]
        if obs is None or len(obs) < self.ee_obs_idx + 3:
            return np.zeros(3, dtype=np.float32)

        idx = self.ee_obs_idx
        ee_x, ee_y, ee_z = float(obs[idx]), float(obs[idx+1]), float(obs[idx+2])

        # Waypoints (Cartesian targets)
        # Note: IK tends to invert y-axis from home config, so we target
        # y values that the IK will map to actual desired EE positions.
        # EE converges to y~0.08-0.10 regardless of y-target sign.
        # So we use the actual EE y-convergence and focus on x,z accuracy.
        PARTICLE_POS = (0.55, -0.03, 0.18)  # particle region
        PUSH_TARGET = (0.55, 0.10, 0.18)    # past platform edge
        RETRACT_POS = (0.55, -0.05, 0.25)   # behind particles, lifted

        if t < 20:
            target = PARTICLE_POS
        elif t < 50:
            target = PUSH_TARGET
        elif t < 60:
            target = RETRACT_POS
        else:
            target = PUSH_TARGET

        # Proportional control: action = K * (target - current)
        # Action scale is 0.05m per unit, so K=1/0.05=20 per meter
        # But action is clipped to [-1,1], so effective reach is 0.05m/step
        K = 15.0  # gain (action units per meter of error)
        dx = np.clip(K * (target[0] - ee_x), -1.0, 1.0)
        dy = np.clip(K * (target[1] - ee_y), -1.0, 1.0)
        dz = np.clip(K * (target[2] - ee_z), -1.0, 1.0)

        return np.array([dx, dy, dz], dtype=np.float32)
