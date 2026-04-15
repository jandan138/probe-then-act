"""Joint-space residual wrapper: scripted joint trajectory + learned correction.

Bypasses Cartesian IK entirely.  The scripted base is a dense joint-position
trajectory extracted from the scoop-tool waypoint sequence in
``run_scripted_baseline.py``.  The RL policy outputs small joint-space
corrections (residuals) that are added to the base trajectory.

    a_applied = q_base[t] + residual_scale * delta_q

This avoids the single-step DLS coupling artifact that causes y-axis
inversion (see docs/40_investigations/IK_MINIMAL_REPRO.md).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Scoop-tool waypoints (7-DOF, from run_scripted_baseline.py)
# ---------------------------------------------------------------------------

HOME_S = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
EXTEND_FWD_S = [0.0, 0.5, 0.0, -1.8, 0.0, 1.8, 0.0]
HOVER_SOURCE_S = [0.0, 1.0, 0.0, -1.5, 0.0, 1.5, 0.0]
SCOOP_START_S = [-0.15, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
SCOOP_MID_S = [0.0, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
SCOOP_PAST_S = [0.15, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
LIFT_LOW_S = [0.0, 1.1, 0.0, -1.2, 0.0, 1.2, 0.0]
LIFT_S = [0.0, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]
TRAVERSE_S = [0.7, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]
DEPOSIT_S = [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 0.0]
DUMP_S = [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 1.5]

# Edge-push waypoints (from run_sequence_e_scoop)
BEHIND_EP = [-0.10, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]
PUSH_END_EP = [0.40, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]

# Franka joint limits (from panda_scoop.xml)
JOINT_LIMITS_LOW = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32
)
JOINT_LIMITS_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32
)


def _interpolate(start: list, end: list, n_steps: int) -> np.ndarray:
    """Linearly interpolate between two joint configs, returning (n_steps, 7)."""
    s = np.array(start, dtype=np.float32)
    e = np.array(end, dtype=np.float32)
    alphas = np.linspace(0.0, 1.0, n_steps + 1)[1:]  # exclude 0
    return s[None, :] * (1 - alphas[:, None]) + e[None, :] * alphas[:, None]


def build_scoop_trajectory() -> np.ndarray:
    """Build the dense joint trajectory for scoop-lift-traverse-deposit.

    Returns array of shape (T, 7) where T ~ 200-300 steps.
    """
    segments = [
        (HOME_S, EXTEND_FWD_S, 20),
        (EXTEND_FWD_S, HOVER_SOURCE_S, 20),
        (HOVER_SOURCE_S, SCOOP_START_S, 30),
        (SCOOP_START_S, SCOOP_MID_S, 30),
        (SCOOP_MID_S, SCOOP_PAST_S, 20),
        (SCOOP_PAST_S, LIFT_LOW_S, 15),
        (LIFT_LOW_S, LIFT_S, 15),
        (LIFT_S, TRAVERSE_S, 30),
        (TRAVERSE_S, DEPOSIT_S, 20),
        (DEPOSIT_S, DUMP_S, 15),
    ]
    pieces = [_interpolate(s, e, n) for s, e, n in segments]
    return np.concatenate(pieces, axis=0)


def build_edge_push_trajectory() -> np.ndarray:
    """Build the dense joint trajectory for edge-push (3-pass).

    Returns array of shape (T, 7) where T ~ 400 steps.
    """
    pieces = []
    # Approach
    pieces.append(_interpolate(HOME_S, EXTEND_FWD_S, 20))
    pieces.append(_interpolate(EXTEND_FWD_S, BEHIND_EP, 30))
    # 3 push passes
    for i in range(3):
        pieces.append(_interpolate(BEHIND_EP, PUSH_END_EP, 100))
        if i < 2:
            pieces.append(_interpolate(PUSH_END_EP, BEHIND_EP, 30))
    # Settle: hold final position for 80 steps to let particles fall into target
    settle = np.tile(pieces[-1][-1:], (80, 1))
    pieces.append(settle)
    return np.concatenate(pieces, axis=0)


class JointResidualWrapper(gymnasium.Wrapper):
    """``q_applied = q_base[t] + residual_scale * delta_q``

    Bypasses the task's Cartesian IK.  Directly sets joint positions on the
    robot via ``set_qpos`` + ``scene.step()``, then reads observations and
    reward from the task.

    Parameters
    ----------
    env:
        A GenesisGymWrapper (or chain containing one).
    residual_scale:
        Max joint residual in radians per action unit.
    trajectory:
        Which base trajectory to use: ``"scoop"`` or ``"edge_push"``.
    settle_steps:
        Number of physics sub-steps after each set_qpos.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        residual_scale: float = 0.1,
        trajectory: str = "edge_push",
        settle_steps: int = 3,
    ) -> None:
        super().__init__(env)

        self.residual_scale = residual_scale
        self.settle_steps = settle_steps

        # Build base trajectory
        if trajectory == "scoop":
            self._q_base = build_scoop_trajectory()
        else:
            self._q_base = build_edge_push_trajectory()

        self._horizon = len(self._q_base)
        self._step = 0

        # Access the underlying task (through wrapper chain)
        self._task = self._get_task()

        # Action space: 7D joint residuals in [-1, 1]
        self.action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # Observation space: base obs + q_base(t) (7) + step_fraction (1)
        base_obs_dim = self.env.observation_space.shape[0]
        self._base_obs_dim = base_obs_dim
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim + 7 + 1,),
            dtype=np.float32,
        )

    def _get_task(self):
        """Walk the wrapper chain to find the ScoopTransferTask."""
        e = self.env
        while hasattr(e, "env"):
            if hasattr(e, "task"):
                return e.task
            e = e.env
        if hasattr(e, "task"):
            return e.task
        raise RuntimeError("Could not find ScoopTransferTask in wrapper chain")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step = 0
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment_obs(obs), info

    def step(
        self,
        residual_action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Get base joint config for this timestep
        t = min(self._step, self._horizon - 1)
        q_base = self._q_base[t]

        # Compute total qpos
        residual = self.residual_scale * np.asarray(residual_action, dtype=np.float32)
        q_total = q_base + residual

        # Clamp to joint limits
        q_total = np.clip(q_total, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

        # Apply directly to robot (bypass IK)
        q_tensor = torch.tensor(q_total, dtype=torch.float32, device="cuda")
        self._task.robot.set_qpos(q_tensor)
        for _ in range(self.settle_steps):
            self._task.scene.step()
            if hasattr(self._task, "post_physics_update"):
                self._task.post_physics_update()

        # Update task step counter to keep reward/done logic in sync
        self._task._step_count = self._step + 1

        # Read observations, reward, done from the task
        try:
            obs_dict = self._task.get_observations()
        except Exception:
            # NaN fallback
            obs_np = np.zeros(self._base_obs_dim, dtype=np.float32)
            self._step += 1
            return self._augment_obs(obs_np), -10.0, True, False, {"nan_crash": True}

        obs_np = (
            self.env._obs_dict_to_numpy(obs_dict)
            if hasattr(self.env, "_obs_dict_to_numpy")
            else self._flatten_obs(obs_dict)
        )

        # NaN guard
        if np.isnan(obs_np).any():
            obs_np = np.zeros_like(obs_np)
            self._step += 1
            return self._augment_obs(obs_np), -10.0, True, False, {"nan_crash": True}

        reward = float(self._task.compute_reward())
        done = self._task.is_done()
        info = self._task.compute_metrics()
        info["step"] = self._step + 1
        info["base_t"] = t

        self._step += 1

        # Gymnasium terminated/truncated
        terminated = done and info.get("success_rate", 0.0) >= 0.5
        truncated = done and not terminated

        return self._augment_obs(obs_np), reward, terminated, truncated, info

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Append q_base[t] and step_fraction to observations."""
        t = min(self._step, self._horizon - 1)
        q_base = self._q_base[t].astype(np.float32)
        step_frac = np.array([self._step / self._horizon], dtype=np.float32)
        return np.concatenate([base_obs, q_base, step_frac])

    def _flatten_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Fallback obs flattening if _obs_dict_to_numpy not available."""
        parts = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            if isinstance(val, torch.Tensor):
                parts.append(val.detach().cpu().float().numpy().flatten())
            elif isinstance(val, (int, float)):
                parts.append(np.array([val], dtype=np.float32))
        flat = np.concatenate(parts).astype(np.float32)
        if len(flat) < self._base_obs_dim:
            flat = np.pad(flat, (0, self._base_obs_dim - len(flat)))
        elif len(flat) > self._base_obs_dim:
            flat = flat[: self._base_obs_dim]
        return flat
