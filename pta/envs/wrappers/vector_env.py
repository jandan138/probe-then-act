"""GenesisBatchedVecEnv -- SB3-compatible VecEnv using Genesis GPU batching.

Builds a single Genesis scene with ``scene.build(n_envs=B)`` and
runs all environments in parallel via a single ``scene.step()`` call.
Handles per-env auto-reset, batched observations/rewards/dones, and
tensor-to-numpy conversion for Stable-Baselines3 compatibility.

Usage::

    from pta.envs.wrappers.vector_env import GenesisBatchedVecEnv
    vec_env = GenesisBatchedVecEnv(num_envs=4, scene_config=cfg)
    obs = vec_env.reset()
    obs, rewards, dones, infos = vec_env.step(actions)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium
import numpy as np
import torch

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs

import genesis as gs
from genesis.utils.geom import xyz_to_quat, transform_quat_by_quat

from pta.envs.builders.scene_builder import SceneBuilder, SceneComponents


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_TASK_CONFIG: Dict[str, Any] = {
    "horizon": 500,
    "action_dim": 7,
    "action_scale_pos": 0.05,
    "action_scale_rot": 0.05,
    "action_scale_grip": 0.04,
    "w_transfer": 1.0,
    "w_spill": -0.5,
    "w_time": -0.001,
    "w_success_bonus": 5.0,
    "success_threshold": 0.5,
    "default_qpos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    "dls_lambda": 0.01,
}

_OBS_DIM = 37   # 9 qpos + 9 qvel + 3 ee + 4 ee_quat + 3 lf + 3 rf + 1 step_frac + pad
_ACTION_DIM = 7


class GenesisBatchedVecEnv(VecEnv):
    """SB3-compatible VecEnv backed by a single GPU-batched Genesis scene.

    All B environments share one ``gs.Scene`` built with
    ``scene.build(n_envs=B)``.  Physics are advanced by a single
    ``scene.step()`` kernel launch, yielding large GPU utilisation gains
    over B independent ``DummyVecEnv`` instances.
    """

    def __init__(
        self,
        num_envs: int,
        task_config: Dict[str, Any] | None = None,
        scene_config: Dict[str, Any] | None = None,
    ) -> None:
        self.num_envs_actual = num_envs
        self.device = gs.device

        # Merge configs
        self._task_cfg = {**_DEFAULT_TASK_CONFIG, **(task_config or {})}
        self._scene_cfg = dict(scene_config or {})
        self._scene_cfg["n_envs"] = num_envs

        # Build scene
        builder = SceneBuilder()
        self.sc: SceneComponents = builder.build_scene(self._scene_cfg)
        self.scene = self.sc.scene
        self.robot = self.sc.robot
        self.particles = self.sc.particles
        self.camera = self.sc.camera

        # Link references
        self._ee_link = self.sc.ee_link
        self._lf_link = self.sc.left_finger_link
        self._rf_link = self.sc.right_finger_link
        self._arm_dof_idx = self.sc.arm_dof_idx
        self._finger_dof_idx = self.sc.finger_dof_idx

        # Task params
        self._horizon = self._task_cfg["horizon"]
        self._action_dim = self._task_cfg["action_dim"]
        self._act_scale_pos = self._task_cfg["action_scale_pos"]
        self._act_scale_rot = self._task_cfg["action_scale_rot"]
        self._act_scale_grip = self._task_cfg["action_scale_grip"]
        self._success_threshold = self._task_cfg["success_threshold"]
        self._dls_lambda = self._task_cfg["dls_lambda"]

        # Reward weights
        self._w_transfer = self._task_cfg["w_transfer"]
        self._w_spill = self._task_cfg["w_spill"]
        self._w_time = self._task_cfg["w_time"]
        self._w_success_bonus = self._task_cfg["w_success_bonus"]

        # Default qpos (shared across envs)
        self._default_qpos = torch.tensor(
            self._task_cfg["default_qpos"],
            dtype=torch.float32, device=self.device,
        )

        # Container AABBs
        sp = self.sc.source_pos
        ss = self.sc.source_size
        self._source_bbox_min = torch.tensor(
            [sp[0] - ss[0] / 2, sp[1] - ss[1] / 2, 0.0],
            device=self.device, dtype=torch.float32,
        )
        self._source_bbox_max = torch.tensor(
            [sp[0] + ss[0] / 2, sp[1] + ss[1] / 2, sp[2] + ss[2] + 0.10],
            device=self.device, dtype=torch.float32,
        )
        tp = self.sc.target_pos
        ts = self.sc.target_size
        self._target_bbox_min = torch.tensor(
            [tp[0] - ts[0] / 2, tp[1] - ts[1] / 2, 0.0],
            device=self.device, dtype=torch.float32,
        )
        self._target_bbox_max = torch.tensor(
            [tp[0] + ts[0] / 2, tp[1] + ts[1] / 2, tp[2] + ts[2] + 0.10],
            device=self.device, dtype=torch.float32,
        )

        self._total_particles = self.particles._n_particles

        # Per-env buffers (GPU)
        self._step_counts = torch.zeros(
            num_envs, dtype=torch.int32, device=self.device,
        )
        self._gripper_widths = torch.full(
            (num_envs,), 0.04, dtype=torch.float32, device=self.device,
        )

        # Observation buffer (GPU) -- filled every step
        self._obs_buf = torch.zeros(
            (num_envs, _OBS_DIM), dtype=torch.float32, device=self.device,
        )

        # Save initial scene state for per-env resets
        self._init_state = self.scene.get_state()

        # SB3 VecEnv init
        obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32,
        )
        act_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float32,
        )
        super().__init__(num_envs, obs_space, act_space)

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns (num_envs, obs_dim) numpy."""
        self.scene.reset()
        self._step_counts.zero_()
        self._gripper_widths.fill_(0.04)

        # Set robot to home config for all envs
        init_qpos_batch = self._default_qpos.unsqueeze(0).expand(
            self.num_envs_actual, -1,
        )
        self.robot.set_qpos(init_qpos_batch)

        # Settle
        for _ in range(5):
            self.scene.step()

        self._update_obs()
        return self._obs_buf.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Store actions for the next step_wait call."""
        self._pending_actions = torch.tensor(
            actions, dtype=torch.float32, device=self.device,
        )

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Execute one physics step and return (obs, rewards, dones, infos)."""
        actions = self._pending_actions  # (B, 7)
        B = self.num_envs_actual

        self._step_counts += 1

        # --- Parse and scale actions ---
        delta_pos = actions[:, :3] * self._act_scale_pos    # (B, 3)
        delta_rot = actions[:, 3:6] * self._act_scale_rot   # (B, 3)
        grip_cmd = actions[:, 6] * self._act_scale_grip     # (B,)

        # --- IK: compute target joint positions ---
        qpos = self._batched_ik(delta_pos, delta_rot)  # (B, n_dof)

        # --- Update gripper ---
        self._gripper_widths = torch.clamp(
            self._gripper_widths + grip_cmd, 0.0, 0.04,
        )
        qpos[:, self._finger_dof_idx] = self._gripper_widths.unsqueeze(-1)

        # --- Control and step ---
        self.robot.control_dofs_position(qpos)
        self.scene.step()

        # --- Observations ---
        self._update_obs()

        # --- Reward (batched) ---
        rewards = self._compute_batched_reward()  # (B,)

        # --- Done flags ---
        dones = self._compute_batched_done()  # (B,) bool

        # --- Build infos ---
        infos: List[Dict[str, Any]] = []
        for i in range(B):
            info: Dict[str, Any] = {"step": int(self._step_counts[i].item())}
            if dones[i]:
                # SB3 expects terminal_observation in info on done
                info["terminal_observation"] = self._obs_buf[i].cpu().numpy()
            infos.append(info)

        # --- Auto-reset done envs ---
        if dones.any():
            self._reset_envs(dones)

        return (
            self._obs_buf.cpu().numpy(),
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            infos,
        )

    def close(self) -> None:
        """Release Genesis resources."""
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs_actual

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def seed(self, seed=None):
        pass

    # ------------------------------------------------------------------
    # Internal: per-env reset
    # ------------------------------------------------------------------

    def _reset_envs(self, done_mask: torch.Tensor) -> None:
        """Auto-reset environments where done_mask is True.

        Uses Genesis's ``envs_idx`` parameter for selective reset.
        """
        self._step_counts.masked_fill_(done_mask, 0)
        self._gripper_widths.masked_fill_(done_mask, 0.04)

        # Reset robot qpos for done envs
        init_batch = self._default_qpos.unsqueeze(0).expand(
            self.num_envs_actual, -1,
        )
        self.robot.set_qpos(init_batch, envs_idx=done_mask, zero_velocity=True)

        # Update obs for reset envs (will be overwritten next step_wait)
        self._update_obs()

    # ------------------------------------------------------------------
    # Internal: batched observation
    # ------------------------------------------------------------------

    def _update_obs(self) -> None:
        """Fill ``self._obs_buf`` with batched observations.

        All getters return (B, dim) tensors when built with n_envs > 0.
        """
        qpos = self.robot.get_qpos()           # (B, 9)
        qvel = self.robot.get_dofs_velocity()   # (B, 9) or (B, n_dof)
        ee_pos = self._ee_link.get_pos()        # (B, 3)
        ee_quat = self._ee_link.get_quat()      # (B, 4)
        lf_pos = self._lf_link.get_pos()        # (B, 3)
        rf_pos = self._rf_link.get_pos()        # (B, 3)

        # Ensure 2D
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        if qvel.dim() == 1:
            qvel = qvel.unsqueeze(0)
        if ee_pos.dim() == 1:
            ee_pos = ee_pos.unsqueeze(0)
        if ee_quat.dim() == 1:
            ee_quat = ee_quat.unsqueeze(0)
        if lf_pos.dim() == 1:
            lf_pos = lf_pos.unsqueeze(0)
        if rf_pos.dim() == 1:
            rf_pos = rf_pos.unsqueeze(0)

        # Step fraction: (B, 1)
        step_frac = (
            self._step_counts.float() / max(self._horizon, 1)
        ).unsqueeze(-1)

        # Concatenate proprio features
        proprio = torch.cat([qpos, qvel, ee_pos, ee_quat, lf_pos, rf_pos], dim=-1)
        raw_obs = torch.cat([proprio, step_frac], dim=-1)  # (B, proprio_dim+1)

        # Pad or truncate to _OBS_DIM
        obs_dim_actual = raw_obs.shape[-1]
        if obs_dim_actual < _OBS_DIM:
            pad = torch.zeros(
                self.num_envs_actual, _OBS_DIM - obs_dim_actual,
                dtype=torch.float32, device=self.device,
            )
            self._obs_buf = torch.cat([raw_obs, pad], dim=-1)
        elif obs_dim_actual > _OBS_DIM:
            self._obs_buf = raw_obs[:, :_OBS_DIM]
        else:
            self._obs_buf = raw_obs

    # ------------------------------------------------------------------
    # Internal: batched IK (Damped Least Squares)
    # ------------------------------------------------------------------

    def _batched_ik(
        self, delta_pos: torch.Tensor, delta_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Batched DLS IK: returns (B, n_dof) target joint positions."""
        current_qpos = self.robot.get_qpos()   # (B, n_dof)
        if current_qpos.dim() == 1:
            current_qpos = current_qpos.unsqueeze(0)

        # Delta pose (B, 6)
        delta_pose = torch.cat([delta_pos, delta_rot], dim=-1)

        # Jacobian: (B, 6, n_dof)
        jacobian = self.robot.get_jacobian(link=self._ee_link)
        if jacobian.dim() == 2:
            jacobian = jacobian.unsqueeze(0)

        jac_T = jacobian.transpose(1, 2)        # (B, n_dof, 6)
        lam_I = (self._dls_lambda ** 2) * torch.eye(
            jacobian.shape[1], device=self.device,
        )  # (6, 6)

        # DLS: delta_q = J^T (J J^T + lambda^2 I)^{-1} delta_x
        jjt = jacobian @ jac_T + lam_I          # (B, 6, 6)
        delta_q = (
            jac_T @ torch.linalg.solve(jjt, delta_pose.unsqueeze(-1))
        ).squeeze(-1)                            # (B, n_dof)

        return current_qpos + delta_q

    # ------------------------------------------------------------------
    # Internal: batched reward
    # ------------------------------------------------------------------

    def _compute_batched_reward(self) -> torch.Tensor:
        """Compute reward for all envs. Returns (B,) float tensor."""
        B = self.num_envs_actual

        # Particle positions: (B, n_particles, 3)
        particle_pos = self.particles.get_particles_pos()
        if particle_pos.dim() == 2:
            particle_pos = particle_pos.unsqueeze(0)

        # Target count per env
        in_target = (
            (particle_pos >= self._target_bbox_min)
            & (particle_pos <= self._target_bbox_max)
        ).all(dim=-1)                            # (B, n_particles)
        n_in_target = in_target.float().sum(dim=-1)  # (B,)

        # Spill count per env
        in_source = (
            (particle_pos >= self._source_bbox_min)
            & (particle_pos <= self._source_bbox_max)
        ).all(dim=-1)
        outside = ~(in_source | in_target)
        n_spilled = outside.float().sum(dim=-1)  # (B,)

        total = max(self._total_particles, 1)
        transfer_frac = n_in_target / total
        spill_frac = n_spilled / total

        reward = (
            self._w_transfer * transfer_frac
            + self._w_spill * spill_frac
            + self._w_time
        )

        # Distance shaping: EE -> source center
        ee_pos = self._ee_link.get_pos()         # (B, 3)
        if ee_pos.dim() == 1:
            ee_pos = ee_pos.unsqueeze(0)
        source_center = torch.tensor(
            [self.sc.source_pos[0], self.sc.source_pos[1],
             self.sc.source_pos[2] + 0.05],
            device=self.device, dtype=torch.float32,
        )
        dist = torch.norm(ee_pos - source_center, dim=-1)  # (B,)
        reward += -0.1 * dist

        # Success bonus
        reward += (transfer_frac >= self._success_threshold).float() * self._w_success_bonus

        return reward

    # ------------------------------------------------------------------
    # Internal: batched done
    # ------------------------------------------------------------------

    def _compute_batched_done(self) -> torch.Tensor:
        """Compute done flags for all envs. Returns (B,) bool tensor."""
        # Horizon check
        horizon_done = self._step_counts >= self._horizon

        # Early success: >= 95% transferred
        particle_pos = self.particles.get_particles_pos()
        if particle_pos.dim() == 2:
            particle_pos = particle_pos.unsqueeze(0)
        in_target = (
            (particle_pos >= self._target_bbox_min)
            & (particle_pos <= self._target_bbox_max)
        ).all(dim=-1)
        n_in_target = in_target.float().sum(dim=-1)
        early_done = (n_in_target / max(self._total_particles, 1)) >= 0.95

        return horizon_done | early_done
