"""ScoopTransferTask -- Scoop material from source and transfer to target.

Primary task for the Probe-Then-Act paper.

Episode flow:
  1. Probe phase  -- short diagnostic motions to identify material.
  2. Scoop phase  -- insert tool into source container, acquire material.
  3. Transfer phase -- move tool to target container, deposit material.

Success: >= ``success_threshold`` fraction of material reaches the target
without excessive spill or contact failure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

import genesis as gs
from genesis.utils.geom import xyz_to_quat, transform_quat_by_quat

from pta.envs.tasks.base_task import BaseTask
from pta.envs.builders.scene_builder import SceneBuilder, SceneComponents


# ---------------------------------------------------------------------------
# Default task configuration
# ---------------------------------------------------------------------------

_DEFAULT_TASK_CONFIG: Dict[str, Any] = {
    # Episode
    "horizon": 500,
    "ctrl_dt": 2e-3,
    # Action
    "action_dim": 7,  # dx, dy, dz, droll, dpitch, dyaw, gripper
    "action_scale_pos": 0.05,  # metres per action unit (increased for reachability)
    "action_scale_rot": 0.05,  # radians per action unit
    "action_scale_grip": 0.04,  # gripper width per action unit
    # Reward weights
    "w_transfer": 1.0,
    "w_spill": -0.5,
    "w_time": -0.001,
    "w_success_bonus": 5.0,
    # Success
    "success_threshold": 0.3,  # 30% for edge-push tiny task
    # Robot defaults
    "default_qpos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    # IK
    "ik_method": "dls",  # "dls" or "gs"
    "dls_lambda": 0.01,
    # Bowl-only sticky fallback. Defaults stay OFF so edge paths remain unchanged.
    "bowl_sticky_fallback_enabled": False,
    "bowl_sticky_top_slack": 0.02,
    "bowl_sticky_detection_margin": 0.012,
    "bowl_sticky_velocity_damping": 0.5,
    "bowl_sticky_zero_outward_velocity": True,
    "bowl_sticky_max_snap": 0.01,
    "bowl_sticky_region_min": (-0.034, 0.013, 0.022),
    "bowl_sticky_region_max": (0.034, 0.067, 0.10),
    "bowl_constraint_fallback_enabled": False,
    "bowl_constraint_stiffness": 1e6,
}


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(-1)
    rw, rx, ry, rz = rhs.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vec[..., :1])
    vec_quat = torch.cat((zeros, vec), dim=-1)
    return _quat_multiply(_quat_multiply(quat, vec_quat), _quat_conjugate(quat))[
        ..., 1:
    ]


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(_quat_conjugate(quat), vec)


def _should_apply_bowl_sticky_fallback(
    *,
    enabled: bool,
    tool_type: str,
    task_layout: str,
    phase: str,
) -> bool:
    return (
        enabled
        and tool_type in {"bowl", "bowl_highwall"}
        and task_layout == "flat"
        and phase == "carry"
    )


def _should_apply_bowl_constraint_fallback(
    *,
    enabled: bool,
    tool_type: str,
    task_layout: str,
    phase: str,
) -> bool:
    return (
        enabled
        and tool_type in {"bowl", "bowl_highwall"}
        and task_layout == "flat"
        and phase == "carry"
    )


def _project_points_into_local_box(
    local_points: torch.Tensor,
    region_min: torch.Tensor,
    region_max: torch.Tensor,
    max_snap: float,
) -> torch.Tensor:
    clipped = torch.minimum(torch.maximum(local_points, region_min), region_max)
    if max_snap <= 0:
        return clipped
    delta = torch.clamp(clipped - local_points, min=-max_snap, max=max_snap)
    return local_points + delta


class ScoopTransferTask(BaseTask):
    """Scoop-and-Transfer manipulation task."""

    def __init__(
        self,
        scene_components: SceneComponents | None = None,
        config: Dict[str, Any] | None = None,
        scene_config: Dict[str, Any] | None = None,
    ) -> None:
        """Build the task.

        Parameters
        ----------
        scene_components:
            Pre-built scene.  If ``None``, a new scene is built using
            :class:`SceneBuilder`.
        config:
            Task-level config (horizon, reward weights, thresholds, ...).
        scene_config:
            Passed to :class:`SceneBuilder` when *scene_components* is None.
        """
        cfg = {**_DEFAULT_TASK_CONFIG, **(config or {})}

        if scene_components is None:
            builder = SceneBuilder()
            scene_components = builder.build_scene(scene_config)

        super().__init__(scene_components, cfg)

        self.sc = scene_components
        self.scene = self.sc.scene
        self.robot = self.sc.robot
        self.particles = self.sc.particles
        self.camera = self.sc.camera

        # Cached link references
        self._ee_link = self.sc.ee_link
        self._left_finger_link = self.sc.left_finger_link
        self._right_finger_link = self.sc.right_finger_link
        self._arm_dof_idx = self.sc.arm_dof_idx
        self._finger_dof_idx = self.sc.finger_dof_idx

        self._n_arm_dof = len(self._arm_dof_idx)
        self._has_fingers = len(self._finger_dof_idx) > 0

        # Task params
        self._horizon = cfg["horizon"]
        self._action_dim = cfg["action_dim"]
        self._act_scale_pos = cfg["action_scale_pos"]
        self._act_scale_rot = cfg["action_scale_rot"]
        self._act_scale_grip = cfg["action_scale_grip"]
        self._success_threshold = cfg["success_threshold"]

        # Default qpos depends on whether fingers exist
        raw_qpos = cfg["default_qpos"]
        if not self._has_fingers and len(raw_qpos) > self._n_arm_dof:
            # Strip finger entries for scoop mode
            raw_qpos = raw_qpos[: self._n_arm_dof]
        self._default_qpos = torch.tensor(
            raw_qpos,
            dtype=torch.float32,
            device=gs.device,
        )

        # Reward weights
        self._w_transfer = cfg["w_transfer"]
        self._w_spill = cfg["w_spill"]
        self._w_time = cfg["w_time"]
        self._w_success_bonus = cfg["w_success_bonus"]

        # IK config
        self._ik_method = cfg.get("ik_method", "dls")
        self._dls_lambda = cfg.get("dls_lambda", 0.01)

        # Container bounding boxes for metric computation
        # Source AABB: extend z-min to ground (0.0) to capture settled particles
        sp = self.sc.source_pos
        ss = self.sc.source_size
        self._source_bbox_min = torch.tensor(
            [
                sp[0] - ss[0] / 2,
                sp[1] - ss[1] / 2,
                0.0,
            ],  # z=0 to capture ground-settled particles
            device=gs.device,
            dtype=torch.float32,
        )
        self._source_bbox_max = torch.tensor(
            [sp[0] + ss[0] / 2, sp[1] + ss[1] / 2, sp[2] + ss[2] + 0.10],
            device=gs.device,
            dtype=torch.float32,
        )

        tp = self.sc.target_pos
        ts = self.sc.target_size
        # Clamp target bbox y_min to platform edge to avoid overlap with source
        platform_edge_y = sp[1] + ss[1] / 2
        target_y_min = max(tp[1] - ts[1] / 2, platform_edge_y)
        self._target_bbox_min = torch.tensor(
            [tp[0] - ts[0] / 2, target_y_min, 0.0],
            device=gs.device,
            dtype=torch.float32,
        )
        self._target_bbox_max = torch.tensor(
            [tp[0] + ts[0] / 2, tp[1] + ts[1] / 2, tp[2] + ts[2] + 0.10],
            device=gs.device,
            dtype=torch.float32,
        )

        # Save initial state for fast resets
        self._init_state = self.scene.get_state()

        # Total particle count (cached)
        self._total_particles = self.particles._n_particles

        # Step counter
        self._step_count = 0

        # Gripper width state
        self._gripper_width = 0.04  # open

        # Delta-reward state (reset each episode)
        self._prev_transfer_frac = 0.0
        self._prev_mean_particle_y = None
        self._success_triggered = False

        self._tool_type = getattr(self.sc, "tool_type", "gripper")
        self._task_layout = getattr(self.sc, "task_layout", "edge_push")
        self._bowl_transport_phase = "off"
        self._bowl_sticky_fallback_enabled = bool(cfg["bowl_sticky_fallback_enabled"])
        self._bowl_sticky_top_slack = float(cfg["bowl_sticky_top_slack"])
        self._bowl_sticky_detection_margin = float(cfg["bowl_sticky_detection_margin"])
        self._bowl_sticky_velocity_damping = float(cfg["bowl_sticky_velocity_damping"])
        self._bowl_sticky_zero_outward_velocity = bool(
            cfg["bowl_sticky_zero_outward_velocity"]
        )
        self._bowl_sticky_max_snap = float(cfg["bowl_sticky_max_snap"])
        self._bowl_sticky_region_min = torch.tensor(
            cfg["bowl_sticky_region_min"], device=gs.device, dtype=torch.float32
        )
        self._bowl_sticky_region_max = torch.tensor(
            cfg["bowl_sticky_region_max"], device=gs.device, dtype=torch.float32
        )
        self._bowl_constraint_fallback_enabled = bool(
            cfg["bowl_constraint_fallback_enabled"]
        )
        self._bowl_constraint_stiffness = float(cfg["bowl_constraint_stiffness"])
        self._bowl_constraints_active = False

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset episode: restore initial state, robot to home pose.

        Returns
        -------
        dict[str, Tensor]
            Initial observation from the sensor stack.
        """
        self.scene.reset()
        self._step_count = 0
        self._gripper_width = 0.04
        self._prev_transfer_frac = 0.0
        self._prev_mean_particle_y = None
        self._success_triggered = False
        self._bowl_transport_phase = "off"
        self._clear_bowl_particle_constraints()

        # Set robot to home configuration
        self.robot.set_qpos(self._default_qpos)

        # Step a few times to settle
        for _ in range(5):
            self.scene.step()

        return self.get_observations()

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """Apply *action*, step physics, return (obs, reward, done, info).

        Parameters
        ----------
        action:
            Robot action tensor, shape ``(7,)``:
            ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``.
        """
        self._step_count += 1

        # Parse action
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.shape[-1] < self._action_dim:
            # Pad with zeros
            pad = torch.zeros(
                self._action_dim - action.shape[-1],
                device=action.device,
                dtype=action.dtype,
            )
            action = torch.cat([action, pad])

        # Scale action
        delta_pos = action[:3] * self._act_scale_pos
        delta_rot = action[3:6] * self._act_scale_rot

        # Compute target joint positions via IK
        qpos = self._compute_ik(delta_pos, delta_rot)

        if self._has_fingers:
            grip_cmd = action[6] * self._act_scale_grip
            # Update gripper
            self._gripper_width = float(
                max(0.0, min(0.04, self._gripper_width + grip_cmd.item()))
            )
            grip_val = self._gripper_width
            # Set full qpos (arm + gripper)
            full_qpos = qpos.clone()
            full_qpos[self._finger_dof_idx] = grip_val
            self.robot.control_dofs_position(full_qpos)
        else:
            # Scoop mode: arm-only, no gripper
            self.robot.control_dofs_position(qpos[: self._n_arm_dof])

        # Step physics
        try:
            self.scene.step()
            self.post_physics_update()
        except Exception:
            # NaN constraint forces — treat as episode failure, return zeros
            dummy_obs = {
                "proprio": torch.zeros(37, device=gs.device),
                "step_fraction": torch.tensor([1.0], device=gs.device),
            }
            return dummy_obs, -10.0, True, {"step": self._step_count, "nan_crash": True}

        # Compute outputs
        obs = self.get_observations()

        # NaN guard on observations
        if torch.isnan(obs["proprio"]).any():
            obs["proprio"] = torch.zeros_like(obs["proprio"])
            return obs, -10.0, True, {"step": self._step_count, "nan_crash": True}
        reward = self.compute_reward()
        done = self.is_done()
        info = self.compute_metrics()
        info["step"] = self._step_count

        return obs, reward, done, info

    def set_bowl_transport_phase(self, phase: str) -> None:
        self._bowl_transport_phase = phase

    def post_physics_update(self) -> None:
        self._maybe_apply_bowl_constraint_fallback()
        self._maybe_apply_bowl_sticky_fallback()

    def _bowl_candidate_mask(self) -> torch.Tensor | None:
        particle_pos = self.particles.get_particles_pos()
        if particle_pos.dim() == 3:
            particle_pos = particle_pos[0]

        ee_pos = self._ee_link.get_pos()
        ee_quat = self._ee_link.get_quat()
        if ee_pos.dim() > 1:
            ee_pos = ee_pos.squeeze(0)
            ee_quat = ee_quat.squeeze(0)

        local_pos = _quat_rotate_inverse(ee_quat, particle_pos - ee_pos.unsqueeze(0))
        detect_min = self._bowl_sticky_region_min - self._bowl_sticky_detection_margin
        detect_max = self._bowl_sticky_region_max.clone()
        detect_max[2] += (
            self._bowl_sticky_top_slack + self._bowl_sticky_detection_margin
        )
        candidate_mask = ((local_pos >= detect_min) & (local_pos <= detect_max)).all(
            dim=-1
        )
        if not bool(candidate_mask.any()):
            return None
        return candidate_mask.unsqueeze(0)

    def _clear_bowl_particle_constraints(self) -> None:
        if self._bowl_constraints_active:
            self.particles.remove_particle_constraints()
            self._bowl_constraints_active = False

    def _maybe_apply_bowl_constraint_fallback(self) -> None:
        if not _should_apply_bowl_constraint_fallback(
            enabled=self._bowl_constraint_fallback_enabled,
            tool_type=self._tool_type,
            task_layout=self._task_layout,
            phase=self._bowl_transport_phase,
        ):
            self._clear_bowl_particle_constraints()
            return

        candidate_mask = self._bowl_candidate_mask()
        if candidate_mask is None:
            self._clear_bowl_particle_constraints()
            return

        self.particles.set_particle_constraints(
            candidate_mask,
            self._ee_link.idx,
            stiffness=self._bowl_constraint_stiffness,
        )
        self._bowl_constraints_active = True

    def _maybe_apply_bowl_sticky_fallback(self) -> None:
        if not _should_apply_bowl_sticky_fallback(
            enabled=self._bowl_sticky_fallback_enabled,
            tool_type=self._tool_type,
            task_layout=self._task_layout,
            phase=self._bowl_transport_phase,
        ):
            return

        particle_pos = self.particles.get_particles_pos()
        particle_vel = self.particles.get_particles_vel()
        if particle_pos.dim() == 3:
            particle_pos = particle_pos[0]
            particle_vel = particle_vel[0]

        ee_pos = self._ee_link.get_pos()
        ee_quat = self._ee_link.get_quat()
        if ee_pos.dim() > 1:
            ee_pos = ee_pos.squeeze(0)
            ee_quat = ee_quat.squeeze(0)

        local_pos = _quat_rotate_inverse(ee_quat, particle_pos - ee_pos.unsqueeze(0))
        local_vel = _quat_rotate_inverse(ee_quat, particle_vel)

        candidate_mask_2d = self._bowl_candidate_mask()
        if candidate_mask_2d is None:
            return

        candidate_mask = candidate_mask_2d.squeeze(0)
        idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
        cand_local_pos = local_pos[idx]
        target_max = self._bowl_sticky_region_max.clone()
        target_max[2] += self._bowl_sticky_top_slack
        projected_local_pos = _project_points_into_local_box(
            cand_local_pos,
            self._bowl_sticky_region_min,
            target_max,
            self._bowl_sticky_max_snap,
        )
        projected_world_pos = ee_pos.unsqueeze(0) + _quat_rotate(
            ee_quat, projected_local_pos
        )

        cand_local_vel = local_vel[idx] * self._bowl_sticky_velocity_damping
        if self._bowl_sticky_zero_outward_velocity:
            cand_local_vel[:, 2] = torch.minimum(
                cand_local_vel[:, 2],
                torch.zeros_like(cand_local_vel[:, 2]),
            )
        projected_world_vel = _quat_rotate(ee_quat, cand_local_vel)

        self.particles.set_particles_pos(projected_world_pos, particles_idx_local=idx)
        self.particles.set_particles_vel(projected_world_vel, particles_idx_local=idx)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Return student-accessible observations.

        Contains proprioception (joint positions, velocities, EE pose)
        and task-relevant information (no privileged material params).
        """
        # Joint positions and velocities
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_dofs_velocity()

        # End-effector pose
        ee_pos = self._ee_link.get_pos()
        ee_quat = self._ee_link.get_quat()

        # Flatten all into a single tensor for the student
        obs_components = []
        for t in [qpos, qvel, ee_pos, ee_quat]:
            if t.dim() == 0:
                obs_components.append(t.unsqueeze(0))
            elif t.dim() > 1:
                obs_components.append(t.squeeze(0))
            else:
                obs_components.append(t)

        # Add finger positions only if gripper mode
        if self._has_fingers:
            lf_pos = self._left_finger_link.get_pos()
            rf_pos = self._right_finger_link.get_pos()
            for t in [lf_pos, rf_pos]:
                if t.dim() == 0:
                    obs_components.append(t.unsqueeze(0))
                elif t.dim() > 1:
                    obs_components.append(t.squeeze(0))
                else:
                    obs_components.append(t)

        proprio = torch.cat(obs_components, dim=-1)

        # Step fraction
        step_frac = torch.tensor(
            [self._step_count / max(self._horizon, 1)],
            device=gs.device,
            dtype=torch.float32,
        )

        obs = {
            "proprio": proprio,
            "step_fraction": step_frac,
        }
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self) -> float:
        """Delta-based reward shaping for edge-push scoop-and-transfer.

        All shaping signals are *incremental* so that already-transferred
        particles do not generate sustained reward, giving PPO meaningful
        advantage variance between active-push and do-nothing trajectories.
        """
        ee_pos = self._ee_link.get_pos()
        if ee_pos.dim() > 1:
            ee_pos = ee_pos.squeeze(0)

        # Particle positions
        particle_pos = self.particles.get_particles_pos()
        if particle_pos.dim() == 3:
            particle_pos = particle_pos[0]

        n_in_target = self._count_particles_in_target()
        n_spilled = self._count_spilled_particles()
        total = max(self._total_particles, 1)
        transfer_frac = n_in_target / total
        spill_frac = n_spilled / total

        # --- Phase 1: Approach — small dense guidance toward source ---
        source_center = torch.tensor(
            [
                self.sc.source_pos[0],
                self.sc.source_pos[1],
                self.sc.source_pos[2] + 0.05,
            ],
            device=gs.device,
            dtype=torch.float32,
        )
        dist_to_source = torch.norm(ee_pos - source_center).item()
        r_approach = -0.01 * dist_to_source

        # --- Phase 2: Push — DELTA mean particle y (only active pushing) ---
        mean_y = particle_pos[:, 1].mean().item()
        if self._prev_mean_particle_y is None:
            self._prev_mean_particle_y = mean_y
        delta_y = mean_y - self._prev_mean_particle_y
        r_push = 5.0 * max(0.0, delta_y)
        self._prev_mean_particle_y = mean_y

        # --- Phase 3: Transfer — DELTA fraction (only NEW transfers) ---
        delta_transfer = transfer_frac - self._prev_transfer_frac
        r_transfer = 20.0 * max(0.0, delta_transfer)
        self._prev_transfer_frac = transfer_frac

        # --- Penalties ---
        r_spill = -2.0 * spill_frac
        r_time = -0.001

        # --- Success bonus (one-shot) ---
        r_success = 0.0
        if transfer_frac >= self._success_threshold and not self._success_triggered:
            r_success = 10.0
            self._success_triggered = True

        reward = r_approach + r_push + r_transfer + r_spill + r_time + r_success
        return float(reward)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> Dict[str, float]:
        """Return ``success_rate``, ``transfer_efficiency``, ``spill_ratio``."""
        n_in_target = self._count_particles_in_target()
        n_spilled = self._count_spilled_particles()
        total = max(self._total_particles, 1)

        transfer_eff = n_in_target / total
        spill_ratio = n_spilled / total
        success = 1.0 if transfer_eff >= self._success_threshold else 0.0

        # Count particles near EE (on tool)
        ee_pos = self._ee_link.get_pos()
        if ee_pos.dim() > 1:
            ee_pos = ee_pos.squeeze(0)
        particle_pos = self.particles.get_particles_pos()
        if particle_pos.dim() == 3:
            particle_pos = particle_pos[0]
        dist_to_ee = torch.norm(particle_pos - ee_pos.unsqueeze(0), dim=-1)
        n_on_tool = int((dist_to_ee < 0.06).sum().item())

        return {
            "success_rate": success,
            "transfer_efficiency": transfer_eff,
            "spill_ratio": spill_ratio,
            "n_in_target": n_in_target,
            "n_spilled": n_spilled,
            "n_on_tool": n_on_tool,
            "total_particles": total,
        }

    def is_done(self) -> bool:
        """Done when horizon exceeded or early termination triggered."""
        if self._step_count >= self._horizon:
            return True
        # Early success: all particles transferred
        n_in_target = self._count_particles_in_target()
        if n_in_target / max(self._total_particles, 1) >= 0.95:
            return True
        return False

    # ------------------------------------------------------------------
    # Particle counting
    # ------------------------------------------------------------------

    def _count_particles_in_target(self) -> int:
        """Count MPM particles inside the target container AABB."""
        pos = self.particles.get_particles_pos()  # (n_particles, 3) for n_envs=0
        if pos.dim() == 3:
            pos = pos[0]
        inside = ((pos >= self._target_bbox_min) & (pos <= self._target_bbox_max)).all(
            dim=-1
        )
        return int(inside.sum().item())

    def _count_spilled_particles(self) -> int:
        """Count MPM particles outside both containers."""
        pos = self.particles.get_particles_pos()
        if pos.dim() == 3:
            pos = pos[0]
        in_source = (
            (pos >= self._source_bbox_min) & (pos <= self._source_bbox_max)
        ).all(dim=-1)
        in_target = (
            (pos >= self._target_bbox_min) & (pos <= self._target_bbox_max)
        ).all(dim=-1)
        outside = ~(in_source | in_target)
        return int(outside.sum().item())

    # ------------------------------------------------------------------
    # IK helpers
    # ------------------------------------------------------------------

    def _compute_ik(
        self, delta_pos: torch.Tensor, delta_rot: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint positions from EE delta pose via IK.

        Returns full qpos (including gripper DOFs as-is).
        """
        current_qpos = self.robot.get_qpos()
        if current_qpos.dim() == 1:
            current_qpos = current_qpos.unsqueeze(0)

        ee_pos = self._ee_link.get_pos()
        ee_quat = self._ee_link.get_quat()

        if ee_pos.dim() == 1:
            ee_pos = ee_pos.unsqueeze(0)
        if ee_quat.dim() == 1:
            ee_quat = ee_quat.unsqueeze(0)
        if delta_pos.dim() == 1:
            delta_pos = delta_pos.unsqueeze(0)
        if delta_rot.dim() == 1:
            delta_rot = delta_rot.unsqueeze(0)

        target_pos = ee_pos + delta_pos

        # Convert euler delta to quaternion
        quat_delta = xyz_to_quat(delta_rot, rpy=True, degrees=False)
        target_quat = transform_quat_by_quat(quat_delta, ee_quat)

        if self._ik_method == "gs":
            # Use Genesis built-in IK -- squeeze back to 1D for single-env
            target_pos_ik = target_pos.squeeze(0)
            target_quat_ik = target_quat.squeeze(0)
            qpos = self.robot.inverse_kinematics(
                link=self._ee_link,
                pos=target_pos_ik,
                quat=target_quat_ik,
                dofs_idx_local=self._arm_dof_idx,
            )
        else:
            # Damped least squares IK
            # delta_pos, delta_rot are (1, 3) at this point
            delta_pose = torch.cat(
                [delta_pos.squeeze(0), delta_rot.squeeze(0)], dim=-1
            )  # (6,)
            jacobian = self.robot.get_jacobian(
                link=self._ee_link
            )  # (6, n_dof) for n_envs=0, or (B, 6, n_dof)

            # Handle both single-env (2D) and batched (3D) Jacobian
            if jacobian.dim() == 2:
                # Single-env: jacobian is (6, n_dof)
                jac_T = jacobian.T  # (n_dof, 6)
                lam = self._dls_lambda
                lam_I = (lam**2) * torch.eye(jacobian.shape[0], device=gs.device)
                delta_q = (
                    jac_T @ torch.inverse(jacobian @ jac_T + lam_I) @ delta_pose
                )  # (n_dof,)
                qpos = current_qpos.squeeze(0) + delta_q
            else:
                # Batched: jacobian is (B, 6, n_dof)
                jac_T = jacobian.transpose(1, 2)
                lam = self._dls_lambda
                lam_I = (lam**2) * torch.eye(jacobian.shape[1], device=gs.device)
                delta_q = (
                    jac_T
                    @ torch.inverse(jacobian @ jac_T + lam_I)
                    @ delta_pose.unsqueeze(0).unsqueeze(-1)
                ).squeeze(-1)
                qpos = current_qpos + delta_q
                if qpos.dim() == 2 and qpos.shape[0] == 1:
                    qpos = qpos.squeeze(0)

        return qpos
