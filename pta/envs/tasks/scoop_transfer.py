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
    "action_scale_pos": 0.05,   # metres per action unit (increased for reachability)
    "action_scale_rot": 0.05,   # radians per action unit
    "action_scale_grip": 0.04,  # gripper width per action unit
    # Reward weights
    "w_transfer": 1.0,
    "w_spill": -0.5,
    "w_time": -0.001,
    "w_success_bonus": 5.0,
    # Success
    "success_threshold": 0.5,  # fraction of particles in target
    # Robot defaults
    "default_qpos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    # IK
    "ik_method": "dls",  # "dls" or "gs"
    "dls_lambda": 0.01,
}


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
            raw_qpos = raw_qpos[:self._n_arm_dof]
        self._default_qpos = torch.tensor(
            raw_qpos, dtype=torch.float32, device=gs.device,
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
            [sp[0] - ss[0] / 2, sp[1] - ss[1] / 2, 0.0],  # z=0 to capture ground-settled particles
            device=gs.device, dtype=torch.float32,
        )
        self._source_bbox_max = torch.tensor(
            [sp[0] + ss[0] / 2, sp[1] + ss[1] / 2, sp[2] + ss[2] + 0.10],
            device=gs.device, dtype=torch.float32,
        )

        tp = self.sc.target_pos
        ts = self.sc.target_size
        self._target_bbox_min = torch.tensor(
            [tp[0] - ts[0] / 2, tp[1] - ts[1] / 2, 0.0],  # z=0 for settled particles
            device=gs.device, dtype=torch.float32,
        )
        self._target_bbox_max = torch.tensor(
            [tp[0] + ts[0] / 2, tp[1] + ts[1] / 2, tp[2] + ts[2] + 0.10],
            device=gs.device, dtype=torch.float32,
        )

        # Save initial state for fast resets
        self._init_state = self.scene.get_state()

        # Total particle count (cached)
        self._total_particles = self.particles._n_particles

        # Step counter
        self._step_count = 0

        # Gripper width state
        self._gripper_width = 0.04  # open

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
                device=action.device, dtype=action.dtype,
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
            self.robot.control_dofs_position(qpos[:self._n_arm_dof])

        # Step physics
        try:
            self.scene.step()
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
            device=gs.device, dtype=torch.float32,
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
        """Staged reward shaping for scoop-and-transfer.

        Four phases guide the policy through the manipulation sequence:
          1. Approach: EE moves toward the source container
          2. Scoop:    EE dips into particles (below particle surface)
          3. Lift:     particles near the EE are raised above source rim
          4. Transfer: particles reach the target container

        Each phase provides dense shaping so PPO gets gradient signal
        even before achieving actual transfer.
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

        # --- Phase 1: Approach source container ---
        source_center = torch.tensor(
            [self.sc.source_pos[0], self.sc.source_pos[1],
             self.sc.source_pos[2] + 0.10],  # slightly above particle surface
            device=gs.device, dtype=torch.float32,
        )
        dist_to_source = torch.norm(ee_pos - source_center).item()
        r_approach = -0.1 * dist_to_source

        # --- Phase 2: Scoop — reward for EE being low (inside material) ---
        # Particle surface is ~0.14m. Reward EE for going below that.
        particle_surface_z = self.sc.source_pos[2] + 0.10  # ~0.15
        ee_z = ee_pos[2].item()
        # Only reward scooping when EE is near the source (x,y)
        ee_xy = ee_pos[:2]
        source_xy = torch.tensor(
            [self.sc.source_pos[0], self.sc.source_pos[1]],
            device=gs.device, dtype=torch.float32,
        )
        dist_xy = torch.norm(ee_xy - source_xy).item()
        if dist_xy < 0.15:  # within source container radius
            depth_below_surface = max(0.0, particle_surface_z - ee_z)
            r_scoop = 0.3 * min(depth_below_surface, 0.08)  # cap at 0.024
        else:
            r_scoop = 0.0

        # --- Phase 3: Lift — reward particles near EE being raised ---
        # Count particles within 0.06m of EE (on the tool)
        dist_to_ee = torch.norm(particle_pos - ee_pos.unsqueeze(0), dim=-1)
        near_ee_mask = dist_to_ee < 0.06
        n_near_ee = near_ee_mask.sum().item()
        if n_near_ee > 0:
            # Reward for particles near EE being above the source rim
            source_rim_z = self.sc.source_pos[2] + 0.08 + 0.02  # wall_height + margin
            particles_near_ee_z = particle_pos[near_ee_mask, 2]
            n_lifted = (particles_near_ee_z > source_rim_z).sum().item()
            r_lift = 0.5 * (n_lifted / total)
        else:
            r_lift = 0.0

        # --- Phase 4: Transfer — reward for particles in target ---
        r_transfer = self._w_transfer * transfer_frac

        # --- Phase 4b: Distance to target (when carrying particles) ---
        if n_near_ee > 2:
            target_center = torch.tensor(
                [self.sc.target_pos[0], self.sc.target_pos[1],
                 self.sc.target_pos[2] + 0.15],
                device=gs.device, dtype=torch.float32,
            )
            dist_to_target = torch.norm(ee_pos - target_center).item()
            r_carry = -0.05 * dist_to_target  # encourage moving to target
        else:
            r_carry = 0.0

        # --- Penalties ---
        r_spill = self._w_spill * spill_frac
        r_time = self._w_time

        # --- Success bonus ---
        r_success = self._w_success_bonus if transfer_frac >= self._success_threshold else 0.0

        reward = r_approach + r_scoop + r_lift + r_transfer + r_carry + r_spill + r_time + r_success
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
        inside = (
            (pos >= self._target_bbox_min) & (pos <= self._target_bbox_max)
        ).all(dim=-1)
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
                lam_I = (lam ** 2) * torch.eye(
                    jacobian.shape[0], device=gs.device
                )
                delta_q = (
                    jac_T @ torch.inverse(jacobian @ jac_T + lam_I)
                    @ delta_pose
                )  # (n_dof,)
                qpos = current_qpos.squeeze(0) + delta_q
            else:
                # Batched: jacobian is (B, 6, n_dof)
                jac_T = jacobian.transpose(1, 2)
                lam = self._dls_lambda
                lam_I = (lam ** 2) * torch.eye(
                    jacobian.shape[1], device=gs.device
                )
                delta_q = (
                    jac_T @ torch.inverse(jacobian @ jac_T + lam_I)
                    @ delta_pose.unsqueeze(0).unsqueeze(-1)
                ).squeeze(-1)
                qpos = current_qpos + delta_q
                if qpos.dim() == 2 and qpos.shape[0] == 1:
                    qpos = qpos.squeeze(0)

        return qpos
