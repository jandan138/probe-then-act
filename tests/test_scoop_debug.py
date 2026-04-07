"""Debug scoop trajectory: print scoop position and particle stats at each phase."""

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import torch
import numpy as np
import genesis as gs

from pta.envs.tasks.scoop_transfer import ScoopTransferTask
from pta.scripts.run_scripted_baseline import (
    interpolate_waypoints, settle,
    HOME_S, EXTEND_FWD_S, HOVER_SOURCE_S, SCOOP_START_S, SCOOP_MID_S,
    SCOOP_PAST_S, LIFT_LOW_S, LIFT_S, TRAVERSE_S, DEPOSIT_S, DUMP_S,
)

scene_cfg = {
    "particle_material": "sand",
    "n_envs": 0,
    "tool_type": "scoop",
    "source_wall_height": 0.0,
    "target_wall_height": 0.0,
    # Move target closer to source for shorter push distance
    "source_pos": (0.5, 0.0, 0.05),
    "target_pos": (0.5, 0.20, 0.05),  # was 0.35, now 0.20 — closer
}
task_cfg = {"horizon": 2000}
env = ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)

env.reset()

def report(label):
    """Print scoop pos, particle stats, and metrics."""
    ee_pos = env._ee_link.get_pos()
    if ee_pos.dim() > 1:
        ee_pos = ee_pos.squeeze(0)

    particle_pos = env.particles.get_particles_pos()
    if particle_pos.dim() == 3:
        particle_pos = particle_pos[0]

    # Particles near scoop
    dist_to_ee = torch.norm(particle_pos - ee_pos.unsqueeze(0), dim=-1)
    n_near = (dist_to_ee < 0.06).sum().item()
    n_near_10cm = (dist_to_ee < 0.10).sum().item()

    # Particle position stats
    p_y_mean = particle_pos[:, 1].mean().item()
    p_y_min = particle_pos[:, 1].min().item()
    p_y_max = particle_pos[:, 1].max().item()
    p_z_mean = particle_pos[:, 2].mean().item()

    # How many particles are in target y range (0.29-0.41)?
    in_target_y = ((particle_pos[:, 1] >= 0.29) & (particle_pos[:, 1] <= 0.41)).sum().item()
    # In target x range (0.44-0.56)?
    in_target_xy = (
        (particle_pos[:, 0] >= 0.44) & (particle_pos[:, 0] <= 0.56) &
        (particle_pos[:, 1] >= 0.29) & (particle_pos[:, 1] <= 0.41)
    ).sum().item()

    metrics = env.compute_metrics()

    print(f"  [{label}]")
    print(f"    Scoop pos: x={ee_pos[0]:.3f} y={ee_pos[1]:.3f} z={ee_pos[2]:.3f}")
    print(f"    Particles near scoop (6cm): {n_near}, (10cm): {n_near_10cm}")
    print(f"    Particle y: mean={p_y_mean:.3f} min={p_y_min:.3f} max={p_y_max:.3f}  z_mean={p_z_mean:.3f}")
    print(f"    In target y-range: {in_target_y}  In target xy: {in_target_xy}")
    print(f"    n_on_tool={metrics['n_on_tool']}  n_in_target={metrics['n_in_target']}  n_spilled={metrics['n_spilled']}")
    print(f"    Target AABB: min={env._target_bbox_min.tolist()} max={env._target_bbox_max.tolist()}")
    print()

report("INITIAL")

# Phase 1: Approach
interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 20, settle_per_step=1)
interpolate_waypoints(env, EXTEND_FWD_S, HOVER_SOURCE_S, 20, settle_per_step=1)
report("HOVER_SOURCE")

interpolate_waypoints(env, HOVER_SOURCE_S, SCOOP_START_S, 40, settle_per_step=2)
report("SCOOP_START")

# Phase 2: Sweep through material
interpolate_waypoints(env, SCOOP_START_S, SCOOP_MID_S, 60, settle_per_step=2)
report("SCOOP_MID (after sweep)")

settle(env, 30)
report("After settle")

# Phase 3: Continue pushing toward target (now at y=0.20 instead of 0.35)
# Target AABB y-range: 0.14-0.26 (target at y=0.20, half-size 0.06)
PUSH_GAP_S = [0.2, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
interpolate_waypoints(env, SCOOP_PAST_S, PUSH_GAP_S, 60, settle_per_step=3)
report("PUSH_GAP")

# Push into target
PUSH_TARGET_S = [0.4, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
interpolate_waypoints(env, PUSH_GAP_S, PUSH_TARGET_S, 60, settle_per_step=3)
report("PUSH_TARGET")

# Push through target
PUSH_THROUGH_S = [0.6, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
interpolate_waypoints(env, PUSH_TARGET_S, PUSH_THROUGH_S, 40, settle_per_step=3)
report("PUSH_THROUGH")

settle(env, 60)
report("FINAL")

settle(env, 40)
report("FINAL")
