"""Test push with PD control instead of set_qpos for smoother motion."""

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import torch
import numpy as np
import genesis as gs

from pta.envs.tasks.scoop_transfer import ScoopTransferTask


def pd_push(env, target_qpos_list, n_steps=100):
    """Drive robot to target via PD control, stepping physics n_steps times."""
    target = torch.tensor(target_qpos_list, dtype=torch.float32, device=gs.device)
    for _ in range(n_steps):
        env.robot.control_dofs_position(target)
        env.scene.step()


scene_cfg = {
    "particle_material": "sand", "n_envs": 0, "tool_type": "scoop",
    "source_wall_height": 0.0, "target_wall_height": 0.0,
    "source_pos": (0.55, -0.05, 0.05),
    "target_pos": (0.55, 0.15, 0.05),
    "target_size": (0.30, 0.20, 0.005),
    "particle_pos": (0.55, -0.05, 0.10),
}
env = ScoopTransferTask(config={"horizon": 5000}, scene_config=scene_cfg)
env.reset()

# Set PD gains
env.robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]))
env.robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200]))

def report(label):
    ee_pos = env._ee_link.get_pos()
    if ee_pos.dim() > 1: ee_pos = ee_pos.squeeze(0)
    pp = env.particles.get_particles_pos()
    if pp.dim() == 3: pp = pp[0]
    m = env.compute_metrics()
    in_target_y = ((pp[:, 1] >= 0.05) & (pp[:, 1] <= 0.25)).sum().item()
    print(f"  [{label}] scoop=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})"
          f"  in_target={m['n_in_target']}  spilled={m['n_spilled']}  in_target_y={in_target_y}"
          f"  p_y_mean={pp[:,1].mean():.3f}")

# 1. Approach
BEHIND = [-0.20, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
pd_push(env, [0, 0, 0, -1.57, 0, 1.57, -0.785], 200)  # home
report("HOME")

pd_push(env, [0, 0.5, 0, -1.8, 0, 1.8, 0], 200)  # extend
report("EXTEND")

pd_push(env, BEHIND, 300)  # position behind pile
report("BEHIND")

# 2. Push forward — single long push
TARGET_PUSH = [0.6, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
pd_push(env, TARGET_PUSH, 800)  # slow push over 800 steps
report("PUSH_END")

# Let settle
pd_push(env, TARGET_PUSH, 200)
report("SETTLED")

print(f"\nFinal metrics:")
m = env.compute_metrics()
for k, v in m.items():
    print(f"  {k}: {v}")
