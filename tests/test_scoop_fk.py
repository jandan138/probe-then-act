"""Probe scoop FK: print EE position at various joint configs to calibrate waypoints."""

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import torch
import numpy as np
import genesis as gs

gs.init(backend=gs.gpu, precision="32", logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=2e-3, substeps=25),
    rigid_options=gs.options.RigidOptions(
        dt=2e-3, constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True, enable_joint_limit=True,
    ),
    show_viewer=False,
)

scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True, coup_friction=0.5),
    morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
)

robot = scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True, coup_friction=1.0),
    morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda_scoop.xml", pos=(0.0, 0.0, 0.0)),
)

scene.build(n_envs=0)

robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]))
robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200]))

scoop_link = robot.get_link("scoop")

# Test configurations (7-DOF, no fingers)
configs = {
    "HOME":              [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "EXTEND_FWD":        [0.0, 0.5, 0.0, -1.8, 0.0, 1.8, 0.0],
    "HOVER_SOURCE":      [0.0, 1.0, 0.0, -1.5, 0.0, 1.5, 0.0],
    # Try different J2 values to find the right scoop height
    "J2=1.2_J4=-1.0":   [0.0, 1.2, 0.0, -1.0, 0.0, 1.1, 0.0],
    "J2=1.3_J4=-1.0":   [0.0, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0],
    "J2=1.4_J4=-0.9":   [0.0, 1.4, 0.0, -0.9, 0.0, 1.0, 0.0],
    "J2=1.5_J4=-0.9":   [0.0, 1.5, 0.0, -0.9, 0.0, 1.0, 0.0],
    "START_-y_J2=1.3":  [-0.15, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0],
    "MID_J2=1.3":       [0.0, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0],
    "LIFT_LOW":          [0.0, 1.1, 0.0, -1.2, 0.0, 1.2, 0.0],
    "LIFT_HIGH":         [0.0, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0],
    "TRAV_target":       [0.7, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0],
    "DEPOSIT_target":    [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 0.0],
    "DUMP_target":       [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 1.5],
}

print(f"{'Config':<22s}  {'X':>8s}  {'Y':>8s}  {'Z':>8s}")
print("-" * 55)

for name, qpos_list in configs.items():
    qpos = torch.tensor(qpos_list, dtype=torch.float32, device=gs.device)
    robot.set_qpos(qpos)
    for _ in range(10):
        scene.step()
    pos = scoop_link.get_pos()
    if pos.dim() > 1:
        pos = pos.squeeze(0)
    print(f"{name:<22s}  {pos[0].item():>8.4f}  {pos[1].item():>8.4f}  {pos[2].item():>8.4f}")

print()
print("Source container: centre (0.5, 0.0, 0.05), particles at z~0.10")
print("Target container: centre (0.5, 0.35, 0.05)")
