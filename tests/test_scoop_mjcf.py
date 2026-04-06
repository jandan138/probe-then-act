"""Minimal test: verify panda_scoop.xml loads in Genesis with 7 DOFs."""

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import genesis as gs

gs.init(backend=gs.gpu, precision="32", logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=2e-3, substeps=25),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.1, -0.5, -0.05),
        upper_bound=(1.0, 0.8, 0.8),
        grid_density=64,
    ),
    rigid_options=gs.options.RigidOptions(
        dt=2e-3,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=False,
)

# Add ground
scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True, coup_friction=0.5),
    morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
)

# Add robot with scoop
robot = scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True, coup_friction=1.0),
    morph=gs.morphs.MJCF(
        file="xml/franka_emika_panda/panda_scoop.xml",
        pos=(0.0, 0.0, 0.0),
    ),
)

# Add some sand particles for testing
sand = scene.add_entity(
    material=gs.materials.MPM.Sand(),
    morph=gs.morphs.Box(pos=(0.5, 0.0, 0.10), size=(0.10, 0.10, 0.04)),
    surface=gs.surfaces.Default(color=(0.9, 0.8, 0.5, 1.0), vis_mode="particle"),
)

scene.build(n_envs=0)

# Check DOFs
print(f"Robot DOFs: {robot.n_dofs}")
assert robot.n_dofs == 7, f"Expected 7 DOFs, got {robot.n_dofs}"

# Check links
link_names = [l.name for l in robot.links]
print(f"Link names: {link_names}")
assert "scoop" in link_names, f"'scoop' link not found in {link_names}"

# Get scoop link
scoop_link = robot.get_link("scoop")
scoop_pos = scoop_link.get_pos()
print(f"Scoop position (home): {scoop_pos}")

# Set PD gains (7 DOFs)
import numpy as np
robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]))
robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200]))

# Set home qpos
import torch
home_qpos = torch.tensor([0, 0, 0, -1.57079, 0, 1.57079, -0.7853], device=gs.device)
robot.set_qpos(home_qpos)

# Step physics
for i in range(50):
    scene.step()

scoop_pos_after = scoop_link.get_pos()
print(f"Scoop position (after 50 steps): {scoop_pos_after}")

# Check no NaN
assert not torch.isnan(scoop_pos_after).any(), "NaN detected in scoop position!"

print("\n=== panda_scoop.xml test PASSED ===")
print(f"  DOFs: {robot.n_dofs}")
print(f"  Scoop link found: True")
print(f"  No NaN: True")
print(f"  Particles: {sand._n_particles}")
