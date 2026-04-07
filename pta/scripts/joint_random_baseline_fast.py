"""Joint-space random baseline — single-init version.

Only initializes Genesis ONCE, runs 3 modes sequentially within
the same episode (reset between modes via task.reset()).
"""
from __future__ import annotations
import sys, numpy as np, torch
sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import genesis as gs
from pta.envs.builders.scene_builder import SceneBuilder
from pta.envs.tasks.scoop_transfer import ScoopTransferTask

JOINT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

def run(task, mode, seed=42, n_steps=500):
    task.reset()
    for _ in range(50):
        task.scene.step()  # settle

    rng = np.random.RandomState(seed)
    q = HOME.copy()
    for _ in range(n_steps):
        if mode == "uniform":
            q = rng.uniform(JOINT_LOW, JOINT_HIGH).astype(np.float32)
        elif mode == "walk":
            q = np.clip(q + rng.normal(0, 0.05, 7).astype(np.float32), JOINT_LOW, JOINT_HIGH)
        elif mode == "near_table":
            q = HOME.copy()
            q[0] = rng.uniform(-0.3, 0.9)
            q[1] = rng.uniform(0.8, 1.7)
            q[3] = rng.uniform(-1.5, -0.5)
            q[5] = rng.uniform(0.5, 1.5)
            q = np.clip(q, JOINT_LOW, JOINT_HIGH)

        task.robot.set_qpos(torch.tensor(q, dtype=torch.float32, device="cuda"))
        for _ in range(3):
            task.scene.step()

    for _ in range(80):
        task.scene.step()
    return task.compute_metrics()

material = sys.argv[1] if len(sys.argv) > 1 else "sand"
sc = SceneBuilder().build_scene({
    "tool_type": "scoop", "task_layout": "edge_push",
    "particle_material": material, "n_envs": 0,
})
task = ScoopTransferTask(sc)

print(f"\nJoint-space random baselines — material: {material}")
print(f"{'Mode':<15} {'Seed':<6} {'Transfer%':<10} {'Spill%':<10} {'n_target':<10}")
print("-" * 55)

for mode in ["uniform", "walk", "near_table"]:
    for seed in [42, 43, 44]:
        m = run(task, mode, seed)
        print(f"{mode:<15} {seed:<6} {m['transfer_efficiency']*100:<10.1f} {m['spill_ratio']*100:<10.1f} {m['n_in_target']:<10}")
