"""Joint-space random baseline for Config D validation.

Tests whether random joint-space motions (bypassing IK) can accidentally
push particles off the platform. If random flailing achieves significant
transfer, the task is too easy and Config D is invalid.

Uses set_qpos() directly (same as JointResidualWrapper) to ensure the
robot actually moves, unlike Sequence C which goes through broken IK.
"""
from __future__ import annotations

import argparse
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import genesis as gs
from pta.envs.builders.scene_builder import SceneBuilder
from pta.envs.tasks.scoop_transfer import ScoopTransferTask


# Franka joint limits (from panda_scoop.xml)
JOINT_LIMITS_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_LIMITS_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def run_joint_random(task, n_steps=500, seed=42, mode="uniform"):
    """Run random joint-space actions via set_qpos.

    Modes:
      - "uniform": sample uniformly within joint limits each step
      - "walk": random walk from home pose (small perturbations)
      - "near_table": bias toward extended-arm configs near particle level
    """
    rng = np.random.RandomState(seed)
    task.reset()

    # Let particles settle first
    for _ in range(50):
        task.scene.step()

    q = HOME.copy()

    for t in range(n_steps):
        if mode == "uniform":
            # Fully random joint config within limits
            q = rng.uniform(JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH).astype(np.float32)
        elif mode == "walk":
            # Random walk: small perturbations from current config
            dq = rng.normal(0, 0.05, size=7).astype(np.float32)
            q = np.clip(q + dq, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        elif mode == "near_table":
            # Bias J2 toward extended (1.0-1.7) and J1 toward sweep range (-0.3, 0.9)
            q = HOME.copy()
            q[0] = rng.uniform(-0.3, 0.9)   # J1: sweep range
            q[1] = rng.uniform(0.8, 1.7)     # J2: extended/low
            q[3] = rng.uniform(-1.5, -0.5)   # J4: elbow
            q[5] = rng.uniform(0.5, 1.5)     # J6: wrist
            q = np.clip(q, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

        q_tensor = torch.tensor(q, dtype=torch.float32, device="cuda")
        task.robot.set_qpos(q_tensor)
        for _ in range(3):  # settle steps (same as JointResidualWrapper)
            task.scene.step()

    # Final settle
    for _ in range(80):
        task.scene.step()

    return task.compute_metrics()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="sand")
    parser.add_argument("--n-episodes", type=int, default=5)
    args = parser.parse_args()

    scene_cfg = {
        "tool_type": "scoop",
        "task_layout": "edge_push",
        "particle_material": args.material,
        "n_envs": 0,
    }

    builder = SceneBuilder()
    sc = builder.build_scene(scene_cfg)
    task = ScoopTransferTask(sc)

    modes = ["uniform", "walk", "near_table"]

    print(f"\nMaterial: {args.material}")
    print(f"{'Mode':<15} {'Episode':<8} {'Transfer%':<12} {'Spill%':<10} {'n_in_target':<12}")
    print("-" * 60)

    for mode in modes:
        transfers = []
        for ep in range(args.n_episodes):
            # Must re-init Genesis for each episode since set_qpos doesn't reset particles
            if ep > 0 or mode != modes[0]:
                gs._initialized = False
                builder = SceneBuilder()
                sc = builder.build_scene(scene_cfg)
                task = ScoopTransferTask(sc)

            metrics = run_joint_random(task, n_steps=500, seed=42 + ep, mode=mode)
            t_eff = metrics["transfer_efficiency"]
            s_rat = metrics["spill_ratio"]
            n_in = metrics["n_in_target"]
            transfers.append(t_eff)
            print(f"{mode:<15} {ep+1:<8} {t_eff*100:<12.1f} {s_rat*100:<10.1f} {n_in:<12}")

        mean_t = np.mean(transfers) * 100
        std_t = np.std(transfers) * 100
        print(f"{'>> ' + mode:<15} {'mean':<8} {mean_t:<12.1f} ± {std_t:.1f}")
        print()


if __name__ == "__main__":
    main()
