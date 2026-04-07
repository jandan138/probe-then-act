"""Scoop material sweep -- quantify particle retention on scoop during traverse.

Tests whether a scoop-based transfer (insert → lift → traverse → deposit)
discriminates materials better than edge-push. The key question: how many
particles does each material retain on the scoop during horizontal traverse?

Previous finding: particles DON'T ADHERE to rigid scoop during MPM traverse.
This script quantifies exactly how many are captured vs fall off, per material.

Variants tested:
  1. Standard Sequence A scoop (existing waypoints)
  2. Slow traverse (2x interpolation steps)
  3. Low lift (minimal lift height above source)
  4. Tilted scoop (J7 angled to retain particles during traverse)

Usage::

    source /home/zhuzihou/dev/Genesis/.venv/bin/activate
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
    python pta/scripts/scoop_material_sweep.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Waypoints (7-DOF scoop, from run_scripted_baseline.py)
# ---------------------------------------------------------------------------
HOME_S = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
EXTEND_FWD_S = [0.0, 0.5, 0.0, -1.8, 0.0, 1.8, 0.0]
HOVER_SOURCE_S = [0.0, 1.0, 0.0, -1.5, 0.0, 1.5, 0.0]

# Scoop at particle level (-y edge of source)
SCOOP_START_S = [-0.15, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
SCOOP_MID_S = [0.0, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]
SCOOP_PAST_S = [0.15, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]

# Slight lift to secure particles in scoop cup
LIFT_LOW_S = [0.0, 1.1, 0.0, -1.2, 0.0, 1.2, 0.0]
# Full lift above source rim
LIFT_S = [0.0, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]
# Traverse to over target while lifted
TRAVERSE_S = [0.7, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]
# Lower over target
DEPOSIT_S = [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 0.0]
# Tilt scoop to dump
DUMP_S = [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 1.5]

# Variant waypoints
# Low-lift: only lift to z~0.065 instead of z~0.160
LIFT_MINIMAL_S = [0.0, 1.2, 0.0, -1.1, 0.0, 1.15, 0.0]
TRAVERSE_LOW_S = [0.7, 1.2, 0.0, -1.1, 0.0, 1.15, 0.0]
DEPOSIT_LOW_S = [0.7, 1.2, 0.0, -1.1, 0.0, 1.15, 0.0]

# Tilted scoop: J7 rotated during traverse to cup particles
LIFT_LOW_TILT_S = [0.0, 1.1, 0.0, -1.2, 0.0, 1.2, -0.8]
TRAVERSE_TILT_S = [0.7, 0.8, 0.0, -1.5, 0.0, 1.5, -0.8]
DEPOSIT_TILT_S = [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, -0.8]
DUMP_TILT_S = [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 1.5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interpolate_waypoints(env, start_qpos, end_qpos, n_steps, settle_per_step=1):
    start = torch.tensor(start_qpos, dtype=torch.float32, device="cuda")
    end = torch.tensor(end_qpos, dtype=torch.float32, device="cuda")
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        qpos = start * (1 - alpha) + end * alpha
        env.robot.set_qpos(qpos)
        for _ in range(settle_per_step):
            env.scene.step()


def settle(env, n_steps=20):
    for _ in range(n_steps):
        env.scene.step()


def snapshot_metrics(env, label=""):
    """Get metrics at a specific point in the trajectory."""
    metrics = env.compute_metrics()
    return {
        "label": label,
        "n_on_tool": metrics["n_on_tool"],
        "n_in_target": metrics["n_in_target"],
        "transfer_efficiency": metrics["transfer_efficiency"],
        "spill_ratio": metrics["spill_ratio"],
        "total_particles": metrics["total_particles"],
    }


# ---------------------------------------------------------------------------
# Scoop variants
# ---------------------------------------------------------------------------

def run_standard_scoop(env):
    """Standard Sequence A scoop: sweep → lift → traverse → deposit → dump."""
    env.reset()
    snapshots = []

    # Approach
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 20, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, HOVER_SOURCE_S, 20, settle_per_step=1)

    # Scoop through source
    interpolate_waypoints(env, HOVER_SOURCE_S, SCOOP_START_S, 30, settle_per_step=1)
    interpolate_waypoints(env, SCOOP_START_S, SCOOP_MID_S, 40, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_MID_S, SCOOP_PAST_S, 30, settle_per_step=2)

    # Lift low
    interpolate_waypoints(env, SCOOP_PAST_S, LIFT_LOW_S, 30, settle_per_step=2)
    settle(env, 10)
    snapshots.append(snapshot_metrics(env, "after_lift_low"))

    # Full lift
    interpolate_waypoints(env, LIFT_LOW_S, LIFT_S, 30, settle_per_step=2)
    settle(env, 10)
    snapshots.append(snapshot_metrics(env, "after_lift_full"))

    # Traverse (midpoint check)
    mid_traverse = [x * 0.5 + y * 0.5 for x, y in zip(LIFT_S, TRAVERSE_S)]
    interpolate_waypoints(env, LIFT_S, mid_traverse, 30, settle_per_step=2)
    snapshots.append(snapshot_metrics(env, "traverse_midpoint"))

    # Complete traverse
    interpolate_waypoints(env, mid_traverse, TRAVERSE_S, 30, settle_per_step=2)

    # Deposit
    interpolate_waypoints(env, TRAVERSE_S, DEPOSIT_S, 20, settle_per_step=2)
    snapshots.append(snapshot_metrics(env, "at_deposit"))

    # Dump
    interpolate_waypoints(env, DEPOSIT_S, DUMP_S, 20, settle_per_step=2)
    settle(env, 40)

    final = env.compute_metrics()
    snapshots.append(snapshot_metrics(env, "final"))
    return final, snapshots


def run_slow_traverse(env):
    """Slow traverse: 2x interpolation steps during traverse phase."""
    env.reset()
    snapshots = []

    # Approach + scoop (same as standard)
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 20, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, HOVER_SOURCE_S, 20, settle_per_step=1)
    interpolate_waypoints(env, HOVER_SOURCE_S, SCOOP_START_S, 30, settle_per_step=1)
    interpolate_waypoints(env, SCOOP_START_S, SCOOP_MID_S, 40, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_MID_S, SCOOP_PAST_S, 30, settle_per_step=2)

    # Lift
    interpolate_waypoints(env, SCOOP_PAST_S, LIFT_LOW_S, 30, settle_per_step=2)
    settle(env, 10)
    snapshots.append(snapshot_metrics(env, "after_lift_low"))
    interpolate_waypoints(env, LIFT_LOW_S, LIFT_S, 30, settle_per_step=2)

    # SLOW traverse (2x steps, 3 settle per step)
    mid_traverse = [x * 0.5 + y * 0.5 for x, y in zip(LIFT_S, TRAVERSE_S)]
    interpolate_waypoints(env, LIFT_S, mid_traverse, 60, settle_per_step=3)
    snapshots.append(snapshot_metrics(env, "traverse_midpoint"))
    interpolate_waypoints(env, mid_traverse, TRAVERSE_S, 60, settle_per_step=3)

    # Deposit + dump
    interpolate_waypoints(env, TRAVERSE_S, DEPOSIT_S, 20, settle_per_step=2)
    snapshots.append(snapshot_metrics(env, "at_deposit"))
    interpolate_waypoints(env, DEPOSIT_S, DUMP_S, 20, settle_per_step=2)
    settle(env, 40)

    final = env.compute_metrics()
    snapshots.append(snapshot_metrics(env, "final"))
    return final, snapshots


def run_low_lift(env):
    """Low lift: keep scoop close to surface during traverse."""
    env.reset()
    snapshots = []

    # Approach + scoop
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 20, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, HOVER_SOURCE_S, 20, settle_per_step=1)
    interpolate_waypoints(env, HOVER_SOURCE_S, SCOOP_START_S, 30, settle_per_step=1)
    interpolate_waypoints(env, SCOOP_START_S, SCOOP_MID_S, 40, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_MID_S, SCOOP_PAST_S, 30, settle_per_step=2)

    # Minimal lift only
    interpolate_waypoints(env, SCOOP_PAST_S, LIFT_MINIMAL_S, 20, settle_per_step=2)
    settle(env, 10)
    snapshots.append(snapshot_metrics(env, "after_lift_low"))

    # Low traverse
    mid_traverse = [x * 0.5 + y * 0.5 for x, y in zip(LIFT_MINIMAL_S, TRAVERSE_LOW_S)]
    interpolate_waypoints(env, LIFT_MINIMAL_S, mid_traverse, 30, settle_per_step=2)
    snapshots.append(snapshot_metrics(env, "traverse_midpoint"))
    interpolate_waypoints(env, mid_traverse, TRAVERSE_LOW_S, 30, settle_per_step=2)

    # Deposit
    interpolate_waypoints(env, TRAVERSE_LOW_S, DEPOSIT_LOW_S, 20, settle_per_step=2)
    snapshots.append(snapshot_metrics(env, "at_deposit"))

    # Dump (tilt)
    DUMP_LOW_S = list(DEPOSIT_LOW_S)
    DUMP_LOW_S[6] = 1.5
    interpolate_waypoints(env, DEPOSIT_LOW_S, DUMP_LOW_S, 20, settle_per_step=2)
    settle(env, 40)

    final = env.compute_metrics()
    snapshots.append(snapshot_metrics(env, "final"))
    return final, snapshots


def run_tilted_scoop(env):
    """Tilted scoop: J7 angled during traverse to retain particles."""
    env.reset()
    snapshots = []

    # Approach + scoop
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 20, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, HOVER_SOURCE_S, 20, settle_per_step=1)
    interpolate_waypoints(env, HOVER_SOURCE_S, SCOOP_START_S, 30, settle_per_step=1)
    interpolate_waypoints(env, SCOOP_START_S, SCOOP_MID_S, 40, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_MID_S, SCOOP_PAST_S, 30, settle_per_step=2)

    # Lift with tilt
    interpolate_waypoints(env, SCOOP_PAST_S, LIFT_LOW_TILT_S, 30, settle_per_step=2)
    settle(env, 10)
    snapshots.append(snapshot_metrics(env, "after_lift_low"))

    # Tilted traverse
    mid_traverse = [x * 0.5 + y * 0.5 for x, y in zip(LIFT_LOW_TILT_S, TRAVERSE_TILT_S)]
    interpolate_waypoints(env, LIFT_LOW_TILT_S, mid_traverse, 40, settle_per_step=2)
    snapshots.append(snapshot_metrics(env, "traverse_midpoint"))
    interpolate_waypoints(env, mid_traverse, TRAVERSE_TILT_S, 40, settle_per_step=2)

    # Deposit + dump (untilt then dump)
    interpolate_waypoints(env, TRAVERSE_TILT_S, DEPOSIT_TILT_S, 20, settle_per_step=2)
    snapshots.append(snapshot_metrics(env, "at_deposit"))
    interpolate_waypoints(env, DEPOSIT_TILT_S, DUMP_TILT_S, 20, settle_per_step=2)
    settle(env, 40)

    final = env.compute_metrics()
    snapshots.append(snapshot_metrics(env, "final"))
    return final, snapshots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_EPISODES = 3

VARIANTS = {
    "standard": run_standard_scoop,
    "slow_traverse": run_slow_traverse,
    "low_lift": run_low_lift,
    "tilted_scoop": run_tilted_scoop,
}


def run_single_material(material):
    """Run all variants for a single material. Returns (rows, snapshots)."""
    import argparse

    output_dir = "results/scoop_material_sweep"
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    all_snapshots = []

    print("=" * 70)
    print(f"Material: {material}")
    print("=" * 70)

    print(f"  Building env (flat layout, scoop tool, {material}) ...")
    t0 = time.time()

    from pta.envs.tasks.scoop_transfer import ScoopTransferTask

    scene_cfg = {
        "particle_material": material,
        "n_envs": 0,
        "tool_type": "scoop",
        "task_layout": "flat",
        "particle_pos": (0.5, 0.0, 0.12),
        "particle_size": (0.10, 0.10, 0.03),
    }
    task_cfg = {"horizon": 500}

    env = ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)

    build_time = time.time() - t0
    print(f"  OK ({build_time:.1f}s, {env._total_particles} particles)")
    print()

    for variant_name, variant_fn in VARIANTS.items():
        print(f"  --- Variant: {variant_name} ---")

        for ep in range(N_EPISODES):
            t_ep = time.time()
            try:
                final_metrics, snapshots = variant_fn(env)
            except Exception as e:
                print(f"    ep {ep+1}: FAILED ({e})")
                import traceback
                traceback.print_exc()
                continue

            dt = time.time() - t_ep
            te = final_metrics.get("transfer_efficiency", 0.0)
            sr = final_metrics.get("spill_ratio", 0.0)
            n_in = final_metrics.get("n_in_target", 0)
            n_tool = final_metrics.get("n_on_tool", 0)
            total = final_metrics.get("total_particles", 0)

            print(
                f"    ep {ep+1}/{N_EPISODES}  "
                f"transfer={te:.4f}  spill={sr:.4f}  "
                f"in_target={n_in}  on_tool={n_tool}  "
                f"total={total}  ({dt:.1f}s)"
            )

            all_rows.append({
                "material": material,
                "variant": variant_name,
                "episode": ep + 1,
                "transfer_efficiency": te,
                "spill_ratio": sr,
                "n_in_target": n_in,
                "n_on_tool": n_tool,
                "total_particles": total,
            })

            for snap in snapshots:
                all_snapshots.append({
                    "material": material,
                    "variant": variant_name,
                    "episode": ep + 1,
                    "phase": snap["label"],
                    "n_on_tool": snap["n_on_tool"],
                    "n_in_target": snap["n_in_target"],
                    "transfer_efficiency": snap["transfer_efficiency"],
                    "spill_ratio": snap["spill_ratio"],
                })

        print()

    # Save per-material results (append-safe)
    sweep_csv = os.path.join(output_dir, f"scoop_sweep_{material}.csv")
    snap_csv = os.path.join(output_dir, f"scoop_snapshots_{material}.csv")

    if all_rows:
        with open(sweep_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Saved {len(all_rows)} rows to {sweep_csv}")

    if all_snapshots:
        with open(snap_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_snapshots[0].keys()))
            writer.writeheader()
            writer.writerows(all_snapshots)
        print(f"  Saved {len(all_snapshots)} snapshot rows to {snap_csv}")

    return all_rows, all_snapshots


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", type=str, required=True,
                        choices=["sand", "snow", "elastoplastic"],
                        help="Material to test")
    args = parser.parse_args()
    run_single_material(args.material)
    print("Done.")


if __name__ == "__main__":
    main()
