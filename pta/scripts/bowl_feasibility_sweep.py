from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List, Tuple

from run_scripted_baseline import (
    BOWL_APPROACH_S,
    BOWL_CAPTURE_S,
    BOWL_INSERT_S,
    BOWL_LIFT_S,
    BOWL_POUR_S,
    BOWL_SETTLE_S,
    BOWL_TRAVERSE_FAST_S,
    BOWL_TRAVERSE_MID_S,
    EXTEND_FWD_S,
    HOME_S,
    _bowl_traverse_steps,
    interpolate_waypoints,
    settle,
)


MATERIALS = ["sand", "snow", "elastoplastic"]
TRAVERSE_SPEEDS = [0.2, 0.5, 1.0]
EPISODES_PER = 5


def snapshot_metrics(env: Any, phase: str, traverse_speed: float) -> Dict[str, Any]:
    metrics = env.compute_metrics()
    return {
        "phase": phase,
        "traverse_speed": traverse_speed,
        "n_on_tool": metrics["n_on_tool"],
        "n_in_target": metrics["n_in_target"],
        "transfer_efficiency": metrics["transfer_efficiency"],
        "spill_ratio": metrics["spill_ratio"],
        "total_particles": metrics["total_particles"],
    }


def run_bowl_episode(
    env: Any, traverse_speed: float
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env.reset()
    snapshots: List[Dict[str, Any]] = []
    traverse_steps = _bowl_traverse_steps(traverse_speed)

    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 15, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, BOWL_APPROACH_S, 30, settle_per_step=2)
    snapshots.append(snapshot_metrics(env, "approach", traverse_speed))

    interpolate_waypoints(env, BOWL_APPROACH_S, BOWL_INSERT_S, 45, settle_per_step=3)
    snapshots.append(snapshot_metrics(env, "capture", traverse_speed))

    interpolate_waypoints(env, BOWL_INSERT_S, BOWL_CAPTURE_S, 35, settle_per_step=3)
    settle(env, 10)
    snapshots.append(snapshot_metrics(env, "lift_capture", traverse_speed))

    interpolate_waypoints(env, BOWL_CAPTURE_S, BOWL_LIFT_S, 35, settle_per_step=3)
    settle(env, 10)
    snapshots.append(snapshot_metrics(env, "lift_full", traverse_speed))

    mid_steps = traverse_steps // 2
    interpolate_waypoints(
        env, BOWL_LIFT_S, BOWL_TRAVERSE_MID_S, mid_steps, settle_per_step=2
    )
    snapshots.append(snapshot_metrics(env, "traverse_mid", traverse_speed))

    interpolate_waypoints(
        env,
        BOWL_TRAVERSE_MID_S,
        BOWL_TRAVERSE_FAST_S,
        traverse_steps - mid_steps,
        settle_per_step=2,
    )
    settle(env, 10)
    snapshots.append(snapshot_metrics(env, "at_target", traverse_speed))

    interpolate_waypoints(env, BOWL_TRAVERSE_FAST_S, BOWL_POUR_S, 30, settle_per_step=3)
    snapshots.append(snapshot_metrics(env, "pour", traverse_speed))

    interpolate_waypoints(env, BOWL_POUR_S, BOWL_SETTLE_S, 20, settle_per_step=3)
    settle(env, 40)
    final_metrics = env.compute_metrics()
    snapshots.append(snapshot_metrics(env, "final", traverse_speed))

    final_metrics["traverse_speed"] = traverse_speed
    final_metrics["traverse_steps"] = traverse_steps
    return final_metrics, snapshots


def build_env(material: str) -> Any:
    from pta.envs.tasks.scoop_transfer import ScoopTransferTask

    scene_cfg = {
        "particle_material": material,
        "n_envs": 0,
        "tool_type": "bowl",
        "task_layout": "flat",
        "particle_pos": (0.5, 0.0, 0.12),
        "particle_size": (0.10, 0.10, 0.03),
    }
    task_cfg = {"horizon": 500}
    return ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)


def main() -> None:
    output_dir = "results/bowl_feasibility_sweep"
    os.makedirs(output_dir, exist_ok=True)

    sweep_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []

    print("=" * 72)
    print("Bowl Feasibility Sweep")
    print("=" * 72)
    print(f"  Materials: {MATERIALS}")
    print(f"  Traverse speeds: {TRAVERSE_SPEEDS}")
    print(f"  Episodes per combo: {EPISODES_PER}")
    print()

    for material in MATERIALS:
        print(f"Material: {material}")
        env = build_env(material)
        print(f"  Built env: {env._total_particles} particles")

        for speed in TRAVERSE_SPEEDS:
            print(f"  Traverse speed: {speed:.1f} m/s")
            speed_rows: List[Dict[str, Any]] = []

            for episode in range(EPISODES_PER):
                t0 = time.time()
                final_metrics, snapshots = run_bowl_episode(env, speed)
                dt = time.time() - t0

                row = {
                    "material": material,
                    "traverse_speed": speed,
                    "episode": episode + 1,
                    "transfer_efficiency": final_metrics["transfer_efficiency"],
                    "spill_ratio": final_metrics["spill_ratio"],
                    "n_in_target": final_metrics["n_in_target"],
                    "n_spilled": final_metrics["n_spilled"],
                    "n_on_tool": final_metrics["n_on_tool"],
                    "total_particles": final_metrics["total_particles"],
                    "traverse_steps": final_metrics["traverse_steps"],
                }
                sweep_rows.append(row)
                speed_rows.append(row)

                for snapshot in snapshots:
                    snapshot_rows.append(
                        {
                            "material": material,
                            "episode": episode + 1,
                            **snapshot,
                        }
                    )

                print(
                    f"    ep {episode + 1}/{EPISODES_PER}  "
                    f"transfer={row['transfer_efficiency']:.4f}  "
                    f"spill={row['spill_ratio']:.4f}  "
                    f"on_tool={row['n_on_tool']}  "
                    f"in_target={row['n_in_target']}  "
                    f"({dt:.1f}s)"
                )

            mean_transfer = sum(r["transfer_efficiency"] for r in speed_rows) / len(
                speed_rows
            )
            mean_spill = sum(r["spill_ratio"] for r in speed_rows) / len(speed_rows)
            print(f"    mean transfer={mean_transfer:.4f}  mean spill={mean_spill:.4f}")

        del env
        print()

    sweep_path = os.path.join(output_dir, "bowl_sweep.csv")
    snapshot_path = os.path.join(output_dir, "bowl_snapshots.csv")

    with open(sweep_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "material",
                "traverse_speed",
                "episode",
                "transfer_efficiency",
                "spill_ratio",
                "n_in_target",
                "n_spilled",
                "n_on_tool",
                "total_particles",
                "traverse_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(sweep_rows)

    with open(snapshot_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "material",
                "episode",
                "phase",
                "traverse_speed",
                "n_on_tool",
                "n_in_target",
                "transfer_efficiency",
                "spill_ratio",
                "total_particles",
            ],
        )
        writer.writeheader()
        writer.writerows(snapshot_rows)

    print(f"Saved {len(sweep_rows)} rows to {sweep_path}")
    print(f"Saved {len(snapshot_rows)} rows to {snapshot_path}")


if __name__ == "__main__":
    main()
