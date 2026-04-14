from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from bowl_transport_diagnosis import phase_snapshot_row, write_csv
from pta.envs.tasks.scoop_transfer import ScoopTransferTask
from run_scripted_baseline import (
    BOWL_APPROACH_S,
    BOWL_CAPTURE_S,
    BOWL_INSERT_S,
    BOWL_LIFT_S,
    BOWL_TRAVERSE_FAST_S,
    BOWL_TRAVERSE_MID_S,
    EXTEND_FWD_S,
    HOME_S,
    interpolate_waypoints,
    settle,
)


def set_transport_phase(env: ScoopTransferTask, phase: str) -> None:
    if hasattr(env, "set_bowl_transport_phase"):
        env.set_bowl_transport_phase(phase)


def build_env(tool_type: str, seed: int) -> ScoopTransferTask:
    scene_cfg = {
        "particle_material": "sand",
        "n_envs": 0,
        "tool_type": tool_type,
        "task_layout": "flat",
        "particle_pos": (0.5, 0.0, 0.12),
        "particle_size": (0.10, 0.10, 0.03),
        "bowl_contact_quality_enabled": True,
        "bowl_enable_cpic": True,
        "bowl_substeps_override": 40,
        "bowl_robot_coup_friction": 6.0,
        "bowl_robot_coup_softness": 0.0005,
        "bowl_robot_sdf_cell_size": 0.002,
    }
    task_cfg = {"horizon": 1600}
    torch.manual_seed(seed)
    return ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)


def run_branch(
    *,
    env: ScoopTransferTask,
    branch_name: str,
    traverse_steps_first_half: int,
    traverse_steps_second_half: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def snap(label: str) -> None:
        rows.append(
            {
                "branch": branch_name,
                **phase_snapshot_row(
                    env,
                    phase=label,
                    traverse_speed=0.2,
                    step_idx=len(rows),
                ),
            }
        )

    env.reset()
    set_transport_phase(env, "off")
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 15, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, BOWL_APPROACH_S, 30, settle_per_step=2)
    interpolate_waypoints(env, BOWL_APPROACH_S, BOWL_INSERT_S, 45, settle_per_step=3)
    snap("capture_approach")
    interpolate_waypoints(env, BOWL_INSERT_S, BOWL_CAPTURE_S, 35, settle_per_step=3)
    settle(env, 10)
    snap("capture_end")

    interpolate_waypoints(env, BOWL_CAPTURE_S, BOWL_LIFT_S, 40, settle_per_step=3)
    settle(env, 10)
    snap("lift_full")

    set_transport_phase(env, "carry")
    early_waypoint = [
        x * 0.5 + y * 0.5 for x, y in zip(BOWL_LIFT_S, BOWL_TRAVERSE_MID_S)
    ]
    interpolate_waypoints(
        env, BOWL_LIFT_S, early_waypoint, traverse_steps_first_half, settle_per_step=2
    )
    settle(env, 10)
    snap("carry_early")
    interpolate_waypoints(
        env,
        early_waypoint,
        BOWL_TRAVERSE_MID_S,
        traverse_steps_first_half,
        settle_per_step=2,
    )
    settle(env, 10)
    snap("carry_mid")

    late_waypoint = [
        x * 0.5 + y * 0.5 for x, y in zip(BOWL_TRAVERSE_MID_S, BOWL_TRAVERSE_FAST_S)
    ]
    interpolate_waypoints(
        env,
        BOWL_TRAVERSE_MID_S,
        late_waypoint,
        traverse_steps_second_half,
        settle_per_step=2,
    )
    settle(env, 10)
    snap("carry_late")
    interpolate_waypoints(
        env,
        late_waypoint,
        BOWL_TRAVERSE_FAST_S,
        traverse_steps_second_half,
        settle_per_step=2,
    )
    settle(env, 10)
    snap("pre_pour")

    set_transport_phase(env, "off")
    return rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    by_branch: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_branch.setdefault(row["branch"], {})[row["phase"]] = row

    for branch, phases in by_branch.items():
        lift_full = phases["lift_full"]
        carry_early = phases["carry_early"]
        carry_mid = phases["carry_mid"]
        pre_pour = phases["pre_pour"]
        summary.append(
            {
                "branch": branch,
                "lift_full_n_on_tool": lift_full["n_on_tool"],
                "carry_early_n_on_tool": carry_early["n_on_tool"],
                "carry_mid_n_on_tool": carry_mid["n_on_tool"],
                "pre_pour_n_on_tool": pre_pour["n_on_tool"],
                "carry_early_inside_bowl": carry_early["inside_bowl_local"],
                "carry_early_above_rim": carry_early["above_rim_local"],
                "carry_early_front_escape": carry_early["front_escape_band"],
                "carry_early_side_escape": carry_early["side_escape_band"],
                "early_loss": lift_full["n_on_tool"] - carry_early["n_on_tool"],
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scoop_debug/carry_onset_highwall_discriminator",
    )
    parser.add_argument("--seed", type=int, default=91)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    branches = [
        ("baseline_nominal", "bowl", 15, 15),
        ("highwall_nominal", "bowl_highwall", 15, 15),
        ("baseline_slow", "bowl", 60, 60),
        ("highwall_slow", "bowl_highwall", 60, 60),
    ]

    all_rows: List[Dict[str, Any]] = []
    for idx, (name, tool_type, first_half, second_half) in enumerate(branches):
        env = build_env(tool_type=tool_type, seed=args.seed + idx)
        all_rows.extend(
            run_branch(
                env=env,
                branch_name=name,
                traverse_steps_first_half=first_half,
                traverse_steps_second_half=second_half,
            )
        )
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(out_dir / "phase_metrics.csv", all_rows)
    summary = summarize(all_rows)
    write_csv(out_dir / "branch_summary.csv", summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
