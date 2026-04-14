from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from PIL import Image
except Exception:
    Image = None

from pta.envs.builders.scene_builder import (
    _DEFAULT_CONFIG,
    _bowl_contact_quality_active,
    _resolve_mpm_options_kwargs,
    _resolve_robot_material_kwargs,
    _resolve_scene_substeps,
)
from pta.envs.tasks.scoop_transfer import (
    ScoopTransferTask,
    _quat_rotate_inverse,
    _should_apply_bowl_constraint_fallback,
    _should_apply_bowl_sticky_fallback,
)
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


BASELINE_BOWL_SCENE = {
    "particle_material": "sand",
    "n_envs": 0,
    "tool_type": "bowl",
}

FLAT_BOWL_SCENE = {
    "particle_material": "sand",
    "n_envs": 0,
    "tool_type": "bowl",
    "task_layout": "flat",
    "particle_pos": (0.5, 0.0, 0.12),
    "particle_size": (0.10, 0.10, 0.03),
}

BOWL_LIFT_LOW_S = [0.5 * (a + b) for a, b in zip(BOWL_CAPTURE_S, BOWL_LIFT_S)]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def merged_scene_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    return {**_DEFAULT_CONFIG, **overrides}


def effective_scene_runtime_config(scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = merged_scene_config(scene_cfg)
    return {
        "tool_type": merged["tool_type"],
        "task_layout": merged["task_layout"],
        "bowl_contact_quality_active": _bowl_contact_quality_active(merged),
        "effective_substeps": _resolve_scene_substeps(merged),
        "effective_mpm_options": _resolve_mpm_options_kwargs(merged),
        "effective_robot_material": _resolve_robot_material_kwargs(merged),
    }


def effective_task_runtime_config(
    scene_cfg: Dict[str, Any], task_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    merged_scene = merged_scene_config(scene_cfg)
    sticky_available = _should_apply_bowl_sticky_fallback(
        enabled=bool(task_cfg.get("bowl_sticky_fallback_enabled", False)),
        tool_type=merged_scene["tool_type"],
        task_layout=merged_scene["task_layout"],
        phase="carry",
    )
    constraint_available = _should_apply_bowl_constraint_fallback(
        enabled=bool(task_cfg.get("bowl_constraint_fallback_enabled", False)),
        tool_type=merged_scene["tool_type"],
        task_layout=merged_scene["task_layout"],
        phase="carry",
    )
    return {
        "tool_type": merged_scene["tool_type"],
        "task_layout": merged_scene["task_layout"],
        "bowl_sticky_fallback_available": sticky_available,
        "bowl_sticky_runtime_activation_requires_phase": "carry"
        if sticky_available
        else None,
        "effective_bowl_sticky_params": task_cfg if sticky_available else {},
        "bowl_constraint_fallback_available": constraint_available,
        "bowl_constraint_runtime_activation_requires_phase": "carry"
        if constraint_available
        else None,
        "effective_bowl_constraint_params": {
            "bowl_constraint_fallback_enabled": task_cfg.get(
                "bowl_constraint_fallback_enabled", False
            ),
            "bowl_constraint_stiffness": task_cfg.get("bowl_constraint_stiffness"),
        }
        if constraint_available
        else {},
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gpu_snapshot() -> Dict[str, str]:
    def run_query(command: List[str]) -> str:
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            text = completed.stdout.strip()
            return text
        except Exception as exc:
            return f"ERROR: {exc}"

    return {
        "gpu": run_query(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "apps": run_query(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        ),
    }


def build_env(
    scene_cfg: Dict[str, Any],
    horizon: int,
    seed: int,
    task_cfg: Dict[str, Any] | None = None,
) -> ScoopTransferTask:
    np.random.seed(seed)
    torch.manual_seed(seed)
    merged_task_cfg = {"horizon": horizon, **(task_cfg or {})}
    return ScoopTransferTask(config=merged_task_cfg, scene_config=scene_cfg)


def env_snapshot(env: ScoopTransferTask) -> Dict[str, Any]:
    return {
        "total_particles": int(env._total_particles),
        "source_pos": tuple(env.sc.source_pos),
        "source_size": tuple(env.sc.source_size),
        "target_pos": tuple(env.sc.target_pos),
        "target_size": tuple(env.sc.target_size),
        "source_bbox_min": env._source_bbox_min.tolist(),
        "source_bbox_max": env._source_bbox_max.tolist(),
        "target_bbox_min": env._target_bbox_min.tolist(),
        "target_bbox_max": env._target_bbox_max.tolist(),
    }


def render_rgb(env: ScoopTransferTask) -> np.ndarray | None:
    try:
        rgb, _, _, _ = env.camera.render(
            rgb=True,
            depth=False,
            segmentation=False,
            normal=False,
        )
    except Exception:
        return None

    img = (
        rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    )
    if img.ndim == 4:
        img = img[0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_frame(env: ScoopTransferTask, path: Path) -> None:
    img = render_rgb(env)
    if img is None or Image is None:
        return
    Image.fromarray(img).save(path)


def plot_curves(rows: List[Dict[str, Any]], path: Path) -> None:
    if plt is None or not rows:
        return
    steps = [row["hold_step"] for row in rows]
    n_on_tool = [row["n_on_tool"] for row in rows]
    spill = [row["spill_ratio"] for row in rows]
    transfer = [row["transfer_efficiency"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(steps, n_on_tool)
    axes[0].set_ylabel("n_on_tool")
    axes[1].plot(steps, spill)
    axes[1].set_ylabel("spill_ratio")
    axes[2].plot(steps, transfer)
    axes[2].set_ylabel("transfer_eff")
    axes[2].set_xlabel("hold_step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metric_row(env: ScoopTransferTask, label: str, hold_step: int) -> Dict[str, Any]:
    metrics = env.compute_metrics()
    return {
        "label": label,
        "hold_step": hold_step,
        "n_on_tool": int(metrics["n_on_tool"]),
        "n_in_target": int(metrics["n_in_target"]),
        "n_spilled": int(metrics["n_spilled"]),
        "transfer_efficiency": float(metrics["transfer_efficiency"]),
        "spill_ratio": float(metrics["spill_ratio"]),
        "total_particles": int(metrics["total_particles"]),
    }


def phase_snapshot_row(
    env: ScoopTransferTask,
    phase: str,
    traverse_speed: float,
    step_idx: int,
) -> Dict[str, Any]:
    metrics = env.compute_metrics()
    particle_pos = env.particles.get_particles_pos()
    if particle_pos.dim() == 3:
        particle_pos = particle_pos[0]

    ee_pos = env._ee_link.get_pos()
    ee_quat = env._ee_link.get_quat()
    if ee_pos.dim() > 1:
        ee_pos = ee_pos.squeeze(0)
        ee_quat = ee_quat.squeeze(0)

    local_pos = _quat_rotate_inverse(ee_quat, particle_pos - ee_pos.unsqueeze(0))
    region_min = env._bowl_sticky_region_min
    region_max = env._bowl_sticky_region_max
    top_z = region_max[2] + env._bowl_sticky_top_slack

    x_in = (local_pos[:, 0] >= region_min[0]) & (local_pos[:, 0] <= region_max[0])
    y_in = (local_pos[:, 1] >= region_min[1]) & (local_pos[:, 1] <= region_max[1])
    z_in = (local_pos[:, 2] >= region_min[2]) & (local_pos[:, 2] <= top_z)
    inside_mask = x_in & y_in & z_in
    above_rim_mask = (
        x_in & y_in & (local_pos[:, 2] > top_z) & (local_pos[:, 2] <= top_z + 0.08)
    )
    front_escape_mask = (
        (local_pos[:, 0] >= region_min[0] - 0.01)
        & (local_pos[:, 0] <= region_max[0] + 0.01)
        & (local_pos[:, 1] > region_max[1])
        & (local_pos[:, 1] <= region_max[1] + 0.06)
        & (local_pos[:, 2] >= region_min[2] - 0.02)
        & (local_pos[:, 2] <= top_z + 0.08)
    )
    side_escape_mask = (
        (
            (
                (local_pos[:, 0] >= region_min[0] - 0.06)
                & (local_pos[:, 0] < region_min[0])
            )
            | (
                (local_pos[:, 0] > region_max[0])
                & (local_pos[:, 0] <= region_max[0] + 0.06)
            )
        )
        & (local_pos[:, 1] >= region_min[1] - 0.01)
        & (local_pos[:, 1] <= region_max[1] + 0.01)
        & (local_pos[:, 2] >= region_min[2] - 0.02)
        & (local_pos[:, 2] <= top_z + 0.08)
    )
    below_bowl_mask = (
        x_in
        & y_in
        & (local_pos[:, 2] < region_min[2])
        & (local_pos[:, 2] >= region_min[2] - 0.08)
    )

    return {
        "step_idx": step_idx,
        "phase": phase,
        "traverse_speed": traverse_speed,
        "n_on_tool": int(metrics["n_on_tool"]),
        "transfer_efficiency": float(metrics["transfer_efficiency"]),
        "spill_ratio": float(metrics["spill_ratio"]),
        "inside_bowl_local": int(inside_mask.sum().item()),
        "above_rim_local": int(above_rim_mask.sum().item()),
        "front_escape_band": int(front_escape_mask.sum().item()),
        "side_escape_band": int(side_escape_mask.sum().item()),
        "below_bowl_band": int(below_bowl_mask.sum().item()),
    }


def shifted_waypoint(qpos: List[float], joint7_delta: float) -> List[float]:
    out = list(qpos)
    out[6] += joint7_delta
    return out


def set_transport_phase(env: ScoopTransferTask, phase: str) -> None:
    if hasattr(env, "set_bowl_transport_phase"):
        env.set_bowl_transport_phase(phase)


def advance_scene(env: ScoopTransferTask, n_steps: int = 1) -> None:
    for _ in range(n_steps):
        env.scene.step()
        if hasattr(env, "post_physics_update"):
            env.post_physics_update()


def scene_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "bowl_contact_quality_enabled": args.bowl_contact_quality_enabled,
        "bowl_enable_cpic": args.bowl_enable_cpic,
    }
    if args.substeps is not None:
        cfg["bowl_substeps_override"] = args.substeps
    if args.robot_coup_friction is not None:
        cfg["bowl_robot_coup_friction"] = args.robot_coup_friction
    if args.robot_coup_softness is not None:
        cfg["bowl_robot_coup_softness"] = args.robot_coup_softness
    if args.robot_sdf_cell_size is not None:
        cfg["bowl_robot_sdf_cell_size"] = args.robot_sdf_cell_size
    if args.robot_sdf_min_res is not None:
        cfg["bowl_robot_sdf_min_res"] = args.robot_sdf_min_res
    if args.robot_sdf_max_res is not None:
        cfg["bowl_robot_sdf_max_res"] = args.robot_sdf_max_res
    return cfg


def task_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "bowl_sticky_fallback_enabled": args.sticky_fallback,
        "bowl_sticky_top_slack": args.sticky_top_slack,
        "bowl_sticky_detection_margin": args.sticky_detection_margin,
        "bowl_sticky_velocity_damping": args.sticky_velocity_damping,
        "bowl_sticky_zero_outward_velocity": args.sticky_zero_outward_velocity,
        "bowl_sticky_max_snap": args.sticky_max_snap,
        "bowl_constraint_fallback_enabled": args.constraint_fallback,
        "bowl_constraint_stiffness": args.constraint_stiffness,
    }


def move_to_stage(
    env: ScoopTransferTask,
    stage: str,
    traverse_speed: float,
    traverse_joint7_delta: float,
) -> Dict[str, Any]:
    traverse_steps = _bowl_traverse_steps(traverse_speed)
    lift_low = shifted_waypoint(BOWL_LIFT_LOW_S, traverse_joint7_delta)
    lift_full = shifted_waypoint(BOWL_LIFT_S, traverse_joint7_delta)
    traverse_mid = shifted_waypoint(BOWL_TRAVERSE_MID_S, traverse_joint7_delta)
    traverse_fast = shifted_waypoint(BOWL_TRAVERSE_FAST_S, traverse_joint7_delta)
    pour = shifted_waypoint(BOWL_POUR_S, traverse_joint7_delta)
    settle_pose = shifted_waypoint(BOWL_SETTLE_S, traverse_joint7_delta)

    env.reset()
    set_transport_phase(env, "off")
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 15, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, BOWL_APPROACH_S, 30, settle_per_step=2)
    interpolate_waypoints(env, BOWL_APPROACH_S, BOWL_INSERT_S, 45, settle_per_step=3)
    interpolate_waypoints(env, BOWL_INSERT_S, BOWL_CAPTURE_S, 35, settle_per_step=3)
    settle(env, 10)

    if stage == "lift_low":
        interpolate_waypoints(env, BOWL_CAPTURE_S, lift_low, 20, settle_per_step=3)
        settle(env, 10)
        return {"traverse_steps": traverse_steps, "stage_label": "lift_low"}

    interpolate_waypoints(env, BOWL_CAPTURE_S, lift_low, 20, settle_per_step=3)
    settle(env, 10)
    interpolate_waypoints(env, lift_low, lift_full, 20, settle_per_step=3)
    settle(env, 10)

    if stage == "lift_full":
        return {"traverse_steps": traverse_steps, "stage_label": "lift_full"}

    mid_steps = traverse_steps // 2
    set_transport_phase(env, "carry")
    interpolate_waypoints(env, lift_full, traverse_mid, mid_steps, settle_per_step=2)
    settle(env, 10)
    set_transport_phase(env, "off")

    if stage == "traverse_mid":
        return {"traverse_steps": traverse_steps, "stage_label": "traverse_mid"}

    set_transport_phase(env, "carry")
    interpolate_waypoints(
        env, traverse_mid, traverse_fast, traverse_steps - mid_steps, settle_per_step=2
    )
    settle(env, 10)
    set_transport_phase(env, "off")
    set_transport_phase(env, "pour")
    interpolate_waypoints(env, traverse_fast, pour, 30, settle_per_step=3)
    interpolate_waypoints(env, pour, settle_pose, 20, settle_per_step=3)
    set_transport_phase(env, "off")
    settle(env, 40)
    return {"traverse_steps": traverse_steps, "stage_label": "final"}


def run_hold_trial(
    env: ScoopTransferTask,
    label: str,
    stage: str,
    output_dir: Path,
    hold_steps: int,
    trial_idx: int,
    traverse_speed: float,
) -> Dict[str, Any]:
    trial_dir = output_dir / label / f"trial_{trial_idx:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stage_info = move_to_stage(
        env, stage, traverse_speed=traverse_speed, traverse_joint7_delta=0.0
    )
    rows = [metric_row(env, stage_info["stage_label"], 0)]
    save_frame(env, trial_dir / "lift_end.png")
    for hold_step in range(1, hold_steps + 1):
        advance_scene(env)
        rows.append(metric_row(env, "hold", hold_step))
        if hold_step == hold_steps // 2:
            save_frame(env, trial_dir / "hold_mid.png")
    save_frame(env, trial_dir / "hold_end.png")
    write_csv(trial_dir / "metrics.csv", rows)
    plot_curves(rows, trial_dir / "metrics.png")
    final = rows[-1]
    summary = {
        "diagnosis_group": label,
        "trial": trial_idx,
        "hold_steps": hold_steps,
        "traverse_speed": traverse_speed,
        "duration_sec": round(time.time() - t0, 2),
        "start_n_on_tool": rows[0]["n_on_tool"],
        "mid_n_on_tool": rows[len(rows) // 2]["n_on_tool"],
        "end_n_on_tool": final["n_on_tool"],
        "end_transfer_efficiency": final["transfer_efficiency"],
        "end_spill_ratio": final["spill_ratio"],
        "dropped_to_zero": int(final["n_on_tool"] == 0),
    }
    return summary


def run_retention_suite(
    output_dir: Path,
    scene_cfg: Dict[str, Any],
    task_cfg: Dict[str, Any],
    repeats: int,
    hold_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    retention_dir = output_dir / "retention"
    retention_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        ("lift_only_hold", "lift_low", 0.2),
        ("lift_full_hold", "lift_full", 0.2),
        ("traverse_slow", "traverse_mid", 0.2),
    ]
    summaries: List[Dict[str, Any]] = []
    for group_idx, (label, stage, speed) in enumerate(groups):
        env = build_env(
            scene_cfg, horizon=1200, seed=seed + group_idx, task_cfg=task_cfg
        )
        for trial in range(1, repeats + 1):
            summaries.append(
                run_hold_trial(
                    env=env,
                    label=label,
                    stage=stage,
                    output_dir=retention_dir,
                    hold_steps=hold_steps,
                    trial_idx=trial,
                    traverse_speed=speed,
                )
            )
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(retention_dir / "retention_summary.csv", summaries)
    return summaries


def run_transport_trial(
    env: ScoopTransferTask,
    traverse_speed: float,
    traverse_joint7_delta: float,
) -> Dict[str, Any]:
    t0 = time.time()
    traverse_steps = _bowl_traverse_steps(traverse_speed)
    lift_low = shifted_waypoint(BOWL_LIFT_LOW_S, traverse_joint7_delta)
    lift_full = shifted_waypoint(BOWL_LIFT_S, traverse_joint7_delta)
    traverse_mid = shifted_waypoint(BOWL_TRAVERSE_MID_S, traverse_joint7_delta)
    traverse_fast = shifted_waypoint(BOWL_TRAVERSE_FAST_S, traverse_joint7_delta)
    pour = shifted_waypoint(BOWL_POUR_S, traverse_joint7_delta)
    settle_pose = shifted_waypoint(BOWL_SETTLE_S, traverse_joint7_delta)

    env.reset()
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 15, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, BOWL_APPROACH_S, 30, settle_per_step=2)
    interpolate_waypoints(env, BOWL_APPROACH_S, BOWL_INSERT_S, 45, settle_per_step=3)
    interpolate_waypoints(env, BOWL_INSERT_S, BOWL_CAPTURE_S, 35, settle_per_step=3)
    settle(env, 10)

    interpolate_waypoints(env, BOWL_CAPTURE_S, lift_low, 20, settle_per_step=3)
    settle(env, 10)
    interpolate_waypoints(env, lift_low, lift_full, 20, settle_per_step=3)
    settle(env, 10)
    lift_full_metrics = env.compute_metrics()

    mid_steps = traverse_steps // 2
    set_transport_phase(env, "carry")
    interpolate_waypoints(env, lift_full, traverse_mid, mid_steps, settle_per_step=2)
    settle(env, 10)
    mid_metrics = env.compute_metrics()

    interpolate_waypoints(
        env, traverse_mid, traverse_fast, traverse_steps - mid_steps, settle_per_step=2
    )
    settle(env, 10)
    set_transport_phase(env, "off")
    set_transport_phase(env, "pour")
    interpolate_waypoints(env, traverse_fast, pour, 30, settle_per_step=3)
    interpolate_waypoints(env, pour, settle_pose, 20, settle_per_step=3)
    set_transport_phase(env, "off")
    settle(env, 40)
    final_metrics = env.compute_metrics()

    return {
        "traverse_speed": traverse_speed,
        "traverse_joint7_delta": traverse_joint7_delta,
        "duration_sec": round(time.time() - t0, 2),
        "lift_full_n_on_tool": int(lift_full_metrics["n_on_tool"]),
        "mid_traverse_n_on_tool": int(mid_metrics["n_on_tool"]),
        "final_n_on_tool": int(final_metrics["n_on_tool"]),
        "final_transfer_efficiency": float(final_metrics["transfer_efficiency"]),
        "final_spill_ratio": float(final_metrics["spill_ratio"]),
        "dropped_midway": int(
            lift_full_metrics["n_on_tool"] > 0 and mid_metrics["n_on_tool"] == 0
        ),
    }


def run_scan(
    output_dir: Path, scene_cfg: Dict[str, Any], task_cfg: Dict[str, Any], seed: int
) -> List[Dict[str, Any]]:
    scan_dir = output_dir / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    combos = [
        {"name": "slow_base", "speed": 0.2, "joint7_delta": 0.0},
        {"name": "mid_base", "speed": 0.5, "joint7_delta": 0.0},
        {"name": "slow_backtilt", "speed": 0.2, "joint7_delta": -0.3},
        {"name": "mid_backtilt", "speed": 0.5, "joint7_delta": -0.3},
    ]
    rows: List[Dict[str, Any]] = []
    env = build_env(scene_cfg, horizon=1400, seed=seed, task_cfg=task_cfg)
    for combo in combos:
        row = run_transport_trial(
            env=env,
            traverse_speed=combo["speed"],
            traverse_joint7_delta=combo["joint7_delta"],
        )
        rows.append({"name": combo["name"], **row})
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_csv(scan_dir / "scan_summary.csv", rows)
    return rows


def run_phase_diagnostic(
    output_dir: Path,
    scene_cfg: Dict[str, Any],
    task_cfg: Dict[str, Any],
    seed: int,
    traverse_speed: float,
) -> Dict[str, Any]:
    phase_dir = output_dir / "phase_diagnostic"
    phase_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(scene_cfg, horizon=1600, seed=seed, task_cfg=task_cfg)

    rows: List[Dict[str, Any]] = []

    def snap(label: str) -> None:
        rows.append(
            phase_snapshot_row(
                env,
                phase=label,
                traverse_speed=traverse_speed,
                step_idx=len(rows),
            )
        )

    traverse_steps = _bowl_traverse_steps(traverse_speed)
    mid_steps = traverse_steps // 2
    early_steps = max(1, mid_steps // 2)
    late_steps = max(1, (traverse_steps - mid_steps) // 2)

    env.reset()
    set_transport_phase(env, "off")
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 15, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, BOWL_APPROACH_S, 30, settle_per_step=2)
    interpolate_waypoints(env, BOWL_APPROACH_S, BOWL_INSERT_S, 45, settle_per_step=3)
    snap("capture_approach")
    interpolate_waypoints(env, BOWL_INSERT_S, BOWL_CAPTURE_S, 35, settle_per_step=3)
    settle(env, 10)
    snap("capture_end")

    interpolate_waypoints(env, BOWL_CAPTURE_S, BOWL_LIFT_LOW_S, 20, settle_per_step=3)
    settle(env, 10)
    snap("lift_low")
    interpolate_waypoints(env, BOWL_LIFT_LOW_S, BOWL_LIFT_S, 20, settle_per_step=3)
    settle(env, 10)
    snap("lift_full")

    set_transport_phase(env, "carry")
    early_waypoint = [
        x * 0.5 + y * 0.5 for x, y in zip(BOWL_LIFT_S, BOWL_TRAVERSE_MID_S)
    ]
    interpolate_waypoints(
        env, BOWL_LIFT_S, early_waypoint, early_steps, settle_per_step=2
    )
    settle(env, 10)
    snap("carry_early")

    interpolate_waypoints(
        env,
        early_waypoint,
        BOWL_TRAVERSE_MID_S,
        max(1, mid_steps - early_steps),
        settle_per_step=2,
    )
    settle(env, 10)
    snap("carry_mid")

    late_waypoint = [
        x * 0.5 + y * 0.5 for x, y in zip(BOWL_TRAVERSE_MID_S, BOWL_TRAVERSE_FAST_S)
    ]
    interpolate_waypoints(
        env, BOWL_TRAVERSE_MID_S, late_waypoint, late_steps, settle_per_step=2
    )
    settle(env, 10)
    snap("carry_late")

    interpolate_waypoints(
        env,
        late_waypoint,
        BOWL_TRAVERSE_FAST_S,
        max(1, (traverse_steps - mid_steps) - late_steps),
        settle_per_step=2,
    )
    settle(env, 10)
    snap("pre_pour")

    set_transport_phase(env, "off")
    set_transport_phase(env, "pour")
    interpolate_waypoints(env, BOWL_TRAVERSE_FAST_S, BOWL_POUR_S, 30, settle_per_step=3)
    settle(env, 10)
    snap("pour_tilt")
    interpolate_waypoints(env, BOWL_POUR_S, BOWL_SETTLE_S, 20, settle_per_step=3)
    set_transport_phase(env, "off")
    settle(env, 40)
    snap("final")

    write_csv(phase_dir / "phase_metrics.csv", rows)
    summary = {
        "traverse_speed": traverse_speed,
        "phases": rows,
    }
    write_json(phase_dir / "phase_summary.json", summary)
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def config_report(
    output_dir: Path,
    material: str,
    seed: int,
    scene_overrides: Dict[str, Any],
    task_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_scene = dict(BASELINE_BOWL_SCENE)
    baseline_scene["particle_material"] = material
    baseline_scene.update(scene_overrides)
    flat_scene = dict(FLAT_BOWL_SCENE)
    flat_scene["particle_material"] = material
    flat_scene.update(scene_overrides)

    baseline_env = build_env(baseline_scene, horizon=500, seed=seed, task_cfg=task_cfg)
    flat_env = build_env(flat_scene, horizon=500, seed=seed + 1, task_cfg=task_cfg)
    report = {
        "baseline_bowl": {
            "requested_scene_config": merged_scene_config(baseline_scene),
            "effective_scene_runtime": effective_scene_runtime_config(baseline_scene),
            "requested_task_config": task_cfg,
            "effective_task_runtime": effective_task_runtime_config(
                baseline_scene, task_cfg
            ),
            "env": env_snapshot(baseline_env),
        },
        "flat_bowl": {
            "requested_scene_config": merged_scene_config(flat_scene),
            "effective_scene_runtime": effective_scene_runtime_config(flat_scene),
            "requested_task_config": task_cfg,
            "effective_task_runtime": effective_task_runtime_config(
                flat_scene, task_cfg
            ),
            "env": env_snapshot(flat_env),
        },
    }
    write_json(output_dir / "merged_scene_configs.json", report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    del baseline_env
    del flat_env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", type=str, default="sand")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode", choices=["all", "config", "retention", "scan", "phase"], default="all"
    )
    parser.add_argument("--output-root", type=str, default="results/scoop_debug")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--regime-label", type=str, default="native")
    parser.add_argument("--phase-traverse-speed", type=float, default=0.2)
    parser.add_argument("--bowl-contact-quality-enabled", action="store_true")
    parser.add_argument("--bowl-enable-cpic", action="store_true")
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--robot-coup-friction", type=float, default=None)
    parser.add_argument("--robot-coup-softness", type=float, default=None)
    parser.add_argument("--robot-sdf-cell-size", type=float, default=None)
    parser.add_argument("--robot-sdf-min-res", type=int, default=None)
    parser.add_argument("--robot-sdf-max-res", type=int, default=None)
    parser.add_argument("--sticky-fallback", action="store_true")
    parser.add_argument("--constraint-fallback", action="store_true")
    parser.add_argument("--constraint-stiffness", type=float, default=1e6)
    parser.add_argument("--sticky-top-slack", type=float, default=0.02)
    parser.add_argument("--sticky-detection-margin", type=float, default=0.012)
    parser.add_argument("--sticky-velocity-damping", type=float, default=0.5)
    parser.add_argument("--sticky-max-snap", type=float, default=0.01)
    parser.add_argument(
        "--sticky-zero-outward-velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.output_dir or str(Path(args.output_root) / utc_now())
    output_dir = Path(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_overrides = scene_cfg_from_args(args)
    task_cfg = task_cfg_from_args(args)
    flat_scene = dict(FLAT_BOWL_SCENE)
    flat_scene["particle_material"] = args.material
    flat_scene.update(scene_overrides)

    metadata = {
        "command": " ".join(sys.argv),
        "start_time_utc": utc_now(),
        "mode": args.mode,
        "material": args.material,
        "seed": args.seed,
        "hold_steps": args.hold_steps,
        "repeats": args.repeats,
        "regime_label": args.regime_label,
        "scene_overrides": scene_overrides,
        "task_config": task_cfg,
        "effective_scene_runtime": effective_scene_runtime_config(flat_scene),
        "effective_task_runtime": effective_task_runtime_config(flat_scene, task_cfg),
        "gpu_before": gpu_snapshot(),
        "impact_main_training": "unknown",
    }
    write_json(output_dir / "run_metadata_start.json", metadata)

    outputs: Dict[str, Any] = {}
    if args.mode in {"all", "config"}:
        outputs["config"] = config_report(
            output_dir,
            args.material,
            args.seed,
            scene_overrides,
            task_cfg,
        )
    if args.mode in {"all", "retention"}:
        outputs["retention"] = run_retention_suite(
            output_dir=output_dir,
            scene_cfg=flat_scene,
            task_cfg=task_cfg,
            repeats=args.repeats,
            hold_steps=args.hold_steps,
            seed=args.seed,
        )
    if args.mode in {"all", "scan"}:
        outputs["scan"] = run_scan(
            output_dir=output_dir,
            scene_cfg=flat_scene,
            task_cfg=task_cfg,
            seed=args.seed + 100,
        )
    if args.mode in {"all", "phase"}:
        outputs["phase"] = run_phase_diagnostic(
            output_dir=output_dir,
            scene_cfg=flat_scene,
            task_cfg=task_cfg,
            seed=args.seed + 200,
            traverse_speed=args.phase_traverse_speed,
        )

    metadata["end_time_utc"] = utc_now()
    metadata["gpu_after"] = gpu_snapshot()
    metadata["output_dir"] = str(output_dir)
    write_json(output_dir / "run_metadata_end.json", metadata)
    write_json(output_dir / "outputs_index.json", outputs)
    print(f"Saved diagnosis outputs to {output_dir}")


if __name__ == "__main__":
    main()
