from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

from pta.scripts.bowl_transport_diagnosis import render_rgb
from pta.envs.tasks.scoop_transfer import ScoopTransferTask
from pta.scripts.run_scripted_baseline import (
    BOWL_APPROACH_S,
    BOWL_CAPTURE_S,
    BOWL_INSERT_S,
    BOWL_LIFT_S,
    BOWL_TRAVERSE_FAST_S,
    BOWL_TRAVERSE_MID_S,
    EXTEND_FWD_S,
    HOME_S,
)


VIEWER_INITIAL_SETTLE_STEPS = 40


def set_transport_phase(env: ScoopTransferTask, phase: str) -> None:
    if hasattr(env, "set_bowl_transport_phase"):
        env.set_bowl_transport_phase(phase)


def save_current_frame(env: ScoopTransferTask, frames_dir: Path, frame_idx: int) -> int:
    img = render_rgb(env)
    if img is None:
        return frame_idx
    Image.fromarray(img).save(frames_dir / f"frame_{frame_idx:04d}.png")
    return frame_idx + 1


def interpolate_and_capture(
    env: ScoopTransferTask,
    start_qpos: List[float],
    end_qpos: List[float],
    n_steps: int,
    settle_per_step: int,
    frames_dir: Path,
    frame_idx: int,
    capture_every: int = 2,
) -> int:
    start = torch.tensor(start_qpos, dtype=torch.float32, device="cuda")
    end = torch.tensor(end_qpos, dtype=torch.float32, device="cuda")
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        qpos = start * (1 - alpha) + end * alpha
        env.robot.set_qpos(qpos)
        for _ in range(settle_per_step):
            env.scene.step()
            if hasattr(env, "post_physics_update"):
                env.post_physics_update()
        if i % capture_every == 0 or i == n_steps - 1:
            frame_idx = save_current_frame(env, frames_dir, frame_idx)
    return frame_idx


def execute_segment_and_capture(
    env: ScoopTransferTask,
    start_qpos: List[float],
    end_qpos: List[float],
    n_steps: int,
    settle_per_step: int,
    frames_dir: Path,
    frame_idx: int,
    mode: str,
    capture_every: int = 2,
) -> int:
    start = torch.tensor(start_qpos, dtype=torch.float32, device="cuda")
    end = torch.tensor(end_qpos, dtype=torch.float32, device="cuda")
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        qpos = start * (1 - alpha) + end * alpha
        if mode == "pd":
            env.robot.control_dofs_position(qpos)
        else:
            env.robot.set_qpos(qpos)
        for _ in range(settle_per_step):
            env.scene.step()
            if hasattr(env, "post_physics_update"):
                env.post_physics_update()
        if i % capture_every == 0 or i == n_steps - 1:
            frame_idx = save_current_frame(env, frames_dir, frame_idx)
    return frame_idx


def settle_and_capture(
    env: ScoopTransferTask,
    n_steps: int,
    frames_dir: Path,
    frame_idx: int,
    capture_every: int = 4,
) -> int:
    for i in range(n_steps):
        env.scene.step()
        if hasattr(env, "post_physics_update"):
            env.post_physics_update()
        if i % capture_every == 0 or i == n_steps - 1:
            frame_idx = save_current_frame(env, frames_dir, frame_idx)
    return frame_idx


def build_viewer_scene_config(tool_type: str) -> Dict[str, object]:
    return {
        "particle_material": "sand",
        "n_envs": 0,
        "tool_type": tool_type,
        "task_layout": "flat",
        "source_pos": (0.5, 0.0, 0.05),
        "source_size": (0.24, 0.20, 0.005),
        "source_wall_height": 0.13,
        "particle_pos": (0.5, 0.0, 0.12),
        "particle_size": (0.15, 0.13, 0.05),
        "bowl_contact_quality_enabled": True,
        "bowl_enable_cpic": True,
        "bowl_substeps_override": 40,
        "bowl_robot_coup_friction": 6.0,
        "bowl_robot_coup_softness": 0.0005,
        "bowl_robot_sdf_cell_size": 0.002,
    }


def build_bowl_viewer_waypoints() -> Dict[str, List[float]]:
    return {
        "approach": [-0.14, 0.98, 0.0, -1.48, 0.0, 1.48, -0.55],
        "insert": [-0.02, 1.26, 0.0, -1.06, 0.0, 1.06, -1.08],
        "capture": [0.04, 1.08, 0.0, -1.24, 0.0, 1.22, -0.72],
        "lift": [0.04, 0.80, 0.0, -1.52, 0.0, 1.48, -0.70],
    }


def build_bowl_carry_plan(
    *, first_half_steps: int, second_half_steps: int
) -> List[Dict[str, object]]:
    viewer_waypoints = build_bowl_viewer_waypoints()
    early_waypoint = [0.16, 0.84, 0.0, -1.52, 0.0, 1.48, -0.70]
    late_waypoint = [0.54, 0.86, 0.0, -1.48, 0.0, 1.44, -0.70]
    return [
        {
            "start": viewer_waypoints["lift"],
            "end": early_waypoint,
            "steps": first_half_steps,
            "settle_after": 12,
            "mode": "qpos",
        },
        {
            "start": early_waypoint,
            "end": BOWL_TRAVERSE_MID_S,
            "steps": first_half_steps,
            "settle_after": 12,
            "mode": "qpos",
        },
        {
            "start": BOWL_TRAVERSE_MID_S,
            "end": late_waypoint,
            "steps": second_half_steps,
            "settle_after": 12,
            "mode": "qpos",
        },
        {
            "start": late_waypoint,
            "end": BOWL_TRAVERSE_FAST_S,
            "steps": second_half_steps,
            "settle_after": 12,
            "mode": "qpos",
        },
    ]


def build_bowl_viewer_motion_plan(
    *, first_half_steps: int, second_half_steps: int
) -> List[Dict[str, object]]:
    viewer_waypoints = build_bowl_viewer_waypoints()
    return [
        {
            "start": HOME_S,
            "end": EXTEND_FWD_S,
            "steps": 15,
            "settle_per_step": 1,
            "mode": "qpos",
        },
        {
            "start": EXTEND_FWD_S,
            "end": viewer_waypoints["approach"],
            "steps": 30,
            "settle_per_step": 2,
            "mode": "pd",
        },
        {
            "start": viewer_waypoints["approach"],
            "end": viewer_waypoints["insert"],
            "steps": 45,
            "settle_per_step": 3,
            "mode": "pd",
        },
        {
            "start": viewer_waypoints["insert"],
            "end": viewer_waypoints["capture"],
            "steps": 35,
            "settle_per_step": 3,
            "mode": "pd",
        },
        {
            "start": viewer_waypoints["capture"],
            "end": viewer_waypoints["lift"],
            "steps": 40,
            "settle_per_step": 3,
            "mode": "pd",
        },
        {
            "start": viewer_waypoints["lift"],
            "end": viewer_waypoints["lift"],
            "steps": 0,
            "settle_per_step": 0,
            "mode": "qpos",
        },
    ]


def build_env(tool_type: str, seed: int) -> ScoopTransferTask:
    scene_cfg = build_viewer_scene_config(tool_type)
    torch.manual_seed(seed)
    return ScoopTransferTask(config={"horizon": 1600}, scene_config=scene_cfg)


CAMERA_VIEWS: Dict[str, Dict[str, Tuple[float, float, float] | float]] = {
    "wide": {
        "pos": (1.0, -2.5, 2.0),
        "lookat": (1.0, 0.0, 0.0),
        "fov": 35,
    },
    "side_close": {
        "pos": (0.80, -0.32, 0.20),
        "lookat": (0.57, 0.07, 0.08),
        "fov": 28,
    },
    "front_close": {
        "pos": (0.56, -0.06, 0.16),
        "lookat": (0.56, 0.10, 0.08),
        "fov": 22,
    },
    "top_close": {
        "pos": (0.58, 0.00, 0.42),
        "lookat": (0.58, 0.08, 0.05),
        "fov": 26,
    },
    "mouth_ultraclose": {
        "pos": (0.61, -0.015, 0.125),
        "lookat": (0.575, 0.085, 0.085),
        "fov": 14,
    },
    "inside_ultraclose": {
        "pos": (0.575, -0.005, 0.205),
        "lookat": (0.575, 0.060, 0.060),
        "fov": 16,
    },
}


def set_camera_view(env: ScoopTransferTask, view_name: str) -> None:
    view = CAMERA_VIEWS[view_name]
    env.camera.set_pose(
        pos=view["pos"],
        lookat=view["lookat"],
    )
    env.camera._fov = view["fov"]


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_branch(
    *,
    branch_name: str,
    view_name: str,
    tool_type: str,
    first_half_steps: int,
    second_half_steps: int,
    seed: int,
    output_dir: Path,
) -> None:
    env = build_env(tool_type=tool_type, seed=seed)
    env.reset()
    set_camera_view(env, view_name)
    frames_dir = output_dir / branch_name / view_name / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    viewer_waypoints = build_bowl_viewer_waypoints()

    set_transport_phase(env, "off")
    frame_idx = save_current_frame(env, frames_dir, frame_idx)
    frame_idx = settle_and_capture(
        env, VIEWER_INITIAL_SETTLE_STEPS, frames_dir, frame_idx
    )
    for segment in build_bowl_viewer_motion_plan(
        first_half_steps=first_half_steps, second_half_steps=second_half_steps
    )[:4]:
        frame_idx = execute_segment_and_capture(
            env,
            segment["start"],
            segment["end"],
            segment["steps"],
            segment["settle_per_step"],
            frames_dir,
            frame_idx,
            segment["mode"],
        )
    frame_idx = settle_and_capture(env, 10, frames_dir, frame_idx)
    frame_idx = execute_segment_and_capture(
        env,
        viewer_waypoints["capture"],
        viewer_waypoints["lift"],
        40,
        3,
        frames_dir,
        frame_idx,
        "pd",
    )
    frame_idx = settle_and_capture(env, 10, frames_dir, frame_idx)

    set_transport_phase(env, "carry")
    for segment in build_bowl_carry_plan(
        first_half_steps=first_half_steps, second_half_steps=second_half_steps
    ):
        frame_idx = execute_segment_and_capture(
            env,
            segment["start"],
            segment["end"],
            segment["steps"],
            2,
            frames_dir,
            frame_idx,
            segment["mode"],
        )
        frame_idx = settle_and_capture(
            env, segment["settle_after"], frames_dir, frame_idx
        )
    set_transport_phase(env, "off")

    encode_video(
        frames_dir,
        output_dir / branch_name / view_name / f"{branch_name}_{view_name}.mp4",
        fps=20,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scoop_debug/carry_onset_videos",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Subset of camera views to render. Defaults to all built-in views.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    branches: List[Tuple[str, str, int, int]] = [
        ("baseline_nominal", "bowl", 15, 15),
        ("highwall_nominal", "bowl_highwall", 15, 15),
        ("baseline_slow", "bowl", 60, 60),
        ("highwall_slow", "bowl_highwall", 60, 60),
    ]
    view_names = args.views or [
        "wide",
        "side_close",
        "front_close",
        "top_close",
        "mouth_ultraclose",
        "inside_ultraclose",
    ]
    for idx, (name, tool_type, first_half, second_half) in enumerate(branches):
        for view_name in view_names:
            run_branch(
                branch_name=name,
                view_name=view_name,
                tool_type=tool_type,
                first_half_steps=first_half,
                second_half_steps=second_half,
                seed=args.seed + idx,
                output_dir=output_dir,
            )
    print(output_dir)


if __name__ == "__main__":
    main()
