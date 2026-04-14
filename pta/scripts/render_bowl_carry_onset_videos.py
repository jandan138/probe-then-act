from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

from bowl_transport_diagnosis import render_rgb
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
)


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

    set_transport_phase(env, "off")
    frame_idx = save_current_frame(env, frames_dir, frame_idx)
    frame_idx = interpolate_and_capture(
        env, HOME_S, EXTEND_FWD_S, 15, 1, frames_dir, frame_idx
    )
    frame_idx = interpolate_and_capture(
        env, EXTEND_FWD_S, BOWL_APPROACH_S, 30, 2, frames_dir, frame_idx
    )
    frame_idx = interpolate_and_capture(
        env, BOWL_APPROACH_S, BOWL_INSERT_S, 45, 3, frames_dir, frame_idx
    )
    frame_idx = interpolate_and_capture(
        env, BOWL_INSERT_S, BOWL_CAPTURE_S, 35, 3, frames_dir, frame_idx
    )
    frame_idx = settle_and_capture(env, 10, frames_dir, frame_idx)
    frame_idx = interpolate_and_capture(
        env, BOWL_CAPTURE_S, BOWL_LIFT_S, 40, 3, frames_dir, frame_idx
    )
    frame_idx = settle_and_capture(env, 10, frames_dir, frame_idx)

    set_transport_phase(env, "carry")
    early_waypoint = [
        x * 0.5 + y * 0.5 for x, y in zip(BOWL_LIFT_S, BOWL_TRAVERSE_MID_S)
    ]
    frame_idx = interpolate_and_capture(
        env, BOWL_LIFT_S, early_waypoint, first_half_steps, 2, frames_dir, frame_idx
    )
    frame_idx = settle_and_capture(env, 10, frames_dir, frame_idx)
    frame_idx = interpolate_and_capture(
        env,
        early_waypoint,
        BOWL_TRAVERSE_MID_S,
        first_half_steps,
        2,
        frames_dir,
        frame_idx,
    )
    frame_idx = settle_and_capture(env, 10, frames_dir, frame_idx)

    late_waypoint = [
        x * 0.5 + y * 0.5 for x, y in zip(BOWL_TRAVERSE_MID_S, BOWL_TRAVERSE_FAST_S)
    ]
    frame_idx = interpolate_and_capture(
        env,
        BOWL_TRAVERSE_MID_S,
        late_waypoint,
        second_half_steps,
        2,
        frames_dir,
        frame_idx,
    )
    frame_idx = settle_and_capture(env, 10, frames_dir, frame_idx)
    frame_idx = interpolate_and_capture(
        env,
        late_waypoint,
        BOWL_TRAVERSE_FAST_S,
        second_half_steps,
        2,
        frames_dir,
        frame_idx,
    )
    frame_idx = settle_and_capture(env, 10, frames_dir, frame_idx)
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
