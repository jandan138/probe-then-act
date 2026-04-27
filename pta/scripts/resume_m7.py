"""Resume M7 PPO training from an SB3 step checkpoint.

This is intentionally separate from ``train_m7.py`` because the original
script always creates a fresh PPO model. Failed DLC jobs that already wrote
durable step checkpoints must load the checkpoint and continue with
``reset_num_timesteps=False``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

_STEP_CHECKPOINT_RE = re.compile(r"^m7_pta_(\d+)_steps\.zip$")


def default_run_name(ablation: str, seed: int) -> str:
    """Return the run name used by ``train_m7.py`` for an M7 ablation."""
    if ablation == "none":
        variant = "pta"
    elif ablation == "no_probe":
        variant = "pta_noprobe"
    elif ablation == "no_belief":
        variant = "pta_nobelief"
    else:
        raise ValueError(f"unsupported ablation: {ablation}")
    return f"m7_{variant}_seed{seed}"


def latest_step_checkpoint(checkpoint_dir: Path) -> Path:
    """Select the highest numeric ``m7_pta_<step>_steps.zip`` checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("m7_pta_*_steps.zip"):
        match = _STEP_CHECKPOINT_RE.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(f"No step checkpoints found in {checkpoint_dir}")

    return max(candidates, key=lambda item: item[0])[1]


def remaining_timesteps(current_timesteps: int, target_timesteps: int) -> int:
    """Return how many timesteps SB3 should learn after loading a checkpoint."""
    if current_timesteps > target_timesteps:
        raise ValueError(
            f"checkpoint is already past target: "
            f"{current_timesteps} > {target_timesteps}"
        )
    return target_timesteps - current_timesteps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume M7 PPO training from a step checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_probe", "no_belief"],
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint .zip path. Relative paths are resolved from repo root.",
    )
    parser.add_argument("--target-timesteps", type=int, default=500_000)
    parser.add_argument("--material", type=str, default="sand")
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--n-probes", type=int, default=3)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Set >0 to enable EvalCallback. Default off to reduce DLC writes.",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def _resolve_resume_path(args: argparse.Namespace, checkpoint_dir: Path) -> Path:
    if args.resume_from is None:
        path = latest_step_checkpoint(checkpoint_dir)
    else:
        path = args.resume_from
        if not path.is_absolute():
            path = _PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def _load_checkpoint_timesteps(path: Path) -> int:
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    return int(model.num_timesteps)


def main() -> None:
    args = parse_args()
    run_name = default_run_name(args.ablation, args.seed)
    log_dir = _PROJECT_ROOT / "logs" / run_name
    checkpoint_dir = _PROJECT_ROOT / "checkpoints" / run_name
    resume_path = _resolve_resume_path(args, checkpoint_dir)

    print("=" * 60)
    print("Probe-Then-Act: M7 Resume Training")
    print("=" * 60)
    print(f"  Ablation:       {args.ablation}")
    print(f"  Seed:           {args.seed}")
    print(f"  Resume from:    {resume_path}")
    print(f"  Target steps:   {args.target_timesteps:,}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Eval freq:      {args.eval_freq}")
    print()

    if args.dry_run:
        current_timesteps = _load_checkpoint_timesteps(resume_path)
        remaining = remaining_timesteps(current_timesteps, args.target_timesteps)
        print(f"Loaded checkpoint steps: {current_timesteps:,}")
        print(f"Remaining train steps:   {remaining:,}")
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pta.scripts.train_m7 import make_m7_env
    from pta.training.utils.checkpoint_io import save_sb3_checkpoint
    from pta.training.utils.seed import set_seed

    set_seed(args.seed)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    task_config = {
        "horizon": args.horizon,
        "success_threshold": 0.3,
    }
    scene_config = {
        "tool_type": "scoop",
        "n_envs": 0,
        "particle_material": args.material,
    }

    def _make_env():
        return make_m7_env(
            task_config=task_config,
            scene_config=scene_config,
            seed=args.seed,
            joint_residual_scale=args.residual_scale,
            latent_dim=args.latent_dim,
            n_probes=args.n_probes,
            ablation=args.ablation,
        )

    def _make_eval_env():
        return make_m7_env(
            task_config=task_config,
            scene_config=scene_config,
            seed=args.seed + 1000,
            joint_residual_scale=args.residual_scale,
            latent_dim=args.latent_dim,
            n_probes=args.n_probes,
            ablation=args.ablation,
        )

    vec_env = None
    eval_env = None
    try:
        vec_env = DummyVecEnv([_make_env])
        model = PPO.load(
            str(resume_path),
            env=vec_env,
            tensorboard_log=None,
            device="auto",
        )
        model.tensorboard_log = None

        current_timesteps = int(model.num_timesteps)
        remaining = remaining_timesteps(current_timesteps, args.target_timesteps)
        print(f"Loaded checkpoint steps: {current_timesteps:,}")
        print(f"Remaining train steps:   {remaining:,}")

        callbacks = []
        if args.save_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=args.save_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix="m7_pta",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )
        if args.eval_freq > 0:
            eval_env = DummyVecEnv([_make_eval_env])
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(checkpoint_dir / "best"),
                    log_path=str(log_dir / "resume_eval"),
                    eval_freq=args.eval_freq,
                    n_eval_episodes=args.n_eval_episodes,
                    deterministic=True,
                    render=False,
                )
            )

        callback = CallbackList(callbacks) if callbacks else None
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=args.progress_bar,
            )
        else:
            print("Checkpoint already matches target; saving final copy.")

        final_path = checkpoint_dir / "m7_pta_final"
        save_sb3_checkpoint(
            model,
            final_path,
            metadata={
                "method": "m7",
                "ablation": args.ablation,
                "seed": args.seed,
                "material": args.material,
                "residual_scale": args.residual_scale,
                "latent_dim": args.latent_dim,
                "n_probes": args.n_probes,
                "horizon": args.horizon,
                "resume_from": str(resume_path),
                "start_timesteps": current_timesteps,
                "target_timesteps": args.target_timesteps,
                "final_num_timesteps": int(model.num_timesteps),
                "eval_freq": args.eval_freq,
                "stage": "m7_rl_resume",
            },
        )
        print(f"Final model saved to {final_path}")
    finally:
        if eval_env is not None:
            eval_env.close()
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    main()
