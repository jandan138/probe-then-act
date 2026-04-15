"""Train M1 (Reactive PPO) and M8 (Teacher PPO) baselines.

Usage::

    # Gate 4 retest (Config D, 200K steps)
    python pta/scripts/train_baselines.py --method gate4 --total-timesteps 200000

    # M1 Reactive PPO (no privileged obs)
    python pta/scripts/train_baselines.py --method m1 --seed 42 --total-timesteps 500000

    # M8 Teacher PPO (with privileged obs)
    python pta/scripts/train_baselines.py --method m8 --seed 42 --total-timesteps 500000

    # Run all M1 seeds
    for seed in 42 0 1; do
        python pta/scripts/train_baselines.py --method m1 --seed $seed --total-timesteps 500000
    done
"""

from __future__ import annotations

import argparse
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train M1/M8/Gate4 baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["gate4", "m1", "m8"],
        help="Which method to train",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--material", type=str, default="sand")
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to SB3 checkpoint zip to resume training from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pta.training.rl.train_teacher import train_teacher

    method = args.method
    seed = args.seed

    # Common config for all methods
    base_config = {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "entropy_coef": 0.0,
        "use_sde": True,
        "sde_sample_freq": 4,
        "policy_kwargs": {
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            "log_std_init": -1.0,
        },
        "total_timesteps": args.total_timesteps,
        "eval_freq": args.eval_freq,
        "save_freq": 50_000,
        "seed": seed,
        "verbose": 1,
        # Joint residual wrapper (validated control stack)
        "use_joint_residual": True,
        "joint_residual_scale": args.residual_scale,
        "joint_residual_trajectory": "edge_push",
        # Scene config (Config D)
        "scene_config": {
            "tool_type": "scoop",
            "n_envs": 0,
            "particle_material": args.material,
        },
        "task_config": {
            "horizon": args.horizon,
            "success_threshold": 0.3,
        },
    }

    if method == "gate4":
        # Gate 4 retest: with privileged obs, shorter training
        config = {
            **base_config,
            "use_privileged": True,
            "total_timesteps": args.total_timesteps,
            "policy_kwargs": {
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
                "log_std_init": -0.5,  # more exploration for Gate 4
            },
            "run_name": f"gate4_configD_seed{seed}",
            "log_dir": str(_PROJECT_ROOT / "logs" / f"gate4_configD_seed{seed}"),
            "checkpoint_dir": str(_PROJECT_ROOT / "checkpoints" / f"gate4_configD_seed{seed}"),
            "tensorboard_log": str(_PROJECT_ROOT / "logs" / f"gate4_configD_seed{seed}" / "tb"),
        }
    elif method == "m1":
        # M1 Reactive PPO: NO privileged obs
        config = {
            **base_config,
            "use_privileged": False,
            "run_name": f"m1_reactive_seed{seed}",
            "log_dir": str(_PROJECT_ROOT / "logs" / f"m1_reactive_seed{seed}"),
            "checkpoint_dir": str(_PROJECT_ROOT / "checkpoints" / f"m1_reactive_seed{seed}"),
            "tensorboard_log": str(_PROJECT_ROOT / "logs" / f"m1_reactive_seed{seed}" / "tb"),
        }
    elif method == "m8":
        # M8 Teacher PPO: WITH privileged obs
        config = {
            **base_config,
            "use_privileged": True,
            "run_name": f"m8_teacher_seed{seed}",
            "log_dir": str(_PROJECT_ROOT / "logs" / f"m8_teacher_seed{seed}"),
            "checkpoint_dir": str(_PROJECT_ROOT / "checkpoints" / f"m8_teacher_seed{seed}"),
            "tensorboard_log": str(_PROJECT_ROOT / "logs" / f"m8_teacher_seed{seed}" / "tb"),
        }

    if args.resume_from:
        config["resume_from"] = args.resume_from

    print("=" * 60)
    print(f"Probe-Then-Act: {method.upper()} Training")
    print("=" * 60)
    print(f"  Method:     {method}")
    print(f"  Timesteps:  {config['total_timesteps']:,}")
    print(f"  Seed:       {seed}")
    print(f"  Material:   {args.material}")
    print(f"  Privileged: {config.get('use_privileged', True)}")
    print(f"  Residual:   scale={args.residual_scale}")
    if args.resume_from:
        print(f"  Resume:     {args.resume_from}")
    print()

    model = train_teacher(config)
    print(f"{method.upper()} training complete (seed={seed}).")
    return model


if __name__ == "__main__":
    main()
