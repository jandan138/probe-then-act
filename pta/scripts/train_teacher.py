"""CLI entry point for teacher training.

Usage::

    # Quick test (500k steps, ~1-2h with Genesis MPM)
    python pta/scripts/train_teacher.py --total-timesteps 500000 --seed 42

    # Full training (10M steps)
    python pta/scripts/train_teacher.py --total-timesteps 10000000

    # Custom material
    python pta/scripts/train_teacher.py --material snow --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for teacher training."""
    parser = argparse.ArgumentParser(
        description="Train a privileged teacher policy with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training
    parser.add_argument(
        "--total-timesteps", type=int, default=500_000,
        help="Total environment timesteps for training",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # PPO hyper-parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=128, help="Steps per env before update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")

    # Environment
    parser.add_argument(
        "--material", type=str, default="sand",
        choices=["sand", "snow", "elastoplastic", "liquid"],
        help="MPM material family for training",
    )
    parser.add_argument(
        "--horizon", type=int, default=200,
        help="Episode length (steps per episode)",
    )

    # Eval / checkpoint
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Eval every N steps")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Checkpoint every N steps")

    # Paths
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Log directory (default: logs/teacher)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Checkpoint directory (default: checkpoints/teacher)",
    )

    # Misc
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0/1/2)")

    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments and launch privileged teacher training."""
    args = parse_args()

    # Set WSL2 rendering env vars if not already set
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = (
            "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
        )

    from pta.training.rl.train_teacher import train_teacher
    from pta.utils.paths import CHECKPOINT_DIR, LOG_DIR

    # Build config dict from args
    config = {
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "learning_rate": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "entropy_coef": args.entropy_coef,
        "eval_freq": args.eval_freq,
        "save_freq": args.save_freq,
        "verbose": args.verbose,
        "run_name": f"teacher_ppo_seed{args.seed}",
        # Scene / task config
        "scene_config": {
            "particle_material": args.material,
            "n_envs": 0,
        },
        "task_config": {
            "horizon": args.horizon,
        },
    }

    # Override paths if provided
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
        config["tensorboard_log"] = str(Path(args.log_dir) / "tb")
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir

    print("=" * 60)
    print("Probe-Then-Act: Teacher PPO Training")
    print("=" * 60)
    print(f"  Timesteps:  {args.total_timesteps:,}")
    print(f"  Seed:       {args.seed}")
    print(f"  Material:   {args.material}")
    print(f"  Horizon:    {args.horizon}")
    print(f"  LR:         {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print()

    model = train_teacher(config)
    print("Training complete.")


if __name__ == "__main__":
    main()
