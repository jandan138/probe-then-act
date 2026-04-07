"""Gate 4 launcher: Joint-space residual PPO on edge-push tiny task.

Uses JointResidualWrapper to bypass Cartesian IK entirely.
The policy learns small corrections on top of the scripted edge-push
joint trajectory.

Usage::

    source /home/zhuzihou/dev/Genesis/.venv/bin/activate
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
    python pta/scripts/launch_gate4.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    )


def main():
    from pta.training.rl.train_teacher import train_teacher

    config = {
        # PPO hyperparameters (from sand_tiny_task.yaml)
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
        # Training
        "total_timesteps": 200_000,
        "eval_freq": 5_000,
        "save_freq": 50_000,
        "seed": 42,
        "verbose": 1,
        # Joint residual wrapper
        "use_joint_residual": True,
        "joint_residual_scale": 0.1,
        "joint_residual_trajectory": "edge_push",
        # Scene config
        "scene_config": {
            "tool_type": "scoop",
            "n_envs": 0,
            "particle_material": "sand",
        },
        # Task config
        "task_config": {
            "horizon": 500,
            "success_threshold": 0.3,
        },
        # Logging / checkpointing
        "run_name": "gate4_joint_residual_v1",
        "log_dir": str(_PROJECT_ROOT / "logs" / "gate4_joint_residual_v1"),
        "checkpoint_dir": str(_PROJECT_ROOT / "checkpoints" / "gate4_joint_residual_v1"),
        "tensorboard_log": str(_PROJECT_ROOT / "logs" / "gate4_joint_residual_v1" / "tb"),
    }

    print("=" * 60)
    print("Gate 4: Joint-Space Residual PPO")
    print("=" * 60)
    print(f"  Total timesteps:   {config['total_timesteps']:,}")
    print(f"  Residual scale:    {config['joint_residual_scale']}")
    print(f"  Trajectory:        {config['joint_residual_trajectory']}")
    print(f"  Eval freq:         {config['eval_freq']}")
    print(f"  Seed:              {config['seed']}")
    print()

    model = train_teacher(config)
    print("Gate 4 training complete.")
    return model


if __name__ == "__main__":
    main()
