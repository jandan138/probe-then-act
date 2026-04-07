"""Gate 4 v2: Joint-space residual PPO with larger residual scale.

v1 (scale=0.1) converged to base trajectory at reward -2.0 but didn't
improve beyond it. v2 uses scale=0.2 for wider exploration.

Usage::

    source /home/zhuzihou/dev/Genesis/.venv/bin/activate
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
    python pta/scripts/launch_gate4_v2.py
"""

from __future__ import annotations

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


def main():
    from pta.training.rl.train_teacher import train_teacher

    config = {
        # PPO hyperparameters
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
            "log_std_init": -0.5,  # higher initial std for more exploration
        },
        # Training
        "total_timesteps": 200_000,
        "eval_freq": 5_000,
        "save_freq": 50_000,
        "seed": 42,
        "verbose": 1,
        # Joint residual wrapper — LARGER SCALE
        "use_joint_residual": True,
        "joint_residual_scale": 0.2,  # doubled from v1
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
        # Logging
        "run_name": "gate4_joint_residual_v2",
        "log_dir": str(_PROJECT_ROOT / "logs" / "gate4_joint_residual_v2"),
        "checkpoint_dir": str(_PROJECT_ROOT / "checkpoints" / "gate4_joint_residual_v2"),
        "tensorboard_log": str(_PROJECT_ROOT / "logs" / "gate4_joint_residual_v2" / "tb"),
    }

    print("=" * 60)
    print("Gate 4 v2: Joint-Space Residual PPO (scale=0.2)")
    print("=" * 60)
    print(f"  Total timesteps:   {config['total_timesteps']:,}")
    print(f"  Residual scale:    {config['joint_residual_scale']}")
    print(f"  Log std init:      {config['policy_kwargs']['log_std_init']}")
    print()

    model = train_teacher(config)
    print("Gate 4 v2 training complete.")
    return model


if __name__ == "__main__":
    main()
