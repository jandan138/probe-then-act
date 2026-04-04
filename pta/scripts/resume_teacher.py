"""Resume interrupted M8 Teacher training from checkpoint.

Usage::

    python pta/scripts/resume_teacher.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# WSL2 rendering
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    )


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pta.training.rl.train_teacher import make_env

    seed = 42
    total_timesteps = 100_000
    checkpoint_dir = Path("checkpoints/teacher/seed42")
    log_dir = Path("logs/teacher")
    tb_log = "logs/tensorboard/teacher"

    # Find latest checkpoint
    ckpts = sorted(checkpoint_dir.glob("teacher_*_steps.zip"))
    if not ckpts:
        print("ERROR: No checkpoints found to resume from.")
        sys.exit(1)

    latest = ckpts[-1]
    print(f"Resuming from: {latest}")

    # Create environments
    def _make_env():
        return make_env(
            task_config={"horizon": 200},
            scene_config={"particle_material": "sand", "n_envs": 0},
            seed=seed,
        )

    def _make_eval_env():
        return make_env(
            task_config={"horizon": 200},
            scene_config={"particle_material": "sand", "n_envs": 0},
            seed=seed + 1000,
        )

    vec_env = DummyVecEnv([_make_env])
    eval_env = DummyVecEnv([_make_eval_env])

    # Load model from checkpoint
    model = PPO.load(
        str(latest),
        env=vec_env,
        tensorboard_log=tb_log,
        device="auto",
    )

    print(f"  Loaded num_timesteps: {model.num_timesteps}")
    print(f"  Target total: {total_timesteps}")
    remaining = total_timesteps - model.num_timesteps
    print(f"  Remaining: {remaining}")

    if remaining <= 0:
        print("Training already complete!")
        sys.exit(0)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(checkpoint_dir),
        name_prefix="teacher",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Resume training
    print(f"Resuming teacher PPO for {remaining} more timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    # Save final
    final_path = checkpoint_dir / "teacher_final"
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    eval_env.close()
    vec_env.close()
    print("Teacher training complete!")


if __name__ == "__main__":
    main()
