"""Train a privileged teacher policy using PPO with full-state observations.

Uses Stable-Baselines3 PPO with the GenesisGymWrapper.  For v1 the
teacher receives the same observations as the student (proprioception
+ step fraction).  Privileged observations (hidden material params)
will be added in v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
from pta.training.utils.checkpoint_io import save_sb3_checkpoint
from pta.training.utils.logger import ExperimentLogger
from pta.training.utils.seed import set_seed
from pta.utils.paths import CHECKPOINT_DIR, LOG_DIR


# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------

_DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    # PPO hyper-parameters
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    # Policy
    "policy": "MlpPolicy",
    "policy_kwargs": {
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
    },
    # Training
    "total_timesteps": 500_000,
    "eval_freq": 10_000,
    "save_freq": 50_000,
    "seed": 42,
    # Env
    "task_config": None,
    "scene_config": None,
    # Logging
    "log_dir": str(LOG_DIR / "teacher"),
    "checkpoint_dir": str(CHECKPOINT_DIR / "teacher"),
    "tensorboard_log": str(LOG_DIR / "teacher" / "tb"),
    "run_name": "teacher_ppo",
    "verbose": 1,
}


def make_env(
    task_config: Optional[Dict[str, Any]] = None,
    scene_config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> GenesisGymWrapper:
    """Create a single GenesisGymWrapper instance.

    Parameters
    ----------
    task_config:
        Task-level configuration for ScoopTransferTask.
    scene_config:
        Scene-level configuration for SceneBuilder.
    seed:
        Seed for the environment.

    Returns
    -------
    GenesisGymWrapper
        A ready-to-use Gymnasium environment.
    """
    env = GenesisGymWrapper(
        task_config=task_config,
        scene_config=scene_config,
    )
    env.reset(seed=seed)
    return env


def train_teacher(config: Dict[str, Any]) -> PPO:
    """Train a privileged teacher using PPO with access to ground-truth state.

    The teacher receives privileged observations (e.g. true material IDs,
    exact mass, friction coefficients) that are unavailable at deployment
    time.  Its converged policy later serves as the expert for student
    distillation.

    For v1, the teacher uses the same observations as the student
    (proprioception + step fraction).

    Parameters
    ----------
    config : dict
        Training configuration.  Missing keys are filled from
        ``_DEFAULT_TRAIN_CONFIG``.

    Returns
    -------
    PPO
        The trained PPO model.
    """
    cfg = {**_DEFAULT_TRAIN_CONFIG, **(config or {})}

    seed = cfg["seed"]
    set_seed(seed)

    log_dir = Path(cfg["log_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -- Environment -------------------------------------------------------
    task_config = cfg.get("task_config")
    scene_config = cfg.get("scene_config")

    def _make_env():
        return make_env(
            task_config=task_config,
            scene_config=scene_config,
            seed=seed,
        )

    vec_env = DummyVecEnv([_make_env])

    # -- Eval environment (separate instance) ------------------------------
    def _make_eval_env():
        return make_env(
            task_config=task_config,
            scene_config=scene_config,
            seed=seed + 1000,
        )

    eval_env = DummyVecEnv([_make_eval_env])

    # -- PPO model ---------------------------------------------------------
    policy_kwargs = cfg.get("policy_kwargs", {})

    model = PPO(
        policy=cfg["policy"],
        env=vec_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["entropy_coef"],
        vf_coef=cfg["value_loss_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        normalize_advantage=cfg["normalize_advantage"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg["tensorboard_log"],
        seed=seed,
        verbose=cfg["verbose"],
        device="auto",
    )

    # -- Callbacks ---------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["save_freq"],
        save_path=str(checkpoint_dir),
        name_prefix="scoop_transfer_teacher",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # -- Experiment logger (our own, for extra metadata) -------------------
    logger = ExperimentLogger(
        log_dir=log_dir,
        project="probe-then-act",
        run_name=cfg["run_name"],
        backends=["csv"],
    )
    logger.log_config(cfg)

    # -- Train -------------------------------------------------------------
    total_timesteps = cfg["total_timesteps"]
    print(f"Starting teacher PPO training for {total_timesteps} timesteps")
    print(f"  Log dir:        {log_dir}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Seed:           {seed}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # -- Save final model --------------------------------------------------
    final_path = checkpoint_dir / "scoop_transfer_teacher_final"
    save_sb3_checkpoint(
        model,
        final_path,
        metadata={
            "total_timesteps": total_timesteps,
            "seed": seed,
            "config": cfg,
            "stage": "teacher_rl",
        },
    )
    print(f"Final model saved to {final_path}")

    # -- Cleanup -----------------------------------------------------------
    logger.close()
    eval_env.close()
    vec_env.close()

    return model
