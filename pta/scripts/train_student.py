"""CLI entry point for student / baseline RL training.

Supports three baseline methods from the paper:
  - M1 (reactive_ppo):     Standard PPO, no history -- Reactive baseline
  - M2 (rnn_ppo):          RecurrentPPO from sb3-contrib -- History-based
  - M3 (domain_rand_ppo):  PPO with domain randomisation wrapper

Usage::

    # M1: Reactive PPO baseline
    python pta/scripts/train_student.py --method reactive_ppo --seed 42

    # M2: RNN-PPO baseline (requires sb3-contrib)
    python pta/scripts/train_student.py --method rnn_ppo --seed 42

    # M3: Domain-randomisation PPO baseline
    python pta/scripts/train_student.py --method domain_rand_ppo --seed 42

    # Shorter test run
    python pta/scripts/train_student.py --method reactive_ppo --total-timesteps 100000
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


METHODS = ["reactive_ppo", "rnn_ppo", "domain_rand_ppo"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for student/baseline training."""
    parser = argparse.ArgumentParser(
        description="Train baseline RL policies (M1/M2/M3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Method
    parser.add_argument(
        "--method", type=str, required=True, choices=METHODS,
        help="Baseline method: reactive_ppo (M1), rnn_ppo (M2), domain_rand_ppo (M3)",
    )

    # Training
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

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
        help="MPM material family (ignored for domain_rand_ppo)",
    )
    parser.add_argument("--horizon", type=int, default=200, help="Episode length")

    # Domain randomization options (M3 only)
    parser.add_argument(
        "--dr-families", type=str, nargs="+",
        default=["sand", "snow", "elastoplastic"],
        help="Material families for domain randomisation (M3 only)",
    )
    parser.add_argument(
        "--dr-rebuild", action="store_true",
        help="Rebuild Genesis scene on each reset (expensive, M3 only)",
    )

    # RNN options (M2 only)
    parser.add_argument("--lstm-hidden-size", type=int, default=256, help="LSTM hidden size (M2 only)")
    parser.add_argument("--n-lstm-layers", type=int, default=1, help="Number of LSTM layers (M2 only)")

    # Eval / checkpoint
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Eval every N steps")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Checkpoint every N steps")

    # Paths
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")

    # Misc
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity (0/1/2)")

    return parser.parse_args()


def train_reactive_ppo(config: dict) -> None:
    """M1: Standard PPO with MlpPolicy, no history."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
    from pta.training.utils.checkpoint_io import save_sb3_checkpoint
    from pta.training.utils.seed import set_seed

    seed = config["seed"]
    set_seed(seed)

    log_dir = Path(config["log_dir"])
    ckpt_dir = Path(config["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _make_env():
        return GenesisGymWrapper(
            task_config=config.get("task_config"),
            scene_config=config.get("scene_config"),
        )

    vec_env = DummyVecEnv([_make_env])
    eval_env = DummyVecEnv([_make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config["lr"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["entropy_coef"],
        policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
        tensorboard_log=str(log_dir / "tb"),
        seed=seed,
        verbose=config["verbose"],
        device="auto",
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=config["save_freq"],
            save_path=str(ckpt_dir),
            name_prefix="reactive_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=config["eval_freq"],
            n_eval_episodes=5,
            deterministic=True,
        ),
    ])

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    save_sb3_checkpoint(
        model,
        ckpt_dir / "reactive_ppo_final",
        metadata={"method": "reactive_ppo", "seed": seed, "config": config},
    )

    eval_env.close()
    vec_env.close()
    print(f"M1 (reactive_ppo) training complete. Model saved to {ckpt_dir}")


def train_rnn_ppo(config: dict) -> None:
    """M2: RecurrentPPO with LSTM policy for history-based learning."""
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        print("ERROR: sb3-contrib is required for rnn_ppo (M2).")
        print("Install with: pip install sb3-contrib")
        sys.exit(1)

    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
    from pta.training.utils.checkpoint_io import save_sb3_checkpoint
    from pta.training.utils.seed import set_seed

    seed = config["seed"]
    set_seed(seed)

    log_dir = Path(config["log_dir"])
    ckpt_dir = Path(config["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _make_env():
        return GenesisGymWrapper(
            task_config=config.get("task_config"),
            scene_config=config.get("scene_config"),
        )

    vec_env = DummyVecEnv([_make_env])
    eval_env = DummyVecEnv([_make_env])

    policy_kwargs = {
        "lstm_hidden_size": config.get("lstm_hidden_size", 256),
        "n_lstm_layers": config.get("n_lstm_layers", 1),
        "net_arch": [dict(pi=[256], vf=[256])],
    }

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=config["lr"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["entropy_coef"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir / "tb"),
        seed=seed,
        verbose=config["verbose"],
        device="auto",
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=config["save_freq"],
            save_path=str(ckpt_dir),
            name_prefix="rnn_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=config["eval_freq"],
            n_eval_episodes=5,
            deterministic=True,
        ),
    ])

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    save_sb3_checkpoint(
        model,
        ckpt_dir / "rnn_ppo_final",
        metadata={"method": "rnn_ppo", "seed": seed, "config": config},
    )

    eval_env.close()
    vec_env.close()
    print(f"M2 (rnn_ppo) training complete. Model saved to {ckpt_dir}")


def train_domain_rand_ppo(config: dict) -> None:
    """M3: PPO with domain randomisation over material families."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pta.envs.wrappers.domain_rand_wrapper import DomainRandWrapper
    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
    from pta.training.utils.checkpoint_io import save_sb3_checkpoint
    from pta.training.utils.seed import set_seed

    seed = config["seed"]
    set_seed(seed)

    log_dir = Path(config["log_dir"])
    ckpt_dir = Path(config["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _make_dr_env():
        base_env = GenesisGymWrapper(
            task_config=config.get("task_config"),
            scene_config=config.get("scene_config"),
        )
        return DomainRandWrapper(
            env=base_env,
            families=config.get("dr_families", ["sand", "snow", "elastoplastic"]),
            task_config=config.get("task_config"),
            scene_config_base=config.get("scene_config", {}),
            rebuild_on_reset=config.get("dr_rebuild", False),
            seed=seed,
        )

    vec_env = DummyVecEnv([_make_dr_env])

    # Eval env without domain rand (use fixed material for fair eval)
    def _make_eval_env():
        return GenesisGymWrapper(
            task_config=config.get("task_config"),
            scene_config=config.get("scene_config"),
        )

    eval_env = DummyVecEnv([_make_eval_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config["lr"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["entropy_coef"],
        policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
        tensorboard_log=str(log_dir / "tb"),
        seed=seed,
        verbose=config["verbose"],
        device="auto",
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=config["save_freq"],
            save_path=str(ckpt_dir),
            name_prefix="domain_rand_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=config["eval_freq"],
            n_eval_episodes=5,
            deterministic=True,
        ),
    ])

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    save_sb3_checkpoint(
        model,
        ckpt_dir / "domain_rand_ppo_final",
        metadata={"method": "domain_rand_ppo", "seed": seed, "config": config},
    )

    eval_env.close()
    vec_env.close()
    print(f"M3 (domain_rand_ppo) training complete. Model saved to {ckpt_dir}")


def main() -> None:
    """Parse CLI arguments and launch student/baseline training."""
    args = parse_args()

    # Set WSL2 rendering env vars
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = (
            "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
        )

    from pta.utils.paths import CHECKPOINT_DIR, LOG_DIR

    # Build config
    method = args.method
    default_log_dir = str(LOG_DIR / method)
    default_ckpt_dir = str(CHECKPOINT_DIR / method)

    config = {
        "method": method,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "lr": args.lr,
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
        "log_dir": args.log_dir or default_log_dir,
        "checkpoint_dir": args.checkpoint_dir or default_ckpt_dir,
        # Env
        "scene_config": {
            "particle_material": args.material,
            "n_envs": 0,
        },
        "task_config": {
            "horizon": args.horizon,
        },
        # M2 options
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": args.n_lstm_layers,
        # M3 options
        "dr_families": args.dr_families,
        "dr_rebuild": args.dr_rebuild,
    }

    print("=" * 60)
    print(f"Probe-Then-Act: Baseline Training ({method})")
    print("=" * 60)
    method_names = {
        "reactive_ppo": "M1 -- Reactive PPO (no history)",
        "rnn_ppo": "M2 -- RNN-PPO (LSTM history)",
        "domain_rand_ppo": "M3 -- Domain-Randomisation PPO",
    }
    print(f"  Method:     {method_names[method]}")
    print(f"  Timesteps:  {args.total_timesteps:,}")
    print(f"  Seed:       {args.seed}")
    print(f"  Material:   {args.material}")
    print(f"  Horizon:    {args.horizon}")
    if method == "rnn_ppo":
        print(f"  LSTM:       hidden={args.lstm_hidden_size}, layers={args.n_lstm_layers}")
    if method == "domain_rand_ppo":
        print(f"  DR families: {args.dr_families}")
        print(f"  DR rebuild:  {args.dr_rebuild}")
    print()

    # Dispatch
    dispatch = {
        "reactive_ppo": train_reactive_ppo,
        "rnn_ppo": train_rnn_ppo,
        "domain_rand_ppo": train_domain_rand_ppo,
    }

    dispatch[method](config)
    print("Training complete.")


if __name__ == "__main__":
    main()
