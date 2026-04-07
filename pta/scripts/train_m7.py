"""Train M7 (Probe-Then-Act) — probe + belief encoder + adaptive policy.

Usage::

    # Full M7 (probe + belief)
    python pta/scripts/train_m7.py --seed 42 --total-timesteps 500000

    # Ablation: no probe (z = zeros, tests if probing matters)
    python pta/scripts/train_m7.py --ablation no_probe --seed 42

    # Ablation: no belief (probe runs but z = zeros, tests if encoding matters)
    python pta/scripts/train_m7.py --ablation no_belief --seed 42

    # Run all seeds
    for seed in 42 0 1; do
        python pta/scripts/train_m7.py --seed $seed --total-timesteps 500000
    done

The wrapper stack is:
  GenesisGymWrapper -> JointResidualWrapper -> ProbePhaseWrapper

Key difference from M1/M8:
  - M7 does NOT use PrivilegedObsWrapper (no access to material params)
  - ProbePhaseWrapper runs probe steps at episode start and appends
    a learned latent belief z to observations
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
        description="Train M7 (Probe-Then-Act)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--material", type=str, default="sand")
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--n-probes", type=int, default=3)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_probe", "no_belief"],
        help=(
            "Ablation mode: none=full M7, "
            "no_probe=skip probe (z=zeros), "
            "no_belief=probe but z=zeros"
        ),
    )
    return parser.parse_args()


def make_m7_env(
    task_config=None,
    scene_config=None,
    seed: int = 0,
    joint_residual_scale: float = 0.2,
    joint_residual_trajectory: str = "edge_push",
    latent_dim: int = 16,
    n_probes: int = 3,
    ablation: str = "none",
):
    """Create the M7 environment stack.

    Stack: GenesisGymWrapper -> JointResidualWrapper -> ProbePhaseWrapper

    No PrivilegedObsWrapper -- M7 must infer material from probing, not
    from privileged ground-truth parameters.
    """
    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
    from pta.envs.wrappers.joint_residual_wrapper import JointResidualWrapper
    from pta.envs.wrappers.probe_phase_wrapper import ProbePhaseWrapper

    base_env = GenesisGymWrapper(
        task_config=task_config,
        scene_config=scene_config,
    )

    env = JointResidualWrapper(
        base_env,
        residual_scale=joint_residual_scale,
        trajectory=joint_residual_trajectory,
    )

    env = ProbePhaseWrapper(
        env,
        latent_dim=latent_dim,
        n_probes=n_probes,
        ablation=ablation,
        device="cpu",  # belief encoder on CPU; lightweight for inference
    )

    env.reset(seed=seed)
    return env


def main() -> None:
    args = parse_args()
    seed = args.seed
    ablation = args.ablation

    # Run name encodes ablation variant
    if ablation == "none":
        variant = "pta"
    elif ablation == "no_probe":
        variant = "pta_noprobe"
    else:
        variant = "pta_nobelief"

    run_name = f"m7_{variant}_seed{seed}"
    log_dir = _PROJECT_ROOT / "logs" / run_name
    checkpoint_dir = _PROJECT_ROOT / "checkpoints" / run_name

    print("=" * 60)
    print(f"Probe-Then-Act: M7 Training ({variant})")
    print("=" * 60)
    print(f"  Ablation:    {ablation}")
    print(f"  Timesteps:   {args.total_timesteps:,}")
    print(f"  Seed:        {seed}")
    print(f"  Material:    {args.material}")
    print(f"  Residual:    scale={args.residual_scale}")
    print(f"  Latent dim:  {args.latent_dim}")
    print(f"  N probes:    {args.n_probes}")
    print(f"  Log dir:     {log_dir}")
    print(f"  Checkpoint:  {checkpoint_dir}")
    print()

    # ---- Imports (heavy, deferred) ----------------------------------------
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pta.training.utils.checkpoint_io import save_sb3_checkpoint
    from pta.training.utils.logger import ExperimentLogger
    from pta.training.utils.seed import set_seed

    set_seed(seed)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Env setup --------------------------------------------------------
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
            seed=seed,
            joint_residual_scale=args.residual_scale,
            latent_dim=args.latent_dim,
            n_probes=args.n_probes,
            ablation=ablation,
        )

    def _make_eval_env():
        return make_m7_env(
            task_config=task_config,
            scene_config=scene_config,
            seed=seed + 1000,
            joint_residual_scale=args.residual_scale,
            latent_dim=args.latent_dim,
            n_probes=args.n_probes,
            ablation=ablation,
        )

    vec_env = DummyVecEnv([_make_env])
    eval_env = DummyVecEnv([_make_eval_env])

    # ---- PPO model (same hyperparams as M1/M8 baselines) ------------------
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            "log_std_init": -1.0,
        },
        tensorboard_log=str(log_dir / "tb"),
        seed=seed,
        verbose=1,
        device="auto",
    )

    # ---- Callbacks --------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_dir),
        name_prefix="m7_pta",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # ---- Logger -----------------------------------------------------------
    logger = ExperimentLogger(
        log_dir=log_dir,
        project="probe-then-act",
        run_name=run_name,
        backends=["csv"],
    )
    logger.log_config({
        "method": "m7",
        "ablation": ablation,
        "seed": seed,
        "total_timesteps": args.total_timesteps,
        "material": args.material,
        "residual_scale": args.residual_scale,
        "latent_dim": args.latent_dim,
        "n_probes": args.n_probes,
        "horizon": args.horizon,
    })

    # ---- Train ------------------------------------------------------------
    print(f"Starting M7 PPO training for {args.total_timesteps:,} timesteps")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ---- Save final model -------------------------------------------------
    final_path = checkpoint_dir / "m7_pta_final"
    save_sb3_checkpoint(
        model,
        final_path,
        metadata={
            "total_timesteps": args.total_timesteps,
            "seed": seed,
            "method": "m7",
            "ablation": ablation,
            "latent_dim": args.latent_dim,
            "n_probes": args.n_probes,
            "stage": "m7_rl",
        },
    )
    print(f"Final model saved to {final_path}")

    # ---- Cleanup ----------------------------------------------------------
    logger.close()
    eval_env.close()
    vec_env.close()

    print(f"M7 training complete ({variant}, seed={seed}).")
    return model


if __name__ == "__main__":
    main()
