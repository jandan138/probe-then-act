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
from copy import deepcopy
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


ENCODER_HIDDEN_DIM = 128
ENCODER_NUM_LAYERS = 2


def clone_belief_encoder_state(encoder):
    """Return an independent eval-mode copy of a belief encoder."""
    clone = deepcopy(encoder)
    clone.eval()
    return clone


def create_m7_belief_encoder(trace_dim: int, latent_dim: int):
    from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder

    encoder = LatentBeliefEncoder(
        trace_dim=trace_dim,
        latent_dim=latent_dim,
        hidden_dim=ENCODER_HIDDEN_DIM,
        num_layers=ENCODER_NUM_LAYERS,
    )
    encoder.eval()
    return encoder


def derive_trace_dim_from_env(env) -> int:
    shape = getattr(getattr(env, "observation_space", None), "shape", None)
    if not shape:
        raise ValueError("M7 inner env observation_space must expose a shape")
    return int(shape[0])


def save_m7_policy_with_encoder(model, encoder, path, repo_root, metadata):
    from pta.training.utils.checkpoint_io import (
        m7_encoder_sidecar_paths,
        save_m7_encoder_artifact,
        save_sb3_checkpoint,
    )

    save_sb3_checkpoint(model, Path(path), metadata=metadata)
    policy_path = m7_encoder_sidecar_paths(Path(path)).policy_path
    return save_m7_encoder_artifact(
        encoder=encoder,
        policy_path=policy_path,
        repo_root=repo_root,
        run_metadata=metadata,
    )


class M7BestModelSidecarCallback:
    """EvalCallback-compatible hook that writes encoder sidecars for best_model.zip."""

    def __init__(self, *, encoder, policy_path, repo_root, metadata):
        self.encoder = encoder
        self.policy_path = Path(policy_path)
        self.repo_root = Path(repo_root)
        self.metadata = dict(metadata)
        self.model = None
        self.parent = None
        self.n_calls = 0
        self.num_timesteps = 0
        self.locals = {}
        self.globals = {}

    def init_callback(self, model):
        self.model = model

    def update_locals(self, locals_):
        self.locals.update(locals_)

    def update_child_locals(self, locals_):
        self.update_locals(locals_)

    def on_step(self) -> bool:
        self.n_calls += 1
        if self.parent is not None:
            self.num_timesteps = getattr(self.parent, "num_timesteps", self.num_timesteps)
        from pta.training.utils.checkpoint_io import save_m7_encoder_artifact

        save_m7_encoder_artifact(
            encoder=self.encoder,
            policy_path=self.policy_path,
            repo_root=self.repo_root,
            run_metadata=self.metadata,
        )
        return True


class M7PeriodicCheckpointSidecarCallback:
    """CheckpointCallback-compatible periodic saver for matched full M7 artifacts."""

    def __init__(self, *, encoder, save_freq, save_path, name_prefix, repo_root, metadata):
        self.encoder = encoder
        self.save_freq = int(save_freq)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.repo_root = Path(repo_root)
        self.metadata = dict(metadata)
        self.model = None
        self.parent = None
        self.n_calls = 0
        self.num_timesteps = 0
        self.locals = {}
        self.globals = {}

    def init_callback(self, model):
        self.model = model

    def on_training_start(self, locals_, globals_):
        self.locals = locals_
        self.globals = globals_
        self.num_timesteps = int(getattr(self.model, "num_timesteps", 0))

    def on_rollout_start(self):
        pass

    def on_rollout_end(self):
        pass

    def on_training_end(self):
        pass

    def update_locals(self, locals_):
        self.locals.update(locals_)

    def update_child_locals(self, locals_):
        self.update_locals(locals_)

    def on_step(self) -> bool:
        self.n_calls += 1
        self.num_timesteps = int(getattr(self.model, "num_timesteps", self.num_timesteps))
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps or self.n_calls
            checkpoint_name = f"{self.name_prefix}_{step}_steps"
            metadata = {**self.metadata, "num_timesteps": int(step)}
            save_m7_policy_with_encoder(
                self.model,
                self.encoder,
                self.save_path / checkpoint_name / checkpoint_name,
                repo_root=self.repo_root,
                metadata=metadata,
            )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train M7 (Probe-Then-Act)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--material", type=str, default="sand")
    parser.add_argument("--residual-scale", type=float, default=0.05)
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
    belief_encoder=None,
):
    """Create the M7 environment stack.

    Stack: GenesisGymWrapper -> JointResidualWrapper -> ProbePhaseWrapper

    No PrivilegedObsWrapper -- M7 must infer material from probing, not
    from privileged ground-truth parameters.
    """
    env = make_m7_inner_env(
        task_config=task_config,
        scene_config=scene_config,
        joint_residual_scale=joint_residual_scale,
        joint_residual_trajectory=joint_residual_trajectory,
    )

    env = wrap_m7_inner_env(
        env,
        latent_dim=latent_dim,
        n_probes=n_probes,
        belief_encoder=belief_encoder,
        ablation=ablation,
    )

    env.reset(seed=seed)
    return env


def make_m7_inner_env(
    task_config=None,
    scene_config=None,
    joint_residual_scale: float = 0.2,
    joint_residual_trajectory: str = "edge_push",
):
    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
    from pta.envs.wrappers.joint_residual_wrapper import JointResidualWrapper

    base_env = GenesisGymWrapper(
        task_config=task_config,
        scene_config=scene_config,
    )
    return JointResidualWrapper(
        base_env,
        residual_scale=joint_residual_scale,
        trajectory=joint_residual_trajectory,
    )


def wrap_m7_inner_env(
    env,
    latent_dim: int = 16,
    n_probes: int = 3,
    ablation: str = "none",
    belief_encoder=None,
):
    from pta.envs.wrappers.probe_phase_wrapper import ProbePhaseWrapper

    return ProbePhaseWrapper(
        env,
        latent_dim=latent_dim,
        n_probes=n_probes,
        belief_encoder=belief_encoder,
        ablation=ablation,
        device="cpu",  # belief encoder on CPU; lightweight for inference
    )


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
    trace_dim = None
    canonical_belief_encoder = None

    def _encoder_for_inner_env(env):
        nonlocal trace_dim, canonical_belief_encoder
        if ablation != "none":
            return None
        env_trace_dim = derive_trace_dim_from_env(env)
        if trace_dim is None:
            trace_dim = env_trace_dim
            canonical_belief_encoder = create_m7_belief_encoder(
                trace_dim=trace_dim,
                latent_dim=args.latent_dim,
            )
        elif env_trace_dim != trace_dim:
            raise ValueError(
                f"M7 trace_dim changed across envs: {trace_dim} vs {env_trace_dim}"
            )
        return clone_belief_encoder_state(canonical_belief_encoder)

    def _make_env():
        env = make_m7_inner_env(
            task_config=task_config,
            scene_config=scene_config,
            joint_residual_scale=args.residual_scale,
        )
        env = wrap_m7_inner_env(
            env,
            latent_dim=args.latent_dim,
            n_probes=args.n_probes,
            ablation=ablation,
            belief_encoder=_encoder_for_inner_env(env),
        )
        env.reset(seed=seed)
        return env

    def _make_eval_env():
        env = make_m7_inner_env(
            task_config=task_config,
            scene_config=scene_config,
            joint_residual_scale=args.residual_scale,
        )
        env = wrap_m7_inner_env(
            env,
            latent_dim=args.latent_dim,
            n_probes=args.n_probes,
            ablation=ablation,
            belief_encoder=_encoder_for_inner_env(env),
        )
        env.reset(seed=seed + 1000)
        return env

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
    best_sidecar_callback = None
    if canonical_belief_encoder is not None:
        assert trace_dim is not None
        checkpoint_callback = M7PeriodicCheckpointSidecarCallback(
            encoder=canonical_belief_encoder,
            save_freq=50_000,
            save_path=checkpoint_dir,
            name_prefix="m7_pta",
            repo_root=_PROJECT_ROOT,
            metadata={
                "method": "m7_pta",
                "seed": seed,
                "ablation": ablation,
                "trace_dim": trace_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": ENCODER_HIDDEN_DIM,
                "num_layers": ENCODER_NUM_LAYERS,
                "n_probes": args.n_probes,
                "stage": "periodic",
            },
        )
        best_sidecar_callback = M7BestModelSidecarCallback(
            encoder=canonical_belief_encoder,
            policy_path=checkpoint_dir / "best" / "best_model.zip",
            repo_root=_PROJECT_ROOT,
            metadata={
                "method": "m7_pta",
                "seed": seed,
                "ablation": ablation,
                "trace_dim": trace_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": ENCODER_HIDDEN_DIM,
                "num_layers": ENCODER_NUM_LAYERS,
                "n_probes": args.n_probes,
                "stage": "best",
            },
        )
    else:
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
        callback_on_new_best=best_sidecar_callback,
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
    final_metadata = {
        "total_timesteps": args.total_timesteps,
        "seed": seed,
        "ablation": ablation,
        "latent_dim": args.latent_dim,
        "n_probes": args.n_probes,
    }
    if canonical_belief_encoder is not None:
        final_metadata.update(
            {
                "method": "m7_pta",
                "trace_dim": trace_dim,
                "hidden_dim": ENCODER_HIDDEN_DIM,
                "num_layers": ENCODER_NUM_LAYERS,
                "stage": "final",
            }
        )
        save_m7_policy_with_encoder(
            model,
            canonical_belief_encoder,
            final_path,
            repo_root=_PROJECT_ROOT,
            metadata=final_metadata,
        )
    else:
        final_metadata.update(
            {
                "method": "m7_pta",
                "stage": "final",
                "encoder_mode": "zero-z",
                "legacy_policy_only": False,
                "zero_z_semantics": ablation,
            }
        )
        save_sb3_checkpoint(model, final_path, metadata=final_metadata)
    print(f"Final model saved to {final_path}")

    # ---- Cleanup ----------------------------------------------------------
    logger.close()
    eval_env.close()
    vec_env.close()

    print(f"M7 training complete ({variant}, seed={seed}).")
    return model


if __name__ == "__main__":
    main()
