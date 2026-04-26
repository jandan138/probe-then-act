"""Diagnose M1 Reactive PPO seed variance."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pta.training.rl.train_teacher import make_env


def summarize_eval_curve(seed: int):
    path = _PROJECT_ROOT / "logs" / f"m1_reactive_seed{seed}" / "eval" / "evaluations.npz"
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]  # shape (n_evals, n_episodes)
    mean_reward = results.mean(axis=1)
    std_reward = results.std(axis=1)
    print(f"\n=== Seed {seed} ===")
    print(f"  Eval points: {len(timesteps)}")
    print(f"  Final mean reward: {mean_reward[-1]:.2f} ± {std_reward[-1]:.2f}")
    print(f"  Best mean reward:  {mean_reward.max():.2f} at step {timesteps[mean_reward.argmax()]}")
    print(f"  Worst mean reward: {mean_reward.min():.2f} at step {timesteps[mean_reward.argmin()]}")
    return timesteps, mean_reward, std_reward


def plot_curves():
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed in [42, 0, 1]:
        timesteps, mean_reward, std_reward = summarize_eval_curve(seed)
        ax.plot(timesteps, mean_reward, label=f"seed={seed}")
        ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("M1 Reactive PPO Eval Curves (original eval seeds)")
    ax.legend()
    ax.grid(True)
    out_path = _PROJECT_ROOT / "results" / "analysis" / "m1_seed_variance_eval_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


def fixed_seed_eval(seed: int, eval_seed: int = 9999, n_episodes: int = 5):
    """Load checkpoint for training seed and evaluate on a fixed eval seed."""
    checkpoint = _PROJECT_ROOT / "checkpoints" / f"m1_reactive_seed{seed}" / "scoop_transfer_teacher_500000_steps.zip"
    print(f"\n--- Fixed-seed eval: train_seed={seed}, eval_seed={eval_seed} ---")

    task_config = {"horizon": 500, "success_threshold": 0.3}
    scene_config = {"tool_type": "scoop", "n_envs": 0, "particle_material": "sand"}

    def _make_env():
        return make_env(
            task_config=task_config,
            scene_config=scene_config,
            seed=eval_seed,
            use_joint_residual=True,
            joint_residual_scale=0.2,
            joint_residual_trajectory="edge_push",
            use_privileged=False,
        )

    eval_env = DummyVecEnv([_make_env])
    model = PPO.load(checkpoint, env=eval_env, device="auto")

    rewards = []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            steps += 1
            if steps > 600:
                break
        rewards.append(ep_reward)
        print(f"  Episode {ep}: reward={ep_reward:.2f}, steps={steps}")

    eval_env.close()
    print(f"  Mean over {n_episodes} episodes: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards


def check_initial_conditions(seed: int):
    """Check what the initial joint config and scene look like for edge_push."""
    task_config = {"horizon": 500, "success_threshold": 0.3}
    scene_config = {"tool_type": "scoop", "n_envs": 0, "particle_material": "sand"}

    env = make_env(
        task_config=task_config,
        scene_config=scene_config,
        seed=seed,
        use_joint_residual=True,
        joint_residual_scale=0.2,
        joint_residual_trajectory="edge_push",
        use_privileged=False,
    )
    obs, info = env.reset(seed=seed)
    wrapper = env
    while not hasattr(wrapper, "_q_base") and hasattr(wrapper, "env"):
        wrapper = wrapper.env
    q_base = wrapper._q_base[0]
    print(f"\n=== Initial conditions for seed={seed} ===")
    print(f"  q_base[0]: {q_base}")
    print(f"  obs shape: {obs.shape}")
    print(f"  obs[:14]: {obs[:14]}")

    # Check if initial q_base is near limits
    from pta.envs.wrappers.joint_residual_wrapper import JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH
    margin_low = q_base - JOINT_LIMITS_LOW
    margin_high = JOINT_LIMITS_HIGH - q_base
    print(f"  Margin to low limits:  {margin_low}")
    print(f"  Margin to high limits: {margin_high}")
    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("M1 Reactive PPO Seed Variance Diagnosis")
    print("=" * 60)

    # 1. Plot eval curves
    plot_curves()

    # 2. Fixed-seed eval for all three training seeds
    print("\n" + "=" * 60)
    print("Fixed eval seed = 9999")
    print("=" * 60)
    for s in [42, 0, 1]:
        fixed_seed_eval(s, eval_seed=9999, n_episodes=5)

    # 4. Check initial conditions
    print("\n" + "=" * 60)
    print("Initial condition check")
    print("=" * 60)
    for s in [42, 0, 1]:
        check_initial_conditions(s)
