#!/usr/bin/env python3
"""Stage B: Zero-action reward breakdown — validates hotfix before RL.

Runs the JointResidualWrapper with zero residuals (pure base trajectory)
for N episodes and reports per-component reward breakdown.

Pass criteria:
- episode total reward > 0
- transfer_frac >= 0.25
- spill_frac <= 0.20

Usage:
    python pta/scripts/eval_zero_action.py --episodes 3 --reward-breakdown
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

import numpy as np


def make_env(residual_scale: float = 0.05):
    """Create the full env stack: GenesisGymWrapper → JointResidualWrapper."""
    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
    from pta.envs.wrappers.joint_residual_wrapper import JointResidualWrapper

    base_env = GenesisGymWrapper(
        task_config={"horizon": 500, "success_threshold": 0.3},
        scene_config={
            "tool_type": "scoop",
            "n_envs": 0,
            "particle_material": "sand",
        },
    )
    env = JointResidualWrapper(
        base_env,
        residual_scale=residual_scale,
        trajectory="edge_push",
    )
    return env


def run_episode(env, breakdown: bool = False):
    """Run one episode with zero action, return (total_reward, info, step_rewards)."""
    obs, info = env.reset()
    total_reward = 0.0
    step_rewards = []
    step = 0
    done = False

    while not done:
        action = np.zeros(7, dtype=np.float32)  # zero residual
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_rewards.append(reward)
        done = terminated or truncated
        step += 1

    return total_reward, info, step_rewards, step


def main():
    parser = argparse.ArgumentParser(description="Zero-action reward breakdown")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--reward-breakdown", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 60)
    print("Stage B: Zero-Action Reward Breakdown")
    print("=" * 60)
    print(f"  Episodes:       {args.episodes}")
    print(f"  Residual scale: {args.residual_scale}")
    print()

    env = make_env(residual_scale=args.residual_scale)

    all_rewards = []
    all_transfer = []
    all_spill = []

    for ep in range(args.episodes):
        total_reward, info, step_rewards, n_steps = run_episode(env, args.reward_breakdown)
        transfer_frac = info.get("transfer_efficiency", 0.0)
        spill_frac = info.get("spill_ratio", 0.0)

        all_rewards.append(total_reward)
        all_transfer.append(transfer_frac)
        all_spill.append(spill_frac)

        print(f"Episode {ep + 1}/{args.episodes}:")
        print(f"  Steps:          {n_steps}")
        print(f"  Total reward:   {total_reward:.2f}")
        print(f"  Transfer frac:  {transfer_frac:.4f}")
        print(f"  Spill frac:     {spill_frac:.4f}")
        print(f"  Success:        {info.get('success_rate', 0.0):.0f}")

        if args.reward_breakdown and len(step_rewards) > 10:
            # Show reward at key points
            sr = step_rewards
            print(f"  Reward @step 1:   {sr[0]:.4f}")
            print(f"  Reward @step 100: {sr[min(99, len(sr)-1)]:.4f}")
            print(f"  Reward @step 200: {sr[min(199, len(sr)-1)]:.4f}")
            print(f"  Reward @step 400: {sr[min(399, len(sr)-1)]:.4f}")
            print(f"  Reward @step 490: {sr[min(489, len(sr)-1)]:.4f}")
            print(f"  Mean reward/step: {np.mean(sr):.4f}")
            print(f"  Max reward/step:  {np.max(sr):.4f}")
            print(f"  Min reward/step:  {np.min(sr):.4f}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    mean_reward = np.mean(all_rewards)
    mean_transfer = np.mean(all_transfer)
    mean_spill = np.mean(all_spill)
    print(f"  Mean total reward:  {mean_reward:.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean transfer frac: {mean_transfer:.4f} ± {np.std(all_transfer):.4f}")
    print(f"  Mean spill frac:    {mean_spill:.4f} ± {np.std(all_spill):.4f}")
    print()

    # Pass criteria
    passed = True
    checks = [
        ("Total reward > 0", mean_reward > 0),
        ("Transfer frac >= 0.25", mean_transfer >= 0.25),
        ("Spill frac <= 0.20", mean_spill <= 0.20),
    ]
    for name, ok in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")
        if not ok:
            passed = False

    print()
    if passed:
        print("🟢 Stage B PASSED — Zero-action baseline is healthy")
    else:
        print("🔴 Stage B FAILED — Reward design needs further work")

    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
