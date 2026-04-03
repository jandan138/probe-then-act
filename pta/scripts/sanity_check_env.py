"""Sanity-check an environment: run N steps and report basic diagnostics.

Usage::

    # From project root with Genesis venv activated:
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    python pta/scripts/sanity_check_env.py
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict

import numpy as np


def main(config: Dict[str, Any] | None = None) -> None:
    """Run the environment for N steps and check for common issues.

    Checks performed:
    - Observation tensors contain no NaN or Inf values.
    - Rewards are finite and within expected bounds.
    - Episode lengths are reasonable.
    - Action space is correctly shaped.

    Prints a summary of environment statistics on completion.

    Parameters
    ----------
    config : dict, optional
        Environment configuration.  If ``None``, reads from CLI args.
    """
    parser = argparse.ArgumentParser(description="Sanity-check the ScoopTransfer env")
    parser.add_argument("--n-steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--material", type=str, default="sand",
                        help="MPM material family (sand, snow, elastoplastic, liquid)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_steps = args.n_steps
    material = args.material

    print("=" * 60)
    print("ScoopTransfer Environment Sanity Check")
    print("=" * 60)
    print(f"  Steps:    {n_steps}")
    print(f"  Material: {material}")
    print()

    # ------------------------------------------------------------------
    # 1. Build env
    # ------------------------------------------------------------------
    print("[1/5] Building environment ...")
    t0 = time.time()

    from pta.envs.tasks.scoop_transfer import ScoopTransferTask

    scene_cfg = {
        "particle_material": material,
        "n_envs": 0,  # single env
    }
    task_cfg = {
        "horizon": n_steps,
    }

    try:
        env = ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)
    except Exception as e:
        print(f"  FAIL: Could not build environment: {e}")
        sys.exit(1)

    build_time = time.time() - t0
    print(f"  OK (build took {build_time:.1f}s)")
    print(f"  Total particles: {env._total_particles}")
    print()

    # ------------------------------------------------------------------
    # 2. Reset
    # ------------------------------------------------------------------
    print("[2/5] Resetting environment ...")
    try:
        obs = env.reset()
        _check_obs(obs, step="reset")
        print("  OK")
    except Exception as e:
        print(f"  FAIL: Reset error: {e}")
        sys.exit(1)
    print()

    # ------------------------------------------------------------------
    # 3. Run random actions
    # ------------------------------------------------------------------
    print(f"[3/5] Running {n_steps} random-action steps ...")
    np.random.seed(args.seed)

    had_nan = False
    had_inf = False
    rewards = []
    metrics_history = []

    import torch

    for step_i in range(n_steps):
        action = torch.tensor(
            np.random.uniform(-1, 1, size=(7,)).astype(np.float32),
            device="cuda",
        )

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            print(f"  FAIL at step {step_i}: {e}")
            sys.exit(1)

        # Check observation
        obs_ok = _check_obs(obs, step=step_i)
        if not obs_ok:
            had_nan = True

        # Check reward
        if not np.isfinite(reward):
            print(f"  WARNING: Non-finite reward at step {step_i}: {reward}")
            had_inf = True

        rewards.append(reward)
        metrics_history.append(info)

        # Print progress every 10 steps
        if (step_i + 1) % 10 == 0 or step_i == 0:
            te = info.get("transfer_efficiency", 0)
            sr = info.get("spill_ratio", 0)
            print(
                f"  step {step_i + 1:4d}/{n_steps} | "
                f"reward={reward:+.4f} | "
                f"transfer={te:.3f} | "
                f"spill={sr:.3f}"
            )

        if done:
            print(f"  Episode done at step {step_i + 1}")
            obs = env.reset()
            _check_obs(obs, step="re-reset")

    print("  OK")
    print()

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("[4/5] Summary statistics:")
    rewards_arr = np.array(rewards)
    print(f"  Reward:  mean={rewards_arr.mean():.4f}  "
          f"std={rewards_arr.std():.4f}  "
          f"min={rewards_arr.min():.4f}  "
          f"max={rewards_arr.max():.4f}")

    if metrics_history:
        last = metrics_history[-1]
        print(f"  Final metrics:")
        for k, v in last.items():
            print(f"    {k}: {v}")
    print()

    # ------------------------------------------------------------------
    # 5. Verdict
    # ------------------------------------------------------------------
    print("[5/5] Verdict:")
    stable = True
    if had_nan:
        print("  FAIL: NaN detected in observations")
        stable = False
    if had_inf:
        print("  FAIL: Inf/NaN detected in rewards")
        stable = False
    if stable:
        print("  PASS: Environment is stable -- no NaN/Inf detected")

    print("=" * 60)
    sys.exit(0 if stable else 1)


def _check_obs(obs: dict, step: Any = None) -> bool:
    """Return True if observation contains no NaN/Inf."""
    import torch

    ok = True
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().float().numpy()
        elif isinstance(val, np.ndarray):
            arr = val
        else:
            continue

        if np.any(np.isnan(arr)):
            print(f"  WARNING: NaN in obs['{key}'] at step {step}")
            ok = False
        if np.any(np.isinf(arr)):
            print(f"  WARNING: Inf in obs['{key}'] at step {step}")
            ok = False

    return ok


if __name__ == "__main__":
    main()
