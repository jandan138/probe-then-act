"""Run scripted baselines for the ScoopTransfer environment.

Three motion sequences are evaluated:
  A. Scripted Scoop-and-Deposit -- a hand-tuned open-loop trajectory.
  B. Scripted Probe + Scoop -- 3 probe taps then the same scoop trajectory.
  C. Random baseline -- uniform random actions for comparison.

Each sequence is run for N_EPISODES episodes and per-episode metrics are
recorded: success_rate, transfer_efficiency, spill_ratio.

Usage::

    # From project root with Genesis venv activated:
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
    python pta/scripts/run_scripted_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def make_action(
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    droll: float = 0.0,
    dpitch: float = 0.0,
    dyaw: float = 0.0,
    gripper: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a single 7-D action tensor.

    All values are in [-1, 1] -- they are scaled internally by the env:
      - position:  * 0.01 m
      - rotation:  * 0.05 rad
      - gripper:   * 0.04 width
    """
    return torch.tensor(
        [dx, dy, dz, droll, dpitch, dyaw, gripper],
        dtype=torch.float32,
        device=device,
    )


def repeat_action(action: torch.Tensor, n: int) -> List[torch.Tensor]:
    """Return a list of *n* copies of *action*."""
    return [action.clone() for _ in range(n)]


# ---------------------------------------------------------------------------
# Scripted sequences
# ---------------------------------------------------------------------------

def build_sequence_a(device: str = "cuda") -> List[torch.Tensor]:
    """Sequence A: Scripted Scoop-and-Deposit.

    The Franka starts at its home pose (~0.3 m above table, roughly
    centred).  The source container is at (0.5, 0.0, 0.05) and the
    target at (0.5, 0.35, 0.05).

    Phases:
      1. Approach  -- move EE above source container.
      2. Descend   -- lower into material.
      3. Scoop     -- push forward (along +y) to gather material.
      4. Lift      -- raise EE.
      5. Transport -- move EE towards target container.
      6. Deposit   -- lower and open gripper / tilt to dump.

    Actions are in normalised [-1, 1] space.  With action_scale_pos=0.01,
    an action component of +1.0 moves the EE by +0.01 m per step.
    """
    seq: List[torch.Tensor] = []

    # Phase 1 -- Approach: move toward source (dx ~+0.5, keep y~0)
    # The EE starts near (0.3, 0.0, 0.5) approximately.
    # Move forward (+x) and down (-z) to get above the source.
    seq += repeat_action(make_action(dx=+1.0, dz=-0.8, device=device), 15)

    # Phase 2 -- Descend into material
    seq += repeat_action(make_action(dz=-1.0, device=device), 12)

    # Phase 3 -- Scoop: push along +y while slightly lifting
    # Close gripper to hold material
    seq += repeat_action(
        make_action(dy=+1.0, dz=+0.2, gripper=-1.0, device=device), 15
    )

    # Phase 4 -- Lift with material
    seq += repeat_action(make_action(dz=+1.0, gripper=-1.0, device=device), 15)

    # Phase 5 -- Transport: move toward target container (+y)
    seq += repeat_action(
        make_action(dy=+1.0, dz=+0.2, gripper=-1.0, device=device), 20
    )

    # Phase 6 -- Deposit: lower and open gripper, tilt to dump
    seq += repeat_action(make_action(dz=-0.8, device=device), 10)
    seq += repeat_action(
        make_action(dz=-0.3, gripper=+1.0, dpitch=+0.5, device=device), 15
    )

    # Hold for settling
    seq += repeat_action(make_action(device=device), 10)

    return seq  # ~112 steps, well within 200 horizon


def build_sequence_b(device: str = "cuda") -> List[torch.Tensor]:
    """Sequence B: Scripted Probe + Scoop.

    Begins with 3 gentle probe taps at different positions near the
    source container, then runs the same scoop-and-deposit from
    Sequence A.

    Probe taps are: touch down gently, retract.
    """
    seq: List[torch.Tensor] = []

    # Move toward source first
    seq += repeat_action(make_action(dx=+1.0, dz=-0.5, device=device), 10)

    # Probe tap 1 -- centre of source
    seq += repeat_action(make_action(dz=-1.0, device=device), 5)   # down
    seq += repeat_action(make_action(dz=+1.0, device=device), 5)   # retract

    # Shift slightly in y for tap 2
    seq += repeat_action(make_action(dy=-0.5, device=device), 3)
    seq += repeat_action(make_action(dz=-1.0, device=device), 5)   # down
    seq += repeat_action(make_action(dz=+1.0, device=device), 5)   # retract

    # Shift in +y for tap 3
    seq += repeat_action(make_action(dy=+1.0, device=device), 3)
    seq += repeat_action(make_action(dz=-1.0, device=device), 5)   # down
    seq += repeat_action(make_action(dz=+1.0, device=device), 5)   # retract

    # Now run scoop-and-deposit (adapted -- shorter since we already
    # moved partially toward the source)

    # Descend into material
    seq += repeat_action(make_action(dz=-1.0, device=device), 12)

    # Scoop along +y, close gripper
    seq += repeat_action(
        make_action(dy=+1.0, dz=+0.2, gripper=-1.0, device=device), 15
    )

    # Lift
    seq += repeat_action(make_action(dz=+1.0, gripper=-1.0, device=device), 15)

    # Transport to target
    seq += repeat_action(
        make_action(dy=+1.0, dz=+0.2, gripper=-1.0, device=device), 20
    )

    # Deposit
    seq += repeat_action(make_action(dz=-0.8, device=device), 8)
    seq += repeat_action(
        make_action(dz=-0.3, gripper=+1.0, dpitch=+0.5, device=device), 12
    )

    # Settle
    seq += repeat_action(make_action(device=device), 5)

    return seq  # ~153 steps


def build_sequence_c(
    horizon: int = 200, seed: int = 0, device: str = "cuda"
) -> List[torch.Tensor]:
    """Sequence C: Random baseline -- uniform random actions."""
    rng = np.random.RandomState(seed)
    seq: List[torch.Tensor] = []
    for _ in range(horizon):
        a = rng.uniform(-1, 1, size=(7,)).astype(np.float32)
        seq.append(torch.tensor(a, dtype=torch.float32, device=device))
    return seq


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: Any,
    action_sequence: List[torch.Tensor],
    horizon: int = 200,
) -> Dict[str, float]:
    """Run one episode with the given action sequence.

    Returns the final-step metrics dict.
    """
    env.reset()
    info: Dict[str, float] = {}

    for t in range(min(len(action_sequence), horizon)):
        _, _, done, info = env.step(action_sequence[t])
        if done:
            break

    # If the sequence was shorter than horizon, pad with no-ops
    if not done and len(action_sequence) < horizon:
        noop = make_action(device=action_sequence[0].device)
        for t in range(len(action_sequence), horizon):
            _, _, done, info = env.step(noop)
            if done:
                break

    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scripted baselines for ScoopTransfer"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=5,
        help="Number of episodes per sequence (default: 5)",
    )
    parser.add_argument(
        "--horizon", type=int, default=200,
        help="Max steps per episode (default: 200)",
    )
    parser.add_argument(
        "--material", type=str, default="sand",
        help="MPM material family (default: sand)",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/tables/scripted_baselines.csv",
        help="CSV output path (default: results/tables/scripted_baselines.csv)",
    )
    args = parser.parse_args()

    n_episodes = args.n_episodes
    horizon = args.horizon

    print("=" * 70)
    print("Scripted Baselines -- ScoopTransfer")
    print("=" * 70)
    print(f"  Episodes per sequence: {n_episodes}")
    print(f"  Horizon:               {horizon}")
    print(f"  Material:              {args.material}")
    print(f"  Output CSV:            {args.output}")
    print()

    # Build environment once -- reuse across episodes
    print("[1/4] Building environment ...")
    t0 = time.time()

    from pta.envs.tasks.scoop_transfer import ScoopTransferTask

    scene_cfg = {
        "particle_material": args.material,
        "n_envs": 0,
    }
    task_cfg = {
        "horizon": horizon,
    }

    try:
        env = ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)
    except Exception as e:
        print(f"  FAIL: Could not build environment: {e}")
        sys.exit(1)

    build_time = time.time() - t0
    print(f"  OK ({build_time:.1f}s, {env._total_particles} particles)")
    print()

    # Define sequences
    sequences = {
        "A_scoop_deposit": lambda seed: build_sequence_a(device="cuda"),
        "B_probe_scoop": lambda seed: build_sequence_b(device="cuda"),
        "C_random": lambda seed: build_sequence_c(
            horizon=horizon, seed=seed, device="cuda"
        ),
    }

    all_rows: List[Dict[str, Any]] = []

    # Run each sequence
    print("[2/4] Running episodes ...")
    print()

    for seq_name, seq_builder in sequences.items():
        print(f"  --- Sequence {seq_name} ---")
        ep_metrics: List[Dict[str, float]] = []

        for ep in range(n_episodes):
            action_seq = seq_builder(seed=ep * 100 + 42)
            t_ep = time.time()
            info = run_episode(env, action_seq, horizon=horizon)
            dt = time.time() - t_ep

            success = info.get("success_rate", 0.0)
            te = info.get("transfer_efficiency", 0.0)
            sr = info.get("spill_ratio", 0.0)

            ep_metrics.append({
                "success_rate": success,
                "transfer_efficiency": te,
                "spill_ratio": sr,
            })

            print(
                f"    ep {ep+1}/{n_episodes}  "
                f"success={success:.0f}  "
                f"transfer={te:.4f}  "
                f"spill={sr:.4f}  "
                f"({dt:.1f}s)"
            )

            all_rows.append({
                "sequence": seq_name,
                "episode": ep + 1,
                "success_rate": success,
                "transfer_efficiency": te,
                "spill_ratio": sr,
            })

        # Summary for this sequence
        successes = [m["success_rate"] for m in ep_metrics]
        transfers = [m["transfer_efficiency"] for m in ep_metrics]
        spills = [m["spill_ratio"] for m in ep_metrics]
        print(
            f"    >> mean  success={np.mean(successes):.2f}  "
            f"transfer={np.mean(transfers):.4f} +/- {np.std(transfers):.4f}  "
            f"spill={np.mean(spills):.4f} +/- {np.std(spills):.4f}"
        )
        print()

    # Print summary table
    print("[3/4] Summary Table")
    print("-" * 70)
    print(f"{'Sequence':<20s} {'Success%':>10s} {'TransferEff':>14s} {'SpillRatio':>14s}")
    print("-" * 70)

    for seq_name in sequences:
        rows = [r for r in all_rows if r["sequence"] == seq_name]
        s_mean = np.mean([r["success_rate"] for r in rows]) * 100
        t_mean = np.mean([r["transfer_efficiency"] for r in rows])
        t_std = np.std([r["transfer_efficiency"] for r in rows])
        sp_mean = np.mean([r["spill_ratio"] for r in rows])
        sp_std = np.std([r["spill_ratio"] for r in rows])
        print(
            f"{seq_name:<20s} "
            f"{s_mean:>9.1f}% "
            f"{t_mean:>8.4f}+/-{t_std:<5.4f} "
            f"{sp_mean:>8.4f}+/-{sp_std:<5.4f}"
        )

    print("-" * 70)
    print()

    # Save CSV
    print(f"[4/4] Saving results to {args.output} ...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sequence", "episode", "success_rate",
                         "transfer_efficiency", "spill_ratio"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Saved {len(all_rows)} rows.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
