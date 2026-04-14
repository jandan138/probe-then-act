"""Run scripted baselines for the ScoopTransfer environment.

Four motion strategies are evaluated:
  A. Scripted Scoop-and-Deposit -- waypoint-based joint trajectory that
     sweeps the gripper through the source container material toward the
     target container.
  B. Scripted Probe + Scoop -- 3 gentle probe taps into the material
     followed by the same scoop-and-deposit trajectory.
  C. Random baseline -- uniform random actions for comparison.
  D. No-op baseline -- zero actions to measure the natural particle
     settling baseline for metrics.

IMPORTANT NOTE ON METRICS:
  The spill_ratio metric uses axis-aligned bounding boxes (AABBs) for the
  source and target containers.  MPM particles initialised inside the
  source container settle downward through the thin base plate (0.005 m)
  and rest on the ground at z ~ 0.01-0.03, which falls below the source
  AABB minimum z of 0.04.  As a result, the spill_ratio converges to
  ~1.0 within the first 50-70 steps regardless of robot actions.  The
  no-op baseline (D) demonstrates this: it produces the same spill_ratio
  as random actions.

  Transfer efficiency (fraction of particles inside the target AABB) is
  therefore the more discriminative metric at this stage.  A future fix
  should extend the source AABB z-range downward to capture settled
  particles.

Sequences A and B use direct joint-position waypoints (via set_qpos with
fine interpolation) because:
  1. The default DLS IK action scale (0.01 m/step) is too small for the
     gripper to traverse the ~0.28 m from the home position to the source
     container within the 200-step horizon.
  2. The PD position controller cannot converge to low-z configurations
     (near the table surface) due to gravity compensation limitations.

Sequence C uses the standard 7-D action interface through env.step().

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
from typing import Any, Callable, Dict, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Robot workspace mapping (pre-computed via FK with set_qpos)
# ---------------------------------------------------------------------------
# Franka home qpos:  [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
# EE start position: ~(0.306, 0.0, 0.587)
#
# Source container:   centre (0.5, 0.0, 0.05), particles at z~0.10 initially,
#                     settle to z~0.02 after ~50 steps
# Target container:   centre (0.5, 0.35, 0.05)
# Target AABB:        x [0.44, 0.56], y [0.29, 0.41], z [0.04, 0.20]
#
# J1 controls base rotation -> EE y position
# J2=1.7 with J4~-0.65, J6~0.75 -> arm fully extended, fingertips near z=0.01
#
# J1 -> EE_y mapping (with [J1, 1.7, 0, -0.65, 0, 0.75, 0]):
#   J1=-0.15 -> EE_y=-0.13  (before source in -y)
#   J1= 0.00 -> EE_y=-0.02  (in source)
#   J1= 0.20 -> EE_y= 0.11  (past source +y edge)
#   J1= 0.60 -> EE_y= 0.29  (entering target)
#   J1= 0.80 -> EE_y= 0.39  (in target)

# ---------------------------------------------------------------------------
# Waypoint interpolation
# ---------------------------------------------------------------------------


def interpolate_waypoints(
    env: Any,
    start_qpos: List[float],
    end_qpos: List[float],
    n_steps: int,
    settle_per_step: int = 1,
) -> None:
    """Smoothly interpolate robot joint positions from *start* to *end*.

    Uses ``set_qpos`` with fine linear interpolation.  Between each qpos
    update, ``settle_per_step`` physics steps allow particles to react.
    """
    start = torch.tensor(start_qpos, dtype=torch.float32, device="cuda")
    end = torch.tensor(end_qpos, dtype=torch.float32, device="cuda")
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        qpos = start * (1 - alpha) + end * alpha
        env.robot.set_qpos(qpos)
        for _ in range(settle_per_step):
            env.scene.step()
            if hasattr(env, "post_physics_update"):
                env.post_physics_update()


def settle(env: Any, n_steps: int = 20) -> None:
    """Run physics steps without changing robot pose."""
    for _ in range(n_steps):
        env.scene.step()
        if hasattr(env, "post_physics_update"):
            env.post_physics_update()


# ---------------------------------------------------------------------------
# Joint-space waypoints (9-D: 7 arm joints + 2 finger joints)
# ---------------------------------------------------------------------------
# Finger values: 0.04 = fully open, 0.00 = fully closed.

HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]

# Intermediate: arm extending forward
EXTEND_FWD = [0.0, 0.5, 0.0, -1.8, 0.0, 1.8, 0.0, 0.04, 0.04]

# Above source container
HOVER_ABOVE_SOURCE = [0.0, 1.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.04, 0.04]

# Fingertips at particle level, positioned at -y edge of source
# EE ~ (0.54, -0.13, 0.08), finger tips near z=0.03
SCOOP_START = [-0.15, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]

# Mid-push: sweeping through source centre
SCOOP_MID = [0.0, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]

# Past +y edge of source
SCOOP_PAST_SOURCE = [0.3, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]

# Between source and target
SCOOP_MIDWAY = [0.5, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]

# At target y
SCOOP_AT_TARGET = [0.7, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]

# Past target
SCOOP_PAST_TARGET = [0.85, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]

# Probe waypoints -- intermediate arm extension for gentle contact
PROBE_HOVER = [0.0, 1.2, 0.0, -1.0, 0.0, 1.2, 0.0, 0.04, 0.04]
PROBE_DOWN_C = [0.0, 1.5, 0.0, -0.7, 0.0, 0.9, 0.0, 0.04, 0.04]
PROBE_DOWN_L = [-0.1, 1.5, 0.0, -0.7, 0.0, 0.9, 0.0, 0.04, 0.04]
PROBE_DOWN_R = [0.1, 1.5, 0.0, -0.7, 0.0, 0.9, 0.0, 0.04, 0.04]


# ---------------------------------------------------------------------------
# Scoop-tool waypoints (7-D: 7 arm joints, NO finger joints)
# ---------------------------------------------------------------------------

HOME_S = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
EXTEND_FWD_S = [0.0, 0.5, 0.0, -1.8, 0.0, 1.8, 0.0]
HOVER_SOURCE_S = [0.0, 1.0, 0.0, -1.5, 0.0, 1.5, 0.0]

# Scoop at particle level (-y edge of source)
# J2=1.3, J4=-1.0 -> z=0.052, safely above ground
# FK: (0.596, -0.087, 0.052)
SCOOP_START_S = [-0.15, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]

# Mid-sweep through source center
# FK: (0.603, 0.003, 0.052)
SCOOP_MID_S = [0.0, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]

# Past +y edge of source — stop before going too far
SCOOP_PAST_S = [0.15, 1.3, 0.0, -1.0, 0.0, 1.1, 0.0]

# Slight lift to secure particles in scoop cup
# J2=1.1 -> z=0.080
LIFT_LOW_S = [0.0, 1.1, 0.0, -1.2, 0.0, 1.2, 0.0]

# Full lift above source rim
# J2=0.8 -> z=0.160 (source rim at z~0.13)
LIFT_S = [0.0, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]

# Traverse to over target while lifted
# FK: (0.466, 0.393, 0.160)
TRAVERSE_S = [0.7, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]

# Lower over target
# FK: (0.463, 0.390, 0.080)
DEPOSIT_S = [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 0.0]

# Tilt scoop to dump (rotate J7)
# FK: (0.466, 0.390, 0.100)
DUMP_S = [0.7, 1.1, 0.0, -1.2, 0.0, 1.2, 1.5]


BOWL_APPROACH_S = [-0.10, 1.05, 0.0, -1.45, 0.0, 1.45, -0.55]
BOWL_INSERT_S = [0.00, 1.34, 0.0, -1.02, 0.0, 1.02, -1.20]
BOWL_CAPTURE_S = [0.06, 1.18, 0.0, -1.20, 0.0, 1.18, -0.75]
BOWL_LIFT_S = [0.06, 0.88, 0.0, -1.46, 0.0, 1.42, -0.70]
BOWL_TRAVERSE_FAST_S = [0.68, 0.88, 0.0, -1.46, 0.0, 1.42, -0.70]
BOWL_TRAVERSE_MID_S = [0.38, 0.88, 0.0, -1.46, 0.0, 1.42, -0.70]
BOWL_POUR_S = [0.68, 1.02, 0.0, -1.20, 0.0, 1.20, 1.35]
BOWL_SETTLE_S = [0.68, 0.92, 0.0, -1.28, 0.0, 1.26, 0.45]


# ---------------------------------------------------------------------------
# Scripted sequences
# ---------------------------------------------------------------------------


def run_sequence_a(env: Any, horizon: int = 200) -> Dict[str, float]:
    """Sequence A: Scripted Scoop-and-Deposit via joint waypoints.

    Sweeps the gripper from the -y edge of the source container through
    the material toward the target container at +y.  Uses fine
    interpolation (many small qpos increments) to moderate contact forces.
    """
    env.reset()

    # Phase 1: Position at -y edge of source at particle level
    # Use multi-step approach to avoid violent teleportation
    interpolate_waypoints(env, HOME, EXTEND_FWD, 20, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD, HOVER_ABOVE_SOURCE, 20, settle_per_step=1)
    interpolate_waypoints(env, HOVER_ABOVE_SOURCE, SCOOP_START, 40, settle_per_step=1)

    # Phase 2: Sweep through source material (slow, fine steps)
    interpolate_waypoints(env, SCOOP_START, SCOOP_MID, 40, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_MID, SCOOP_PAST_SOURCE, 30, settle_per_step=2)

    # Phase 3: Continue sweep through gap to target
    interpolate_waypoints(env, SCOOP_PAST_SOURCE, SCOOP_MIDWAY, 30, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_MIDWAY, SCOOP_AT_TARGET, 30, settle_per_step=2)
    interpolate_waypoints(
        env, SCOOP_AT_TARGET, SCOOP_PAST_TARGET, 20, settle_per_step=2
    )

    # Phase 4: Settle particles
    settle(env, 50)

    return env.compute_metrics()


def run_sequence_b(env: Any, horizon: int = 200) -> Dict[str, float]:
    """Sequence B: Scripted Probe + Scoop.

    Performs 3 probe taps (gentle vertical motions into the material at
    different y-positions) to gather contact information, then executes
    the same scoop-and-deposit sweep from Sequence A.
    """
    env.reset()

    # Phase 0: Position for probing
    interpolate_waypoints(env, HOME, EXTEND_FWD, 15, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD, PROBE_HOVER, 15, settle_per_step=1)

    # Probe tap 1: centre
    interpolate_waypoints(env, PROBE_HOVER, PROBE_DOWN_C, 20, settle_per_step=2)
    interpolate_waypoints(env, PROBE_DOWN_C, PROBE_HOVER, 10, settle_per_step=1)

    # Probe tap 2: left (-y)
    interpolate_waypoints(env, PROBE_HOVER, PROBE_DOWN_L, 20, settle_per_step=2)
    interpolate_waypoints(env, PROBE_DOWN_L, PROBE_HOVER, 10, settle_per_step=1)

    # Probe tap 3: right (+y)
    interpolate_waypoints(env, PROBE_HOVER, PROBE_DOWN_R, 20, settle_per_step=2)
    interpolate_waypoints(env, PROBE_DOWN_R, PROBE_HOVER, 10, settle_per_step=1)

    # Phase 1: Scoop-and-deposit (same as A, slightly compressed)
    interpolate_waypoints(env, PROBE_HOVER, SCOOP_START, 30, settle_per_step=1)
    interpolate_waypoints(env, SCOOP_START, SCOOP_MID, 35, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_MID, SCOOP_PAST_SOURCE, 25, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_PAST_SOURCE, SCOOP_MIDWAY, 25, settle_per_step=2)
    interpolate_waypoints(env, SCOOP_MIDWAY, SCOOP_AT_TARGET, 25, settle_per_step=2)
    interpolate_waypoints(
        env, SCOOP_AT_TARGET, SCOOP_PAST_TARGET, 15, settle_per_step=2
    )

    # Phase 2: Settle
    settle(env, 40)

    return env.compute_metrics()


def run_sequence_c(env: Any, horizon: int = 200, seed: int = 0) -> Dict[str, float]:
    """Sequence C: Random baseline -- uniform random 7-D actions via step()."""
    env.reset()

    rng = np.random.RandomState(seed)
    info: Dict[str, float] = {}

    for t in range(horizon):
        a = rng.uniform(-1, 1, size=(7,)).astype(np.float32)
        action = torch.tensor(a, dtype=torch.float32, device="cuda")
        _, _, done, info = env.step(action)
        if done:
            break

    return info


def run_sequence_d(env: Any, horizon: int = 200) -> Dict[str, float]:
    """Sequence D: No-op baseline -- zero actions to show settling effect."""
    env.reset()

    info: Dict[str, float] = {}
    action = torch.zeros(7, dtype=torch.float32, device="cuda")

    for t in range(horizon):
        _, _, done, info = env.step(action)
        if done:
            break

    return info


def run_sequence_e_scoop(env: Any, horizon: int = 200) -> Dict[str, float]:
    """Sequence E: Edge-push with panda_scoop.xml (7-DOF, no fingers).

    Strategy: Source particles sit on an elevated platform. The scoop pushes
    particles off the +y edge of the platform into a target container below.
    Multiple passes to maximize transfer.

    Requires: env built with tool_type="scoop" and task_layout="edge_push".
    """
    env.reset()

    phase_reached = "approach"

    # Waypoints for edge-push (7-DOF, platform at z=0.15)
    # Scoop at J2=0.8 gives z~0.16, right at platform surface
    BEHIND_EP = [-0.10, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]
    PUSH_END_EP = [0.40, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]

    # Phase 1: Approach — position behind particles on platform
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 20, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, BEHIND_EP, 30, settle_per_step=2)

    phase_reached = "push"

    # Phase 2: Multi-pass push — push particles off the platform edge
    for pass_idx in range(3):
        interpolate_waypoints(env, BEHIND_EP, PUSH_END_EP, 100, settle_per_step=3)
        if pass_idx < 2:
            # Retract for another pass
            interpolate_waypoints(env, PUSH_END_EP, BEHIND_EP, 30, settle_per_step=1)

    phase_reached = "dump"

    # Phase 3: Settle — let particles fall into target
    settle(env, 80)

    metrics = env.compute_metrics()
    metrics["phase_reached"] = phase_reached
    return metrics


def _bowl_traverse_steps(traverse_speed: float) -> int:
    if traverse_speed <= 0.25:
        return 120
    if traverse_speed <= 0.6:
        return 70
    return 35


def run_sequence_f_bowl(
    env: Any,
    horizon: int = 200,
    traverse_speed: float = 0.5,
) -> Dict[str, float]:
    env.reset()

    traverse_steps = _bowl_traverse_steps(traverse_speed)
    phase_reached = "approach"

    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 15, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, BOWL_APPROACH_S, 30, settle_per_step=2)

    phase_reached = "insert"
    interpolate_waypoints(env, BOWL_APPROACH_S, BOWL_INSERT_S, 45, settle_per_step=3)

    phase_reached = "capture"
    interpolate_waypoints(env, BOWL_INSERT_S, BOWL_CAPTURE_S, 35, settle_per_step=3)
    settle(env, 10)

    phase_reached = "lift"
    interpolate_waypoints(env, BOWL_CAPTURE_S, BOWL_LIFT_S, 35, settle_per_step=3)
    settle(env, 10)

    phase_reached = "traverse"
    interpolate_waypoints(
        env, BOWL_LIFT_S, BOWL_TRAVERSE_MID_S, traverse_steps // 2, settle_per_step=2
    )
    interpolate_waypoints(
        env,
        BOWL_TRAVERSE_MID_S,
        BOWL_TRAVERSE_FAST_S,
        traverse_steps - traverse_steps // 2,
        settle_per_step=2,
    )
    settle(env, 10)

    phase_reached = "pour"
    interpolate_waypoints(env, BOWL_TRAVERSE_FAST_S, BOWL_POUR_S, 30, settle_per_step=3)
    interpolate_waypoints(env, BOWL_POUR_S, BOWL_SETTLE_S, 20, settle_per_step=3)
    settle(env, 40)

    metrics = env.compute_metrics()
    metrics["phase_reached"] = phase_reached
    metrics["traverse_speed"] = traverse_speed
    metrics["traverse_steps"] = traverse_steps
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scripted baselines for ScoopTransfer"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes per sequence (default: 5)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Max steps per episode (default: 200)",
    )
    parser.add_argument(
        "--material",
        type=str,
        default="sand",
        help="MPM material family (default: sand)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/scripted_baselines.csv",
        help="CSV output path (default: results/tables/scripted_baselines.csv)",
    )
    parser.add_argument(
        "--tool-type",
        type=str,
        default="gripper",
        choices=["gripper", "scoop", "bowl"],
        help="Tool type: gripper (9-DOF), scoop (7-DOF), or bowl (7-DOF)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Run only a specific sequence (A, B, C, D, E). Default: all applicable.",
    )
    parser.add_argument(
        "--source-wall-height",
        type=float,
        default=None,
        help="Override source container wall height (0 for no walls).",
    )
    parser.add_argument(
        "--target-wall-height",
        type=float,
        default=None,
        help="Override target container wall height (0 for no walls).",
    )
    parser.add_argument(
        "--traverse-speed",
        type=float,
        default=0.5,
        help="Traverse speed tag for bowl Sequence F (default: 0.5)",
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
    print(f"  Tool type:             {args.tool_type}")
    print(f"  Output CSV:            {args.output}")
    print()

    # Build environment
    print("[1/4] Building environment ...")
    t0 = time.time()

    from pta.envs.tasks.scoop_transfer import ScoopTransferTask

    scene_cfg = {
        "particle_material": args.material,
        "n_envs": 0,
        "tool_type": args.tool_type,
    }
    if args.source_wall_height is not None:
        scene_cfg["source_wall_height"] = args.source_wall_height
    if args.target_wall_height is not None:
        scene_cfg["target_wall_height"] = args.target_wall_height
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

    # Report bbox info
    print(
        f"  Source AABB z: [{env._source_bbox_min[2]:.3f}, {env._source_bbox_max[2]:.3f}]"
    )
    print(
        f"  Target AABB z: [{env._target_bbox_min[2]:.3f}, {env._target_bbox_max[2]:.3f}]"
    )
    print(f"  NOTE: particles settle to z~0.02 (below source AABB z_min=0.04)")
    print(f"        -> spill_ratio ~1.0 is expected for all baselines")
    print()

    # Define sequences
    all_sequence_runners: Dict[str, Callable] = {
        "A_scoop_deposit": lambda ep: run_sequence_a(env, horizon),
        "B_probe_scoop": lambda ep: run_sequence_b(env, horizon),
        "C_random": lambda ep: run_sequence_c(env, horizon, seed=ep * 100 + 42),
        "D_noop": lambda ep: run_sequence_d(env, horizon),
        "E_scoop_tool": lambda ep: run_sequence_e_scoop(env, horizon),
        "F_bowl_tool": lambda ep: run_sequence_f_bowl(
            env, horizon, traverse_speed=args.traverse_speed
        ),
    }

    # Filter sequences based on args
    if args.sequence:
        seq_key = args.sequence.upper()
        matched = {
            k: v for k, v in all_sequence_runners.items() if k.startswith(seq_key)
        }
        if not matched:
            print(
                f"ERROR: Unknown sequence '{args.sequence}'. Available: A, B, C, D, E, F"
            )
            sys.exit(1)
        sequence_runners = matched
    elif args.tool_type == "scoop":
        sequence_runners = {"E_scoop_tool": all_sequence_runners["E_scoop_tool"]}
    elif args.tool_type == "bowl":
        sequence_runners = {"F_bowl_tool": all_sequence_runners["F_bowl_tool"]}
    else:
        sequence_runners = {
            k: v
            for k, v in all_sequence_runners.items()
            if k not in {"E_scoop_tool", "F_bowl_tool"}
        }

    all_rows: List[Dict[str, Any]] = []

    # Run episodes
    print("[2/4] Running episodes ...")
    print()

    for seq_name, runner in sequence_runners.items():
        print(f"  --- Sequence {seq_name} ---")
        ep_metrics: List[Dict[str, float]] = []

        for ep in range(n_episodes):
            t_ep = time.time()
            info = runner(ep)
            dt = time.time() - t_ep

            success = info.get("success_rate", 0.0)
            te = info.get("transfer_efficiency", 0.0)
            sr = info.get("spill_ratio", 0.0)
            n_in = info.get("n_in_target", 0)
            n_sp = info.get("n_spilled", 0)

            ep_metrics.append(
                {
                    "success_rate": success,
                    "transfer_efficiency": te,
                    "spill_ratio": sr,
                }
            )

            print(
                f"    ep {ep + 1}/{n_episodes}  "
                f"success={success:.0f}  "
                f"transfer={te:.4f}  "
                f"spill={sr:.4f}  "
                f"in_target={n_in}  "
                f"spilled={n_sp}  "
                f"({dt:.1f}s)"
            )

            all_rows.append(
                {
                    "sequence": seq_name,
                    "episode": ep + 1,
                    "success_rate": success,
                    "transfer_efficiency": te,
                    "spill_ratio": sr,
                }
            )

        successes = [m["success_rate"] for m in ep_metrics]
        transfers = [m["transfer_efficiency"] for m in ep_metrics]
        spills = [m["spill_ratio"] for m in ep_metrics]
        print(
            f"    >> mean  success={np.mean(successes):.2f}  "
            f"transfer={np.mean(transfers):.4f} +/- {np.std(transfers):.4f}  "
            f"spill={np.mean(spills):.4f} +/- {np.std(spills):.4f}"
        )
        print()

    # Summary table
    print("[3/4] Summary Table")
    print("-" * 70)
    print(
        f"{'Sequence':<20s} {'Success%':>10s} {'TransferEff':>14s} {'SpillRatio':>14s}"
    )
    print("-" * 70)

    for seq_name in sequence_runners:
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
            fieldnames=[
                "sequence",
                "episode",
                "success_rate",
                "transfer_efficiency",
                "spill_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Saved {len(all_rows)} rows.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
