"""Controller Replay A/B Test.

Compares two methods of replaying the exact same joint trajectory:
  Mode A: set_qpos(q_t) + scene.step()   -- kinematic / teleport
  Mode B: control_dofs_position(q_t) + scene.step()  -- PD position control

Both modes use the dense waypoint trajectory from the scripted baseline
(Sequence A: scoop-and-deposit). The test isolates whether PD tracking
error explains the learner's failure to reproduce scripted performance.

Usage::

    source /home/zhuzihou/dev/Genesis/.venv/bin/activate
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
    python pta/scripts/controller_replay_ab.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Waypoints (9-DOF: 7 arm + 2 finger) — copied from run_scripted_baseline.py
# ---------------------------------------------------------------------------

HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
EXTEND_FWD = [0.0, 0.5, 0.0, -1.8, 0.0, 1.8, 0.0, 0.04, 0.04]
HOVER_ABOVE_SOURCE = [0.0, 1.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.04, 0.04]
SCOOP_START = [-0.15, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]
SCOOP_MID = [0.0, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]
SCOOP_PAST_SOURCE = [0.3, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]
SCOOP_MIDWAY = [0.5, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]
SCOOP_AT_TARGET = [0.7, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]
SCOOP_PAST_TARGET = [0.85, 1.7, 0.0, -0.65, 0.0, 0.75, 0.0, 0.04, 0.04]


# ---------------------------------------------------------------------------
# Build dense trajectory (same as Sequence A in run_scripted_baseline.py)
# ---------------------------------------------------------------------------

def build_dense_trajectory() -> list[list[float]]:
    """Return list of (q_t) waypoints matching Sequence A interpolation.

    Each segment mirrors the interpolation in run_sequence_a:
      HOME -> EXTEND_FWD:           20 steps, settle=1
      EXTEND_FWD -> HOVER:          20 steps, settle=1
      HOVER -> SCOOP_START:         40 steps, settle=1
      SCOOP_START -> MID:           40 steps, settle=2
      MID -> PAST_SOURCE:           30 steps, settle=2
      PAST_SOURCE -> MIDWAY:        30 steps, settle=2
      MIDWAY -> AT_TARGET:          30 steps, settle=2
      AT_TARGET -> PAST_TARGET:     20 steps, settle=2

    For the dense trajectory we produce one q_t per interpolation step.
    The settle_per_step value tells us how many scene.step() calls to do
    per q_t — we record that alongside q_t.
    """
    segments = [
        (HOME, EXTEND_FWD, 20, 1),
        (EXTEND_FWD, HOVER_ABOVE_SOURCE, 20, 1),
        (HOVER_ABOVE_SOURCE, SCOOP_START, 40, 1),
        (SCOOP_START, SCOOP_MID, 40, 2),
        (SCOOP_MID, SCOOP_PAST_SOURCE, 30, 2),
        (SCOOP_PAST_SOURCE, SCOOP_MIDWAY, 30, 2),
        (SCOOP_MIDWAY, SCOOP_AT_TARGET, 30, 2),
        (SCOOP_AT_TARGET, SCOOP_PAST_TARGET, 20, 2),
    ]

    trajectory = []  # list of (qpos_list, settle_steps)
    for start, end, n_steps, settle in segments:
        s = np.array(start, dtype=np.float64)
        e = np.array(end, dtype=np.float64)
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q = s * (1 - alpha) + e * alpha
            trajectory.append((q.tolist(), settle))

    return trajectory


# ---------------------------------------------------------------------------
# Run one mode
# ---------------------------------------------------------------------------

def run_mode(env, trajectory, mode: str) -> list[dict]:
    """Run one mode (A=set_qpos, B=control_dofs_position).

    Returns per-step records with ee position and final metrics.
    """
    env.reset()

    records = []
    global_step = 0

    for q_list, settle_steps in trajectory:
        q_t = torch.tensor(q_list, dtype=torch.float32, device="cuda")

        if mode == "A":
            env.robot.set_qpos(q_t)
        else:
            env.robot.control_dofs_position(q_t)

        for s in range(settle_steps):
            env.scene.step()
            global_step += 1

            # Record EE position
            ee_pos = env._ee_link.get_pos()
            if ee_pos.dim() > 1:
                ee_pos = ee_pos.squeeze(0)
            ee_x, ee_y, ee_z = ee_pos[0].item(), ee_pos[1].item(), ee_pos[2].item()

            # Also record actual qpos for J1 (the y-control joint)
            actual_qpos = env.robot.get_qpos()
            if actual_qpos.dim() > 1:
                actual_qpos = actual_qpos.squeeze(0)
            j1_cmd = q_list[0]
            j1_actual = actual_qpos[0].item()

            records.append({
                "mode": mode,
                "step": global_step,
                "ee_x": ee_x,
                "ee_y": ee_y,
                "ee_z": ee_z,
                "j1_cmd": j1_cmd,
                "j1_actual": j1_actual,
            })

    # Settle phase: 50 steps (same as Sequence A)
    for _ in range(50):
        env.scene.step()
        global_step += 1

        ee_pos = env._ee_link.get_pos()
        if ee_pos.dim() > 1:
            ee_pos = ee_pos.squeeze(0)
        ee_x, ee_y, ee_z = ee_pos[0].item(), ee_pos[1].item(), ee_pos[2].item()

        actual_qpos = env.robot.get_qpos()
        if actual_qpos.dim() > 1:
            actual_qpos = actual_qpos.squeeze(0)
        j1_actual = actual_qpos[0].item()

        records.append({
            "mode": mode,
            "step": global_step,
            "ee_x": ee_x,
            "ee_y": ee_y,
            "ee_z": ee_z,
            "j1_cmd": float("nan"),  # no command during settle
            "j1_actual": j1_actual,
        })

    # Final metrics
    metrics = env.compute_metrics()
    for r in records:
        r["transferred_mass_frac"] = metrics["transfer_efficiency"]
        r["spill_ratio"] = metrics["spill_ratio"]
        r["success"] = int(metrics["success_rate"])

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Controller Replay A/B Test")
    print("=" * 70)

    # Build environment
    print("[1/5] Building environment ...")
    t0 = time.time()

    from pta.envs.tasks.scoop_transfer import ScoopTransferTask

    scene_cfg = {
        "particle_material": "sand",
        "n_envs": 0,
        "tool_type": "gripper",
    }
    task_cfg = {"horizon": 500}

    env = ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)
    print(f"  Built in {time.time() - t0:.1f}s, {env._total_particles} particles")

    # Build dense trajectory
    print("[2/5] Building dense trajectory ...")
    trajectory = build_dense_trajectory()
    total_physics_steps = sum(s for _, s in trajectory) + 50  # +50 settle
    print(f"  {len(trajectory)} waypoints, {total_physics_steps} total physics steps")

    # Mode A: set_qpos
    print("[3/5] Running Mode A (set_qpos) ...")
    t0 = time.time()
    records_a = run_mode(env, trajectory, "A")
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Transfer: {records_a[-1]['transferred_mass_frac']:.4f}")
    print(f"  Spill:    {records_a[-1]['spill_ratio']:.4f}")
    print(f"  Success:  {records_a[-1]['success']}")

    # Mode B: control_dofs_position
    print("[4/5] Running Mode B (control_dofs_position) ...")
    t0 = time.time()
    records_b = run_mode(env, trajectory, "B")
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Transfer: {records_b[-1]['transferred_mass_frac']:.4f}")
    print(f"  Spill:    {records_b[-1]['spill_ratio']:.4f}")
    print(f"  Success:  {records_b[-1]['success']}")

    # Save CSV
    print("[5/5] Saving results ...")
    os.makedirs("results", exist_ok=True)

    csv_path = "results/controller_replay_ab_test.csv"
    all_records = records_a + records_b
    fieldnames = [
        "mode", "step", "ee_x", "ee_y", "ee_z",
        "j1_cmd", "j1_actual",
        "transferred_mass_frac", "spill_ratio", "success",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"  CSV: {csv_path} ({len(all_records)} rows)")

    # Plot
    steps_a = [r["step"] for r in records_a]
    ey_a = [r["ee_y"] for r in records_a]
    ez_a = [r["ee_z"] for r in records_a]
    j1_actual_a = [r["j1_actual"] for r in records_a]

    steps_b = [r["step"] for r in records_b]
    ey_b = [r["ee_y"] for r in records_b]
    ez_b = [r["ee_z"] for r in records_b]
    j1_actual_b = [r["j1_actual"] for r in records_b]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # EE y
    axes[0].plot(steps_a, ey_a, label="Mode A (set_qpos)", linewidth=1.5)
    axes[0].plot(steps_b, ey_b, label="Mode B (control_dofs_position)", linewidth=1.5, linestyle="--")
    axes[0].axhline(y=0.0, color="gray", linestyle=":", alpha=0.5, label="source center y=0")
    axes[0].axhline(y=0.35, color="orange", linestyle=":", alpha=0.5, label="target center y=0.35")
    axes[0].set_ylabel("EE y (m)")
    axes[0].set_title("Controller Replay A/B: End-Effector Y Trajectory")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # EE z
    axes[1].plot(steps_a, ez_a, label="Mode A (set_qpos)", linewidth=1.5)
    axes[1].plot(steps_b, ez_b, label="Mode B (control_dofs_position)", linewidth=1.5, linestyle="--")
    axes[1].set_ylabel("EE z (m)")
    axes[1].set_title("End-Effector Z Trajectory")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # J1 tracking error
    # Align lengths (should be same)
    n = min(len(j1_actual_a), len(j1_actual_b))
    j1_err = [abs(j1_actual_b[i] - j1_actual_a[i]) for i in range(n)]
    axes[2].plot(steps_a[:n], j1_err, color="red", linewidth=1.5)
    axes[2].set_ylabel("|J1 error| (rad)")
    axes[2].set_xlabel("Physics step")
    axes[2].set_title("J1 Tracking Error (Mode B vs Mode A)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "results/controller_replay_ab_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot: {plot_path}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Compute divergence stats
    ey_a_arr = np.array(ey_a)
    ey_b_arr = np.array(ey_b)
    ez_a_arr = np.array(ez_a)
    ez_b_arr = np.array(ez_b)
    n = min(len(ey_a_arr), len(ey_b_arr))

    y_err = np.abs(ey_a_arr[:n] - ey_b_arr[:n])
    z_err = np.abs(ez_a_arr[:n] - ez_b_arr[:n])

    print(f"  EE y divergence:  mean={y_err.mean():.4f}m  max={y_err.max():.4f}m")
    print(f"  EE z divergence:  mean={z_err.mean():.4f}m  max={z_err.max():.4f}m")
    print(f"  Mode A final y:   {ey_a_arr[-1]:.4f}m")
    print(f"  Mode B final y:   {ey_b_arr[-1]:.4f}m")
    print(f"  Mode A transfer:  {records_a[-1]['transferred_mass_frac']:.4f}")
    print(f"  Mode B transfer:  {records_b[-1]['transferred_mass_frac']:.4f}")
    print(f"  Mode A success:   {records_a[-1]['success']}")
    print(f"  Mode B success:   {records_b[-1]['success']}")
    print()

    # Max J1 tracking error
    j1_err_arr = np.array(j1_err)
    print(f"  J1 tracking error: mean={j1_err_arr.mean():.4f}rad  max={j1_err_arr.max():.4f}rad")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
