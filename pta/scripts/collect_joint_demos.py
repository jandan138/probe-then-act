"""Collect scripted joint-space demos for BC warmstart.

Records 20 episodes of the edge-push scripted baseline (Sequence E) using
set_qpos(), saving per-timestep joint commands, EE positions, observations,
rewards, and metrics.

Output: checkpoints/demos/scripted_joint_demos.npz

Usage::

    source /home/zhuzihou/dev/Genesis/.venv/bin/activate
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
    python pta/scripts/collect_joint_demos.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 7-DOF scoop-tool waypoints (from run_scripted_baseline.py)
# ---------------------------------------------------------------------------

HOME_S = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
EXTEND_FWD_S = [0.0, 0.5, 0.0, -1.8, 0.0, 1.8, 0.0]

# Edge-push waypoints
BEHIND_EP = [-0.10, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]
PUSH_END_EP = [0.40, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]


def build_edge_push_trajectory():
    """Build the dense waypoint sequence for Sequence E (edge-push).

    Returns list of (qpos_7d, settle_steps) tuples.
    """
    segments = [
        # Phase 1: Approach
        (HOME_S, EXTEND_FWD_S, 20, 1),
        (EXTEND_FWD_S, BEHIND_EP, 30, 2),
        # Phase 2: Push pass 1
        (BEHIND_EP, PUSH_END_EP, 100, 3),
        (PUSH_END_EP, BEHIND_EP, 30, 1),  # retract
        # Push pass 2
        (BEHIND_EP, PUSH_END_EP, 100, 3),
        (PUSH_END_EP, BEHIND_EP, 30, 1),  # retract
        # Push pass 3 (no retract)
        (BEHIND_EP, PUSH_END_EP, 100, 3),
    ]

    trajectory = []
    for start, end, n_steps, settle in segments:
        s = np.array(start, dtype=np.float64)
        e = np.array(end, dtype=np.float64)
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q = s * (1 - alpha) + e * alpha
            trajectory.append((q.tolist(), settle))
    return trajectory


def run_demo_episode(env, trajectory):
    """Run one demo episode, returning per-step records and episode summary.

    Returns (step_data, episode_meta) where:
      step_data: list of dicts with keys t, q_cmd, ee_pos, reward, transfer, spill, obs
      episode_meta: dict with total_reward, final_transfer, final_spill, success, n_steps
    """
    env.reset()

    # Reset reward tracking state
    env._prev_transfer_frac = 0.0
    env._prev_mean_particle_y = None
    env._success_triggered = False
    env._step_count = 0

    step_data = []
    total_reward = 0.0
    global_step = 0

    for q_list, settle_steps in trajectory:
        q_t = torch.tensor(q_list, dtype=torch.float32, device="cuda")
        env.robot.set_qpos(q_t)

        for _ in range(settle_steps):
            env.scene.step()
            env._step_count += 1
            global_step += 1

            # EE position
            ee_pos = env._ee_link.get_pos()
            if ee_pos.dim() > 1:
                ee_pos = ee_pos.squeeze(0)

            # Observations
            obs = env.get_observations()
            obs_vec = obs["proprio"].cpu().numpy()

            # Reward
            reward = env.compute_reward()
            total_reward += reward

            # Metrics
            metrics = env.compute_metrics()

            step_data.append({
                "t": global_step,
                "q_cmd": np.array(q_list, dtype=np.float32),
                "ee_pos": ee_pos.cpu().numpy(),
                "reward": reward,
                "transferred_mass_frac": metrics["transfer_efficiency"],
                "spill_ratio": metrics["spill_ratio"],
                "obs": obs_vec,
            })

    # Settle phase (80 steps, matching Sequence E)
    last_q = trajectory[-1][0]
    for _ in range(80):
        env.scene.step()
        env._step_count += 1
        global_step += 1

        ee_pos = env._ee_link.get_pos()
        if ee_pos.dim() > 1:
            ee_pos = ee_pos.squeeze(0)

        obs = env.get_observations()
        obs_vec = obs["proprio"].cpu().numpy()
        reward = env.compute_reward()
        total_reward += reward
        metrics = env.compute_metrics()

        step_data.append({
            "t": global_step,
            "q_cmd": np.array(last_q, dtype=np.float32),
            "ee_pos": ee_pos.cpu().numpy(),
            "reward": reward,
            "transferred_mass_frac": metrics["transfer_efficiency"],
            "spill_ratio": metrics["spill_ratio"],
            "obs": obs_vec,
        })

    final_metrics = env.compute_metrics()
    episode_meta = {
        "total_reward": total_reward,
        "final_transfer": final_metrics["transfer_efficiency"],
        "final_spill": final_metrics["spill_ratio"],
        "success": int(final_metrics["success_rate"]),
        "n_steps": global_step,
    }
    return step_data, episode_meta


def main():
    n_episodes = 20

    print("=" * 70)
    print("Collect Scripted Joint Demos")
    print("=" * 70)

    # Build environment
    print(f"[1/3] Building environment (scoop, edge_push) ...")
    t0 = time.time()

    from pta.envs.tasks.scoop_transfer import ScoopTransferTask

    scene_cfg = {
        "particle_material": "sand",
        "n_envs": 0,
        "tool_type": "scoop",
    }
    task_cfg = {"horizon": 2000}  # high horizon so is_done() won't trigger early

    env = ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)
    print(f"  Built in {time.time() - t0:.1f}s, {env._total_particles} particles")

    # Build trajectory
    trajectory = build_edge_push_trajectory()
    total_physics_steps = sum(s for _, s in trajectory) + 80
    print(f"  {len(trajectory)} waypoints, {total_physics_steps} physics steps/episode")

    # Collect demos
    print(f"[2/3] Collecting {n_episodes} episodes ...")
    all_episodes_steps = []
    all_meta = []

    for ep in range(n_episodes):
        t_ep = time.time()
        step_data, meta = run_demo_episode(env, trajectory)
        dt = time.time() - t_ep
        all_episodes_steps.append(step_data)
        all_meta.append(meta)
        print(
            f"  ep {ep+1:2d}/{n_episodes}  "
            f"transfer={meta['final_transfer']:.4f}  "
            f"spill={meta['final_spill']:.4f}  "
            f"reward={meta['total_reward']:.2f}  "
            f"success={meta['success']}  "
            f"steps={meta['n_steps']}  "
            f"({dt:.1f}s)"
        )

    # Pack into arrays
    print("[3/3] Saving ...")

    # Determine dimensions from first episode
    n_steps_max = max(len(ep) for ep in all_episodes_steps)
    q_dim = 7
    ee_dim = 3
    obs_dim = all_episodes_steps[0][0]["obs"].shape[0]

    # Per-step arrays: (n_episodes, max_steps, dim)
    q_cmd_arr = np.zeros((n_episodes, n_steps_max, q_dim), dtype=np.float32)
    ee_pos_arr = np.zeros((n_episodes, n_steps_max, ee_dim), dtype=np.float32)
    obs_arr = np.zeros((n_episodes, n_steps_max, obs_dim), dtype=np.float32)
    reward_arr = np.zeros((n_episodes, n_steps_max), dtype=np.float32)
    transfer_arr = np.zeros((n_episodes, n_steps_max), dtype=np.float32)
    spill_arr = np.zeros((n_episodes, n_steps_max), dtype=np.float32)
    step_count_arr = np.zeros((n_episodes,), dtype=np.int32)

    for ep_idx, step_data in enumerate(all_episodes_steps):
        n = len(step_data)
        step_count_arr[ep_idx] = n
        for t, sd in enumerate(step_data):
            q_cmd_arr[ep_idx, t] = sd["q_cmd"]
            ee_pos_arr[ep_idx, t] = sd["ee_pos"]
            obs_arr[ep_idx, t] = sd["obs"]
            reward_arr[ep_idx, t] = sd["reward"]
            transfer_arr[ep_idx, t] = sd["transferred_mass_frac"]
            spill_arr[ep_idx, t] = sd["spill_ratio"]

    # Episode-level metadata
    meta_total_reward = np.array([m["total_reward"] for m in all_meta], dtype=np.float32)
    meta_final_transfer = np.array([m["final_transfer"] for m in all_meta], dtype=np.float32)
    meta_success = np.array([m["success"] for m in all_meta], dtype=np.int32)

    out_dir = "checkpoints/demos"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scripted_joint_demos.npz")

    np.savez_compressed(
        out_path,
        q_cmd=q_cmd_arr,
        ee_pos=ee_pos_arr,
        obs=obs_arr,
        reward=reward_arr,
        transfer=transfer_arr,
        spill=spill_arr,
        step_count=step_count_arr,
        meta_total_reward=meta_total_reward,
        meta_final_transfer=meta_final_transfer,
        meta_success=meta_success,
    )

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path}")
    print(f"  Shape: ({n_episodes}, {n_steps_max}, ...) obs_dim={obs_dim}")
    print(f"  Size:  {file_size_mb:.2f} MB")

    # Summary stats
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Episodes:         {n_episodes}")
    print(f"  Steps/episode:    {n_steps_max}")
    print(f"  Obs dim:          {obs_dim}")
    print(f"  Avg transfer:     {meta_final_transfer.mean():.4f} +/- {meta_final_transfer.std():.4f}")
    print(f"  Avg reward:       {meta_total_reward.mean():.2f} +/- {meta_total_reward.std():.2f}")
    print(f"  Success rate:     {meta_success.mean():.2f}")
    print(f"  File size:        {file_size_mb:.2f} MB")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
