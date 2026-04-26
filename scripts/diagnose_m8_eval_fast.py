"""Fast M8 diagnosis: run a single short deterministic rollout and print actions/obs."""

import sys
sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO

CHECKPOINT_DIR = Path("/home/zhuzihou/dev/probe-then-act/checkpoints/m8_teacher_seed0")
meta_path = CHECKPOINT_DIR / "scoop_transfer_teacher_final.json"
import json
with open(meta_path) as f:
    meta = json.load(f)
cfg = meta["config"]

seed = cfg["seed"]

def _make_env(seed_offset=0):
    from pta.training.rl.train_teacher import make_env as _make
    return _make(
        task_config=cfg.get("task_config"),
        scene_config=cfg.get("scene_config"),
        seed=seed + 1000 + seed_offset,
        use_reduced_action=cfg.get("use_reduced_action", False),
        action_repeat=cfg.get("action_repeat", 1),
        use_residual=cfg.get("use_residual", False),
        residual_scale=cfg.get("residual_scale", 0.3),
        use_joint_residual=cfg.get("use_joint_residual", True),
        joint_residual_scale=cfg.get("joint_residual_scale", 0.2),
        joint_residual_trajectory=cfg.get("joint_residual_trajectory", "edge_push"),
        use_privileged=cfg.get("use_privileged", True),
    )

# Pick one checkpoint
ckpt_path = CHECKPOINT_DIR / "scoop_transfer_teacher_50000_steps.zip"
model = PPO.load(ckpt_path)
env = _make_env(seed_offset=0)

obs, info = env.reset()
print("Initial obs shape:", obs.shape)
print("Initial obs first 20:", obs[:20])
print("Privileged features:", obs[-7:])

# Access underlying task to get true EE position
task = env.env
while hasattr(task, "env"):
    if hasattr(task, "task"):
        break
    task = task.env

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 50 == 0 or step < 10:
        ee_pos = task.task._ee_link.get_pos()
        if ee_pos.dim() > 1:
            ee_pos = ee_pos.squeeze(0)
        ee_pos_np = ee_pos.cpu().numpy()
        qpos = task.task.robot.get_qpos()
        if qpos.dim() > 1:
            qpos = qpos.squeeze(0)
        # Get particle positions
        ppos = task.task.particles.get_particles_pos()
        if ppos.dim() == 3:
            ppos = ppos[0]
        mean_y = ppos[:, 1].mean().item()
        min_z = ppos[:, 2].min().item()
        n_in_target = task.task._count_particles_in_target()
        n_spilled = task.task._count_spilled_particles()
        total = task.task._total_particles
        n_in_source = total - n_in_target - n_spilled
        print(f"Step {step:3d}: action={np.round(action, 3)}  reward={reward:8.3f}  "
              f"source={n_in_source}  target={n_in_target}  spill={n_spilled}  "
              f"ee_pos={np.round(ee_pos_np, 3)}  mean_y={mean_y:.3f}  min_z={min_z:.3f}")
    if terminated or truncated:
        print(f"Terminated/truncated at step {step}")
        break

# Final summary
final_info = info
print("\n=== FINAL ===")
print(f"Total steps: {step}")
print(f"Success: {final_info.get('success_rate', -1)}")
print(f"Transfer efficiency: {final_info.get('transfer_efficiency', -1):.3f}")
print(f"Spill ratio: {final_info.get('spill_ratio', -1):.3f}")
print(f"n_in_target: {final_info.get('n_in_target', -1)}")
print(f"n_spilled: {final_info.get('n_spilled', -1)}")

# Now test what a ZERO action does (pure base trajectory)
print("\n=== ZERO ACTION BASELINE ===")
env2 = _make_env(seed_offset=1)
obs2, _ = env2.reset()
total_r = 0.0
for step in range(500):
    action = np.zeros(7, dtype=np.float32)
    obs2, reward, terminated, truncated, info = env2.step(action)
    total_r += reward
    if step % 50 == 0:
        n_in_target = info.get("n_in_target", -1)
        n_spilled = info.get("n_spilled", -1)
        print(f"Step {step:3d}: reward={reward:8.3f}  target={n_in_target}  spill={n_spilled}")
    if terminated or truncated:
        break
print(f"Zero-action total reward: {total_r:.2f}")
print(f"Zero-action final transfer={info.get('transfer_efficiency', -1):.3f}  spill={info.get('spill_ratio', -1):.3f}")

# Test random action baseline
print("\n=== RANDOM ACTION BASELINE ===")
env3 = _make_env(seed_offset=2)
obs3, _ = env3.reset()
total_r = 0.0
for step in range(500):
    action = np.random.uniform(-1, 1, size=7).astype(np.float32)
    obs3, reward, terminated, truncated, info = env3.step(action)
    total_r += reward
    if step % 50 == 0:
        n_in_target = info.get("n_in_target", -1)
        n_spilled = info.get("n_spilled", -1)
        print(f"Step {step:3d}: reward={reward:8.3f}  target={n_in_target}  spill={n_spilled}")
    if terminated or truncated:
        break
print(f"Random-action total reward: {total_r:.2f}")
print(f"Random-action final transfer={info.get('transfer_efficiency', -1):.3f}  spill={info.get('spill_ratio', -1):.3f}")
