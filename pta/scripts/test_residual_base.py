"""Test scripted base policy (residual=0): does it push particles?"""
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
from pta.training.rl.train_teacher import make_env

env = make_env(
    task_config={"horizon": 2000},
    scene_config={"task_layout": "edge_push"},
    seed=42,
    use_reduced_action=True,
    action_repeat=25,
    use_residual=True,
    residual_scale=0.3,
)

# Test 1: Zero residual (pure scripted base)
print("=== SCRIPTED BASE ONLY (residual=0) ===")
obs, info = env.reset()
total_reward = 0.0
for step in range(80):
    action = np.zeros(env.action_space.shape)  # zero residual
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if step in {0, 10, 20, 30, 40, 50, 60, 70, 79}:
        # Unwrap to task for metrics
        task = env
        while hasattr(task, 'env'):
            task = task.env
        if hasattr(task, 'task'):
            task = task.task
        metrics = task.compute_metrics() if hasattr(task, 'compute_metrics') else {}
        print(f"  step={step:3d}  reward={reward:8.4f}  "
              f"transfer={metrics.get('transfer_efficiency',0):.4f}  "
              f"spill={metrics.get('spill_ratio',0):.4f}  "
              f"success={metrics.get('success',0):.0f}")
    done = terminated or truncated
    if done:
        break
print(f"  TOTAL: reward={total_reward:.4f}, steps={step+1}")

# Test 2: Random residual
print("\n=== RANDOM RESIDUAL ===")
obs, info = env.reset()
total_reward = 0.0
for step in range(80):
    action = env.action_space.sample()  # random residual
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    if done:
        break
task = env
while hasattr(task, 'env'):
    task = task.env
if hasattr(task, 'task'):
    task = task.task
metrics = task.compute_metrics() if hasattr(task, 'compute_metrics') else {}
print(f"  TOTAL: reward={total_reward:.4f}, steps={step+1}")
print(f"  transfer={metrics.get('transfer_efficiency',0):.4f}  "
      f"spill={metrics.get('spill_ratio',0):.4f}  "
      f"success={metrics.get('success',0):.0f}")
