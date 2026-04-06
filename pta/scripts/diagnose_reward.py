"""Diagnose reward structure: run 1 episode with zero actions, 1 with random.
Prints per-step decomposed reward at key checkpoints."""
import os, sys, torch
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
)

for label, action_fn in [("ZERO", lambda e: np.zeros(e.action_space.shape)),
                          ("RANDOM", lambda e: e.action_space.sample())]:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    checkpoints = {0, 1, 5, 10, 20, 40, 60, 79}
    done = False
    while not done:
        action = action_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        if steps in checkpoints:
            # Unwrap to task
            task = env
            while hasattr(task, 'env'):
                task = task.env
            if hasattr(task, 'task'):
                task = task.task
            metrics = task.compute_metrics() if hasattr(task, 'compute_metrics') else {}
            print(f"[{label}] step={steps:3d}  reward={reward:8.4f}  "
                  f"transfer={metrics.get('transfer_efficiency',0):.4f}  "
                  f"spill={metrics.get('spill_ratio',0):.4f}  "
                  f"success={metrics.get('success',0):.0f}")
        total_reward += reward
        steps += 1
        done = terminated or truncated
    print(f"[{label}] TOTAL: reward={total_reward:.4f}, steps={steps}")
    print()
