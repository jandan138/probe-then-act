"""Diagnose M8 teacher eval by running deterministic rollouts at checkpoints."""

import sys
sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pta.training.rl.train_teacher import make_env

CHECKPOINT_DIR = Path("/home/zhuzihou/dev/probe-then-act/checkpoints/m8_teacher_seed0")
LOG_DIR = Path("/home/zhuzihou/dev/probe-then-act/logs/m8_teacher_seed0")

# Load config from final json metadata
import json
meta_path = CHECKPOINT_DIR / "scoop_transfer_teacher_final.json"
with open(meta_path) as f:
    meta = json.load(f)
cfg = meta["config"]

seed = cfg["seed"]
task_config = cfg.get("task_config")
scene_config = cfg.get("scene_config")

use_reduced_action = cfg.get("use_reduced_action", False)
action_repeat = cfg.get("action_repeat", 1)
use_residual = cfg.get("use_residual", False)
residual_scale = cfg.get("residual_scale", 0.3)
use_joint_residual = cfg.get("use_joint_residual", False)
joint_residual_scale = cfg.get("joint_residual_scale", 0.1)
joint_residual_trajectory = cfg.get("joint_residual_trajectory", "edge_push")
use_privileged = cfg.get("use_privileged", True)

def _make_env(seed_offset=0):
    return make_env(
        task_config=task_config,
        scene_config=scene_config,
        seed=seed + 1000 + seed_offset,
        use_reduced_action=use_reduced_action,
        action_repeat=action_repeat,
        use_residual=use_residual,
        residual_scale=residual_scale,
        use_joint_residual=use_joint_residual,
        joint_residual_scale=joint_residual_scale,
        joint_residual_trajectory=joint_residual_trajectory,
        use_privileged=use_privileged,
    )

# Checkpoints to test
checkpoints = [
    50000,
    100000,
    150000,
    200000,
    250000,
    300000,
    350000,
    400000,
    450000,
    500000,
]

for ckpt_steps in checkpoints:
    ckpt_path = CHECKPOINT_DIR / f"scoop_transfer_teacher_{ckpt_steps}_steps.zip"
    if not ckpt_path.exists():
        print(f"Missing {ckpt_path}")
        continue

    model = PPO.load(ckpt_path)
    env = DummyVecEnv([lambda: _make_env(seed_offset=ckpt_steps)])

    obs = env.reset()
    total_reward = 0.0
    done = False
    step = 0
    infos = []

    while not done and step < 600:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        step += 1
        infos.append(info[0])

    # Summarize episode
    final_info = infos[-1] if infos else {}
    print(f"CKPT {ckpt_steps:6d}: reward={total_reward:8.2f}  steps={step}  "
          f"success={final_info.get('success_rate', -1):.2f}  "
          f"transfer={final_info.get('transfer_efficiency', -1):.3f}  "
          f"spill={final_info.get('spill_ratio', -1):.3f}  "
          f"n_in_target={final_info.get('n_in_target', -1)}")

    env.close()
