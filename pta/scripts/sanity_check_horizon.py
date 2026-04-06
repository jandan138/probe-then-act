"""Sanity check: run 1 random episode, print physics steps vs policy steps."""
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from pta.training.rl.train_teacher import make_env

env = make_env(
    task_config={"horizon": 2000},
    scene_config={"task_layout": "edge_push"},
    seed=42,
    use_reduced_action=True,
    action_repeat=25,
)
obs, info = env.reset()
policy_steps = 0
done = False
total_reward = 0.0
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    policy_steps += 1
    total_reward += reward
    done = terminated or truncated

physics_steps = info.get("step", "?")
print(f"Episode finished:")
print(f"  physics_steps (env._step_count): {physics_steps}")
print(f"  policy_steps (outer loop):       {policy_steps}")
print(f"  expected policy_steps:           {2000 // 25} = 2000/25")
print(f"  total_reward:                    {total_reward:.4f}")
print(f"  match: {'OK' if policy_steps == 2000 // 25 else 'MISMATCH!'}")
