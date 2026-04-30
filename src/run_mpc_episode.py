"""
Run a single episode of the MPC expert in merge-v0.

Usage (from project root, with venv active):
    python3 src/run_mpc_episode.py

Expected output:
    ego stays at y≈4.0 (main highway lane) throughout
    speed holds 15–30 m/s
    episode completes without crash
    MPC call time printed for first 5 steps (~12ms each)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import random
import gymnasium as gym
import highway_env  # noqa: F401
from driver_types import make_cautious, make_normal, make_aggressive
from mpc_expert import mpc_select_action
from reward import NORMAL

env = gym.make("merge-v0", config={
    "vehicles_count": 3,
    "controlled_vehicles": 1,
    "action": {"type": "ContinuousAction"},
})

obs, _ = env.reset()
for v in env.unwrapped.road.vehicles[1:]:
    random.choice([make_cautious, make_normal, make_aggressive])(v)

print(f"{'Step':>4}  {'y':>6}  {'vx':>6}  {'action[0]':>9}")
for step in range(25):
    action = mpc_select_action(env, theta=NORMAL)
    obs, _, terminated, truncated, _ = env.step(action)
    ego = env.unwrapped.road.vehicles[0]
    print(f"{step:>4}  {ego.position[1]:>6.2f}  {ego.speed:>6.2f}  {action[0]:>9.4f}")
    if terminated or truncated:
        status = "crash" if env.unwrapped.vehicle.crashed else "completed"
        print(f"\nEpisode {status} at step {step}.")
        break

env.close()
