import random

import gymnasium as gym
import highway_env  # noqa: F401 — registers highway environments
import time

from driver_types import make_cautious, make_normal, make_aggressive

DRIVER_FNS = [make_cautious, make_normal, make_aggressive]


def assign_driver_types(env: gym.Env) -> None:
    """Randomly assign a driver archetype to each non-ego vehicle."""
    for vehicle in env.unwrapped.road.vehicles[1:]:  # index 0 is always ego
        random.choice(DRIVER_FNS)(vehicle)

env = gym.make("merge-v0", render_mode="human", config={
    "vehicles_count": 3,
    "controlled_vehicles": 1,
    "lanes_count": 2,
    "duration": 20,
    "collision_reward": -10,
    "high_speed_reward": 0.4,
    "render_agent": True,
    "screen_width": 1200,
    "screen_height": 300,
    "scaling": 7,
    "centering_position": [0.3, 0.5],
    "show_trajectories": True,
    "action": {"type": "ContinuousAction"},
})

obs, info = env.reset()
assign_driver_types(env)
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print()

# env.unwrapped is needed to access highway-env internals through gymnasium wrappers
print("Vehicles on road (with assigned driver types):")
for i, v in enumerate(env.unwrapped.road.vehicles):
    print(f"  [{i}] position={v.position}, speed={v.speed:.1f} m/s, lane={v.lane_index}")

for step in range(200):
    action = env.action_space.sample()  # ContinuousAction — will be a float array
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)  # ~20fps — remove this line when generating training data
    if terminated or truncated:
        print(f"\nEpisode ended at step {step} — resetting.")
        obs, info = env.reset()
        assign_driver_types(env)

env.close()
print("\nDone. If you saw a render window with moving cars, Phase 1 is complete.")
