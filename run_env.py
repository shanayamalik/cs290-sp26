import time
import os
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway-env environments


CONFIG = {
    # Keep the observation focused on 3 vehicles: ego + 2 nearby vehicles.
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
    },
    # Highway-Env exposes this config key, but merge-v0 1.10.2 still hard-codes
    # extra cars internally. The helper below prunes the scene to 3 total cars.
    "vehicles_count": 3,
    "controlled_vehicles": 1,
    "duration": 20,
    # Make the render window wide enough to see the merge area.
    "screen_width": 1200,
    "screen_height": 300,
    "scaling": 7,
    "centering_position": [0.3, 0.5],
}


def reset_three_vehicle_scene(env):
    obs, info = env.reset()

    ego = env.unwrapped.vehicle
    other_vehicles = [vehicle for vehicle in env.unwrapped.road.vehicles if vehicle is not ego]
    merging_vehicles = [
        vehicle
        for vehicle in other_vehicles
        if vehicle.lane_index and vehicle.lane_index[0] in {"j", "k"}
    ]
    highway_vehicles = [
        vehicle
        for vehicle in other_vehicles
        if vehicle not in merging_vehicles
    ]

    kept_vehicles = [ego]
    if highway_vehicles:
        kept_vehicles.append(highway_vehicles[0])
    if merging_vehicles:
        kept_vehicles.append(merging_vehicles[0])

    env.unwrapped.road.vehicles = kept_vehicles
    obs = env.unwrapped.observation_type.observe()
    return obs, info


def main():
    env = gym.make("merge-v0", render_mode="human", config=CONFIG)

    obs, info = reset_three_vehicle_scene(env)
    print("Observation shape:", obs.shape, flush=True)
    print("Action space:", env.action_space, flush=True)
    print("Initial vehicles:", len(env.unwrapped.road.vehicles), flush=True)

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Slow the loop down so the movement is visible in the render window.
        time.sleep(0.05)

        if terminated or truncated:
            print(f"Episode ended at step {step}; resetting.", flush=True)
            obs, info = reset_three_vehicle_scene(env)

    env.close()


if __name__ == "__main__":
    main()
