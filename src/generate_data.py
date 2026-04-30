"""
Phase 5: Expert dataset generation.

Runs N_EPISODES of merge-v0 with the MPC expert and saves one record per
timestep. Each record is a dict:

    obs        : (5, 5) float32 array — raw observation matrix (NOT flattened;
                 flatten at training time with obs.reshape(-1))
    action     : (2,) float32 array — [acc_norm, 0.0] from mpc_select_action
    theta_name : str — "cautious" / "normal" / "aggressive"
    step       : int — timestep index within the episode
    episode_id : int — episode index
    crashed    : bool — True if env flagged a crash at episode end
    ego_speed  : float — ego vx at this step (m/s)
    d_min      : float — Euclidean distance to nearest vehicle at this step (m)

Design decisions (Claude + Copilot review, 2026-04-29):
  - Save raw 5×5 obs: can always flatten later, can't un-flatten.
  - Keep crashed episodes with a flag: lets you filter at training time,
    analyze crash rates per driver type, and optionally use early steps of
    crashed episodes as training signal (drop last ~3 steps).
  - Save ego_speed + d_min: enables trajectory plots without decoding obs.
  - theta_name saved as string: human-readable and unambiguous.

Usage:
    python3 src/generate_data.py              # 100 episodes → data/expert_dataset.pkl
    python3 src/generate_data.py --episodes 5 # quick sanity check
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 — registers merge-v0
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from driver_types import make_cautious, make_normal, make_aggressive
from mpc_expert import mpc_select_action, _extract_state, straight_line_trajectory, predict_other_responses
from reward import CAUTIOUS, NORMAL, AGGRESSIVE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EPISODES = 100
MAX_STEPS = 50          # cap per episode; prevents runaway episodes if env doesn't terminate
OUTPUT_PATH = Path("data/expert_dataset.pkl")

DRIVER_FNS = [make_cautious, make_normal, make_aggressive]
THETA_MAP = {
    "cautious": CAUTIOUS,
    "normal": NORMAL,
    "aggressive": AGGRESSIVE,
}

ENV_CONFIG = {
    "vehicles_count": 3,
    "controlled_vehicles": 1,
    "action": {"type": "ContinuousAction"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_d_min(env) -> float:
    """Euclidean distance from ego to the nearest other vehicle (m)."""
    ego = env.unwrapped.road.vehicles[0]
    others = env.unwrapped.road.vehicles[1:]
    if not others:
        return 100.0
    ego_pos = ego.position
    return float(min(
        np.linalg.norm(ego_pos - v.position) for v in others
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(n_episodes: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make("merge-v0", config=ENV_CONFIG)
    dataset = []
    crash_count = 0

    for episode_id in range(n_episodes):
        obs, _ = env.reset()

        # Randomly assign a driver archetype to the ego for this episode
        theta_name = random.choice(list(THETA_MAP.keys()))
        theta = THETA_MAP[theta_name]

        # Randomly assign archetypes to non-ego vehicles
        for v in env.unwrapped.road.vehicles[1:]:
            random.choice(DRIVER_FNS)(v)

        step = 0
        terminated = truncated = False

        while not (terminated or truncated) and step < MAX_STEPS:
            ego_speed = float(env.unwrapped.road.vehicles[0].speed)
            d_min = _get_d_min(env)

            action = mpc_select_action(env, theta=theta)

            record = {
                "obs": obs.copy(),           # (5, 5) float32
                "action": action.copy(),     # (2,) float32
                "theta_name": theta_name,
                "step": step,
                "episode_id": episode_id,
                "crashed": False,            # updated below if crash
                "ego_speed": ego_speed,
                "d_min": d_min,
            }
            dataset.append(record)

            obs, _, terminated, truncated, _ = env.step(action)
            step += 1

        # Mark all records in this episode if it ended in a crash
        if env.unwrapped.vehicle.crashed:
            crash_count += 1
            for r in dataset[-(step):]:
                r["crashed"] = True

        if True:
            print(
                f"Episode {episode_id + 1:>4}/{n_episodes} | "
                f"steps: {step:>3} | "
                f"theta: {theta_name:<10} | "
                f"crashed: {env.unwrapped.vehicle.crashed} | "
                f"dataset size: {len(dataset)}"
            )

    env.close()

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")
    print(f"  Total transitions : {len(dataset)}")
    print(f"  Episodes          : {n_episodes}")
    print(f"  Crashes           : {crash_count} ({100*crash_count/n_episodes:.1f}%)")
    print(f"  Obs shape         : {dataset[0]['obs'].shape}")
    print(f"  Action shape      : {dataset[0]['action'].shape}")
    print(f"  Keys              : {list(dataset[0].keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()
    generate(args.episodes, Path(args.output))
