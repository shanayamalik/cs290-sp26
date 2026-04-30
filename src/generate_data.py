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
    driver_mix : str — non-ego driver mixture used for this dataset
    npc_driver_types : list[str] — assigned non-ego archetypes for the episode

Design decisions (Claude + Copilot review, 2026-04-29):
  - Save raw 5×5 obs: can always flatten later, can't un-flatten.
  - Keep crashed episodes with a flag: lets you filter at training time,
    analyze crash rates per driver type, and optionally use early steps of
    crashed episodes as training signal (drop last ~3 steps).
  - Save ego_speed + d_min: enables trajectory plots without decoding obs.
  - theta_name saved as string: human-readable and unambiguous.

Usage:
    python3 src/generate_data.py --episodes 200 --mix all_normal
    python3 src/generate_data.py --episodes 200 --all-mixes
    python3 src/generate_data.py --episodes 5 --all-mixes  # quick sanity check
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import gymnasium as gym
import highway_env  # noqa: F401 — registers merge-v0
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from driver_types import make_cautious, make_normal, make_aggressive
from mpc_expert import mpc_select_action
from reward import CAUTIOUS, NORMAL, AGGRESSIVE


def _patch_merge_env_continuous_rewards() -> None:
    """
    highway-env 1.10.2 assumes discrete actions in MergeEnv._rewards.
    ContinuousAction passes an array, which makes `action in [0, 2]` raise.
    For continuous control, lane_change_reward should be false because this
    project fixes steering at 0 and uses longitudinal gap management only.
    """
    from highway_env.envs.merge_env import MergeEnv

    if getattr(MergeEnv, "_cs290_continuous_patch", False):
        return

    original_rewards = MergeEnv._rewards

    def rewards_continuous_safe(self, action):
        if isinstance(action, np.ndarray):
            action = -1
        return original_rewards(self, action)

    MergeEnv._rewards = rewards_continuous_safe
    MergeEnv._cs290_continuous_patch = True

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EPISODES = 200
MAX_STEPS = 50          # cap per episode; prevents runaway episodes if env doesn't terminate

DRIVER_FN_MAP = {
    "cautious": make_cautious,
    "normal": make_normal,
    "aggressive": make_aggressive,
}

DRIVER_MIXES = {
    "all_normal": {"normal": 1.00, "cautious": 0.00, "aggressive": 0.00},
    "default_mix": {"normal": 0.60, "cautious": 0.20, "aggressive": 0.20},
    "cautious_heavy": {"normal": 0.40, "cautious": 0.50, "aggressive": 0.10},
    "aggressive_heavy": {"normal": 0.40, "cautious": 0.10, "aggressive": 0.50},
}

OUTPUT_PATHS = {
    mix_name: Path(f"data/expert_dataset_{mix_name}.pkl")
    for mix_name in DRIVER_MIXES
}

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


def _sample_driver_type(driver_mix: dict) -> str:
    """Sample one driver type name from a mixture dict."""
    names = list(driver_mix.keys())
    weights = list(driver_mix.values())
    return random.choices(names, weights=weights, k=1)[0]


def _assign_npc_driver_types(env, driver_mix: dict) -> list:
    """Assign non-ego driver archetypes from a weighted mixture."""
    assigned = []
    for vehicle in env.unwrapped.road.vehicles[1:]:
        driver_type = _sample_driver_type(driver_mix)
        DRIVER_FN_MAP[driver_type](vehicle)
        assigned.append(driver_type)
    return assigned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(n_episodes: int, output_path: Path, mix_name: str) -> None:
    driver_mix = DRIVER_MIXES[mix_name]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _patch_merge_env_continuous_rewards()
    env = gym.make("merge-v0", config=ENV_CONFIG)
    dataset = []
    crash_count = 0

    print(f"\nGenerating dataset: {mix_name}")
    print(f"  Driver mix: {driver_mix}")
    print(f"  Output    : {output_path}")

    for episode_id in range(n_episodes):
        obs, _ = env.reset()

        # Randomly assign a driver archetype to the ego for this episode
        theta_name = random.choice(list(THETA_MAP.keys()))
        theta = THETA_MAP[theta_name]

        # Randomly assign archetypes to non-ego vehicles
        npc_driver_types = _assign_npc_driver_types(env, driver_mix)

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
                "driver_mix": mix_name,
                "npc_driver_types": npc_driver_types.copy(),
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
                f"npc: {','.join(npc_driver_types):<30} | "
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
    if dataset:
        print(f"  Obs shape         : {dataset[0]['obs'].shape}")
        print(f"  Action shape      : {dataset[0]['action'].shape}")
        print(f"  Keys              : {list(dataset[0].keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument(
        "--mix",
        choices=list(DRIVER_MIXES.keys()),
        default="default_mix",
        help="Non-ego driver mixture to use for a single dataset.",
    )
    parser.add_argument(
        "--all-mixes",
        action="store_true",
        help="Generate one dataset for every named non-ego driver mixture.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for a single --mix run. Ignored with --all-mixes.",
    )
    args = parser.parse_args()

    if args.all_mixes:
        for mix_name, output_path in OUTPUT_PATHS.items():
            generate(args.episodes, output_path, mix_name)
    else:
        output_path = Path(args.output) if args.output else OUTPUT_PATHS[args.mix]
        generate(args.episodes, output_path, args.mix)
