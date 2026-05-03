"""
Cross-evaluate behavioral-cloning policies across driver mixtures.

This script expects locally trained weights:
    models/bc_policy_all_normal.pt
    models/bc_policy_default_mix.pt
    models/bc_policy_cautious_heavy.pt
    models/bc_policy_aggressive_heavy.pt

The .pt files are intentionally local artifacts; regenerate them with
src/train_policy.py before running the table.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from eval_policy import ENV_CONFIG, TRAFFIC_MIXES, run_episodes, summarize_results
from generate_data import _patch_merge_env_continuous_rewards
from policy_network import PolicyNetwork

MIXTURES = ["all_normal", "default_mix", "cautious_heavy", "aggressive_heavy"]


class BCAction:
    """Append d_min + step, normalize, and query a trained BC model."""

    def __init__(self, env, model: PolicyNetwork, obs_mean: np.ndarray, obs_std: np.ndarray):
        self.env = env
        self.model = model
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self._step = 0
        self.total_clamp_count = 0
        self.total_action_count = 0

    def reset(self):
        self._step = 0

    def __call__(self, obs):
        ego = self.env.unwrapped.road.vehicles[0]
        others = self.env.unwrapped.road.vehicles[1:]
        d_min = float(min(np.linalg.norm(ego.position - v.position) for v in others)) if others else 100.0
        obs_aug = np.append(obs.reshape(-1), [d_min, float(self._step)]).astype(np.float32)
        obs_norm = (obs_aug - self.obs_mean) / self.obs_std
        action = self.model.predict(obs_norm)
        self.total_action_count += 1
        # Clamp action so ego speed cannot go below zero during rollout.
        ego_speed = float(ego.speed)
        if ego_speed < 2.0 and action[0] < 0:
            action[0] = max(action[0], 0.0)
            self.total_clamp_count += 1
        self._step += 1
        return action


def load_policy(model_path: Path) -> tuple[PolicyNetwork, np.ndarray, np.ndarray]:
    stats_path = model_path.with_suffix(".npz")
    if not model_path.exists() or not stats_path.exists():
        missing = model_path if not model_path.exists() else stats_path
        raise FileNotFoundError(missing)

    model = PolicyNetwork()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    stats = np.load(stats_path)
    return model, stats["mean"], stats["std"]


def evaluate_cell(model_path: Path, test_mix: str, episodes: int, seed: int) -> Optional[dict]:
    try:
        model, obs_mean, obs_std = load_policy(model_path)
    except FileNotFoundError:
        return None

    env = gym.make("merge-v0", config=ENV_CONFIG)
    action_fn = BCAction(env, model, obs_mean, obs_std)
    random.seed(seed)
    np.random.seed(seed)
    results = run_episodes(
        env,
        action_fn,
        episodes,
        seed,
        label=f"{model_path.stem} on {test_mix}",
        driver_fns=TRAFFIC_MIXES[test_mix],
        verbose=False,
    )
    env.close()
    return summarize_results(results)


def format_cell(summary: Optional[dict]) -> str:
    if summary is None:
        return "missing"
    return (
        f"{100*summary['crash_rate']:.0f}% crash, "
        f"R={summary['mean_reward']:.1f}, "
        f"v={summary['mean_speed']:.1f}, "
        f"clamp={100*summary['clamp_activation_rate']:.0f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--include-combined", action="store_true")
    args = parser.parse_args()

    _patch_merge_env_continuous_rewards()

    train_mixes = MIXTURES + (["all"] if args.include_combined else [])
    rows = []
    for train_mix in train_mixes:
        model_path = Path(args.models_dir) / f"bc_policy_{train_mix}.pt"
        row = [train_mix]
        for test_mix in MIXTURES:
            print(f"  Running: train={train_mix} | test={test_mix} ({args.episodes} eps)...",
                  flush=True)
            summary = evaluate_cell(model_path, test_mix, args.episodes, args.seed)
            print(f"    -> {format_cell(summary)}", flush=True)
            row.append(format_cell(summary))
        rows.append(row)

    header = ["trained_on \\ tested_on"] + MIXTURES
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
