"""
Phase 9 final evaluator.

Runs the same deterministic seeds/traffic mix across methods and reports:
merge success, crash rate, short/timeouts, speed, merge step, distance, TTC,
and clamp rate.

Methods:
  - bc       : behavioral cloning policy
  - ppo      : PPO fine-tuned policy
  - baseline : independent 2-agent planning baseline
  - mpc      : full multi-agent MPC expert

Examples:
    python3 src/evaluate.py --episodes 20 --traffic-mix default_mix
    python3 src/evaluate.py --methods baseline mpc --episodes 20
    python3 src/evaluate.py --ppo-model models/ppo_500k_v2_merge.zip
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent))

from baseline import (
    ENV_CONFIG,
    MAX_STEPS,
    MERGE_EXIT_X,
    TRAFFIC_MIXES,
    independent_baseline_action,
    min_distance,
    min_ttc,
    no_reverse_clamp,
)
from generate_data import _patch_merge_env_continuous_rewards
from mpc_expert import mpc_select_action
from policy_network import PolicyNetwork
from reward import NORMAL
from rl_finetune import TanhMeanActorCriticPolicy, load_bc_stats, make_env


def assign_traffic(env, driver_fns, rng: random.Random) -> None:
    for vehicle in env.unwrapped.road.vehicles[1:]:
        rng.choice(driver_fns)(vehicle)


class BCAction:
    def __init__(self, env, model_path: Path):
        self.env = env
        self.model = PolicyNetwork()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        stats = np.load(model_path.with_suffix(".npz"))
        self.obs_mean = stats["mean"]
        self.obs_std = stats["std"]
        self.step = 0
        self.total_action_count = 0
        self.total_clamp_count = 0

    def reset(self):
        self.step = 0

    def __call__(self, obs):
        ego = self.env.unwrapped.road.vehicles[0]
        others = self.env.unwrapped.road.vehicles[1:]
        d_min = float(min(np.linalg.norm(ego.position - v.position) for v in others)) if others else 100.0
        obs_aug = np.append(obs.reshape(-1), [d_min, float(self.step)]).astype(np.float32)
        obs_norm = (obs_aug - self.obs_mean) / self.obs_std
        action = self.model.predict(obs_norm)
        self.total_action_count += 1
        if float(ego.speed) < 2.0 and action[0] < 0:
            action[0] = max(action[0], 0.0)
            self.total_clamp_count += 1
        self.step += 1
        return action


def summarize_episode(ep: int, steps: int, crashed: bool, merge_step,
                      max_x: float, total_reward: float, speeds: list,
                      dmins: list, ttcs: list, clamp_count: int,
                      action_count: int, first_crash_step) -> dict:
    return {
        "episode": ep,
        "steps": steps,
        "crashed": crashed,
        "merge_success": (not crashed) and merge_step is not None,
        "merge_step": merge_step,
        "max_x": max_x,
        "reward": total_reward,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "min_distance": float(np.min(dmins)) if dmins else 100.0,
        "min_ttc": float(np.min(ttcs)) if ttcs else float("inf"),
        "clamp_rate": clamp_count / max(action_count, 1),
        "first_crash_step": first_crash_step,
    }


def evaluate_raw_method(method: str, episodes: int, traffic_mix: str, seed: int,
                        bc_model: Path = None) -> list[dict]:
    _patch_merge_env_continuous_rewards()
    env = gym.make("merge-v0", config=ENV_CONFIG)
    rng = random.Random(seed)
    driver_fns = TRAFFIC_MIXES[traffic_mix]
    results = []

    if method == "bc":
        action_fn = BCAction(env, bc_model)
    elif method == "baseline":
        action_fn = lambda _obs: independent_baseline_action(env, theta=NORMAL)
    elif method == "mpc":
        action_fn = lambda _obs: mpc_select_action(env, theta=NORMAL)
    else:
        raise ValueError(method)

    for ep in range(episodes):
        np.random.seed(seed + ep)
        random.seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        assign_traffic(env, driver_fns, rng)
        if hasattr(action_fn, "reset"):
            action_fn.reset()

        steps = 0
        total_reward = 0.0
        speeds, dmins, ttcs = [], [], []
        max_x = float(env.unwrapped.road.vehicles[0].position[0])
        merge_step = None
        first_crash_step = None
        clamp_count = 0
        action_count = 0
        terminated = truncated = False

        while not (terminated or truncated) and steps < MAX_STEPS:
            ego = env.unwrapped.road.vehicles[0]
            speeds.append(float(ego.speed))
            dmins.append(min_distance(env))
            ttcs.append(min_ttc(env))
            max_x = max(max_x, float(ego.position[0]))
            if merge_step is None and max_x > MERGE_EXIT_X:
                merge_step = steps

            action = action_fn(obs)
            if method in {"baseline", "mpc"}:
                action, clamped = no_reverse_clamp(env, action)
                clamp_count += int(clamped)
                action_count += 1
            else:
                clamp_count = action_fn.total_clamp_count
                action_count = action_fn.total_action_count

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            if env.unwrapped.vehicle.crashed and first_crash_step is None:
                first_crash_step = steps

        ego = env.unwrapped.road.vehicles[0]
        max_x = max(max_x, float(ego.position[0]))
        if merge_step is None and max_x > MERGE_EXIT_X:
            merge_step = steps
        crashed = bool(env.unwrapped.vehicle.crashed)
        results.append(summarize_episode(
            ep, steps, crashed, merge_step, max_x, total_reward,
            speeds, dmins, ttcs, clamp_count, action_count, first_crash_step,
        ))

    env.close()
    return results


def evaluate_ppo(ppo_model: Path, ppo_bc_model: Path, episodes: int,
                 traffic_mix: str, seed: int) -> list[dict]:
    obs_mean, obs_std = load_bc_stats(ppo_bc_model)
    model = PPO.load(
        str(ppo_model),
        custom_objects={"policy_class": TanhMeanActorCriticPolicy},
    )
    expected_dim = int(np.prod(model.observation_space.shape))
    if expected_dim != len(obs_mean):
        raise ValueError(
            f"PPO model expects {expected_dim}-dim obs but current PPO wrapper "
            f"uses {len(obs_mean)} dims. This usually means the zip was trained "
            "before ego_speed was added; use the final v3 PPO model."
        )
    env = make_env(obs_mean, obs_std, traffic_mix, seed)
    results = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        steps = 0
        total_reward = 0.0
        speeds, dmins, ttcs = [], [], []
        max_x = float(env.unwrapped.road.vehicles[0].position[0])
        merge_step = None
        first_crash_step = None
        clamp_before = env.total_clamp_count
        action_before = env.total_action_count
        done = False

        while not done:
            ego = env.unwrapped.road.vehicles[0]
            speeds.append(float(ego.speed))
            dmins.append(min_distance(env))
            ttcs.append(min_ttc(env))
            max_x = max(max_x, float(ego.position[0]))
            if merge_step is None and max_x > MERGE_EXIT_X:
                merge_step = steps

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1
            if info.get("crashed", False) and first_crash_step is None:
                first_crash_step = steps

        ego = env.unwrapped.road.vehicles[0]
        max_x = max(max_x, float(ego.position[0]))
        if merge_step is None and max_x > MERGE_EXIT_X:
            merge_step = steps
        crashed = first_crash_step is not None
        results.append(summarize_episode(
            ep, steps, crashed, merge_step, max_x, total_reward,
            speeds, dmins, ttcs,
            env.total_clamp_count - clamp_before,
            env.total_action_count - action_before,
            first_crash_step,
        ))

    env.close()
    return results


def summarize_method(method: str, results: list[dict]) -> dict:
    n = len(results)
    finite_ttc = [r["min_ttc"] for r in results if np.isfinite(r["min_ttc"])]
    merge_steps = [r["merge_step"] for r in results if r["merge_step"] is not None]
    return {
        "method": method,
        "episodes": n,
        "merge_success_rate": sum(r["merge_success"] for r in results) / n,
        "crash_rate": sum(r["crashed"] for r in results) / n,
        "short_timeout_rate": sum((not r["merge_success"]) and (not r["crashed"]) for r in results) / n,
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_speed": float(np.mean([r["avg_speed"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_merge_step": float(np.mean(merge_steps)) if merge_steps else float("nan"),
        "mean_min_distance": float(np.mean([r["min_distance"] for r in results])),
        "mean_min_ttc": float(np.mean(finite_ttc)) if finite_ttc else float("inf"),
        "clamp_rate": float(np.mean([r["clamp_rate"] for r in results])),
    }


def print_table(summaries: list[dict]) -> None:
    headers = [
        "method", "merge_success_rate", "crash_rate", "short_timeout_rate",
        "mean_speed", "mean_merge_step", "mean_min_distance", "mean_min_ttc",
        "clamp_rate",
    ]
    print("\nFinal evaluation summary")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for s in summaries:
        row = [
            s["method"],
            f"{100*s['merge_success_rate']:.1f}%",
            f"{100*s['crash_rate']:.1f}%",
            f"{100*s['short_timeout_rate']:.1f}%",
            f"{s['mean_speed']:.2f}",
            f"{s['mean_merge_step']:.1f}",
            f"{s['mean_min_distance']:.2f}",
            f"{s['mean_min_ttc']:.2f}",
            f"{100*s['clamp_rate']:.1f}%",
        ]
        print("| " + " | ".join(row) + " |")


def write_csv(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nSaved summary CSV to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bc", "ppo", "baseline", "mpc"],
        default=["bc", "ppo", "baseline", "mpc"],
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--traffic-mix", choices=list(TRAFFIC_MIXES.keys()), default="default_mix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-model", type=str, default="models/bc_policy_default_mix.pt")
    parser.add_argument("--ppo-model", type=str, default="models/ppo_500k_v2_merge.zip")
    parser.add_argument("--ppo-bc-model", type=str, default="models/bc_policy_default_mix.pt")
    parser.add_argument("--output", type=str, default="diagnostics/final_evaluation_summary.csv")
    args = parser.parse_args()

    summaries = []
    for method in args.methods:
        print(f"\n=== Evaluating {method} ({args.episodes} eps, {args.traffic_mix}) ===")
        if method == "bc":
            bc_model = Path(args.bc_model)
            if not bc_model.exists() or not bc_model.with_suffix(".npz").exists():
                print(f"Skipping BC: missing {bc_model} or normalization stats.")
                continue
            results = evaluate_raw_method(method, args.episodes, args.traffic_mix, args.seed, bc_model)
        elif method == "ppo":
            ppo_model = Path(args.ppo_model)
            ppo_bc_model = Path(args.ppo_bc_model)
            if not ppo_model.exists():
                print(f"Skipping PPO: missing {ppo_model}")
                continue
            try:
                results = evaluate_ppo(ppo_model, ppo_bc_model, args.episodes, args.traffic_mix, args.seed)
            except ValueError as exc:
                print(f"Skipping PPO: {exc}")
                continue
        else:
            results = evaluate_raw_method(method, args.episodes, args.traffic_mix, args.seed)
        summaries.append(summarize_method(method, results))

    if not summaries:
        print("No methods evaluated. Check model paths.")
        return
    print_table(summaries)
    write_csv(Path(args.output), summaries)


if __name__ == "__main__":
    main()
