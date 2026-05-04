"""
Phase 8 baseline: naive independent 2-agent planning.

This is the "apply Sadigh et al. naively to a multi-car merge" baseline:
for each non-ego vehicle, predict its response to the ego independently, as if
the other non-ego vehicles were absent. The ego then scores candidate actions
against the union of those independent pairwise predictions.

Our full MPC expert uses iterative best response among all non-ego vehicles.
This baseline intentionally removes human-human interaction chains, isolating
the value of the multi-agent extension.

Usage:
    python3 src/baseline.py --episodes 20 --traffic-mix default_mix
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from best_response import idm_predict, straight_line_trajectory
from driver_types import make_aggressive, make_cautious, make_normal
from generate_data import _patch_merge_env_continuous_rewards
from mpc_expert import (
    ACC_SCALE,
    COLLISION_DIST,
    CRASH_THRESHOLD,
    FALLBACK_BRAKING,
    HORIZON,
    N_SAMPLES,
    N_WAYPOINTS,
    _build_ego_traj,
    _evaluate_sequence,
)
from reward import NORMAL

ENV_CONFIG = {
    "vehicles_count": 3,
    "controlled_vehicles": 1,
    "action": {"type": "ContinuousAction"},
}

TRAFFIC_MIXES = {
    "uniform": [make_cautious, make_normal, make_aggressive],
    "all_normal": [make_normal],
    "default_mix": [make_normal, make_normal, make_normal, make_normal,
                    make_normal, make_normal, make_cautious, make_cautious,
                    make_aggressive, make_aggressive],
    "cautious_heavy": [make_cautious, make_cautious, make_cautious, make_cautious,
                       make_cautious, make_normal, make_normal, make_normal,
                       make_normal, make_aggressive],
    "aggressive_heavy": [make_aggressive, make_aggressive, make_aggressive, make_aggressive,
                         make_aggressive, make_normal, make_normal, make_normal,
                         make_normal, make_cautious],
}

MAX_STEPS = 150
MERGE_EXIT_X = 310.0
_timing_calls = []


def predict_independent_pairwise(env, ego_trajectory: np.ndarray) -> list[np.ndarray]:
    """
    Predict each NPC as a separate 2-agent problem: ego + that NPC only.

    idm_predict accepts a list of all predicted non-ego trajectories. Passing
    only the vehicle's own straight-line trajectory means it can respond to ego,
    but it cannot treat other NPCs as lead vehicles.
    """
    predictions = []
    for vehicle in env.unwrapped.road.vehicles[1:]:
        own_guess = [straight_line_trajectory(vehicle)]
        predictions.append(idm_predict(vehicle, ego_trajectory, own_guess, own_idx=0))
    return predictions


def independent_baseline_action(env, theta: np.ndarray = NORMAL) -> np.ndarray:
    """Choose ego action using independent pairwise NPC response predictions."""
    t0 = time.perf_counter()
    ego = env.unwrapped.road.vehicles[0]
    nominal_ego_traj = straight_line_trajectory(ego)
    predicted_others = predict_independent_pairwise(env, nominal_ego_traj)

    best_acc_norm = 0.0
    best_score = -np.inf
    xp = np.linspace(0, HORIZON - 1, N_WAYPOINTS)
    xi = np.arange(HORIZON)
    structured = [
        np.zeros(N_WAYPOINTS),
        np.full(N_WAYPOINTS, -0.3),
        np.full(N_WAYPOINTS, 0.3),
    ]

    candidates = structured + [
        np.random.normal(0.0, 0.4, size=(N_WAYPOINTS,)).clip(-1.0, 1.0)
        for _ in range(N_SAMPLES)
    ]
    for waypoints in candidates:
        acc_sequence = np.interp(xi, xp, waypoints)
        score, first_acc_norm = _evaluate_sequence(ego, acc_sequence, predicted_others, theta)
        if score > best_score:
            best_score = score
            best_acc_norm = first_acc_norm

    if best_score < CRASH_THRESHOLD:
        action = FALLBACK_BRAKING.copy()
    else:
        winning_sequence = np.full(HORIZON, best_acc_norm)
        winning_ego_traj = _build_ego_traj(ego, winning_sequence)
        verified_others = predict_independent_pairwise(env, winning_ego_traj)
        verified_score, _ = _evaluate_sequence(ego, winning_sequence, verified_others, theta)
        action = FALLBACK_BRAKING.copy() if verified_score < CRASH_THRESHOLD else np.array(
            [float(best_acc_norm), 0.0], dtype=np.float32
        )

    elapsed = time.perf_counter() - t0
    _timing_calls.append(elapsed)
    if len(_timing_calls) <= 5:
        print(f"[baseline timing] call {len(_timing_calls)}: {elapsed:.3f}s  "
              f"best_score={best_score:.1f}  best_acc_norm={best_acc_norm:.3f}")
    return action


def no_reverse_clamp(env, action: np.ndarray) -> tuple[np.ndarray, bool]:
    """Match data-generation safety: do not allow env.step to reverse ego."""
    ego_speed = float(env.unwrapped.road.vehicles[0].speed)
    min_acc_norm = -(ego_speed - 0.05) / ACC_SCALE
    if action[0] < min_acc_norm:
        action = action.copy()
        action[0] = float(np.clip(min_acc_norm, -1.0, 1.0))
        return action, True
    return action, False


def min_distance(env) -> float:
    ego = env.unwrapped.road.vehicles[0]
    others = env.unwrapped.road.vehicles[1:]
    if not others:
        return 100.0
    return float(min(np.linalg.norm(ego.position - v.position) for v in others))


def min_ttc(env) -> float:
    """
    Approximate longitudinal time-to-collision in ego lane.

    Returns inf when no vehicle is closing on ego in the same lane.
    """
    ego = env.unwrapped.road.vehicles[0]
    ttc_values = []
    for v in env.unwrapped.road.vehicles[1:]:
        if abs(float(v.position[1] - ego.position[1])) > 4.0:
            continue
        dx = float(v.position[0] - ego.position[0])
        rel_v = float(ego.speed - v.speed)
        if dx > 0 and rel_v > 0:
            ttc_values.append(dx / rel_v)
        elif dx < 0 and rel_v < 0:
            ttc_values.append(abs(dx / rel_v))
    return float(min(ttc_values)) if ttc_values else float("inf")


def run_baseline(episodes: int, traffic_mix: str, seed: int) -> list[dict]:
    _patch_merge_env_continuous_rewards()
    env = gym.make("merge-v0", config=ENV_CONFIG)
    rng = random.Random(seed)
    driver_fns = TRAFFIC_MIXES[traffic_mix]
    results = []

    for ep in range(episodes):
        np.random.seed(seed + ep)
        random.seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        for vehicle in env.unwrapped.road.vehicles[1:]:
            rng.choice(driver_fns)(vehicle)

        steps = 0
        total_reward = 0.0
        speeds = []
        dmins = []
        ttcs = []
        max_x = float(env.unwrapped.road.vehicles[0].position[0])
        first_merge_step = None
        first_crash_step = None
        clamp_count = 0
        terminated = truncated = False

        while not (terminated or truncated) and steps < MAX_STEPS:
            ego = env.unwrapped.road.vehicles[0]
            speeds.append(float(ego.speed))
            dmins.append(min_distance(env))
            ttcs.append(min_ttc(env))
            max_x = max(max_x, float(ego.position[0]))
            if first_merge_step is None and max_x > MERGE_EXIT_X:
                first_merge_step = steps

            action = independent_baseline_action(env, theta=NORMAL)
            action, clamped = no_reverse_clamp(env, action)
            clamp_count += int(clamped)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            if env.unwrapped.vehicle.crashed and first_crash_step is None:
                first_crash_step = steps

        ego = env.unwrapped.road.vehicles[0]
        max_x = max(max_x, float(ego.position[0]))
        if first_merge_step is None and max_x > MERGE_EXIT_X:
            first_merge_step = steps
        crashed = bool(env.unwrapped.vehicle.crashed)
        merge_success = (not crashed) and first_merge_step is not None
        result = {
            "episode": ep,
            "steps": steps,
            "crashed": crashed,
            "merge_success": merge_success,
            "merge_step": first_merge_step,
            "max_x": max_x,
            "reward": total_reward,
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "min_distance": float(np.min(dmins)) if dmins else 100.0,
            "min_ttc": float(np.min(ttcs)) if ttcs else float("inf"),
            "clamp_rate": clamp_count / max(steps, 1),
            "first_crash_step": first_crash_step,
        }
        results.append(result)
        status = "MERGE OK" if merge_success else ("CRASH" if crashed else "SHORT")
        print(
            f"ep{ep:02d}: steps={steps:3d} max_x={max_x:6.1f} "
            f"crashed={crashed:<5} avg_v={result['avg_speed']:5.1f} "
            f"clamp={100*result['clamp_rate']:4.1f}% [{status}]"
        )

    env.close()
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    finite_ttc = [r["min_ttc"] for r in results if np.isfinite(r["min_ttc"])]
    return {
        "episodes": n,
        "merge_success_rate": sum(r["merge_success"] for r in results) / n,
        "crash_rate": sum(r["crashed"] for r in results) / n,
        "timeout_rate": sum((not r["merge_success"]) and (not r["crashed"]) for r in results) / n,
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_speed": float(np.mean([r["avg_speed"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_merge_step": float(np.mean([r["merge_step"] for r in results if r["merge_step"] is not None]))
        if any(r["merge_step"] is not None for r in results) else float("nan"),
        "mean_min_distance": float(np.mean([r["min_distance"] for r in results])),
        "mean_min_ttc": float(np.mean(finite_ttc)) if finite_ttc else float("inf"),
        "clamp_rate": float(np.mean([r["clamp_rate"] for r in results])),
    }


def print_summary(summary: dict) -> None:
    print("\nIndependent 2-agent baseline summary")
    print(f"  Episodes           : {summary['episodes']}")
    print(f"  Merge success rate : {100*summary['merge_success_rate']:.1f}%")
    print(f"  Crash rate         : {100*summary['crash_rate']:.1f}%")
    print(f"  Timeout/short rate : {100*summary['timeout_rate']:.1f}%")
    print(f"  Mean reward        : {summary['mean_reward']:.2f}")
    print(f"  Mean speed         : {summary['mean_speed']:.2f} m/s")
    print(f"  Mean steps         : {summary['mean_steps']:.1f}")
    print(f"  Mean merge step    : {summary['mean_merge_step']:.1f}")
    print(f"  Mean min distance  : {summary['mean_min_distance']:.2f} m")
    print(f"  Mean min TTC       : {summary['mean_min_ttc']:.2f} s")
    print(f"  Clamp rate         : {100*summary['clamp_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--traffic-mix", choices=list(TRAFFIC_MIXES.keys()), default="default_mix")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Independent 2-agent planning baseline")
    print(f"  traffic_mix={args.traffic_mix} episodes={args.episodes} seed={args.seed}")
    print(f"  merge success threshold: x > {MERGE_EXIT_X:.0f}m")
    results = run_baseline(args.episodes, args.traffic_mix, args.seed)
    print_summary(summarize(results))


if __name__ == "__main__":
    main()
