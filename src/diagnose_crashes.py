"""
Mini crash-diagnostic study for the MPC expert.

This script is intentionally separate from generate_data.py so ablations are
reversible and do not change the main expert dataset path.

Usage:
    python3 src/diagnose_crashes.py --episodes 50
"""

import argparse
import csv
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import mpc_expert
from best_response import predict_other_responses, straight_line_trajectory
from generate_data import (
    DRIVER_FN_MAP,
    DRIVER_MIXES,
    ENV_CONFIG,
    THETA_MAP,
    _assign_npc_driver_types,
    _patch_merge_env_continuous_rewards,
)
from reward import ego_reward


OUTPUT_DIR = Path("diagnostics")
EPISODE_CSV = OUTPUT_DIR / "crash_diagnostic_episodes.csv"
CRASH_CSV = OUTPUT_DIR / "crash_diagnostic_crashes.csv"
SUMMARY_CSV = OUTPUT_DIR / "crash_diagnostic_summary.csv"

VARIANTS = {
    "baseline": {},
    "collision_dist_8": {"collision_dist": 8.0},
    "collision_dist_10": {"collision_dist": 10.0},
    "proximity_2x": {"proximity_scale": 2.0},
    "horizon_10": {"horizon": 10},
    "samples_100": {"n_samples": 100},
    "top5_recompute": {"top_k_recompute": 5},
    "fallback_0p8": {"fallback_braking": -0.8},
    "fallback_1p0": {"fallback_braking": -1.0},
    "collision_penalty_fix": {"collision_weight": 1000.0},
}


def nearest_vehicle_info(env):
    """Return nearest non-ego vehicle info in the current env state."""
    ego = env.unwrapped.road.vehicles[0]
    nearest = None
    nearest_dist = float("inf")
    for vehicle in env.unwrapped.road.vehicles[1:]:
        dist = float(np.linalg.norm(ego.position - vehicle.position))
        if dist < nearest_dist:
            nearest = vehicle
            nearest_dist = dist
    if nearest is None:
        return {
            "distance": 100.0,
            "position": None,
            "speed": None,
            "lane_index": None,
        }
    return {
        "distance": nearest_dist,
        "position": nearest.position.copy(),
        "speed": float(nearest.speed),
        "lane_index": nearest.lane_index,
    }


def scaled_theta(theta, proximity_scale):
    """Scale only the proximity weight for reward ablations."""
    out = theta.copy()
    out[1] *= proximity_scale
    return out


def apply_reward_overrides(theta, config):
    """Apply diagnostic-only reward-vector overrides."""
    out = scaled_theta(theta, config.get("proximity_scale", 1.0))
    if "collision_weight" in config:
        out[4] = config["collision_weight"]
    return out


@contextmanager
def mpc_overrides(config):
    """Temporarily override mpc_expert module constants for one ablation run."""
    original = {
        "COLLISION_DIST": mpc_expert.COLLISION_DIST,
        "HORIZON": mpc_expert.HORIZON,
        "N_SAMPLES": mpc_expert.N_SAMPLES,
        "FALLBACK_BRAKING": mpc_expert.FALLBACK_BRAKING.copy(),
    }
    if "collision_dist" in config:
        mpc_expert.COLLISION_DIST = config["collision_dist"]
    if "horizon" in config:
        mpc_expert.HORIZON = config["horizon"]
    if "n_samples" in config:
        mpc_expert.N_SAMPLES = config["n_samples"]
    if "fallback_braking" in config:
        mpc_expert.FALLBACK_BRAKING = np.array(
            [config["fallback_braking"], 0.0], dtype=np.float32
        )
    try:
        yield
    finally:
        mpc_expert.COLLISION_DIST = original["COLLISION_DIST"]
        mpc_expert.HORIZON = original["HORIZON"]
        mpc_expert.N_SAMPLES = original["N_SAMPLES"]
        mpc_expert.FALLBACK_BRAKING = original["FALLBACK_BRAKING"]


def mpc_select_action_diagnostic(env, theta, config):
    """
    MPC selector with debug info and optional top-k response recomputation.

    Baseline behavior mirrors src/mpc_expert.py, including nominal response reuse
    and the existing verification pass. The top-k variant re-predicts other
    responses for the top candidate ego sequences and chooses by verified score.
    """
    start = time.perf_counter()
    ego = env.unwrapped.road.vehicles[0]
    top_k = config.get("top_k_recompute", 0)

    nominal_ego_traj = straight_line_trajectory(ego)
    predicted_others = predict_other_responses(env, nominal_ego_traj)

    candidates = []
    best_acc_norm = 0.0
    best_score = -np.inf
    best_sequence = None

    xp = np.linspace(0, mpc_expert.HORIZON - 1, mpc_expert.N_WAYPOINTS)
    xi = np.arange(mpc_expert.HORIZON)
    structured = [
        np.zeros(mpc_expert.N_WAYPOINTS),
        np.full(mpc_expert.N_WAYPOINTS, -0.3),
        np.full(mpc_expert.N_WAYPOINTS, 0.3),
    ]

    def evaluate_candidate(waypoints):
        nonlocal best_acc_norm, best_score, best_sequence
        acc_sequence = np.interp(xi, xp, waypoints)
        score, first_acc_norm = mpc_expert._evaluate_sequence(
            ego, acc_sequence, predicted_others, theta
        )
        candidates.append((score, first_acc_norm, acc_sequence.copy()))
        if score > best_score:
            best_score = score
            best_acc_norm = first_acc_norm
            best_sequence = acc_sequence.copy()

    for waypoints in structured:
        evaluate_candidate(waypoints)

    for _ in range(mpc_expert.N_SAMPLES):
        waypoints = np.random.normal(
            0.0, 0.4, size=(mpc_expert.N_WAYPOINTS,)
        ).clip(-1.0, 1.0)
        evaluate_candidate(waypoints)

    fallback_triggered = False
    verified_score = None

    if best_score < mpc_expert.CRASH_THRESHOLD:
        action = mpc_expert.FALLBACK_BRAKING.copy()
        fallback_triggered = True
    elif top_k > 0:
        verified = []
        for score, first_acc_norm, acc_sequence in sorted(
            candidates, key=lambda item: item[0], reverse=True
        )[:top_k]:
            ego_traj = mpc_expert._build_ego_traj(ego, acc_sequence)
            verified_others = predict_other_responses(env, ego_traj)
            v_score, _ = mpc_expert._evaluate_sequence(
                ego, acc_sequence, verified_others, theta
            )
            verified.append((v_score, first_acc_norm, acc_sequence))
        verified_score, best_acc_norm, best_sequence = max(
            verified, key=lambda item: item[0]
        )
        if verified_score < mpc_expert.CRASH_THRESHOLD:
            action = mpc_expert.FALLBACK_BRAKING.copy()
            fallback_triggered = True
        else:
            action = np.array([float(best_acc_norm), 0.0], dtype=np.float32)
    else:
        # Preserve current baseline verification behavior in mpc_expert.py.
        winning_sequence = np.full(mpc_expert.HORIZON, best_acc_norm)
        winning_ego_traj = mpc_expert._build_ego_traj(ego, winning_sequence)
        verified_others = predict_other_responses(env, winning_ego_traj)
        verified_score, _ = mpc_expert._evaluate_sequence(
            ego, winning_sequence, verified_others, theta
        )
        if verified_score < mpc_expert.CRASH_THRESHOLD:
            action = mpc_expert.FALLBACK_BRAKING.copy()
            fallback_triggered = True
        else:
            action = np.array([float(best_acc_norm), 0.0], dtype=np.float32)

    return action, {
        "best_score": float(best_score),
        "verified_score": None if verified_score is None else float(verified_score),
        "fallback_triggered": fallback_triggered,
        "best_acc_norm": float(best_acc_norm),
        "runtime_s": time.perf_counter() - start,
    }


def run_episode(env, mix_name, variant_name, variant_config, episode_id):
    obs, _ = env.reset()
    theta_name = random.choice(list(THETA_MAP.keys()))
    theta = THETA_MAP[theta_name]
    theta = apply_reward_overrides(theta, variant_config)
    npc_driver_types = _assign_npc_driver_types(env, DRIVER_MIXES[mix_name])

    total_reward = 0.0
    min_distance = float("inf")
    first_crash_step = None
    crash_debug = None
    step = 0
    terminated = truncated = False
    last_action = np.array([0.0, 0.0], dtype=np.float32)
    last_debug = {}
    start = time.perf_counter()

    while not (terminated or truncated) and step < 50:
        ego = env.unwrapped.road.vehicles[0]
        nearest_before = nearest_vehicle_info(env)
        min_distance = min(min_distance, nearest_before["distance"])
        ego_before_pos = ego.position.copy()
        ego_before_speed = float(ego.speed)

        action, debug = mpc_select_action_diagnostic(env, theta, variant_config)
        obs, env_reward, terminated, truncated, _ = env.step(action)
        total_reward += float(env_reward)
        last_action = action.copy()
        last_debug = debug

        nearest_after = nearest_vehicle_info(env)
        min_distance = min(min_distance, nearest_after["distance"])

        if env.unwrapped.vehicle.crashed and first_crash_step is None:
            first_crash_step = step
            crash_debug = {
                "variant": variant_name,
                "driver_mix": mix_name,
                "episode_id": episode_id,
                "step": step,
                "theta_name": theta_name,
                "npc_driver_types": "|".join(npc_driver_types),
                "ego_position_before": ego_before_pos.tolist(),
                "ego_speed_before": ego_before_speed,
                "nearest_position_before": (
                    None
                    if nearest_before["position"] is None
                    else nearest_before["position"].tolist()
                ),
                "nearest_speed_before": nearest_before["speed"],
                "nearest_lane_before": nearest_before["lane_index"],
                "distance_before": nearest_before["distance"],
                "distance_after": nearest_after["distance"],
                "action": action.tolist(),
                "best_score": debug["best_score"],
                "verified_score": debug["verified_score"],
                "fallback_triggered": debug["fallback_triggered"],
            }
        step += 1

    final_ego = env.unwrapped.road.vehicles[0]
    episode_runtime = time.perf_counter() - start

    row = {
        "variant": variant_name,
        "driver_mix": mix_name,
        "episode_id": episode_id,
        "crashed": bool(env.unwrapped.vehicle.crashed),
        "episode_length": step,
        "total_reward": total_reward,
        "min_distance": min_distance,
        "final_speed": float(final_ego.speed),
        "first_crash_step": first_crash_step,
        "theta_name": theta_name,
        "npc_driver_types": "|".join(npc_driver_types),
        "last_action": last_action.tolist(),
        "last_best_score": last_debug.get("best_score"),
        "last_verified_score": last_debug.get("verified_score"),
        "last_fallback_triggered": last_debug.get("fallback_triggered"),
        "runtime_s": episode_runtime,
    }
    return row, crash_debug


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for variant_name in VARIANTS:
        variant_rows = [r for r in rows if r["variant"] == variant_name]
        if not variant_rows:
            continue
        out = {
            "variant": variant_name,
            "avg_min_distance": np.mean([r["min_distance"] for r in variant_rows]),
            "avg_reward": np.mean([r["total_reward"] for r in variant_rows]),
            "avg_runtime_per_episode": np.mean([r["runtime_s"] for r in variant_rows]),
        }
        for mix_name in DRIVER_MIXES:
            mix_rows = [r for r in variant_rows if r["driver_mix"] == mix_name]
            out[f"{mix_name}_crash_rate"] = (
                sum(r["crashed"] for r in mix_rows) / len(mix_rows)
                if mix_rows else np.nan
            )
        summary.append(out)
    return summary


def print_metric_note(rows):
    baseline = [r for r in rows if r["variant"] == "baseline"]
    print("\nSTEP 1 - Metric verification")
    print("Crash rate is computed per episode: crashed episodes / total episodes.")
    print("Records/transitions are per timestep, so they are not the crash-rate denominator.")
    if not baseline:
        print("  Baseline was not included in this filtered run.")
        return
    for mix_name in DRIVER_MIXES:
        mix_rows = [r for r in baseline if r["driver_mix"] == mix_name]
        if not mix_rows:
            continue
        crashes = sum(r["crashed"] for r in mix_rows)
        transitions = sum(r["episode_length"] for r in mix_rows)
        clean_transitions = sum(
            r["episode_length"] for r in mix_rows if not r["crashed"]
        )
        print(
            f"  {mix_name}: episodes={len(mix_rows)}, crashes={crashes}, "
            f"crash_rate={100 * crashes / len(mix_rows):.1f}%, "
            f"records={transitions}, clean_records={clean_transitions}"
        )


def print_summary_table(summary):
    print("\nSTEP 5 - Ablation comparison")
    header = [
        "variant",
        "all_normal",
        "default_mix",
        "cautious_heavy",
        "aggressive_heavy",
        "avg_min_d",
        "avg_reward",
        "avg_runtime_s",
    ]
    print(
        f"{header[0]:<18} {header[1]:>10} {header[2]:>11} {header[3]:>15} "
        f"{header[4]:>17} {header[5]:>10} {header[6]:>11} {header[7]:>13}"
    )
    for row in summary:
        print(
            f"{row['variant']:<18} "
            f"{100 * row['all_normal_crash_rate']:>9.1f}% "
            f"{100 * row['default_mix_crash_rate']:>10.1f}% "
            f"{100 * row['cautious_heavy_crash_rate']:>14.1f}% "
            f"{100 * row['aggressive_heavy_crash_rate']:>16.1f}% "
            f"{row['avg_min_distance']:>10.2f} "
            f"{row['avg_reward']:>11.2f} "
            f"{row['avg_runtime_per_episode']:>13.3f}"
        )


def diagnose(summary):
    baseline = next(r for r in summary if r["variant"] == "baseline")
    print("\nSTEP 6 - Initial diagnosis")
    baseline_rates = [
        baseline[f"{mix}_crash_rate"] for mix in DRIVER_MIXES
    ]
    print(
        f"Baseline mean crash rate: {100 * np.mean(baseline_rates):.1f}% "
        f"(range {100 * min(baseline_rates):.1f}% to {100 * max(baseline_rates):.1f}%)."
    )
    improvements = []
    for row in summary:
        if row["variant"] == "baseline":
            continue
        rates = [row[f"{mix}_crash_rate"] for mix in DRIVER_MIXES]
        improvements.append((np.mean(baseline_rates) - np.mean(rates), row["variant"]))
    improvements.sort(reverse=True)
    if improvements:
        best_delta, best_variant = improvements[0]
        print(
            f"Best mean crash-rate improvement came from {best_variant}: "
            f"{100 * best_delta:.1f} percentage points."
        )
        print(
            "Interpretation guide: collision-distance/fallback wins suggest safety "
            "margin is too late; N_SAMPLES/HORIZON wins suggest MPC sampling/planning; "
            "top5_recompute wins suggest prediction mismatch; proximity_2x wins "
            "suggest reward proximity is too weak."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    _patch_merge_env_continuous_rewards()

    rows = []
    crash_rows = []
    total_runs = len(args.variants) * len(DRIVER_MIXES) * args.episodes
    run_idx = 0

    for variant_name in args.variants:
        variant_config = VARIANTS[variant_name]
        with mpc_overrides(variant_config):
            for mix_name in DRIVER_MIXES:
                env = gym.make("merge-v0", config=ENV_CONFIG)
                for episode_id in range(args.episodes):
                    run_idx += 1
                    row, crash_debug = run_episode(
                        env, mix_name, variant_name, variant_config, episode_id
                    )
                    rows.append(row)
                    if crash_debug:
                        crash_rows.append(crash_debug)
                    print(
                        f"[{run_idx:>4}/{total_runs}] {variant_name:<18} "
                        f"{mix_name:<16} ep={episode_id + 1:>3}/{args.episodes} "
                        f"crashed={row['crashed']} len={row['episode_length']:>2} "
                        f"min_d={row['min_distance']:.2f}"
                    )
                env.close()

    episode_fields = [
        "variant", "driver_mix", "episode_id", "crashed", "episode_length",
        "total_reward", "min_distance", "final_speed", "first_crash_step",
        "theta_name", "npc_driver_types", "last_action", "last_best_score",
        "last_verified_score", "last_fallback_triggered", "runtime_s",
    ]
    crash_fields = [
        "variant", "driver_mix", "episode_id", "step", "theta_name",
        "npc_driver_types", "ego_position_before", "ego_speed_before",
        "nearest_position_before", "nearest_speed_before", "nearest_lane_before",
        "distance_before", "distance_after", "action", "best_score",
        "verified_score", "fallback_triggered",
    ]
    write_csv(EPISODE_CSV, rows, episode_fields)
    write_csv(CRASH_CSV, crash_rows, crash_fields)

    summary = summarize(rows)
    summary_fields = [
        "variant",
        "all_normal_crash_rate",
        "default_mix_crash_rate",
        "cautious_heavy_crash_rate",
        "aggressive_heavy_crash_rate",
        "avg_min_distance",
        "avg_reward",
        "avg_runtime_per_episode",
    ]
    write_csv(SUMMARY_CSV, summary, summary_fields)

    print_metric_note(rows)
    print_summary_table(summary)
    diagnose(summary)
    print(f"\nSaved per-episode metrics to {EPISODE_CSV}")
    print(f"Saved crash debug logs to {CRASH_CSV}")
    print(f"Saved summary table to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
