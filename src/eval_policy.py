"""
Phase 6: BC policy rollout evaluation.

Rolls out the trained BC policy in merge-v0 for N episodes and reports:
  - Crash rate
  - Mean environment reward
  - Mean episode length (steps)
  - Mean ego speed (m/s)
  - Mean minimum gap (d_min, m)

Compares against a baseline MPC expert run on the same seeds for reference.

Usage:
    python3 src/eval_policy.py                          # default_mix model, 20 episodes
    python3 src/eval_policy.py --model models/bc_policy_all_normal.pt --episodes 20
    python3 src/eval_policy.py --model models/bc_policy_all_normal.pt --no-mpc-baseline
"""

import argparse
import os
import random
import sys
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
import highway_env  # noqa: F401

from driver_types import make_cautious, make_normal, make_aggressive
from generate_data import _patch_merge_env_continuous_rewards
from mpc_expert import mpc_select_action
from policy_network import PolicyNetwork
from reward import NORMAL

DRIVER_FNS = [make_cautious, make_normal, make_aggressive]

# Named traffic mixes for cross-evaluation
TRAFFIC_MIXES = {
    "uniform":          [make_cautious, make_normal, make_aggressive],
    "all_normal":       [make_normal],
    "default_mix":      [make_normal, make_normal, make_normal, make_normal,
                         make_normal, make_normal, make_cautious, make_cautious,
                         make_aggressive, make_aggressive],  # 60/20/20
    "all_cautious":     [make_cautious],
    "all_aggressive":   [make_aggressive],
    "cautious_heavy":   [make_cautious, make_cautious, make_cautious, make_cautious,
                         make_cautious, make_normal, make_normal, make_normal,
                         make_normal, make_aggressive],  # ~50% cautious
    "aggressive_heavy": [make_aggressive, make_aggressive, make_aggressive, make_aggressive,
                         make_aggressive, make_normal, make_normal, make_normal,
                         make_normal, make_cautious],   # ~50% aggressive
}

ENV_CONFIG = {
    "vehicles_count": 3,
    "controlled_vehicles": 1,
    "action": {"type": "ContinuousAction"},
}

MAX_STEPS = 50


def run_episodes(env, action_fn, n_episodes: int, seed: int, label: str,
                 driver_fns: list = None, verbose: bool = True,
                 plot_path: Path = None):
    """Roll out action_fn for n_episodes, return summary stats."""
    if driver_fns is None:
        driver_fns = DRIVER_FNS
    rng = random.Random(seed)
    results = []

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  {label}")
        print(f"{'─'*55}")
        print(f"{'Ep':>4}  {'steps':>5}  {'crashed':>7}  {'reward':>8}  {'avg_spd':>7}  {'min_gap':>7}")

    first_traj = {"step": [], "speed": [], "min_gap": [], "x": []}

    for ep in range(n_episodes):
        # Use a fixed seed per episode so BC and MPC see identical spawns
        np.random.seed(seed + ep)
        random.seed(seed + ep)

        obs, _ = env.reset()
        # Allow stateful action functions to reset their internal state
        if hasattr(action_fn, "reset"):
            action_fn.reset()
        for v in env.unwrapped.road.vehicles[1:]:
            rng.choice(driver_fns)(v)

        speeds, dmins = [], []
        total_reward = 0.0
        step = 0
        terminated = truncated = False

        while not (terminated or truncated) and step < MAX_STEPS:
            ego = env.unwrapped.road.vehicles[0]
            speeds.append(float(ego.speed))
            if ep == 0:
                first_traj["step"].append(step)
                first_traj["speed"].append(float(ego.speed))
                first_traj["x"].append(float(ego.position[0]))

            others = env.unwrapped.road.vehicles[1:]
            if others:
                d = min(np.linalg.norm(ego.position - v.position) for v in others)
                dmins.append(float(d))
                if ep == 0:
                    first_traj["min_gap"].append(float(d))

            action = action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            step += 1

        crashed = env.unwrapped.vehicle.crashed
        avg_spd = float(np.mean(speeds)) if speeds else 0.0
        min_gap = float(np.min(dmins)) if dmins else 100.0

        if verbose:
            print(
                f"{ep+1:>4}  {step:>5}  {'YES' if crashed else 'no':>7}  "
                f"{total_reward:>8.2f}  {avg_spd:>7.2f}  {min_gap:>7.2f}"
            )
        results.append({
            "crashed": crashed,
            "steps": step,
            "reward": total_reward,
            "avg_spd": avg_spd,
            "min_gap": min_gap,
        })

    summary = summarize_results(results)
    if verbose:
        print_summary(summary, results)
    if plot_path is not None:
        save_trajectory_plot(first_traj, plot_path, label)
    return results


def summarize_results(results: list[dict]) -> dict:
    n = len(results)
    return {
        "n": n,
        "crash_rate": sum(r["crashed"] for r in results) / n,
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_speed": float(np.mean([r["avg_spd"] for r in results])),
        "mean_gap": float(np.mean([r["min_gap"] for r in results])),
        "negative_speed_rate": sum(r["avg_spd"] < 0.0 for r in results) / n,
    }


def print_summary(summary: dict, results: list[dict]) -> None:
    print(f"\n  Summary ({summary['n']} episodes):")
    print(f"    Crash rate : {100*summary['crash_rate']:.1f}%")
    print(f"    Mean reward: {summary['mean_reward']:.2f}")
    print(f"    Mean steps : {summary['mean_steps']:.1f}")
    print(f"    Mean speed : {summary['mean_speed']:.2f} m/s")
    print(f"    Mean min gap: {summary['mean_gap']:.2f} m")
    print(f"    Negative-speed episodes: {100*summary['negative_speed_rate']:.1f}%")
    crashed_steps = sorted([r["steps"] for r in results if r["crashed"]])
    if crashed_steps:
        from collections import Counter
        counts = Counter(crashed_steps)
        breakdown = ", ".join(f"step {s}: {c}x" for s, c in sorted(counts.items()))
        spawn = sum(c for s, c in counts.items() if s <= 2)
        late  = sum(c for s, c in counts.items() if s > 2)
        print(f"    Crash steps: [{breakdown}]")
        print(f"    Spawn (<=2): {spawn}  |  Late (>2): {late}")


def save_trajectory_plot(traj: dict, plot_path: Path, label: str) -> None:
    if not traj["step"]:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        axes[0].plot(traj["step"], traj["speed"])
        axes[0].set_ylabel("speed")
        axes[1].plot(traj["step"], traj["min_gap"])
        axes[1].set_ylabel("min gap")
        axes[2].plot(traj["step"], traj["x"])
        axes[2].set_ylabel("ego x")
        axes[2].set_xlabel("step")
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Trajectory plot saved to: {plot_path}")
    except Exception as exc:
        print(f"Warning: could not save trajectory plot ({exc})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/bc_policy_all_normal.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mpc-baseline", action="store_true")
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save a trajectory plot for episode 1 under plots/.",
    )
    parser.add_argument(
        "--traffic-mix",
        choices=list(TRAFFIC_MIXES.keys()),
        default="uniform",
        help="NPC driver distribution during rollout (default: uniform random across all 3 types).",
    )
    args = parser.parse_args()
    driver_fns = TRAFFIC_MIXES[args.traffic_mix]
    _patch_merge_env_continuous_rewards()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        print("Run train_policy.py first.")
        sys.exit(1)

    # Load BC policy
    model = PolicyNetwork()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Load normalization stats saved alongside the model
    stats_path = model_path.with_suffix(".npz")
    if not stats_path.exists():
        print(f"Error: norm stats not found at {stats_path}. Retrain the model first.")
        sys.exit(1)
    stats = np.load(stats_path)
    obs_mean = stats["mean"]
    obs_std  = stats["std"]
    print(f"Norm stats: {stats_path}  (dim={obs_mean.shape[0]})")

    class BCAction:
        """Stateful callable: appends d_min + step to obs, normalizes, then runs model."""
        def __init__(self):
            self._step = 0

        def reset(self):
            self._step = 0

        def __call__(self, obs):
            ego    = env.unwrapped.road.vehicles[0]
            others = env.unwrapped.road.vehicles[1:]
            d_min  = float(min(
                np.linalg.norm(ego.position - v.position) for v in others
            )) if others else 100.0
            aug     = np.array([d_min, float(self._step)], dtype=np.float32)
            obs_aug = np.append(obs.reshape(-1), aug).astype(np.float32)
            obs_norm = (obs_aug - obs_mean) / obs_std
            action  = model.predict(obs_norm)
            # Clamp action so ego speed cannot go below zero during rollout.
            ego_speed = float(ego.speed)
            if ego_speed < 2.0 and action[0] < 0:
                action[0] = max(action[0], 0.0)
            self._step += 1
            return action

    bc_action = BCAction()

    env = gym.make("merge-v0", config=ENV_CONFIG)

    print(f"Model : {model_path}")
    print(f"Episodes: {args.episodes} | Seed: {args.seed} | Traffic: {args.traffic_mix}")

    # BC rollout
    plot_path = None
    if args.save_plot:
        plot_path = Path("plots") / f"rollout_{model_path.stem}_{args.traffic_mix}_seed{args.seed}.png"
    run_episodes(env, bc_action, args.episodes, args.seed,
                 f"BC policy ({model_path.stem}) | traffic={args.traffic_mix}",
                 driver_fns=driver_fns, plot_path=plot_path)

    # MPC baseline on same seeds
    if not args.no_mpc_baseline:
        def mpc_action(obs):
            return mpc_select_action(env, theta=NORMAL)

        run_episodes(env, mpc_action, args.episodes, args.seed,
                     f"MPC expert (NORMAL, baseline) | traffic={args.traffic_mix}",
                     driver_fns=driver_fns)

    env.close()


if __name__ == "__main__":
    main()
