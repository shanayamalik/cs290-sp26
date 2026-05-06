"""
demo.py  —  Record one episode per method as a GIF.

Saves:
    demo_ppo.gif
    demo_mpc.gif
    demo_baseline.gif
    demo_comparison.gif   (3 strips stacked vertically, labelled)

Usage:
    python3 demo.py [--seed 7] [--traffic-mix default_mix] [--fps 10] [--max-steps 120]

The default seed (7) was chosen because PPO merges cleanly and MPC/Baseline
show clearly different behaviour in the same scene.
"""

import argparse
import os
import random
import sys
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import gymnasium as gym
import highway_env  # noqa: F401
import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from baseline import (
    ENV_CONFIG,
    MERGE_EXIT_X,
    TRAFFIC_MIXES,
    independent_baseline_action,
    no_reverse_clamp,
)
from generate_data import _patch_merge_env_continuous_rewards
from mpc_expert import mpc_select_action
from policy_network import PolicyNetwork
from reward import NORMAL
from rl_finetune import TanhMeanActorCriticPolicy, load_bc_stats, make_env

# ── paths ────────────────────────────────────────────────────────────────────
PPO_MODEL   = Path("models/ppo_500k_v2_merge.zip")
BC_MODEL    = Path("models/bc_policy_default_mix.pt")

RENDER_CONFIG = {
    **ENV_CONFIG,
    "screen_width": 1200,
    "screen_height": 200,
    "scaling": 6.5,
    "centering_position": [0.3, 0.5],
    "show_trajectories": True,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def assign_traffic(env, driver_fns, rng):
    for vehicle in env.unwrapped.road.vehicles[1:]:
        rng.choice(driver_fns)(vehicle)


def add_label(frame: np.ndarray, text: str, color=(255, 255, 100)) -> np.ndarray:
    """Burn a text label into the top-left corner of a frame (no extra deps)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, len(text) * 8 + 10, 22], fill=(0, 0, 0))
        draw.text((5, 4), text, fill=color)
        return np.array(img)
    except Exception:
        return frame   # skip label if PIL fails


def record_episode(method: str, seed: int, traffic_mix: str, max_steps: int,
                   ppo_model_path: Path, bc_model_path: Path) -> list[np.ndarray]:
    """Run one episode of *method* and return frames as a list of RGB arrays."""
    _patch_merge_env_continuous_rewards()
    driver_fns = TRAFFIC_MIXES[traffic_mix]
    rng = random.Random(seed)

    if method == "ppo":
        obs_mean, obs_std = load_bc_stats(bc_model_path)
        from stable_baselines3 import PPO as SB3PPO
        from rl_finetune import MergePPOWrapper, ENV_CONFIG as PPO_ENV_CONFIG
        model = SB3PPO.load(
            str(ppo_model_path),
            custom_objects={"policy_class": TanhMeanActorCriticPolicy},
        )
        render_cfg = {**PPO_ENV_CONFIG, **RENDER_CONFIG}
        base_env = gym.make("merge-v0", render_mode="rgb_array", config=render_cfg)
        env = MergePPOWrapper(base_env, obs_mean, obs_std, traffic_mix=traffic_mix, seed=seed)
        np.random.seed(seed); random.seed(seed)
        obs, _ = env.reset(seed=seed)
        env.render()  # initialize viewer before first step()

        frames = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(add_label(frame, f"PPO  step={steps}"))
            done = terminated or truncated
            steps += 1
        env.close()
        return frames

    else:
        env = gym.make("merge-v0", render_mode="rgb_array", config=RENDER_CONFIG)
        np.random.seed(seed); random.seed(seed)
        obs, _ = env.reset(seed=seed)
        assign_traffic(env, driver_fns, rng)
        env.render()  # initialize viewer before first step()

        if method == "mpc":
            label = "MPC (full IBR)"
            def action_fn(_obs):
                a = mpc_select_action(env, theta=NORMAL)
                a, _ = no_reverse_clamp(env, a)
                return a
        elif method == "baseline":
            label = "Baseline (independent)"
            def action_fn(_obs):
                a = independent_baseline_action(env, theta=NORMAL)
                a, _ = no_reverse_clamp(env, a)
                return a
        else:
            raise ValueError(method)

        frames = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = action_fn(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(add_label(frame, f"{label}  step={steps}"))
            done = terminated or truncated
            steps += 1
        env.close()
        return frames


def save_gif(frames: list[np.ndarray], path: str, fps: int):
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"  Saved {path}  ({len(frames)} frames @ {fps} fps)")


def stack_frames(frames_list: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Stack multiple frame sequences vertically, padding shorter ones."""
    max_len = max(len(f) for f in frames_list)
    stacked = []
    for i in range(max_len):
        rows = []
        for seq in frames_list:
            idx = min(i, len(seq) - 1)  # hold last frame for shorter episodes
            rows.append(seq[idx])
        stacked.append(np.concatenate(rows, axis=0))
    return stacked


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=7)
    parser.add_argument("--traffic-mix", type=str, default="default_mix")
    parser.add_argument("--fps",         type=int, default=10)
    parser.add_argument("--max-steps",   type=int, default=120)
    parser.add_argument("--methods",     nargs="+",
                        default=["ppo", "mpc", "baseline"],
                        help="Subset of: ppo mpc baseline")
    args = parser.parse_args()

    print(f"Recording demo — seed={args.seed}, mix={args.traffic_mix}, "
          f"fps={args.fps}, max_steps={args.max_steps}")
    print()

    all_frames = {}
    for method in args.methods:
        print(f"[{method}] running episode...")
        frames = record_episode(
            method, args.seed, args.traffic_mix, args.max_steps,
            PPO_MODEL, BC_MODEL,
        )
        all_frames[method] = frames
        save_gif(frames, f"demo_{method}.gif", args.fps)

    if len(args.methods) > 1:
        print("\n[comparison] stacking strips...")
        stacked = stack_frames([all_frames[m] for m in args.methods])
        save_gif(stacked, "demo_comparison.gif", args.fps)

    print("\nDone! Open demo_comparison.gif (or individual demo_*.gif) to view.")


if __name__ == "__main__":
    main()
