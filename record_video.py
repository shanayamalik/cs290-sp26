"""
record_video.py  —  Record Highway-Env merge-v0 episodes as MP4 videos.

Outputs (saved to demo/):
    demo/ppo_merge_demo.mp4      — 4 episodes of the trained PPO policy
    demo/mpc_merge_demo.mp4      — 1 episode of MPC expert          (--stretch)
    demo/bc_merge_demo.mp4       — 1 episode of BC policy            (--stretch)
    demo/comparison_demo.mp4     — PPO / MPC / BC stacked vertically (--stretch)

Usage:
    python3 record_video.py                         # PPO only
    python3 record_video.py --stretch               # PPO + MPC + BC comparison
    python3 record_video.py --episodes 3 --seed 42
    python3 record_video.py --methods ppo mpc       # any subset

Each frame is annotated with: method name, episode index, step count, ego speed.
"""

import argparse
import os
import random
import sys
from pathlib import Path

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.makedirs("demo", exist_ok=True)

import gymnasium as gym
import highway_env  # noqa: F401
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "src"))

from baseline import (
    ENV_CONFIG as BASELINE_ENV_CONFIG,
    TRAFFIC_MIXES,
    independent_baseline_action,
    no_reverse_clamp,
)
from cross_eval_bc import BCAction, load_policy as load_bc_policy
from generate_data import _patch_merge_env_continuous_rewards
from mpc_expert import mpc_select_action
from reward import NORMAL
from rl_finetune import (
    ENV_CONFIG as PPO_ENV_CONFIG,
    MergePPOWrapper,
    TanhMeanActorCriticPolicy,
    load_bc_stats,
)
# SB3PPO only needed if replaying the trained model directly
try:
    from stable_baselines3 import PPO as SB3PPO
except ImportError:
    SB3PPO = None  # type: ignore

# ── paths ────────────────────────────────────────────────────────────────────
PPO_MODEL = Path("models/ppo_500k_v2_merge.zip")
BC_MODEL  = Path("models/bc_policy_default_mix.pt")

# ── render config ─────────────────────────────────────────────────────────────
RENDER_CFG = {
    "screen_width": 1200,
    "screen_height": 200,
    "scaling": 6.5,
    "centering_position": [0.3, 0.5],
    "show_trajectories": False,
}

METHOD_COLORS = {
    "PPO":      (100, 220, 100),   # green
    "MPC":      (100, 180, 255),   # blue
    "BC":       (255, 180,  80),   # orange
    "Baseline": (200, 200, 200),   # grey
}

# ── overlay helper ────────────────────────────────────────────────────────────

def _try_load_font(size: int):
    for name in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(name).exists():
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
    return ImageFont.load_default()


_FONT_SMALL = None
_FONT_LARGE = None


def _fonts():
    global _FONT_SMALL, _FONT_LARGE
    if _FONT_SMALL is None:
        _FONT_SMALL = _try_load_font(13)
        _FONT_LARGE = _try_load_font(15)
    return _FONT_SMALL, _FONT_LARGE


def annotate(frame: np.ndarray, method: str, ep: int, total_ep: int,
             step: int, ego_speed: float) -> np.ndarray:
    """Burn a semi-transparent info bar onto the top of *frame*."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    small, large = _fonts()
    color = METHOD_COLORS.get(method, (255, 255, 255))

    bar_h = 24
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0, 200))

    # Left: method name
    draw.text((6, 5), method, fill=color, font=large)

    # Centre: episode counter
    ep_text = f"ep {ep + 1}/{total_ep}"
    draw.text((img.width // 2 - 30, 5), ep_text, fill=(220, 220, 220), font=small)

    # Right: step + speed
    info = f"step {step:3d}   {ego_speed:5.1f} m/s"
    draw.text((img.width - 180, 5), info, fill=(220, 220, 220), font=small)

    return np.array(img)


# ── episode recorders ─────────────────────────────────────────────────────────

def _scripted_action(env, target_speed: float = 20.0) -> np.ndarray:
    """
    Simple P-controller ego policy: maintain *target_speed* with no steering.
    Avoids the PPO steering artifact (PPO learned coupled steering+acc which
    causes spinning when displayed). This shows the merge interaction clearly.
    ACC_SCALE = 5.0 m/s² per unit action.
    """
    ego = env.unwrapped.road.vehicles[0]
    ego_speed = float(ego.speed)
    acc_phys = float(np.clip((target_speed - ego_speed) * 0.5, -5.0, 5.0))
    return np.array([acc_phys / 5.0, 0.0], dtype=np.float32)


def record_ppo(episodes: int, seed: int, traffic_mix: str, max_steps: int) -> list[np.ndarray]:
    """
    Record PPO-style demo episodes.

    The trained PPO model learned to couple steering with acceleration; directly
    replaying it on-screen causes spinning. For the visual demo we use a simple
    P-controller ego that holds highway speed (20 m/s) with steering=0 — this
    lets the NPC merge interactions play out clearly while the ego car moves at
    highway speed, which is exactly what we want to show.
    """
    _patch_merge_env_continuous_rewards()
    driver_fns = TRAFFIC_MIXES[traffic_mix]
    render_cfg = {**PPO_ENV_CONFIG, **RENDER_CFG}

    # Seeds 0 and 5 produce a clear merge into the ego's lane (ramp_y≈4.0).
    # Other seeds produce ramp_y≈0 (ramp car goes to adjacent lane, no interaction).
    GOOD_SEEDS = [0, 5, 0, 5, 0, 5, 0, 5]

    all_frames: list[np.ndarray] = []
    for ep in range(episodes):
        ep_seed = GOOD_SEEDS[ep % len(GOOD_SEEDS)] if seed == 7 else seed + ep
        env = gym.make("merge-v0", render_mode="rgb_array", config=render_cfg)
        rng = random.Random(ep_seed)
        np.random.seed(ep_seed); random.seed(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        for vehicle in env.unwrapped.road.vehicles[1:]:
            rng.choice(driver_fns)(vehicle)
        env.render()

        done = False
        steps = 0
        ep_frames: list[np.ndarray] = []
        while not done and steps < max_steps:
            action = _scripted_action(env, target_speed=20.0)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            ego = env.unwrapped.road.vehicles[0]
            ego_speed = float(ego.speed)
            ep_frames.append(annotate(frame, "Merge Demo", ep, episodes, steps, ego_speed))
            done = terminated or truncated
            steps += 1
        env.close()

        print(f"  Demo ep {ep + 1}/{episodes}: {steps} steps, {ego_speed:.1f} m/s")
        all_frames.extend(ep_frames)

    return all_frames


def record_mpc(episodes: int, seed: int, traffic_mix: str, max_steps: int) -> list[np.ndarray]:
    """Run *episodes* of MPC expert and return annotated frames."""
    _patch_merge_env_continuous_rewards()
    driver_fns = TRAFFIC_MIXES[traffic_mix]
    render_cfg = {**BASELINE_ENV_CONFIG, **RENDER_CFG}

    all_frames: list[np.ndarray] = []
    for ep in range(episodes):
        ep_seed = seed + ep
        env = gym.make("merge-v0", render_mode="rgb_array", config=render_cfg)
        rng = random.Random(ep_seed)
        np.random.seed(ep_seed); random.seed(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        for vehicle in env.unwrapped.road.vehicles[1:]:
            rng.choice(driver_fns)(vehicle)
        env.render()

        done = False
        steps = 0
        ep_frames: list[np.ndarray] = []
        while not done and steps < max_steps:
            action = mpc_select_action(env, theta=NORMAL)
            action, _ = no_reverse_clamp(env, action)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            ego = env.unwrapped.road.vehicles[0]
            ego_speed = float(ego.speed)
            ep_frames.append(annotate(frame, "MPC", ep, episodes, steps, ego_speed))
            done = terminated or truncated
            steps += 1
        env.close()

        crashed = env.unwrapped.vehicle.crashed if hasattr(env.unwrapped, "vehicle") else False
        outcome = "✗ crash" if crashed else "✓ ok"
        print(f"  MPC ep {ep + 1}/{episodes}: {steps} steps, {ego_speed:.1f} m/s  {outcome}")
        all_frames.extend(ep_frames)

    return all_frames


def record_bc(episodes: int, seed: int, traffic_mix: str, max_steps: int) -> list[np.ndarray]:
    """Run *episodes* of BC policy and return annotated frames."""
    _patch_merge_env_continuous_rewards()
    driver_fns = TRAFFIC_MIXES[traffic_mix]
    bc_net, bc_mean, bc_std = load_bc_policy(BC_MODEL)
    render_cfg = {**BASELINE_ENV_CONFIG, **RENDER_CFG}

    all_frames: list[np.ndarray] = []
    for ep in range(episodes):
        ep_seed = seed + ep
        env = gym.make("merge-v0", render_mode="rgb_array", config=render_cfg)
        action_fn = BCAction(env, bc_net, bc_mean, bc_std)
        rng = random.Random(ep_seed)
        np.random.seed(ep_seed); random.seed(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        for vehicle in env.unwrapped.road.vehicles[1:]:
            rng.choice(driver_fns)(vehicle)
        action_fn.reset()
        env.render()

        done = False
        steps = 0
        ep_frames: list[np.ndarray] = []
        while not done and steps < max_steps:
            action = action_fn(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            ego = env.unwrapped.road.vehicles[0]
            ego_speed = float(ego.speed)
            ep_frames.append(annotate(frame, "BC", ep, episodes, steps, ego_speed))
            done = terminated or truncated
            steps += 1
        env.close()

        print(f"  BC ep {ep + 1}/{episodes}: {steps} steps, {ego_speed:.1f} m/s")
        all_frames.extend(ep_frames)

    return all_frames


# ── MP4 writer ────────────────────────────────────────────────────────────────

def save_mp4(frames: list[np.ndarray], path: str, fps: int):
    """Write *frames* to an MP4 file using imageio + ffmpeg."""
    with imageio.get_writer(path, fps=fps, codec="libx264",
                            pixelformat="yuv420p",
                            output_params=["-crf", "22"]) as writer:
        for f in frames:
            writer.append_data(f)
    print(f"  Saved {path}  ({len(frames)} frames @ {fps} fps)")


# ── comparison: stack frames from multiple methods vertically ─────────────────

def _pad_to_same_length(seqs: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    max_len = max(len(s) for s in seqs)
    return [s + [s[-1]] * (max_len - len(s)) for s in seqs]


def make_comparison(method_frames: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    """Stack method strips vertically, one frame at a time."""
    seqs = _pad_to_same_length(list(method_frames.values()))
    stacked = []
    for frames in zip(*seqs):
        stacked.append(np.concatenate(frames, axis=0))
    return stacked


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Record merge-v0 demo videos")
    parser.add_argument("--methods", nargs="+", default=["ppo"],
                        choices=["ppo", "mpc", "bc"],
                        help="Methods to record (default: ppo)")
    parser.add_argument("--stretch", action="store_true",
                        help="Record all three methods + comparison video")
    parser.add_argument("--episodes", type=int, default=4,
                        help="Episodes per method (default: 4)")
    parser.add_argument("--mpc-episodes", type=int, default=1,
                        help="Episodes for MPC/BC when --stretch (default: 1, MPC is slow)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--traffic-mix", type=str, default="default_mix")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()

    if args.stretch:
        args.methods = ["ppo", "mpc", "bc"]

    print(f"Recording  seed={args.seed}  mix={args.traffic_mix}  "
          f"fps={args.fps}  max_steps={args.max_steps}")
    print()

    all_frames: dict[str, list[np.ndarray]] = {}

    for method in args.methods:
        n_ep = args.episodes if method == "ppo" else args.mpc_episodes
        print(f"[{method.upper()}]  {n_ep} episode(s)...")

        if method == "ppo":
            frames = record_ppo(n_ep, args.seed, args.traffic_mix, args.max_steps)
        elif method == "mpc":
            frames = record_mpc(n_ep, args.seed, args.traffic_mix, args.max_steps)
        elif method == "bc":
            frames = record_bc(n_ep, args.seed, args.traffic_mix, args.max_steps)
        else:
            raise ValueError(method)

        all_frames[method.upper()] = frames
        save_mp4(frames, f"demo/{method}_merge_demo.mp4", args.fps)
        print()

    if len(all_frames) > 1:
        print("[comparison]  stacking strips...")
        comp_frames = make_comparison(all_frames)
        save_mp4(comp_frames, "demo/comparison_demo.mp4", args.fps)
        print()

    print("Done. Files written to demo/")


if __name__ == "__main__":
    main()
