"""Diagnostic: check road length and whether terminated ever fires in merge-v0."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
import highway_env  # noqa
import numpy as np
from rl_finetune import make_env, load_bc_stats

obs_mean, obs_std = load_bc_stats(Path("models/bc_policy_default_mix.pt"))
env = make_env(obs_mean, obs_std, "default_mix", seed=0)
raw = env.unwrapped

# Print road network lane endpoints
print("Road network (lane start/end x-coords):")
for from_id, to_dict in raw.road.network.graph.items():
    for to_id, lanes in to_dict.items():
        for lane in lanes:
            try:
                print(f"  {from_id}->{to_id}: start_x={lane.start[0]:.1f}, end_x={lane.end[0]:.1f}, length={lane.length:.1f}m")
            except Exception:
                pass

# Run 10 episodes with a random policy
print("\nEpisode diagnostics (random policy, 10 episodes):")
term_count = 0
trunc_count = 0
final_xs = []

for ep in range(10):
    obs, _ = env.reset(seed=ep)
    done = False
    last_terminated = False
    last_truncated = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_terminated = terminated
        last_truncated = truncated
        steps += 1
    x = float(raw.road.vehicles[0].position[0])
    final_xs.append(x)
    if last_terminated:
        term_count += 1
    if last_truncated:
        trunc_count += 1
    print(f"  ep{ep:02d}: steps={steps:3d}, terminated={last_terminated}, truncated={last_truncated}, final_x={x:.1f}m, crashed={info.get('crashed', False)}")

print(f"\nSummary over 10 episodes:")
print(f"  terminated: {term_count}/10")
print(f"  truncated:  {trunc_count}/10")
print(f"  final_x: min={min(final_xs):.1f}, mean={np.mean(final_xs):.1f}, max={max(final_xs):.1f}")
env.close()
