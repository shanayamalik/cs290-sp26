"""Check if the 500k v2 model reaches x=370 (the terminated threshold)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from stable_baselines3 import PPO
from rl_finetune import make_env, load_bc_stats

obs_mean, obs_std = load_bc_stats(Path("models/bc_policy_default_mix.pt"))
env = make_env(obs_mean, obs_std, "default_mix", seed=0)
model = PPO.load("models/ppo_500k_v2_merge.zip", env=env)

print("10 deterministic eval episodes with 500k v2 model:")
print(f"  terminated threshold: x > 370m")
term_count = 0
for ep in range(10):
    obs, _ = env.reset(seed=ep)
    done = False
    last_t = False
    last_tr = False
    steps = 0
    max_x = float("-inf")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_t = terminated
        last_tr = truncated
        steps += 1
        x = float(env.unwrapped.road.vehicles[0].position[0])
        if x > max_x:
            max_x = x
    final_x = float(env.unwrapped.road.vehicles[0].position[0])
    if last_t:
        term_count += 1
    print(
        f"  ep{ep:02d}: steps={steps:4d}, terminated={last_t}, "
        f"truncated={last_tr}, final_x={final_x:.1f}m, max_x={max_x:.1f}m"
    )

print(f"\nterminated: {term_count}/10")
env.close()
