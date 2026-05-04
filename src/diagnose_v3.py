"""Check if the 500k v3 model crosses x > 310 (merge zone exit) in eval episodes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from rl_finetune import make_env, load_bc_stats

obs_mean, obs_std = load_bc_stats(Path("models/bc_policy_default_mix.pt"))
env = make_env(obs_mean, obs_std, "default_mix", seed=0)
import sys as _sys
model_name = _sys.argv[1] if len(_sys.argv) > 1 else "models/ppo_500k_v3_merge.zip"
model = PPO.load(model_name, env=env)

print(f"20 deterministic eval episodes — {model_name}:")
print(f"  merge zone exit: x > 310m\n")
crossed_310 = 0
crashed_count = 0
for ep in range(20):
    obs, _ = env.reset(seed=ep)
    done = False
    steps = 0
    max_x = float("-inf")
    crashed = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        ego = env.unwrapped.road.vehicles[0]
        x = float(ego.position[0])
        if x > max_x:
            max_x = x
        if ego.crashed:
            crashed = True
    if max_x > 310:
        crossed_310 += 1
    if crashed:
        crashed_count += 1
    tag = "MERGE OK" if (max_x > 310 and not crashed) else ("CRASH" if crashed else "SHORT")
    print(f"  ep{ep:02d}: steps={steps:4d}  max_x={max_x:6.1f}m  crashed={crashed}  [{tag}]")

print(f"\ncrossed x>310 (merge complete): {crossed_310}/20")
print(f"crashed:                         {crashed_count}/20")
env.close()
