# Autonomous Driving with Multi-Agent Lane Merging

**Shanaya Malik, Nathan McNaughton**

ELENG 290: Learning-Enabled Multi-Agent Systems — Spring 2026

Extension of Sadigh et al. (2016) to multi-vehicle lane merging with 3+ agents. Each human driver is assigned a predefined reward archetype (cautious / normal / aggressive). The pipeline has three stages: iterative best-response MPC expert → behavioral cloning → PPO fine-tuning.

Driver type parameters (in `src/driver_types.py`) are hand-tuned based on empirically observed ranges from Treiber et al. (2000) and map onto the Social Value Orientation taxonomy from Schwarting et al. (2019). They are not learned from data.

**Progress:** Phases 1–4 complete (environment setup, driver types, reward functions, MPC expert). Starting Phase 5 (expert dataset generation).
See [implementation_guide.md](implementation_guide.md) for the full step-by-step implementation plan.
See [report_notes.md](report_notes.md) for things to address in the final paper writeup.

---

## Setup

**Requirements:** Python 3.10+, no GPU needed.

```bash
# 1. Clone the repo
git clone https://github.com/shanayamalik/cs290-sp26.git
cd cs290-sp26

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

To verify the environment is working:

```bash
python3 -c "import highway_env, gymnasium, torch, stable_baselines3; print('All dependencies OK')"
```

Then run the environment sanity check (a render window with cars should appear, and the terminal will print the observation shape and vehicle states):

```bash
python3 src/merge_env.py
```

> **macOS note:** highway-env's render window requires a display. If you're on a headless machine, set `render_mode=None` in `merge_env.py` or run with `python3 -c "..."` style invocations.

> **highway-env patch required:** The `merge-v0` env raises a `ValueError` when receiving a continuous action array. After installing dependencies, apply this one-line patch:
> ```bash
> python3 -c "
> import highway_env, pathlib
> f = pathlib.Path(highway_env.__file__).parent / 'envs/merge_env.py'
> txt = f.read_text()
> txt = txt.replace('action in [0, 2]', 'action in [0, 2] if isinstance(action, (int, float)) else False')
> f.write_text(txt)
> print('Patch applied.')
> "
> ```

---

## Reproducing Phase 4 results (MPC expert)

After setup and the patch above, you can run the MPC expert on a single episode:

```bash
# From the project root with venv active
python3 - <<'EOF'
import sys; sys.path.insert(0, 'src')
import gymnasium as gym, highway_env, numpy as np, random
from driver_types import make_cautious, make_normal, make_aggressive
from mpc_expert import mpc_select_action
from reward import NORMAL

env = gym.make('merge-v0', config={
    'vehicles_count': 3,
    'controlled_vehicles': 1,
    'action': {'type': 'ContinuousAction'},
})
obs, _ = env.reset()
for v in env.unwrapped.road.vehicles[1:]:
    random.choice([make_cautious, make_normal, make_aggressive])(v)

print(f"{'Step':>4}  {'y':>6}  {'vx':>6}  {'action[0]':>9}")
for step in range(25):
    action = mpc_select_action(env, theta=NORMAL)
    obs, _, terminated, truncated, _ = env.step(action)
    ego = env.unwrapped.road.vehicles[0]
    print(f"{step:>4}  {ego.position[1]:>6.2f}  {ego.speed:>6.2f}  {action[0]:>9.4f}")
    if terminated or truncated:
        status = 'crash' if env.unwrapped.vehicle.crashed else 'completed'
        print(f"\nEpisode {status} at step {step}.")
        break

env.close()
EOF
```

**Expected output:** ego stays at y≈4.0 (main highway lane) throughout, speed holds 15–30 m/s, episode completes without crash. MPC call time is printed for the first 5 steps (~12 ms each).

To run the best-response prediction tests:

```bash
python3 src/test_best_response.py
# Saves plots to plots/test_single_follower.png and plots/test_chain_propagation.png
```

