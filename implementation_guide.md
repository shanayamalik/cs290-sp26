# Implementation Guide
## Autonomous Driving with Multi-Agent Lane Merging

**Deadline:** Presentation — May 8, 2026 | Final Report — May 15, 2026

> **Before starting:** Follow the setup instructions in [README.md](README.md) to create a virtual environment and install all dependencies. The phases below assume `venv` is active.

---

## ✅ Phase 1: Environment Setup — COMPLETE

See [README.md](README.md). Note: use `env.unwrapped` to access highway-env internals through gymnasium's wrappers — `env.road` will raise an `AttributeError`.

---

## ✅ Phase 2: Driver Types — COMPLETE

See `driver_types.py`. Three types implemented (cautious, normal, aggressive) by overriding IDM parameters on `IDMVehicle` instances. Normal driver uses Treiber et al. (2000) reference values (T=1.5 s, s0=2 m, delta=4); cautious and aggressive are scaled relative to normal. SVO taxonomy mapping from Schwarting et al. (2019) — see module docstring for full citations and DOIs.

Driver types are randomly assigned to non-ego vehicles in `merge_env.py` via `assign_driver_types()` after every `env.reset()`. Vehicles[0] is always ego; vehicles[1:] receive a random archetype.

---

## ✅ Phase 3: Reward Functions — COMPLETE

See [src/reward.py](src/reward.py). The reward is $r_i = \theta_i^\top f(s_t, u_t)$ with six features. The three numbers are for cautious/normal/aggressive. Key design decisions:

- **forward_progress (0.2/0.5/0.9):** normalized feature in [0,1]; ordering encodes SVO angle φ (Schwarting et al. 2019)
- **proximity_penalty (−2.0/−1.0/−0.5):** 1/d² repulsive potential (Khatib 1986), 4:2:1 ratio across types
- **smoothness/jerk:** secondary comfort terms (Bae et al. 2020), intentionally small — tune after MPC works
- **collision (−1000, all types):** dominates any horizon (50 steps × max 0.9 = 45 << 1000); no per-step gain justifies a crash
- **lane_deviation (−0.5/−0.3/−0.1):** y_target must be the *target* lane center, not current lane

---

## ✅ Phase 4: MPC Expert (Iterative Best-Response) — COMPLETE

See `src/best_response.py` and `src/mpc_expert.py`. Committed to `naya` branch (final commit `b38f98d`).

**Key confirmed facts:**
- Ego is the **highway vehicle** (y=4.0), not the merging one. NPC from ramp (y=14.5) merges in. Task = pure longitudinal gap management; `Y_TARGET=4.0`, steering fixed at 0.
- `ACC_SCALE=5.0`, `DT_PLAN=1.0s`, `HORIZON=20`, `N_SAMPLES=50`, `N_WAYPOINTS=6`
- Sampling: 3 structured candidates (constant-speed/brake/accel) + `normal(0, 0.4)` random waypoints
- Verification pass: winning trajectory re-scored against responses re-predicted from that trajectory (~2ms extra)
- `COLLISION_DIST=6.0m` (vehicles are ~5m long; 6m gives a small safety buffer)
- `dt=DT_PLAN=1.0` passed explicitly to `ego_reward` (default was 0.1, inflating jerk 100×)
- `proximity_penalty` feature = `+1/d²` (positive); negative weight makes contribution repulsive
- Lane deviation weights zeroed (ego is not merging; `y = y_target = 4.0` always)

**Full-episode result:** Ego stays at y=4.0, holds 15–29 m/s, traverses full a→b→c→d corridor without crash. ~12ms/call.

---

## ✅ Phase 5: Expert Dataset Generation — COMPLETE

See `src/generate_data.py`. Committed to `naya` branch.

**Key design decisions:**
- Raw 5×5 obs saved (not flattened); flatten at training time with `obs.reshape(-1)`
- Crashed episodes kept with `crashed=True` flag — filter at training time, exclude from BC
- `MAX_STEPS=50` cap prevents runaway episodes (two 400+ step outliers observed without cap)
- `ego_speed` and `d_min` saved per step for trajectory plots without decoding obs
- `theta_name` saved as string for type-conditioned analysis

**Results (200 episodes):**
- 3564 total transitions, 3169 clean (non-crashed)
- 26.5% crash rate — dominated by spawn collisions (terminated within 4–8 steps); MPC had no time to react
- Clean transitions per type: cautious 968, normal 1209, aggressive 992 (well-balanced, ≤1.25× ratio)
- Saved to `data/expert_dataset.pkl` (excluded from git via `.gitignore`)

To regenerate:
```bash
python3 src/generate_data.py --episodes 200   # full run
python3 src/generate_data.py --episodes 5     # sanity check
```

---

## Phase 6: Behavioral Cloning (Policy Distillation)
**Goal:** Train a neural network to imitate the MPC expert via supervised learning.

### Define the policy network

Create `policy_network.py`:

```python
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### Train via behavioral cloning

Create `train_policy.py`:

```python
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from policy_network import PolicyNetwork

with open("expert_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

obs_list, act_list = zip(*dataset)
obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
obs_tensor = obs_tensor.view(obs_tensor.size(0), -1)
act_tensor = torch.tensor(np.array(act_list), dtype=torch.float32)
if act_tensor.dim() == 1:
    act_tensor = act_tensor.unsqueeze(1)

obs_dim, action_dim = obs_tensor.shape[1], act_tensor.shape[1]
model = PolicyNetwork(obs_dim, action_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(obs_tensor, act_tensor), batch_size=64, shuffle=True)

EPOCHS = 50
for epoch in range(EPOCHS):
    total_loss = 0.0
    for obs_batch, act_batch in loader:
        pred = model(obs_batch)
        loss = loss_fn(pred, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "distilled_policy.pt")
print("Saved distilled_policy.pt")
```

**Done when:** Loss decreases over epochs. Roll out the cloned policy in the environment and compare trajectories to the MPC expert side-by-side.

> **Expected limitation:** Compounding errors at long horizons — this is normal and is exactly why RL fine-tuning is the next step.

---

## Phase 7: RL Fine-Tuning
**Goal:** Fine-tune the distilled policy with PPO to minimize total merge completion time.

Warm-starting PPO from the behavioral cloning weights is the methodologically important step — it must be implemented properly, not left as a comment.

### Fine-tune with PPO

Create `rl_finetune.py`:

```python
import gymnasium as gym
import highway_env  # noqa
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from policy_network import PolicyNetwork

env = gym.make("merge-v0", config={"vehicles_count": 3, "controlled_vehicles": 1})

# Train PPO with warm-started weights
model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4, n_steps=2048, batch_size=64)

# Warm-start: load distilled policy weights into PPO's mlp_extractor / action_net
obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1] if len(env.observation_space.shape) > 1 else env.observation_space.shape[0]
action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else 1
distilled = PolicyNetwork(obs_dim, action_dim)
distilled.load_state_dict(torch.load("distilled_policy.pt"))

# Map distilled weights into PPO policy — adjust layer names as needed after inspecting model.policy
# model.policy.mlp_extractor.policy_net[0].weight.data = distilled.net[0].weight.data  # example

model.learn(total_timesteps=500_000)
model.save("rl_finetuned_policy")
print("Saved rl_finetuned_policy")
```

> **Note on warm-starting:** After running this once, inspect `model.policy` to see the exact layer names, then map the distilled weights layer by layer. This is the part to spend time on — document the mapping in the methods section.

**RL reward function:** Penalize total merge completion time across all vehicles + large collision penalty (−1000). Do not reuse the per-driver reward from Phase 3 directly — the RL objective is global efficiency.

---

## Phase 8: Baseline

Create `baseline.py`. The baseline is independent 2-agent planning: the MPC plans separately for each human driver as if no other humans exist (i.e., `predict_other_responses` sees only one non-ego vehicle at a time). Re-run the same evaluation scenarios with this baseline and compare metrics.

---

## Phase 9: Evaluation

Run 20 episodes per scenario. Record for each episode:
- **Merge success** — no collision during episode
- **Merge completion time** — steps until all vehicles clear the merge zone
- **Minimum TTC** — minimum time-to-collision across all timesteps in the episode

### Scenarios

| Scenario | Ego | Others |
|---|---|---|
| Human-AV | RL policy | 2 human IDM vehicles (random type each episode) |
| Human-Mixed AV | RL policy | 1 human IDM + 1 additional RL policy vehicle |
| Multi-AV | RL policy | 2 additional RL policy vehicles (no humans) |

Create `evaluate.py` with a function that accepts a scenario config and runs N episodes, returning the three metrics. Compare Our Method vs. Baseline in a table.

### Plots to generate for the paper
- Bar charts: merge success rate and avg completion time across 3 scenarios × 2 methods
- Line plot: PPO training curve (episode reward vs. timesteps)
- Trajectory plots: a successful merge for each scenario (qualitative)

---

## Phase 10: Paper Writeup

**Format:** IEEE, 5–9 pages. **Due:** May 15, 2026.

### Structure
1. Abstract
2. Introduction — motivation, gap (single-pair → multi-agent), contributions
3. Related Work — Sadigh, Zhang, Schwarting, Huang, **Weil et al.** (add this), + survey paper
4. Problem Formulation — state space, action space, dynamics, reward functions, driver archetypes
5. Approach — three-stage pipeline with equations
6. Experiments — three scenarios, metrics, baseline description
7. Results — tables and plots
8. Discussion — what worked, non-convergence observations, limitations (known driver types), future work (IRL / SVO estimation)
9. Conclusion
10. References

### Key things to state explicitly in the methods section
- Reward functions are hand-defined, not learned (contrast with Huang et al.)
- Driver types are assumed known at planning time (limitation)
- Iterative best-response uses last iterate as fallback when convergence fails
- Warm-starting procedure for PPO from behavioral cloning weights
- Highway-Env (`merge-v0`) is the simulator

---

## Timeline

| Dates | Phase |
|---|---|
| Apr 26 – Apr 28 | Phase 1–2: Environment + driver types running |
| Apr 29 – May 1 | Phase 3–4: Reward functions + MPC expert working |
| May 2 – May 4 | Phase 5: Expert dataset generated, trajectory plots ready |
| May 5 – May 7 | Phase 6–7: Behavioral cloning + RL fine-tuning started |
| **May 8** | **Presentation — need: problem statement, MPC results, BC results, RL in progress** |
| May 9 – May 11 | Phase 8–9: Baseline + evaluation complete |
| May 12 – May 15 | Phase 10: Paper writeup |
