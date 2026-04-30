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

### Step 0: Switch to ContinuousAction — do this first

Before writing any MPC code, add `"action": {"type": "ContinuousAction"}` to the env config in `merge_env.py` and verify that `env.step(np.array([0.0]))` runs without error. Nothing downstream works without this.

### Implementation order

Work through these in sequence, verifying each piece in isolation before moving on:

1. **ContinuousAction** — confirm `env.step()` accepts a float array
2. **`straight_line_trajectory` + `idm_predict`** — test in isolation, plot a few predicted trajectories to sanity-check IDM behavior
3. **Basic MPC loop** — 10 samples, 20-step horizon, confirm ego does something sensible
4. **Scale up** — increase samples and horizon once the basic loop works

### `best_response.py` — trajectory prediction

The algorithm:

1. Initialize all non-ego trajectories as constant-speed, straight-line.
2. For each iteration (K = 3–5): update each driver's predicted trajectory given the others' current predictions.
3. Convergence check: use **per-step maximum difference** (`max_t max_i |τ^(k)_i(t) - τ^(k-1)_i(t)| < ε`), not the full trajectory norm. Full norm scales with horizon length and is too sensitive. `EPSILON = 0.1` (in meters/m/s over a single step).
4. **Fallback:** if convergence is not reached after MAX_ITER, use the last iterate. Non-convergence is expected with aggressive drivers — document it in the paper's methods section.

**Key design decision:** To avoid calling `predict_other_responses` once per MPC sample (50 samples × 3–5 iterations × N vehicles = very slow), **pre-compute the best-response prediction once per timestep** using a nominal ego trajectory (constant current speed), then reuse those predicted other-vehicle trajectories to score all N candidate sequences. Accuracy tradeoff is acceptable — the MPC expert needs to produce useful training data, not be perfectly accurate. This gives a ~50x speedup.

**`idm_predict` bug fixes required:**
- **Positive-gap filter:** only consider vehicles where `other_x > x` (other vehicle is *ahead*). The current plan includes ego_x indiscriminately — if ego is behind this vehicle, that's not a leading vehicle.
- **Lane check:** add a lateral distance threshold (`|other_y - y| < LANE_WIDTH_THRESHOLD`, e.g. 4 m). In `merge-v0`, the merge lane is laterally offset — vehicles beside each other in different lanes should not trigger IDM braking.

```python
import numpy as np

DT = 0.1
HORIZON = 50       # 5 seconds at 0.1s per step
MAX_ITER = 4
EPSILON = 0.1      # per-step max diff in meters or m/s — NOT full trajectory norm
LANE_WIDTH_THRESHOLD = 4.0   # meters; vehicles farther apart laterally ignored by IDM
VEHICLE_LENGTH = 4.5         # meters


def straight_line_trajectory(vehicle, horizon: int = HORIZON, dt: float = DT) -> np.ndarray:
    """Constant-speed, straight-line prediction. Shape: (horizon, 3) = [x, y, vx]."""
    x, y, vx = vehicle.position[0], vehicle.position[1], vehicle.speed
    traj = []
    for _ in range(horizon):
        x += vx * dt
        traj.append([x, y, vx])
    return np.array(traj)


def idm_acceleration(speed: float, target_speed: float, gap: float, time_wanted: float,
                     a_max: float = 1.5, s0: float = 2.0, delta: int = 4) -> float:
    # Default a_max=1.5 matches the normal driver archetype (Treiber 2000 reference values).
    # idm_predict always passes vehicle.COMFORT_ACC_MAX explicitly, so the default
    # only matters if idm_acceleration is called in isolation.
    s_star = s0 + max(0.0, speed * time_wanted)
    acc = a_max * (1 - (speed / max(target_speed, 0.1)) ** delta - (s_star / max(gap, 0.5)) ** 2)
    return float(np.clip(acc, -6.0, 4.0))


def idm_predict(vehicle, ego_trajectory: np.ndarray, all_predicted: list, own_idx: int) -> np.ndarray:
    """
    Predict this vehicle's trajectory given ego and other vehicles' predicted trajectories.
    Bug fixes vs. naive version:
      - Only considers vehicles strictly ahead (positive longitudinal gap).
      - Ignores vehicles in different lanes (lateral distance > LANE_WIDTH_THRESHOLD).
    """
    x, y, vx = vehicle.position[0], vehicle.position[1], vehicle.speed
    time_wanted = getattr(vehicle, "TIME_WANTED", 1.5)
    target_speed = getattr(vehicle, "target_speed", 30.0)
    traj = []

    for t in range(HORIZON):
        ego_pos = ego_trajectory[t] if t < len(ego_trajectory) else ego_trajectory[-1]
        candidate_gaps = []

        # Check ego
        if ego_pos[0] > x and abs(ego_pos[1] - y) < LANE_WIDTH_THRESHOLD:
            candidate_gaps.append(ego_pos[0] - x)

        # Check other non-ego vehicles
        for j, other_traj in enumerate(all_predicted):
            if j == own_idx:
                continue
            other_pos = other_traj[t] if t < len(other_traj) else other_traj[-1]
            if other_pos[0] > x and abs(other_pos[1] - y) < LANE_WIDTH_THRESHOLD:
                candidate_gaps.append(other_pos[0] - x)

        gap = max(min(candidate_gaps) - VEHICLE_LENGTH, 0.5) if candidate_gaps else 100.0
        acc = idm_acceleration(vx, target_speed, gap, time_wanted)
        vx = max(0.0, vx + acc * DT)
        x += vx * DT
        traj.append([x, y, vx])

    return np.array(traj)


def predict_other_responses(env, ego_trajectory: np.ndarray, max_iter: int = MAX_ITER) -> list:
    """
    Given a nominal ego trajectory, iteratively predict non-ego vehicle trajectories.
    Returns a list of predicted trajectories (one per non-ego vehicle).
    Convergence uses per-step max diff, not full trajectory norm.
    """
    non_ego = env.unwrapped.road.vehicles[1:]
    predicted = [straight_line_trajectory(v) for v in non_ego]

    for _ in range(max_iter):
        prev = [p.copy() for p in predicted]
        for i, vehicle in enumerate(non_ego):
            predicted[i] = idm_predict(vehicle, ego_trajectory, predicted, i)
        # Per-step max diff convergence check
        diffs = [np.max(np.abs(predicted[i] - prev[i])) for i in range(len(non_ego))]
        if max(diffs) < EPSILON:
            break  # converged

    return predicted  # use last iterate regardless
```

### `mpc_expert.py` — action selection

**Waypoint interpolation instead of per-step random sampling.** Sampling 50 random sequences over a 50-dimensional space produces mostly garbage trajectories. Instead, sample 6 acceleration *waypoints* and linearly interpolate to the full horizon — reduces the search space from 50-dim to 6-dim, which 50 samples can actually cover meaningfully.

**Steering fixed at 0.** The merge task is a gap-acceptance timing problem, not a lateral positioning problem. Sampling both steering and acceleration doubles the search space with no benefit until longitudinal control is confirmed as the bottleneck. The MPC output is shape `(2,)` = `[0.0, acceleration_normalized]` to match the confirmed action space `Box(-1, 1, (2,), float32)`. Note: waypoint acceleration values are in raw m/s²; they must be normalized to `[-1, 1]` before being passed to `env.step()`.

**Pre-compute other-vehicle responses once** using a nominal ego trajectory (constant current speed), then reuse across all N candidate sequences. This is the ~50x speedup noted above.

**State dict helper:** `_extract_state` builds the dict expected by `ego_reward` from simulated positions — `d_min` uses Euclidean distance (not just longitudinal), so two vehicles at the same x but different y do not produce a spuriously small gap.

**Timing:** add a timing wrapper around the first few `mpc_select_action` calls. At 1 s/step × 200 steps/episode × 100 episodes = 5+ hours. If a single call exceeds ~0.5 s, reduce `N_SAMPLES` or `HORIZON` before starting data generation.

```python
import time
import numpy as np
from reward import ego_reward, NORMAL
from best_response import predict_other_responses, straight_line_trajectory, HORIZON, DT

N_SAMPLES = 50      # start with 10 while debugging; increase to 50–100 for data generation
N_WAYPOINTS = 6     # number of acceleration waypoints to sample (interpolated to full horizon)
COLLISION_DIST = 3.0  # meters; ego positions within this distance count as collision
ACC_SCALE = 4.0     # maps raw m/s² to [-1, 1]: normalized = raw / ACC_SCALE


def mpc_select_action(env, theta: np.ndarray = NORMAL, _time_calls: list = []) -> np.ndarray:
    t0 = time.perf_counter()
    ego = env.unwrapped.road.vehicles[0]

    # Pre-compute other-vehicle responses once using nominal (constant-speed) ego trajectory
    nominal_ego_traj = straight_line_trajectory(ego)
    predicted_others = predict_other_responses(env, nominal_ego_traj)

    # y_target: center of the target lane (the lane ego is merging into, not current lane)
    # In merge-v0 the main lane is at y=0; adjust if needed based on env inspection
    y_target = 0.0

    best_acc = 0.0
    best_score = -np.inf

    for _ in range(N_SAMPLES):
        # Sample acceleration waypoints in raw m/s², interpolate to full horizon
        waypoints = np.random.uniform(-ACC_SCALE, ACC_SCALE * 0.75, size=(N_WAYPOINTS,))
        actions = np.interp(
            np.arange(HORIZON),
            np.linspace(0, HORIZON - 1, N_WAYPOINTS),
            waypoints,
        )
        score, first_acc = evaluate_sequence(ego, actions, predicted_others, theta, y_target)
        if score > best_score:
            best_score = score
            best_acc = first_acc

    elapsed = time.perf_counter() - t0
    _time_calls.append(elapsed)
    if len(_time_calls) <= 5:
        print(f"[MPC timing] call {len(_time_calls)}: {elapsed:.3f}s")

    # Return (2,) float32 action: steering=0, acceleration normalized to [-1, 1]
    return np.array([0.0, best_acc / ACC_SCALE], dtype=np.float32)


def evaluate_sequence(ego, actions: np.ndarray, predicted_others: list,
                      theta: np.ndarray, y_target: float):
    x, y, vx = ego.position[0], ego.position[1], ego.speed
    ego_traj = []
    for acc in actions:
        vx = max(0.0, vx + acc * DT)
        x += vx * DT
        ego_traj.append([x, y, vx])
    ego_traj = np.array(ego_traj)

    total_reward = 0.0
    pa = np.array([0.0])
    for t, acc in enumerate(actions):
        state = _extract_state(ego_traj[t], predicted_others, t, y_target)
        total_reward += ego_reward(state, np.array([acc]), pa, theta)
        pa = np.array([acc])

    return total_reward, actions[0]


def _extract_state(ego_pos: np.ndarray, others: list, t: int, y_target: float) -> dict:
    """Build the state dict expected by ego_reward from simulated positions."""
    distances = []
    for traj in others:
        other_pos = traj[t] if t < len(traj) else traj[-1]
        distances.append(float(np.linalg.norm(ego_pos[:2] - other_pos[:2])))
    d_min = min(distances) if distances else 100.0
    collision = int(d_min < COLLISION_DIST)
    return {
        "vx": float(ego_pos[2]),
        "d_min": d_min,
        "y": float(ego_pos[1]),
        "y_target": y_target,
        "collision": collision,
    }


def _min_distance(ego_pos: np.ndarray, others: list, t: int) -> float:
    distances = []
    for traj in others:
        other_pos = traj[t] if t < len(traj) else traj[-1]
        distances.append(float(np.linalg.norm(ego_pos[:2] - other_pos[:2])))
    return min(distances) if distances else 100.0
```

**Done when:** You can call `mpc_select_action(env)` at each timestep and the ego vehicle navigates the merge without crashing across multiple episodes. Plot a few rollouts before moving to Phase 5.

---

## Phase 5: Expert Dataset Generation
**Goal:** Collect (observation, action) pairs from the MPC expert across diverse scenarios.

### Generate and save the dataset

Create `generate_data.py`:

```python
import numpy as np
import pickle
import random
import gymnasium as gym
import highway_env  # noqa

from mpc_expert import mpc_select_action
from driver_types import make_cautious, make_normal, make_aggressive
from reward import CAUTIOUS, NORMAL, AGGRESSIVE

N_EPISODES = 100  # reduce to 20 for a quick sanity check first
driver_fns = [make_cautious, make_normal, make_aggressive]
thetas = [CAUTIOUS, NORMAL, AGGRESSIVE]

env = gym.make("merge-v0", config={"vehicles_count": 3, "controlled_vehicles": 1})
dataset = []  # list of (observation, action) pairs

for episode in range(N_EPISODES):
    obs, info = env.reset()
    ego_theta = random.choice(thetas)

    for vehicle in env.road.vehicles[1:]:
        random.choice(driver_fns)(vehicle)

    terminated = truncated = False
    while not (terminated or truncated):
        action = mpc_select_action(env, theta=ego_theta)
        dataset.append((obs.copy().flatten(), action.copy()))
        obs, _, terminated, truncated, _ = env.step(action)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{N_EPISODES} — dataset size: {len(dataset)}")

env.close()

with open("expert_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print(f"Dataset saved: {len(dataset)} transitions")
```

**Done when:** `expert_dataset.pkl` exists and contains tens of thousands of (obs, action) pairs. Generate a few trajectory plots to use as preliminary results figures.

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
