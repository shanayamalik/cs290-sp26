# Implementation Guide
## Autonomous Driving with Multi-Agent Lane Merging

**Deadline:** Presentation — May 8, 2026 | Final Report — May 15, 2026

This guide is the authoritative implementation plan. Follow phases in order. Each phase has a clear goal, concrete files to create, and a definition of done.

---

## Dependencies

```bash
pip install highway-env gymnasium numpy matplotlib scipy torch stable-baselines3
```

No GPU required. Everything runs on CPU.

**Simulator:** Highway-Env (`merge-v0`) — do not build a custom simulator.

---

## Phase 1: Environment Setup
**Goal:** Confirm you can run the merge environment, control vehicles, and read all agent states.

### Step 1 — Run merge-v0 with random actions

Create `merge_env.py`:

```python
import gymnasium as gym
import highway_env  # noqa: F401 — registers environments

env = gym.make("merge-v0", render_mode="human", config={
    "vehicles_count": 3,
    "controlled_vehicles": 1,
    "lanes_count": 2,
    "duration": 20,
    "collision_reward": -10,
    "high_speed_reward": 0.4,
})

obs, info = env.reset()
print("Observation shape:", obs.shape)

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

**Done when:** A render window opens, the ego vehicle (green) moves, and the observation shape prints without error.

### Step 2 — Inspect vehicle states

After `env.reset()`, print `env.road.vehicles` to confirm you can read positions, speeds, and lane indices for all vehicles. The ego vehicle is always `env.road.vehicles[0]`.

---

## Phase 2: Driver Types
**Goal:** Define three reward archetypes. Assign them to non-ego vehicles using IDM parameters.

### Step 3 — Implement driver types

Create `driver_types.py`:

```python
from highway_env.vehicle.behavior import IDMVehicle

def make_cautious(vehicle: IDMVehicle) -> None:
    """Large headway, low speed, high smoothness."""
    vehicle.COMFORT_ACC_MAX = 2.0   # m/s^2
    vehicle.COMFORT_ACC_MIN = -3.0
    vehicle.TIME_WANTED = 2.5       # seconds headway
    vehicle.DISTANCE_WANTED = 10.0  # meters
    vehicle.DELTA = 4               # sensitivity exponent

def make_normal(vehicle: IDMVehicle) -> None:
    """Balanced behavior."""
    vehicle.COMFORT_ACC_MAX = 3.0
    vehicle.COMFORT_ACC_MIN = -5.0
    vehicle.TIME_WANTED = 1.5
    vehicle.DISTANCE_WANTED = 5.0
    vehicle.DELTA = 4

def make_aggressive(vehicle: IDMVehicle) -> None:
    """Short headway, high speed priority."""
    vehicle.COMFORT_ACC_MAX = 4.0
    vehicle.COMFORT_ACC_MIN = -6.0
    vehicle.TIME_WANTED = 1.0       # keep >= 1.0 to avoid spurious collisions early on
    vehicle.DISTANCE_WANTED = 2.0
    vehicle.DELTA = 4
```

> **Note:** These are fixed, hand-defined weight vectors — not learned via IRL. This is a deliberate scope reduction. The methods section should state this explicitly and note that real-time driver type estimation (e.g., SVO inference per Schwarting et al.) is future work.

### Step 4 — Assign driver types after reset

In any script that creates an environment, assign types to non-ego vehicles after `env.reset()`:

```python
import random
from driver_types import make_cautious, make_normal, make_aggressive

driver_fns = [make_cautious, make_normal, make_aggressive]

obs, info = env.reset()
for vehicle in env.road.vehicles[1:]:  # index 0 is always ego
    random.choice(driver_fns)(vehicle)
```

**Done when:** Running a few episodes shows visually distinct behavior — cautious vehicles hang back, aggressive ones push into gaps.

---

## Phase 3: Reward Functions
**Goal:** Implement a feature-based reward function that supports all three driver types.

### Step 5 — Implement reward features

Create `reward.py`. The reward for agent $i$ is $r_i = \theta_i^\top f(s_t, u_t)$, where $f$ has six components:

```python
import numpy as np

# Weight vectors: [forward_progress, proximity_penalty, smoothness, jerk, collision, lane_deviation]
CAUTIOUS   = np.array([0.2, -2.0, -0.5, -0.3, -1000.0, -0.5])
NORMAL     = np.array([0.5, -1.0, -0.3, -0.2, -1000.0, -0.3])
AGGRESSIVE = np.array([0.9, -0.3, -0.1, -0.1, -1000.0, -0.1])


def compute_features(state: dict, action: np.ndarray, prev_action: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    state keys expected: vx, d_min, y, y_target
    action: [acceleration]
    """
    v_desired = 30.0  # m/s
    forward_progress = state["vx"] / v_desired
    proximity_penalty = -1.0 / max(state["d_min"] ** 2, 0.1)
    smoothness = -(action[0] ** 2)
    jerk = -((action[0] - prev_action[0]) / dt) ** 2
    collision = -1.0 if state.get("collision", False) else 0.0
    lane_deviation = -((state["y"] - state["y_target"]) ** 2)

    return np.array([forward_progress, proximity_penalty, smoothness, jerk, collision, lane_deviation])


def ego_reward(state: dict, action: np.ndarray, prev_action: np.ndarray, theta: np.ndarray, dt: float = 0.1) -> float:
    f = compute_features(state, action, prev_action, dt)
    return float(theta @ f)
```

---

## Phase 4: MPC Expert (Iterative Best-Response)
**Goal:** Build an MPC that generates high-quality training trajectories by modeling interdependent driver reactions.

This is the most time-intensive phase. Prioritize it for the May 8 presentation — the MPC trajectories alone are sufficient preliminary results.

### Step 6 — Iterative best-response prediction

Create `best_response.py`. The algorithm:

1. Initialize all non-ego trajectories as constant-speed, straight-line.
2. For each iteration (K = 3–5): update each driver's predicted trajectory given the others' current predictions.
3. If $\|\tau^{(k)} - \tau^{(k-1)}\| < \epsilon$, stop early.
4. **Fallback:** if convergence is not reached, use the last iterate. Non-convergence is expected with aggressive drivers — document it.

```python
import numpy as np

DT = 0.1
HORIZON = 50  # 5 seconds at 0.1s per step
MAX_ITER = 4
EPSILON = 1e-3


def straight_line_trajectory(vehicle, horizon: int = HORIZON, dt: float = DT) -> np.ndarray:
    """Constant-speed, straight-line prediction."""
    traj = []
    x, y, vx = vehicle.position[0], vehicle.position[1], vehicle.speed
    for _ in range(horizon):
        x += vx * dt
        traj.append([x, y, vx])
    return np.array(traj)


def idm_acceleration(speed: float, target_speed: float, gap: float, time_wanted: float) -> float:
    a_max = 3.0
    a_comfort = 5.0
    delta = 4
    s0 = 2.0
    s_star = s0 + max(0, speed * time_wanted)
    acc = a_max * (1 - (speed / max(target_speed, 0.1)) ** delta - (s_star / max(gap, 0.5)) ** 2)
    return float(np.clip(acc, -6.0, 4.0))


def predict_other_responses(env, ego_trajectory: np.ndarray, max_iter: int = MAX_ITER) -> list[np.ndarray]:
    """
    Given a planned ego trajectory, iteratively predict non-ego vehicle trajectories.
    Returns a list of predicted trajectories (one per non-ego vehicle).
    """
    non_ego = env.road.vehicles[1:]
    predicted = [straight_line_trajectory(v) for v in non_ego]

    for _ in range(max_iter):
        prev = [p.copy() for p in predicted]
        for i, vehicle in enumerate(non_ego):
            predicted[i] = idm_predict(vehicle, ego_trajectory, predicted, i)
        if all(np.linalg.norm(predicted[i] - prev[i]) < EPSILON for i in range(len(non_ego))):
            break  # converged

    return predicted  # use last iterate regardless


def idm_predict(vehicle, ego_trajectory: np.ndarray, all_predicted: list[np.ndarray], own_idx: int) -> np.ndarray:
    traj = []
    x, y, vx = vehicle.position[0], vehicle.position[1], vehicle.speed
    time_wanted = getattr(vehicle, "TIME_WANTED", 1.5)
    target_speed = getattr(vehicle, "target_speed", 30.0)

    for t in range(HORIZON):
        # Find the nearest vehicle ahead (ego + other predicted)
        ego_x = ego_trajectory[t, 0] if t < len(ego_trajectory) else ego_trajectory[-1, 0]
        gaps = [ego_x - x]
        for j, other_traj in enumerate(all_predicted):
            if j == own_idx:
                continue
            other_x = other_traj[t, 0] if t < len(other_traj) else other_traj[-1, 0]
            if other_x > x:
                gaps.append(other_x - x)
        gap = max(min(gaps) - 4.5, 0.5)  # 4.5m vehicle length

        acc = idm_acceleration(vx, target_speed, gap, time_wanted)
        vx = max(0.0, vx + acc * DT)
        x += vx * DT
        traj.append([x, y, vx])

    return np.array(traj)
```

### Step 7 — MPC action selection

Create `mpc_expert.py`. For each timestep, sample N action sequences over the planning horizon, simulate ego dynamics, score using `ego_reward`, return the best first action.

```python
import numpy as np
from reward import ego_reward, NORMAL
from best_response import predict_other_responses, HORIZON, DT

N_SAMPLES = 50  # increase to 100+ for better expert quality


def mpc_select_action(env, theta: np.ndarray = NORMAL) -> np.ndarray:
    best_action = np.array([0.0])
    best_score = -np.inf

    ego = env.road.vehicles[0]
    prev_action = np.array([0.0])

    for _ in range(N_SAMPLES):
        actions = np.random.uniform(-4.0, 3.0, size=(HORIZON,))
        score, first_action = evaluate_sequence(env, actions, theta, prev_action)
        if score > best_score:
            best_score = score
            best_action = np.array([first_action])

    return best_action


def evaluate_sequence(env, actions: np.ndarray, theta: np.ndarray, prev_action: np.ndarray) -> tuple[float, float]:
    ego = env.road.vehicles[0]
    x, y, vx = ego.position[0], ego.position[1], ego.speed
    y_target = y  # simplification: stay in current lane

    ego_traj = []
    for acc in actions:
        vx = max(0.0, vx + acc * DT)
        x += vx * DT
        ego_traj.append([x, y, vx])
    ego_traj = np.array(ego_traj)

    predicted_others = predict_other_responses(env, ego_traj)

    total_reward = 0.0
    pa = prev_action.copy()
    for t, acc in enumerate(actions):
        d_min = _min_distance(ego_traj[t], predicted_others, t)
        state = {"vx": ego_traj[t, 2], "d_min": d_min, "y": y_target, "y_target": y_target}
        total_reward += ego_reward(state, np.array([acc]), pa, theta)
        pa = np.array([acc])

    return total_reward, actions[0]


def _min_distance(ego_pos: np.ndarray, others: list[np.ndarray], t: int) -> float:
    distances = []
    for traj in others:
        other_pos = traj[t] if t < len(traj) else traj[-1]
        distances.append(np.linalg.norm(ego_pos[:2] - other_pos[:2]))
    return min(distances) if distances else 100.0
```

**Done when:** You can call `mpc_select_action(env)` at each timestep and the ego vehicle navigates the merge without crashing across multiple episodes.

---

## Phase 5: Expert Dataset Generation
**Goal:** Collect (observation, action) pairs from the MPC expert across diverse scenarios.

### Step 8 — Generate and save the dataset

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

### Step 9 — Define the policy network

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

### Step 10 — Train via behavioral cloning

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

### Step 11 — Fine-tune with PPO

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
