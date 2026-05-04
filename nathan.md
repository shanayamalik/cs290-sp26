# Phase 7 PPO Fine-Tuning — Changes, Results, and Status

**For Codex review** | ELENG 290, `shanu` branch | May 3, 2026

---

## What Was Done (Phase 7: PPO Fine-Tuning from BC Warm Start)

Starting from Codex's `src/rl_finetune.py` (Nathan's commit `ad89ef4`), Shanaya ran a full
debugging and training session. Here is everything that was changed and why, in order.

---

## Bug Fixes to `src/rl_finetune.py`

### 1. BC model path
- **Before:** `DEFAULT_BC_MODEL = Path("models/bc_policy_all.pt")`
- **After:** `DEFAULT_BC_MODEL = Path("models/bc_policy_default_mix.pt")`
- **Why:** The `all.pt` model didn't exist. `default_mix.pt` is the correct warm-start model.

### 2. Observation space shape mismatch (27 → 28)
- **Before:** `observation_space = spaces.Box(..., shape=(27,), ...)`
- **After:** `shape=(28,)` — ego speed added as 28th feature
- **Why:** PPO needs to observe ego speed directly. In highway-env's ego-relative frame, the ego's
  own velocity appears as ~0 in the raw obs. Without it, PPO cannot modulate acceleration smoothly
  and oscillates. `_augment_obs` appends `float(ego.speed)`.

### 3. BC stats padding (27 → 28)
- `load_bc_stats` pads the 27-dim BC mean/std with `[0.0]` and `[1.0]` so ego speed passes
  through unnormalized.

### 4. Warm-start first layer fix
- `warm_start_actor` copies BC weights into columns `[:27]` of PPO's first linear layer and
  zeros column `[27:]`, so PPO starts behaving identically to BC before training.

### 5. Speed limit in ENV_CONFIG
- Added `"speed_limit": 25` to cap physics and prevent reward hacking (PPO was flooring the
  accelerator, reaching 37+ m/s with 0% clamp rate under the old reward).

### 6. Reward hacking fix
- **Before:** `0.08 * min(speed, 25)` — PPO maximized speed to 25 m/s unconditionally
- **After:** `0.04 * min(speed, 20)` + `-0.15 * (speed - 20)` if speed > 20
- Keeps the policy at a safe highway speed rather than pinning the accelerator.

### 7. MAX_STEPS: 50 → 150
- **Problem discovered via diagnostic:** `terminated` never fires in `merge-v0`. With MAX_STEPS=50
  and `dt=1.0s` per step (`policy_frequency=1`, `simulation_frequency=15`), at 20 m/s the ego
  covers 20m per step → 1000m over 50 steps from spawn (~70m). So the step budget was NOT the
  reason `terminated` never fired — the ego had enough budget to reach the road end.
- The real cause: `MergeEnv._is_terminated()` checks `position[0] > 370`, but the road curves
  (sine-lane merge ramp geometry) such that `position[0]` peaks at ~333–343m and then decreases
  as the vehicle follows the curve. **The threshold of 370 is geometrically unreachable.**
- **Fix applied:** Increased to 150 anyway (gives more training signal via milestones and
  truncation reward). The completion threshold issue is addressed separately (see Open Questions).
- **Diagnostic scripts:** `src/diagnose_termination.py` (random policy) and
  `src/diagnose_termination2.py` (trained model) confirm 10/10 episodes end `truncated=True`.

### 8. Milestone rewards (dense progress signal)
- Added `milestones_hit: set` and `milestone_spacing = 50.0m` to `MergePPOWrapper`.
- Reset each episode. `+5.0` reward each time ego crosses a new 50m milestone.
- Cannot be farmed (only fires once per milestone per episode).
- Gives PPO a breadcrumb trail through the merge zone rather than relying solely on the sparse
  completion reward.

### 9. Truncation-shaped reward
- When step limit is hit (`truncated=True, terminated=False`): `reward += 0.05 * max(x - spawn_x, 0.0)`
- Gives PPO a gradient for reaching further down the road even on incomplete episodes.

### 10. Checkpoint callback updated
- `save_freq=100_000` (was 25k) — appropriate for 500k runs.
- Saves at `models/ppo_500k_v2_merge_100000_steps.zip` through `_500000_steps.zip`.

---

## Training Runs and Results

All runs: `--traffic-mix default_mix`, warm-started from `bc_policy_default_mix.pt`, seed=0.

| Run | Steps | Episodes | Crash | Mean Reward | Mean Speed | Mean Steps | Clamp |
|-----|-------|----------|-------|-------------|------------|------------|-------|
| BC baseline | — | — | — | — | 3.8 m/s | — | 54% |
| Smoke (old) | 5k | 10 | 0.0% | — | 14.6 m/s | ~50 | 0% |
| 20k PPO | 20k | 20 | 0.0% | 58.36 | 23.75 m/s | ~50 | 0% |
| 100k PPO | 100k | 20 | 0.0% | 77.60 | 21.35 m/s | ~50 | 0% |
| 500k v1 (milestones, MAX_STEPS=50) | 500k | 50 | 0.0% | 117.25 | 19.69 m/s | 49.5 | 0% |
| **500k v2 (milestones + MAX_STEPS=150)** | 500k | 50 | 0.0% | **250.86** | **20.22 m/s** | 150.0 | **0%** |

**Note:** `mean_steps=150.0` in the final run means the step budget is still the binding
constraint — `terminated` appears to never fire in `merge-v0` regardless of x-position. The ego
is reaching the end of the road but the env does not set `terminated=True`. The reward numbers
across runs are not directly comparable (reward function changed); use **speed** and **crash rate**
as primary metrics in the paper.

---

## Model Files (gitignored, local only)

- `models/ppo_100k_merge.zip` — 100k baseline
- `models/ppo_500k_merge.zip` — 500k v1 (milestones, old MAX_STEPS=50)
- `models/ppo_500k_v2_merge.zip` — **best model** (milestones + MAX_STEPS=150)
- `models/ppo_500k_v2_merge_{100,200,300,400,500}k_steps.zip` — learning curve checkpoints

---

## Open Questions / Next Steps for Codex

1. **`terminated` never fires — root cause confirmed.** `MergeEnv._is_terminated()` source:
   ```python
   def _is_terminated(self):
       return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
   def _is_truncated(self):
       return False
   ```
   The road's sine-lane geometry means `position[0]` peaks at ~333–343m and then decreases as
   the vehicle follows the curve — the 370m threshold is geometrically unreachable in our config.
   **Proposed fix (not yet implemented):** override in `MergePPOWrapper.step()`:
   ```python
   if x > 330.0 and not crashed:
       terminated = True
       reward += 20.0  # completion bonus
   ```
   This needs a 500k rerun. Expected to be the last fix needed to get completion events.

2. **Phase 8 — independent 2-agent baseline (most important missing piece for the paper).**
   We do not yet have the MPC/independent planner baseline implemented. The paper needs a
   3×4 table (method × traffic mix) comparing MPC, BC, and PPO on crash rate, mean speed,
   and mean reward across all 4 traffic mixes, each evaluated on 50 seeds. BC eval can run
   from `src/cross_eval_bc.py`; PPO eval is in `rl_finetune.py`; MPC baseline needs a
   dedicated eval script.

3. **Immediate priority order (Claude Opus recommendation):**
   1. Implement x > 330 completion fix and rerun 500k → get merge completion events
   2. Plot training curve from 5 checkpoints (100k–500k) → presentation figure
   3. Run unified 3-method × 4-traffic-mix cross-eval → paper results table

---

## Branch Status

- Branch: `shanu` (6 commits ahead of `main`)
- All code changes committed, pushed to `origin/shanu`
- Model `.zip` files gitignored (stored locally only)

---

