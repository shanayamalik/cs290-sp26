# Phase 7 PPO Fine-Tuning — Changes, Results, and Status

**For Codex review** | ELENG 290, `shanu` branch | May 3, 2026

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

## Status — Results Not Final, Rerun in Progress

The 500k v2 results above (crash=0%, speed=20.22 m/s, reward=250.86) are real but **not the
final numbers** — we are rerunning at 500k again (v3) with one more fix before locking results.

**Why we're rerunning:**

We discovered that `MergeEnv._is_terminated()` checks `position[0] > 370`, but the road's
sine-lane geometry causes `position[0]` to peak at ~333–343m and then *decrease* as the vehicle
follows the curve. The 370m threshold is geometrically unreachable — the `+20` completion reward
and the `terminated=True` branch have **never fired in any of our training runs.**

We are overriding the termination condition in `MergePPOWrapper.step()` with a reachable
threshold based on our diagnostic data:
```python
if x > 330.0 and not crashed:
    terminated = True
    reward += 20.0  # completion bonus
```

**Why 500k again (not 100k):** the 500k v2 run showed PPO needs the full budget to consistently
reach x ≈ 340m. At 100k the policy was still exploring and didn't reliably make it that far.
500k gives the value function enough updates to make the completion bonus actually useful as a
learning signal.

The v3 run (~90 min) will be the first run where PPO can actually learn to complete the merge.
We'll update this file with results once it finishes.

**Code is on the `shanu` branch** — all changes committed and pushed. Nathan can see the full
diff and the current `src/rl_finetune.py` there.

---

## After v3 Completes — Shanaya Will

1. Update this file with v3 eval results (50 episodes)
2. Plot training curve from v2 checkpoints (100k–500k) → presentation figure
3. Coordinate with Nathan on Phase 8 (MPC/BC/PPO cross-eval) once v3 results are in

---

## Branch Status

- Branch: `shanu` (pushed to `origin/shanu`)
- All code committed and pushed — Nathan can view on GitHub
- Model `.zip` files are gitignored (large binaries, stored locally only)

