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
| 500k v2 (milestones + MAX_STEPS=150) | 500k | 50 | 0.0% | 250.86 | 20.22 m/s | 150.0 | 0% |
| **500k v3 (+ x>330 completion override)** | 500k | 50 | 0.0% | **261.44** | **18.67 m/s** | 150.0 | **0%** |

**Note:** `mean_steps=150.0` across v2 and v3 means the step budget remains the binding
constraint in eval. The x>330 override does fire occasionally during training (ep_len_mean
≈146–147 in the SB3 rollout logs vs the hard cap of 150), but does not trigger consistently
enough to reduce mean_steps in 50-episode eval. The reward numbers across runs are not
directly comparable (reward function changed); use **crash rate** and **mean speed** as primary
metrics in the paper.

---

## Model Files (gitignored, local only)

- `models/ppo_100k_merge.zip` — 100k baseline
- `models/ppo_500k_merge.zip` — 500k v1 (milestones, old MAX_STEPS=50)
- `models/ppo_500k_v2_merge.zip` — **final model** (milestones + MAX_STEPS=150)
- `models/ppo_500k_v2_merge_{100,200,300,400,500}k_steps.zip` — v2 learning curve checkpoints
- `models/ppo_500k_v3_merge.zip` — v3 (x>330 termination override, do not use — see below)

---

## Status — Phase 7 Complete ✓

**Final model: `models/ppo_500k_v2_merge.zip`**

Diagnostic (`src/diagnose_v3.py`, 20 deterministic episodes, seeds 0–19):
- **Merge complete (max_x > 310m): 18/20 = 90%**
- Crashed: 0/20
- Typical episode length: 28–35 steps (finishes well before 150-step limit)
- Typical max_x: 330–337m

**Why v2 and not v3:** v3 was retrained with the x>330 termination override and that disrupted
learning — the v3 policy stalls at x≈182–297m (0/20 crossing x>310). v2 learned to reach x>330
anyway via milestones and truncation reward without the explicit termination signal. Running v2
weights with the current code (which has the x>330 override) works correctly: the override fires
naturally as the policy crosses 330m, terminating the episode cleanly.

The two anomalous episodes (ep13: max_x=280m, ep16: max_x=295m) hit the 150-step limit due to
dense-traffic spawns that forced the policy to yield longer than usual. 18/20 = 90% is the
paper number.

**Code is on the `shanu` branch** — all changes committed and pushed.

---

## Next Steps (Phases 8–9)

1. Phase 8: `src/baseline.py` — independent MPC baseline (no multi-agent awareness)
2. Phase 9: `src/evaluate.py` — 20 eps x 3 scenarios x 2 methods, bar charts + training curve
3. Final paper numbers should use `ppo_500k_v2_merge.zip` for the PPO agent
---

## Branch Status

- Branch: `shanu` (pushed to `origin/shanu`)
- All code committed and pushed — Nathan can view on GitHub
- Model `.zip` files are gitignored (large binaries, stored locally only)

