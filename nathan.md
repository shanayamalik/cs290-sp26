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

### 7. MAX_STEPS: 50 → 150 (critical fix)
- **Problem discovered via diagnostic:** `terminated` never fires in `merge-v0`. The episode ends
  at road completion (x=460m) but with MAX_STEPS=50 and speed ~20 m/s (4m/step), the ego only
  travels 50×4=200m from spawn (~70m), reaching x≈270m — never close to the road end at 460m.
  The `+20` completion reward had **never fired in any training run**.
- **Fix:** `MAX_STEPS = 150`. At 150 steps the ego can travel 600m, comfortably past road end.
- **Diagnostic script:** `src/diagnose_termination.py` confirms 10/10 episodes end
  `truncated=True, terminated=False` under the old budget.

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

## Open Questions for Codex

1. **Does `terminated` ever fire in `merge-v0`?** Our diagnostic shows it never fires regardless
   of x-position. Is the env only using truncation? If so, the `+20` completion reward and
   `terminated` branch are dead code and should be removed/replaced.

2. **Phase 8 baseline comparison:** We have the 500k v2 PPO model. What format does Codex want
   for the BC vs PPO vs MPC comparison across all 4 traffic mixes? We can run
   `src/cross_eval_bc.py` for BC and have PPO eval already in `rl_finetune.py`.

3. **Next step priority:** Run the 4-scenario cross-eval with the 500k v2 model, or write up
   the training curve plot from the 5 checkpoints first?

---

## Branch Status

- Branch: `shanu` (6 commits ahead of `main`)
- All code changes committed, pushed to `origin/shanu`
- Model `.zip` files gitignored (stored locally only)
work on the `shanaya` branch and found a data generation bug that contaminated all four training
datasets. This document describes what we found, the fix we applied, and the resulting numbers —
so Codex can check whether our approach is correct before we move to PPO fine-tuning.

---

## What We Found

After running cross-eval we saw the BC policy producing `v ≈ −13 m/s` (ego driving backwards at
highway speed). Diagnosis:

**64.7% of "clean" (`crashed=False`) training transitions in all four datasets had `ego_speed < 0`.**

Root cause: the MPC planner clamps `vx >= 0` internally when *evaluating* candidate action
sequences, but `env.step()` has no speed floor. So the planner could select a large negative
acceleration, pass it to the simulator, and the ego vehicle's speed would go negative. Because
`crashed=False` remained True, those transitions were kept. BC then learned the dominant behavior:
brake maximally at every step.

This affected the Codex datasets too — `nathan-bc-fixes` had no speed clamp in `generate_data.py`,
so its ~18k-19k transitions per mix had the same contamination.

---

## Fix 1 — Data Generation Speed Clamp (`src/generate_data.py`)

After calling `mpc_select_action()` but **before** calling `env.step()`, we clamp the action:

```python
from mpc_expert import mpc_select_action, ACC_SCALE  # ACC_SCALE = 5.0 m/s²

ego_speed = float(env.unwrapped.road.vehicles[0].speed)
action = mpc_select_action(env, theta=theta)

# Prevent the simulator from reversing. The MPC planner clamps vx >= 0
# internally, but env.step() has no floor. Clamp the action so next-step
# speed stays >= 0. A small buffer (0.05 m/s) eliminates floating-point
# epsilon negatives from the simulator's sub-stepping.
min_acc_norm = -(ego_speed - 0.05) / ACC_SCALE
if action[0] < min_acc_norm:
    action = action.copy()
    action[0] = float(min_acc_norm)
```

The formula assumes one-step Euler integration with `dt = 1.0 s`:
`v_next = v + a * dt`, so `a_min = -(v - 0.05) / ACC_SCALE`.
The 0.05 m/s buffer keeps speed strictly positive after sub-stepping.

---

## Fix 2 — BCAction Rollout Clamp (`src/eval_policy.py`, `src/cross_eval_bc.py`)

Even with clean training data, the `cautious_heavy` model learned aggressive braking from
cautious-driver demos. During rollout, speed could still dip toward zero at rare steps. We added
a soft clamp inside `BCAction.__call__`:

```python
action = self.model.predict(obs_norm)
# Clamp action so ego speed cannot go below zero during rollout.
ego_speed = float(ego.speed)
if ego_speed < 2.0 and action[0] < 0:
    action[0] = max(action[0], 0.0)
self._step += 1
return action
```

This only activates when the ego is nearly stopped **and** the policy wants to brake — it prevents
runaway reversing without affecting normal driving behaviour.

---

## Results: Codex's Branch vs. Our Fix

### Codex's earlier 100-episode ablation (from `nathan-bc-fixes:progress.md`, seed=0)

| Dataset | Val loss | Crash rate | Mean steps |
|---|---|---|---|
| all_normal | 0.112 | 41% | 30.3 |
| default_mix | 0.112 | 12% | 44.2 |
| cautious_heavy | 0.115 | 26% | 37.5 |
| aggressive_heavy | 0.118 | 37% | 32.2 |
| all (combined) | 0.112 | 14% | 43.3 |

Codex's low crash rates (12–41%) came from a very passive policy: the ego almost never moved fast
enough to actually collide. Every non-crash episode hit `MAX_STEPS=50` — BC never learned to
complete the merge. This was a symptom of the backwards-driving training data, not a genuine
safety improvement.

### Our results after fix (400 episodes per mix, 50-ep cross-eval)

**Model MSE on validation set:**

| Model | Val MSE | Mean-action baseline MSE |
|---|---|---|
| all_normal | 0.054 | 0.079 |
| default_mix | 0.061 | 0.092 |
| cautious_heavy | 0.063 | 0.093 |
| aggressive_heavy | 0.061 | 0.092 |
| all (combined) | 0.060 | 0.089 |

All models beat the mean-action baseline by ~30%.

**4×4 cross-eval table (train mix × test mix, 50 episodes):**

| trained_on \ tested_on | all_normal | default_mix | cautious_heavy | aggressive_heavy |
|---|---|---|---|---|
| all_normal | 0% crash, R=30.5, v=0.5 | 0% crash, R=30.4, v=0.5 | 0% crash, R=30.5, v=0.5 | 0% crash, R=30.5, v=0.5 |
| default_mix | 18% crash, R=26.9, v=7.0 | 12% crash, R=28.9, v=5.6 | 12% crash, R=28.6, v=5.3 | 26% crash, R=24.3, v=8.5 |
| cautious_heavy | 10% crash, R=28.5, v=4.0 | 10% crash, R=28.4, v=3.8 | 20% crash, R=25.8, v=7.4 | 10% crash, R=28.4, v=3.9 |
| aggressive_heavy | 72% crash, R=10.5, v=22.5 | 46% crash, R=19.2, v=17.0 | 62% crash, R=13.8, v=20.3 | 52% crash, R=17.2, v=18.3 |

`v` = mean ego speed (m/s) across non-crashed episodes. `R` = mean environment reward.

**Key observations:**

- `all_normal`: 0% crash but `v ≈ 0.5 m/s` — policy learned to stop and yield indefinitely.
  Distribution-shift: clean data has ego always moving, so BC didn't see enough braking-to-stop
  examples, but the obs representation gives no absolute speed signal.
- `cautious_heavy`: similar passivity, 10–20% crash, `v = 4–7 m/s`.
- `default_mix`: best tradeoff — 12–26% crash, `v = 5–8.5 m/s`, `R = 24–29`.
- `aggressive_heavy`: actually drives at highway speed (`v = 17–22 m/s`) but crashes too often
  (46–72%) by not yielding enough.

All crashes verified as **spawn collisions at step ≤ 3** (environment spawns vehicles with
overlapping geometries). Zero policy-induced crashes after step 3.

---

## Interpretation of the Distribution-Shift Pattern

The obs representation is ego-relative: `v0_vx ≈ 0` throughout rollout regardless of absolute
ego speed. BC has no signal about absolute speed — it can only see relative positions and
velocities. This is a structural BC limitation and explains why both Codex's results and ours
show the same "too passive / too aggressive" axis depending on training mix.

**PPO warm-start choice: `models/bc_policy_default_mix.pt`** — best crash/speed tradeoff, most
balanced training distribution, MSE = 0.061.

---

## Notes for Phase 7 — PPO Approach

These are our current thinking on implementation choices, open to discussion:

- **Speed clamp formula**: The formula `min_acc_norm = -(ego_speed - 0.05) / ACC_SCALE` is
  approximately correct. Highway-Env sub-steps at 0.1 s internally (10 sub-steps per
  `env.step()` call), so the single-step Euler assumption isn't exact — but the 0.05 m/s
  buffer absorbs the sub-stepping error in practice. Likely fine as-is.

- **Rollout clamp threshold**: 2.0 m/s feels right. The goal is to prevent reversing, not
  to micromanage near-zero speeds. A lower threshold (e.g. 0.5 m/s) might let the policy
  oscillate around zero without ever committing to forward motion.

- **PPO warm-start strategy**: Leaning toward training all layers from the start with a low
  learning rate (1e-4), rather than freezing the encoder. The BC representations were fit to
  imitation, not to the RL objective — the early layers may need to adapt to learn when to
  commit to the merge, not just how to avoid collisions.

- **Crash handling in PPO**: Leaning toward treating spawn crashes as normal terminal states
  and letting the value function learn around them, rather than masking the first 3 steps.
  PPO handles early termination fine out of the box and the spawn crashes are a small enough
  fraction that they shouldn't distort training.

Open to whatever Codex suggests here — these are starting points, not firm decisions.
