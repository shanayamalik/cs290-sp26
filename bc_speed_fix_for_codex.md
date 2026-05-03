# BC Speed Bug Fix — Changes, Reasoning, and Results

**For Codex review** | ELENG 290, `shanaya` branch | May 2, 2026

---

## Background

Codex suggested the BC implementation on the `nathan-bc-fixes` branch (400 episodes per driver mix,
`src/policy_network.py`, `src/train_policy.py`, `src/cross_eval_bc.py`). We built on top of that
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

## Questions for Codex

1. **Speed clamp formula**: Is `min_acc_norm = -(ego_speed - 0.05) / ACC_SCALE` correct for
   one-step Euler with `dt = 1.0 s`? Highway-Env uses internal sub-stepping at `dt = 0.1 s`
   (10 sub-steps per `env.step()` call) — does this change the formula?

2. **Rollout clamp threshold**: We used 2.0 m/s as the activation threshold. Is that too
   conservative? A lower value (e.g., 0.5 m/s) would let the policy brake more freely and
   might better reflect the expert's cautious behaviour.

3. **PPO warm-start strategy**: Should we freeze BC encoder layers for the first N steps, or
   train all layers from the start with a reduced learning rate (e.g., 1e-4)? The BC actor
   is already a reasonable initialisation — we want to preserve the yielding behaviour while
   letting PPO learn when to commit to the merge.

4. **Crash handling in PPO**: All BC crashes are spawn collisions at step ≤ 3. Should PPO
   training mask out episode reward for the first 3 steps, or just accept those crashes as
   terminal states and let the value function learn around them?
