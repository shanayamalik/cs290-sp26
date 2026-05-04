# Progress Update — May 3, 2026

**Shanaya Malik, Nathan McNaughton** | ELENG 290 — Multi-Agent Lane Merging

---

## Project

Extending Sadigh et al. (2016) from a single AV–human pair to multi-vehicle lane merging with 3+ agents. Three-stage pipeline: iterative best-response MPC expert → behavioral cloning → PPO fine-tuning. Simulator: Highway-Env `merge-v0`.

**Deadline:** Presentation May 8, Final Report May 15.

---

## Completed: Phases 1–5

**Phase 1 — Environment.** Highway-Env merge scenario running. Observation: 4 vehicles × 5 features. Action: `Box(−1, 1, (2,))` (continuous acceleration + steering). One episode = one merge attempt: ego starts on the main highway, one NPC merges from the ramp, episode ends on road exit or crash.

**Phase 2 — Driver types** (`src/driver_types.py`). Three IDM archetypes assigned randomly to non-ego vehicles after each reset. Parameters hand-tuned from Treiber et al. (2000) baseline, mapped to the SVO taxonomy from Schwarting et al. (2019):

| Parameter | Cautious | Normal | Aggressive |
|---|---|---|---|
| Acceleration (m/s²) | 1.2 | 1.5 | 2.5 |
| Braking (m/s²) | −1.5 | −2.0 | −4.0 |
| Time headway (s) | 2.5 | 1.5 | 1.0 |
| Min gap (m) | 8.0 | 2.0 | 1.5 |

**Phase 3 — Reward functions** (`src/reward.py`). Feature-based reward $r_i = \theta_i^\top f(s_t, u_t)$ with six features: forward progress, proximity penalty (1/d² repulsive potential, Khatib 1986), smoothness, jerk, collision (−1000, universal), and lane deviation. Weight vectors encode SVO ordering — cautious up-weights proximity (−2.0), aggressive up-weights forward progress (0.9). Collision penalty set so no cumulative per-step gain can compensate for a crash (50 × 0.9 = 45 ≪ 1000). April 30 fix: collision feature is now `+1` on collision, so the `−1000` weight creates a true penalty; the previous `−1` feature accidentally rewarded collisions.

**Phase 4 — MPC expert** (`src/mpc_expert.py`, `src/best_response.py`). Iterative best-response MPC over a 20-step horizon. Ego task is pure longitudinal gap management (steering fixed at 0; ego is the highway vehicle at y=4.0, not the merging one). Key details: 50 samples per call (3 structured + 47 random normal), verification pass re-scores the winner, ~12ms/call. Full-episode result: ego holds 15–29 m/s, traverses road without crash.

Bugs found and fixed during Phase 4 (documented in `mpc_expert_results.txt`):
- `Y_TARGET` was 0.0 (ramp lane) instead of 4.0 (highway lane) — ego was steering into the ramp
- `dt=0.1` default in `ego_reward` inflated jerk penalty 100× — caused erratic acceleration
- `COLLISION_DIST=3.0m` too small for 5m vehicles — crash detection missed imminent collisions
- Proximity feature sign bug: `−1/d²` with weight `−1.0` produced an attractive potential (rewarded closeness); fixed to `+1/d²`
- Collision feature sign bug: `−1` with weight `−1000` produced `+1000` reward for predicted collisions; fixed to `+1`

**Phase 5 — Expert dataset** (`src/generate_data.py`). Collected (obs, action) pairs from the MPC expert across 400 episodes (= 400 merge attempts) per named non-ego driver mixture. The ego reward type is still randomized across cautious/normal/aggressive; the mixture controls only surrounding traffic.

Dataset results (regenerated May 2 with speed-clamp fix — see Phase 6 notes):

| Dataset | Non-ego driver mix | Records | Clean records | Crashes | Crash rate |
|---|---|---:|---:|---:|---:|
| `data/expert_dataset_all_normal.pkl` | 100% normal | 18,629 | 18,519 | 10 / 400 | 2.5% |
| `data/expert_dataset_default_mix.pkl` | 60% normal, 20% cautious, 20% aggressive | 18,346 | 18,248 | 11 / 400 | 2.8% |
| `data/expert_dataset_cautious_heavy.pkl` | 40% normal, 50% cautious, 10% aggressive | 18,630 | 18,524 | 12 / 400 | 3.0% |
| `data/expert_dataset_aggressive_heavy.pkl` | 40% normal, 10% cautious, 50% aggressive | 18,523 | 18,460 | 8 / 400 | 2.0% |

Total clean BC data: **73,751 transitions** across 1,600 merge attempts. Train on clean records only (`crashed=False`).

---

## Completed: Phase 6 — Behavioral Cloning

**Files:** `src/policy_network.py`, `src/train_policy.py`, `src/eval_policy.py`, `src/cross_eval_bc.py`

**Obs input:** 27-dim — raw 5×5 obs (25) + d_min + step. Per-feature normalization fitted on training split only.

**Architecture:** MLP 27→256→256→128→2, tanh output. 106,114 params. MSE loss on [acc, steer].

**Bug found and fixed (May 2):** 64.7% of previously "clean" training transitions had `ego_speed < 0` (ego driving backwards). Root cause: the MPC planner clamps `vx >= 0` internally when evaluating candidates, but `env.step()` has no speed floor. The planner could select a large negative acceleration, pass it to the simulator, and the ego reversed. The `crashed=False` filter kept all these records. BC then learned the dominant behavior: brake maximally at every step.

Fix applied in `src/generate_data.py` — proportional speed clamp before `env.step()`:
```python
min_acc_norm = -(ego_speed - 0.05) / ACC_SCALE
if action[0] < min_acc_norm:
    action[0] = float(min_acc_norm)
```
A rollout clamp was also added to `BCAction.__call__` in `src/eval_policy.py` and `src/cross_eval_bc.py`: if `ego_speed < 2.0` and the policy wants to brake, clamp acceleration to 0. All four datasets were regenerated at 400 episodes each and all five models retrained on the clean data.

**Model MSE (retrained May 2 on clean 400-ep data):**

| Model | Val MSE | Mean-action baseline MSE |
|---|---|---|
| all_normal | 0.054 | 0.079 |
| default_mix | 0.061 | 0.092 |
| cautious_heavy | 0.063 | 0.093 |
| aggressive_heavy | 0.061 | 0.092 |
| all (combined) | 0.060 | 0.089 |

All models beat the mean-action baseline by ~30%.

**4×4 cross-eval (50 eps per cell):**

| trained_on \ tested_on | all_normal | default_mix | cautious_heavy | aggressive_heavy |
|---|---|---|---|---|
| all_normal | 0% crash, R=30.5, v=0.5 | 0% crash, R=30.4, v=0.5 | 0% crash, R=30.5, v=0.5 | 0% crash, R=30.5, v=0.5 |
| default_mix | 18% crash, R=26.9, v=7.0 | 12% crash, R=28.9, v=5.6 | 12% crash, R=28.6, v=5.3 | 26% crash, R=24.3, v=8.5 |
| cautious_heavy | 10% crash, R=28.5, v=4.0 | 10% crash, R=28.4, v=3.8 | 20% crash, R=25.8, v=7.4 | 10% crash, R=28.4, v=3.9 |
| aggressive_heavy | 72% crash, R=10.5, v=22.5 | 46% crash, R=19.2, v=17.0 | 62% crash, R=13.8, v=20.3 | 52% crash, R=17.2, v=18.3 |

`v` = mean ego speed (m/s). All crashes verified as spawn collisions at step ≤ 3. Zero policy-induced crashes after step 3.

**Interpretation:** Distribution-shift pattern is clear — more cautious training data produces a more passive policy. `all_normal` learned to stop and yield indefinitely (v=0.5 m/s). `aggressive_heavy` drives at highway speed but crashes too often. `default_mix` is the best tradeoff: 12–26% crash (all spawn geometry), v=5–8.5 m/s, R=24–29.

**PPO warm-start target:** use the combined BC model, `models/bc_policy_all.pt` + `models/bc_policy_all.npz`. The earlier note suggested `default_mix`, but clamp-rate logging showed the combined model was stronger across all mixtures: 0% crash in the 50-episode cross-eval, reasonable speed, and lower clamp dependence than the passive all-normal/cautious models.

---

## In Progress: Phase 7 — PPO Fine-Tuning

**Implemented:** `src/rl_finetune.py`

**What it does:**
- Wraps Highway-Env `merge-v0` with the same 27-dim observation representation used by BC: flattened 5x5 obs + `d_min` + `step`.
- Loads BC normalization stats from `models/bc_policy_all.npz`.
- Uses a PPO actor architecture matching BC: 27 -> 256 -> 256 -> 128 -> 2.
- Warm-starts PPO actor weights from `models/bc_policy_all.pt`.
- Uses a tanh-bounded PPO action mean so deterministic PPO actions match the BC convention of actions in `[-1, 1]`.
- Applies the same no-reverse action safety clamp and reports clamp rate.
- Uses a shaped RL reward that penalizes crash/reversing/clamp dependence and rewards forward speed/progress.

**Smoke test run:** `python3 src/rl_finetune.py --timesteps 5000 --eval-episodes 10 --traffic-mix default_mix --out models/ppo_smoke_merge`

| Check | Before PPO | After 5k PPO steps |
|---|---:|---:|
| Crash rate | 0.0% | 0.0% |
| Mean reward | 8.00 | 155.72 |
| Mean steps | 50.0 | 50.0 |
| Mean speed | 4.00 m/s | 34.98 m/s |
| Clamp rate | 58.8% | 0.0% |

**Cross-mixture smoke eval after 5k PPO steps, 10 episodes each:** 0% crash on all_normal/default_mix/cautious_heavy/aggressive_heavy, mean speed about 35 m/s, clamp rate 0%.

**Interpretation:** PPO wiring is working and quickly fixes the BC failure mode of slow/clamped driving. This is only a smoke result, not final Phase 7. The policy is now very fast and still hits the 50-step cap, so next we need a longer run plus a better completion metric/reward term to confirm it is actually completing the merge task efficiently rather than just driving fast.

---

## Questions for GSI

1. **Crash episode training signal:** We're excluding crash-flagged records from BC training. After the reward fix, crash rates are much lower (1.8%-2.5% across the four fixed datasets). Is clean-only still the best training choice, or should we keep pre-crash steps from otherwise crashed episodes?

2. **PPO warm-starting strategy:** For RL fine-tuning, should we freeze the BC encoder layers for the first few thousand steps, or let all layers train from the start with a low learning rate (1e-4)?
   
3. How many episodes/merges do you suggest? We now have 400 merges per mixture and roughly 18k-19k transitions per fixed dataset.

4. Question: Our MPC expert fixes steering to 0.0 — the ego is a highway vehicle that only needs to manage its speed, not change lanes. As a result, our BC network effectively only learns acceleration. Should we keep this as-is, or is there value in learning a steering signal too (e.g., for lane-keeping robustness in the PPO fine-tuning phase)?

5. can we meet with u again pls

---

## In Progress: Phase 8 — Baseline

**Implemented:** `src/baseline.py`

**Baseline definition:** naive independent 2-agent planning. This is the Sadigh-style comparison point: for each non-ego vehicle, predict its response to the ego independently as if the other non-ego vehicles were absent. The ego then scores actions against the collection of those independent pairwise predictions. This preserves robot-human strategic planning but removes human-human interaction chains.

**Why this baseline matters:** our project claims value from extending interaction-aware planning from one human to multiple interacting vehicles. The baseline asks what happens if we apply the original 2-agent idea naively to a multi-car merge.

**Default-mix smoke result:** `python3 src/baseline.py --episodes 20 --traffic-mix default_mix --seed 0`

| Metric | Independent 2-agent baseline |
|---|---:|
| Merge success rate (`x > 310`, no crash) | 90.0% |
| Crash rate | 0.0% |
| Short/timeout rate | 10.0% |
| Mean speed | 5.85 m/s |
| Mean steps | 65.7 |
| Mean merge step | 50.1 |
| Mean min distance | 5.43 m |
| Mean min TTC | 0.59 s |
| Clamp rate | 37.5% |

**Interpretation:** the baseline can merge in most default-mix episodes, but it is slow and relies on the no-reverse clamp often. This gives us a meaningful comparison against PPO: PPO should preserve safety while improving speed/commitment and reducing clamp dependence.
