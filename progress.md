# Progress Update — April 30, 2026

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

Fixed-reward dataset results:

| Dataset | Non-ego driver mix | Records | Clean records | Crashes | Crash rate |
|---|---|---:|---:|---:|---:|
| `data/expert_dataset_all_normal.pkl` | 100% normal | 18718 | 18616 | 10 / 400 | 2.5% |
| `data/expert_dataset_default_mix.pkl` | 60% normal, 20% cautious, 20% aggressive | 18983 | 18913 | 8 / 400 | 2.0% |
| `data/expert_dataset_cautious_heavy.pkl` | 40% normal, 50% cautious, 10% aggressive | 19003 | 18933 | 10 / 400 | 2.5% |
| `data/expert_dataset_aggressive_heavy.pkl` | 40% normal, 10% cautious, 50% aggressive | 19060 | 18980 | 7 / 400 | 1.8% |

Total clean BC data: **75,442 transitions** across 1,600 merge attempts. For behavioral cloning, train on clean records first (`crashed=False`) and keep the crash-flagged records for analysis/safety filtering.

---

## Phase 6 — Behavioral Cloning: Implemented, Needs Full Rerun

**Files:** `src/policy_network.py`, `src/train_policy.py`, `src/eval_policy.py`, `src/cross_eval_bc.py`

**Obs input:** 27-dim — raw 5×5 obs (25) + d_min + step. Per-feature normalization fitted on training split only.

**Architecture:** MLP 27→256→256→128→2, tanh output. 106,114 params. MSE loss on [acc, steer].

**What we changed and why:**

- Kept the existing BC implementation instead of starting over, because the structure matches the project plan: supervised imitation of MPC expert actions from clean trajectories.
- Updated `src/train_policy.py` to save loss curves (`plots/*_loss.csv/png`) and print validation action diagnostics. This lets us prove loss decreases, check predicted acceleration/steering ranges, and catch constant or saturated policies.
- Confirmed BC trains only on clean records (`crashed=False`). This follows mentor feedback because our crash episodes are instant failures, not useful evasion examples.
- Updated `src/eval_policy.py` to use the same continuous-action Highway-Env patch as data generation, add exact `default_mix` traffic, report reward, report negative-speed episodes, and optionally save a trajectory plot. This matters because crash rate alone can hide a bad policy that avoids crashes by braking/reversing.
- Added `src/cross_eval_bc.py` for the 4x4 train-mixture x test-mixture evaluation. This directly supports the next sanity check: whether models trained on one traffic mixture generalize to the others.
- Updated README / implementation guide language so BC is described as a PPO warm-start stage, not as the final driving controller.

**Current status:** BC code is implemented and aligned with the implementation guide, but the BC phase is not fully "done" until we rerun the full training/evaluation with the new diagnostics. The expected role of BC is to initialize PPO, not to be the final policy.

**Smoke test after fixes:** A 2-epoch all-normal training run executed successfully. It saved the loss CSV/PNG and printed action stats. Predicted actions stayed inside `[-1, 1]`, acceleration had nonzero variation, and steering stayed near zero as expected because the MPC expert always uses steering = 0.

**Earlier 100-episode ablation results (seed=0, uniform NPC traffic):**

| Dataset | Val loss | Crash rate | Mean steps |
|---|---|---|---|
| all_normal | 0.11195 | 41% | 30.3 |
| default_mix | 0.11245 | 12% | 44.2 |
| cautious_heavy | 0.11508 | 26% | 37.5 |
| aggressive_heavy | 0.11825 | 37% | 32.2 |
| all combined | 0.11194 | 14% | 43.3 |

All crashes are at step 2 (spawn collisions, verified). BC causes 0 policy-induced crashes.

**Key finding:** BC learned safe deceleration but never commits to the merge. Every non-crash episode hits MAX_STEPS=50; MPC completes in 15–25 steps. BC copies the "yield" half of expert demos without the goal-directed "push through" — classic distribution shift.

**2x2 cross-eval (50 eps, seed=0):** all_normal model + normal traffic: 34%/33.7 steps; all_normal + aggressive: 44%/27.9; aggressive_heavy + normal: 34%/32.7; aggressive_heavy + aggressive: 28%/33.7. All crashes at step 2. Crash rate variation = spawn geometry noise, not policy failure.

**PPO warm-start target:** train `models/bc_policy_all.pt` + `.npz` locally with `python3 src/train_policy.py --dataset all`; `.pt` files are local artifacts, so regenerate before evaluation if missing.

---

## In Progress: Phase 7 — PPO Fine-Tuning

---

## Questions for GSI

1. **Crash episode training signal:** We're excluding crash-flagged records from BC training. After the reward fix, crash rates are much lower (1.8%-2.5% across the four fixed datasets). Is clean-only still the best training choice, or should we keep pre-crash steps from otherwise crashed episodes?

2. **PPO warm-starting strategy:** For RL fine-tuning, should we freeze the BC encoder layers for the first few thousand steps, or let all layers train from the start with a low learning rate (1e-4)?
   
3. How many episodes/merges do you suggest? We now have 400 merges per mixture and roughly 18k-19k transitions per fixed dataset.

4. Question: Our MPC expert fixes steering to 0.0 — the ego is a highway vehicle that only needs to manage its speed, not change lanes. As a result, our BC network effectively only learns acceleration. Should we keep this as-is, or is there value in learning a steering signal too (e.g., for lane-keeping robustness in the PPO fine-tuning phase)?

5. can we meet with u again pls
