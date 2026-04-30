# Progress Update — April 29, 2026

**Shanaya Malik, Nathan McNaughton** | ELENG 290 — Multi-Agent Lane Merging

---

## Project

Extending Sadigh et al. (2016) from a single AV–human pair to multi-vehicle lane merging with 3+ agents. Three-stage pipeline: iterative best-response MPC expert → behavioral cloning → PPO fine-tuning. Simulator: Highway-Env `merge-v0`.

---

## Completed: Phases 1–3

**Phase 1 — Environment.** Highway-Env merge scenario running. Observation: 5 vehicles × 5 features. Action: Discrete(5) (will switch to ContinuousAction for MPC). Dependencies verified (highway-env, gymnasium, torch, stable-baselines3).

**Phase 2 — Driver types** (`src/driver_types.py`). Three IDM archetypes assigned randomly to non-ego vehicles after each reset. Parameters hand-tuned from Treiber et al. (2000) baseline, mapped to the SVO taxonomy from Schwarting et al. (2019):

| Parameter | Cautious | Normal | Aggressive |
|---|---|---|---|
| Acceleration (m/s²) | 1.2 | 1.5 | 2.5 |
| Braking (m/s²) | −1.5 | −2.0 | −4.0 |
| Time headway (s) | 2.5 | 1.5 | 1.0 |
| Min gap (m) | 8.0 | 2.0 | 1.5 |

**Phase 3 — Reward functions** (`src/reward.py`). Feature-based reward $r_i = \theta_i^\top f(s_t, u_t)$ with six features: forward progress, proximity penalty (1/d² repulsive potential, Khatib 1986), smoothness, jerk, collision (−1000, universal), and lane deviation. Weight vectors encode SVO ordering — cautious up-weights proximity (−2.0), aggressive up-weights forward progress (0.9). Collision penalty set so no cumulative per-step gain can compensate for a crash (50 × 0.9 = 45 ≪ 1000).

---

## In Progress: Phase 4 — MPC Expert

Building the iterative best-response MPC over a 50-step (5 s) horizon. Two known issues to resolve first:

1. **Action space mismatch.** `merge-v0` defaults to Discrete(5); MPC outputs continuous acceleration. Need to add `"action": {"type": "ContinuousAction"}` to the env config.
2. **Gap computation bug.** `idm_predict` can produce negative gaps when the ego is behind another vehicle. Fix: filter to positive gaps before taking the minimum.