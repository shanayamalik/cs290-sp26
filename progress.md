# Progress Update — April 29, 2026

**Shanaya Malik, Nathan McNaughton** | ELENG 290 — Multi-Agent Lane Merging

---

## Project

Extending Sadigh et al. (2016) from a single AV–human pair to multi-vehicle lane merging with 3+ agents. Three-stage pipeline: iterative best-response MPC expert → behavioral cloning → PPO fine-tuning. Simulator: Highway-Env `merge-v0`.

**Deadline:** Presentation May 8, Final Report May 15.

---

## Completed: Phases 1–5

**Phase 1 — Environment.** Highway-Env merge scenario running. Observation: 5 vehicles × 5 features. Action: `Box(−1, 1, (2,))` (continuous acceleration + steering). One episode = one merge attempt: ego starts on the main highway, one NPC merges from the ramp, episode ends on road exit or crash.

**Phase 2 — Driver types** (`src/driver_types.py`). Three IDM archetypes assigned randomly to non-ego vehicles after each reset. Parameters hand-tuned from Treiber et al. (2000) baseline, mapped to the SVO taxonomy from Schwarting et al. (2019):

| Parameter | Cautious | Normal | Aggressive |
|---|---|---|---|
| Acceleration (m/s²) | 1.2 | 1.5 | 2.5 |
| Braking (m/s²) | −1.5 | −2.0 | −4.0 |
| Time headway (s) | 2.5 | 1.5 | 1.0 |
| Min gap (m) | 8.0 | 2.0 | 1.5 |

**Phase 3 — Reward functions** (`src/reward.py`). Feature-based reward $r_i = \theta_i^\top f(s_t, u_t)$ with six features: forward progress, proximity penalty (1/d² repulsive potential, Khatib 1986), smoothness, jerk, collision (−1000, universal), and lane deviation. Weight vectors encode SVO ordering — cautious up-weights proximity (−2.0), aggressive up-weights forward progress (0.9). Collision penalty set so no cumulative per-step gain can compensate for a crash (50 × 0.9 = 45 ≪ 1000).

**Phase 4 — MPC expert** (`src/mpc_expert.py`, `src/best_response.py`). Iterative best-response MPC over a 20-step horizon. Ego task is pure longitudinal gap management (steering fixed at 0; ego is the highway vehicle at y=4.0, not the merging one). Key details: 50 samples per call (3 structured + 47 random normal), verification pass re-scores the winner, ~12ms/call. Full-episode result: ego holds 15–29 m/s, traverses road without crash.

Bugs found and fixed during Phase 4 (documented in `mpc_expert_results.txt`):
- `Y_TARGET` was 0.0 (ramp lane) instead of 4.0 (highway lane) — ego was steering into the ramp
- `dt=0.1` default in `ego_reward` inflated jerk penalty 100× — caused erratic acceleration
- `COLLISION_DIST=3.0m` too small for 5m vehicles — crash detection missed imminent collisions
- Proximity feature sign bug: `−1/d²` with weight `−1.0` produced an attractive potential (rewarded closeness); fixed to `+1/d²`

**Phase 5 — Expert dataset** (`src/generate_data.py`). Collected (obs, action) pairs from the MPC expert across 200 episodes (= 200 merge attempts) with randomized driver type assignments.

Results:
- 3564 total transitions, **3169 clean** (non-crashed), ~16 steps/episode average
- 26.5% crash rate — dominated by spawn collisions (env places vehicles too close at reset; MPC has no time to react within 4–8 steps). MPC behavior is not the cause.
- Clean transitions per driver type: cautious 968 / normal 1209 / aggressive 992 (≤1.25× imbalance — acceptable for BC)
- Saved to `data/expert_dataset.pkl` (binary, excluded from git)

> **On episode count:** Each episode is one merge attempt (~16 steps at 1 step/s = ~16 seconds of simulated driving). 200 episodes × 16 steps = 3200 transitions before filtering. After removing crashed episodes, 3169 clean transitions remain. This is the minimum we felt comfortable with for a 4-layer network (~25k parameters after obs flattening to 25-dim input); 200 episodes gives ~1000 clean examples per driver type.

---

## In Progress: Phase 6 — Behavioral Cloning

Next step: train a policy network (`src/policy_network.py`) to imitate the MPC expert via supervised regression on the 3169 clean transitions. Input: flattened 5×5 obs (25-dim). Output: 2-dim continuous action (acceleration, steering). Architecture: 4-layer MLP (25 → 256 → 256 → 128 → 2), trained with MSE loss.

---

## Questions for GSI

1. **Crash episode training signal:** We're excluding crashed episodes from BC training (26.5% of episodes, all spawn collisions within 4–8 steps). Is it worth keeping steps *before* the crash as additional signal, or is clean-only the safer approach?

2. **PPO warm-starting strategy:** For RL fine-tuning, should we freeze the BC encoder layers for the first few thousand steps, or let all layers train from the start with a low learning rate (1e-4)?  

3. **Evaluation scope:** Our three planned scenarios (Human-AV, Human-Mixed, Multi-AV) vary who the *other* vehicles are, but the ego's task is identical in all three — longitudinal gap management on the highway. Would you prefer we keep the three-scenario comparison, or focus on one scenario and vary driver type combinations more extensively?
