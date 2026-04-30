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

1. **Dataset size:** We collected 3169 clean transitions across 200 episodes (one episode = one merge). Is this sufficient for behavioral cloning, or would you recommend 500+ episodes? The main constraint is compute: each episode requires ~200 MPC calls at 12ms each (~2.4s/episode), so 500 episodes ≈ 20 minutes.

2. **Crash rate exclusion:** 26.5% of episodes ended in collision within the first 4–8 steps due to initial vehicle placement by the simulator (not MPC behavior). We plan to exclude all crashed episodes from the BC training set. Is it worth instead keeping the first N steps of crashed episodes (up to the crash) as additional training signal, or is clean-only the safer approach?

3. **BC architecture:** We're using a 4-layer MLP (25→256→256→128→2) with MSE loss. Any guidance on whether action normalization (tanh output layer) or a separate head per action dimension would be preferable for this continuous control setting?

4. **Phase 7 warm-starting:** For PPO fine-tuning, we plan to load BC weights as the policy initialization and freeze the encoder layers for the first few thousand steps. Does this match the approach you'd recommend, or should we let all layers train from the start?