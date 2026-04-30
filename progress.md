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

**Phase 5 — Expert dataset** (`src/generate_data.py`). Collected (obs, action) pairs from the MPC expert across 200 episodes (= 200 merge attempts) with named non-ego driver mixtures. The ego reward type is still randomized across cautious/normal/aggressive; the mixture controls only surrounding traffic.

Fixed-reward dataset results:

| Dataset | Non-ego driver mix | Records | Clean records | Crashes | Crash rate |
|---|---|---:|---:|---:|---:|
| `data/expert_dataset_all_normal.pkl` | 100% normal | 9703 | 9695 | 1 / 200 | 0.5% |
| `data/expert_dataset_default_mix.pkl` | 60% normal, 20% cautious, 20% aggressive | 9485 | 9477 | 1 / 200 | 0.5% |
| `data/expert_dataset_cautious_heavy.pkl` | 40% normal, 50% cautious, 10% aggressive | 9316 | 9258 | 7 / 200 | 3.5% |
| `data/expert_dataset_aggressive_heavy.pkl` | 40% normal, 10% cautious, 50% aggressive | 9517 | 9451 | 7 / 200 | 3.5% |

> **On episode count:** Each episode is one merge attempt capped at 50 steps. After the collision reward fix, most episodes survive much longer, so 200 episodes now produces roughly 9k–10k transitions per mixture instead of the old ~3k bugged dataset. For behavioral cloning, train on clean records first (`crashed=False`) and keep the crash-flagged records for analysis/safety filtering.

---

## In Progress: Phase 6 — Behavioral Cloning

Next step: train a policy network (`src/policy_network.py`) to imitate the MPC expert via supervised regression on the clean transitions from one or more fixed-reward datasets. Input: flattened 5×5 obs (25-dim). Output: 2-dim continuous action (acceleration, steering). Architecture: 4-layer MLP (25 → 256 → 256 → 128 → 2), trained with MSE loss.

---

## Questions for GSI

1. **Crash episode training signal:** We're excluding crash-flagged records from BC training. After the reward fix, crash rates are much lower (0.5%, 0.5%, and 3.5% across the three fixed datasets). Is clean-only still the best training choice, or should we keep pre-crash steps from otherwise crashed episodes?

2. **PPO warm-starting strategy:** For RL fine-tuning, should we freeze the BC encoder layers for the first few thousand steps, or let all layers train from the start with a low learning rate (1e-4)?
   
3. How many episodes/merges do you suggest? We now have 200 merges per mixture and roughly 9k-10k transitions per fixed dataset.

4. Question: Our MPC expert fixes steering to 0.0 — the ego is a highway vehicle that only needs to manage its speed, not change lanes. As a result, our BC network effectively only learns acceleration. Should we keep this as-is, or is there value in learning a steering signal too (e.g., for lane-keeping robustness in the PPO fine-tuning phase)?

5. can we meet with u again pls
