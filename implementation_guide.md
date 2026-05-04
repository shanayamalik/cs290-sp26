# Implementation Guide
## Autonomous Driving with Multi-Agent Lane Merging

**Deadline:** Presentation — May 8, 2026 | Final Report — May 15, 2026

> **Before starting:** Follow the setup instructions in [README.md](README.md) to create a virtual environment and install all dependencies. The phases below assume `venv` is active.

---

## ✅ Phase 1: Environment Setup — COMPLETE

See [README.md](README.md). Note: use `env.unwrapped` to access highway-env internals through gymnasium's wrappers — `env.road` will raise an `AttributeError`.

---

## ✅ Phase 2: Driver Types — COMPLETE

See `driver_types.py`. Three types implemented (cautious, normal, aggressive) by overriding IDM parameters on `IDMVehicle` instances. Normal driver uses Treiber et al. (2000) reference values (T=1.5 s, s0=2 m, delta=4); cautious and aggressive are scaled relative to normal. SVO taxonomy mapping from Schwarting et al. (2019) — see module docstring for full citations and DOIs.

Driver types are randomly assigned to non-ego vehicles in `merge_env.py` via `assign_driver_types()` after every `env.reset()`. Vehicles[0] is always ego; vehicles[1:] receive a random archetype.

---

## ✅ Phase 3: Reward Functions — COMPLETE

See [src/reward.py](src/reward.py). The reward is $r_i = \theta_i^\top f(s_t, u_t)$ with six features. The three numbers are for cautious/normal/aggressive. Key design decisions:

- **forward_progress (0.2/0.5/0.9):** normalized feature in [0,1]; ordering encodes SVO angle φ (Schwarting et al. 2019)
- **proximity_penalty (−2.0/−1.0/−0.5):** 1/d² repulsive potential (Khatib 1986), 4:2:1 ratio across types
- **smoothness/jerk:** secondary comfort terms (Bae et al. 2020), intentionally small — tune after MPC works
- **collision (−1000, all types):** dominates any horizon (50 steps × max 0.9 = 45 << 1000); no per-step gain justifies a crash
- **lane_deviation (−0.5/−0.3/−0.1):** y_target must be the *target* lane center, not current lane

**April 30 fix:** collision is encoded as `+1` when a predicted collision occurs and `0` otherwise. With the universal `−1000` weight, this produces a large negative penalty. The previous encoding used `−1`, which accidentally made collision contribute `+1000` to the reward.

---

## ✅ Phase 4: MPC Expert (Iterative Best-Response) — COMPLETE

See `src/best_response.py` and `src/mpc_expert.py`. Committed to `naya` branch (final commit `b38f98d`).

**Key confirmed facts:**
- Ego is the **highway vehicle** (y=4.0), not the merging one. NPC from ramp (y=14.5) merges in. Task = pure longitudinal gap management; `Y_TARGET=4.0`, steering fixed at 0.
- `ACC_SCALE=5.0`, `DT_PLAN=1.0s`, `HORIZON=20`, `N_SAMPLES=50`, `N_WAYPOINTS=6`
- Sampling: 3 structured candidates (constant-speed/brake/accel) + `normal(0, 0.4)` random waypoints
- Verification pass: winning trajectory re-scored against responses re-predicted from that trajectory (~2ms extra)
- `COLLISION_DIST=6.0m` (vehicles are ~5m long; 6m gives a small safety buffer)
- `dt=DT_PLAN=1.0` passed explicitly to `ego_reward` (default was 0.1, inflating jerk 100×)
- `proximity_penalty` feature = `+1/d²` (positive); negative weight makes contribution repulsive
- Lane deviation weights zeroed (ego is not merging; `y = y_target = 4.0` always)

**Full-episode result:** Ego stays at y=4.0, holds 15–29 m/s, traverses full a→b→c→d corridor without crash. ~12ms/call.

---

## ✅ Phase 5: Expert Dataset Generation — COMPLETE

See `src/generate_data.py`. Committed to `shanaya` branch.

**Key design decisions:**
- Raw 5×5 obs saved (not flattened); flatten at training time with `obs.reshape(-1)`
- Crashed episodes kept with `crashed=True` flag — filter at training time, exclude from BC
- `MAX_STEPS=50` cap prevents runaway episodes
- `ego_speed` and `d_min` saved per step for trajectory plots without decoding obs
- `theta_name` saved as string for type-conditioned analysis
- **Speed clamp added (May 2):** `min_acc_norm = -(ego_speed - 0.05) / ACC_SCALE` applied before `env.step()` to prevent simulator from reversing the ego. See Phase 6 for root cause.

**Dataset results (400 episodes each, regenerated May 2 with speed-clamp fix):**

| Dataset | Non-ego driver mix | Records | Clean records | Crashes | Crash rate |
|---|---|---:|---:|---:|---:|
| `data/expert_dataset_all_normal.pkl` | 100% normal | 18,629 | 18,519 | 10 / 400 | 2.5% |
| `data/expert_dataset_default_mix.pkl` | 60% normal, 20% cautious, 20% aggressive | 18,346 | 18,248 | 11 / 400 | 2.8% |
| `data/expert_dataset_cautious_heavy.pkl` | 40% normal, 50% cautious, 10% aggressive | 18,630 | 18,524 | 12 / 400 | 3.0% |
| `data/expert_dataset_aggressive_heavy.pkl` | 40% normal, 10% cautious, 50% aggressive | 18,523 | 18,460 | 8 / 400 | 2.0% |

To regenerate:
```bash
python3 src/generate_data.py --episodes 400 --all-mixes
python3 src/generate_data.py --episodes 5 --all-mixes  # sanity check
```

---

## ✅ Phase 6: Behavioral Cloning — COMPLETE

See `src/policy_network.py`, `src/train_policy.py`, `src/eval_policy.py`, and `src/cross_eval_bc.py`.

**Obs augmentation:** Raw 5×5 obs (25 features) augmented with `d_min` (nearest vehicle gap) and `step` (episode progress) → 27-dim input. Normalized per-feature using training-split mean/std only. Stats saved to `models/bc_policy_{dataset}.npz`.

**Architecture:** MLP 27→256→256→128→2, tanh output. 106,114 params.

**Training:** 100 epochs, batch=256, lr=1e-3, ReduceLROnPlateau (patience=10, factor=0.5), MSE loss, 90/10 split. Trains on clean (`crashed=False`) records only.

**Bug found and fixed (May 2):** 64.7% of previously "clean" transitions had `ego_speed < 0`. The MPC planner clamps `vx >= 0` internally but `env.step()` has no floor. Fix: proportional speed clamp in `generate_data.py` before `env.step()`, and a soft rollout clamp in `BCAction.__call__` (`if ego_speed < 2.0 and action[0] < 0: action[0] = 0`). All datasets regenerated, all models retrained. Full details in `nathan.md`.

**Model MSE (retrained May 2 on clean 400-ep data):**

| Dataset | Records | Val MSE | Baseline MSE |
|---|---:|---:|---:|
| all_normal | 18,519 | 0.054 | 0.079 |
| default_mix | 18,248 | 0.061 | 0.092 |
| cautious_heavy | 18,524 | 0.063 | 0.093 |
| aggressive_heavy | 18,460 | 0.061 | 0.092 |
| **all (combined)** | **73,751** | **0.060** | **0.089** |

All models beat the mean-action baseline by ~30%.

**4×4 cross-eval (50 episodes per cell):**

| trained_on \ tested_on | all_normal | default_mix | cautious_heavy | aggressive_heavy |
|---|---|---|---|---|
| all_normal | 0% crash, R=30.5, v=0.5 | 0% crash, R=30.4, v=0.5 | 0% crash, R=30.5, v=0.5 | 0% crash, R=30.5, v=0.5 |
| default_mix | 18% crash, R=26.9, v=7.0 | 12% crash, R=28.9, v=5.6 | 12% crash, R=28.6, v=5.3 | 26% crash, R=24.3, v=8.5 |
| cautious_heavy | 10% crash, R=28.5, v=4.0 | 10% crash, R=28.4, v=3.8 | 20% crash, R=25.8, v=7.4 | 10% crash, R=28.4, v=3.9 |
| aggressive_heavy | 72% crash, R=10.5, v=22.5 | 46% crash, R=19.2, v=17.0 | 62% crash, R=13.8, v=20.3 | 52% crash, R=17.2, v=18.3 |

All crashes verified as spawn collisions at step ≤ 3. Zero policy-induced crashes after step 3. `default_mix` is the best PPO warm-start choice (12–26% crash, R=24–29, v=5–8.5 m/s).

To reproduce:
```bash
python3 src/train_policy.py --dataset all         # combined (PPO warm-start)
python3 src/train_policy.py --dataset all_normal  # ablation (repeat for other mixes)
python3 src/eval_policy.py --model models/bc_policy_default_mix.pt --episodes 20 --traffic-mix default_mix
python3 src/cross_eval_bc.py --episodes 50
```

---

## ✅ Phase 7: RL Fine-Tuning — COMPLETE

**Final model:** `models/ppo_500k_v2_merge.zip` (local only, not git-tracked)

**Key implementation details** (see `src/rl_finetune.py` and `nathan.md` for full bug list):
- 28-dim obs: raw 5×5 highway-env obs (25) + d_min + step_count + ego_speed
- Warm-started from `bc_policy_default_mix.pt`
- Milestone rewards (+5 every 50m), truncation-shaped reward, MAX_STEPS=150
- Termination override: `x > 330m` (built-in `x > 370m` threshold is geometrically unreachable due to road curvature)
- 500k timesteps, lr=1e-4, PPO with TanhMean policy

**Results (50-ep eval, `diagnose_v3.py`, deterministic):**
- 18/20 merge completions (x > 310m), 0/20 crashes, ~28–35 steps per episode

```bash
# Reproduce training (~90 min on Apple M2)
python3 src/rl_finetune.py --timesteps 500000 --eval-episodes 50 --traffic-mix default_mix --out models/ppo_500k_v2_merge

# Verify final model
python3 src/diagnose_v3.py models/ppo_500k_v2_merge.zip
```
python3 src/rl_finetune.py --timesteps 100000 --eval-episodes 50 --traffic-mix default_mix --out models/ppo_finetuned_merge
```

**Implementation notes:** The wrapper uses the same 27-dim normalized observation as BC, warm-starts the PPO actor from BC weights, and uses a tanh-bounded PPO action mean so deterministic PPO actions match BC's `[-1, 1]` output convention. The script reports crash rate, reward, mean speed, mean steps, and no-reverse clamp rate before and after training.

**Current smoke result:** 5k PPO steps improved default-mix rollout from slow/clamped BC behavior to forward driving: 0% crash over 10 episodes, mean speed ≈ 35 m/s, clamp rate 0%. This confirms the PPO path is wired correctly. It is not the final Phase 7 result yet because mean episode length still hits the 50-step cap; next tune completion reward/metric and run a longer training job.

---

## ✅ Phase 8: Baseline — COMPLETE

Implemented in `src/baseline.py`. Independent 2-agent planning: MPC plans separately for each non-ego vehicle as if the other non-ego vehicles don't exist (no human–human interaction chains). Isolates the value of the multi-agent extension.

```bash
python3 src/baseline.py --episodes 50 --traffic-mix default_mix --seed 0
```

Default-mix result (50 eps): 90% merge success, 2% crash, mean speed 6.56 m/s, mean merge step 48.7, clamp rate 34.6%.

---

## ✅ Phase 9: Evaluation — COMPLETE

Implemented in `src/evaluate.py`. Runs BC, PPO, baseline, and MPC on the same seeds and traffic mix.

```bash
# Single mix
python3 src/evaluate.py --episodes 50 --traffic-mix default_mix --seed 0

# All 4 mixes
for mix in default_mix all_normal cautious_heavy aggressive_heavy; do
  python3 src/evaluate.py --episodes 50 --traffic-mix $mix --seed 0 --output diagnostics/final_eval_${mix}.csv
done
```

**Final results across traffic mixes (PPO merge success / crash rate):**

| Traffic Mix | BC | PPO | Baseline | MPC |
|-------------|-----|-----|----------|-----|
| default_mix | 0% / 10% | **96% / 0%** | 90% / 2% | 92% / 2% |
| all_normal | 0% / 8% | **84% / 0%** | 78% / 6% | 78% / 2% |
| cautious_heavy | 0% / 8% | **90% / 0%** | 86% / 6% | 86% / 2% |
| aggressive_heavy | 0% / 10% | **96% / 0%** | 90% / 0% | 92% / 0% |

Results CSVs saved to `diagnostics/`. PPO outperforms all methods across all mixes.

## Phase 9: Evaluation

Run 20 episodes per scenario. Record for each episode:
- **Merge success** — no collision during episode
- **Merge completion time** — steps until all vehicles clear the merge zone
- **Minimum TTC** — minimum time-to-collision across all timesteps in the episode

### Scenarios

| Scenario | Ego | Others |
|---|---|---|
| Human-AV | RL policy | 2 human IDM vehicles (random type each episode) |
| Human-Mixed AV | RL policy | 1 human IDM + 1 additional RL policy vehicle |
| Multi-AV | RL policy | 2 additional RL policy vehicles (no humans) |

Implemented in `src/evaluate.py`. It runs BC, PPO, independent 2-agent baseline, and full MPC on the same seeds/traffic mix, then writes a summary table/CSV.

```bash
python3 src/evaluate.py --episodes 20 --traffic-mix default_mix --seed 0 \
  --ppo-model models/ppo_500k_v3_merge.zip
```

If the final PPO zip is not local, run the non-PPO methods while waiting for the file:

```bash
python3 src/evaluate.py --methods bc baseline mpc --episodes 20 --traffic-mix default_mix
```

The evaluator skips missing or incompatible PPO model files with a clear message. The old local `ppo_smoke_merge.zip` used the pre-ego-speed 27-dim wrapper, while the final v3 PPO model uses the 28-dim wrapper.

### Plots to generate for the paper
- Bar charts: merge success rate and avg completion time across 3 scenarios × 2 methods
- Line plot: PPO training curve (episode reward vs. timesteps)
- Trajectory plots: a successful merge for each scenario (qualitative)

---

## Phase 10: Paper Writeup

**Format:** IEEE, 5–9 pages. **Due:** May 15, 2026.

### Structure
1. Abstract
2. Introduction — motivation, gap (single-pair → multi-agent), contributions
3. Related Work — Sadigh, Zhang, Schwarting, Huang, **Weil et al.** (add this), + survey paper
4. Problem Formulation — state space, action space, dynamics, reward functions, driver archetypes
5. Approach — three-stage pipeline with equations
6. Experiments — three scenarios, metrics, baseline description
7. Results — tables and plots
8. Discussion — what worked, non-convergence observations, limitations (known driver types), future work (IRL / SVO estimation)
9. Conclusion
10. References

### Key things to state explicitly in the methods section
- Reward functions are hand-defined, not learned (contrast with Huang et al.)
- Driver types are assumed known at planning time (limitation)
- Iterative best-response uses last iterate as fallback when convergence fails
- Warm-starting procedure for PPO from behavioral cloning weights
- Highway-Env (`merge-v0`) is the simulator

---

## Timeline

| Dates | Phase |
|---|---|
| Apr 26 – Apr 28 | Phase 1–2: Environment + driver types running |
| Apr 29 – May 1 | Phase 3–4: Reward functions + MPC expert working |
| May 2 – May 4 | Phase 5–7: Expert dataset, BC, PPO fine-tuning complete |
| May 4 – May 4 | Phase 8–9: Baseline + final evaluation complete |
| **May 8** | **Presentation** |
| May 9 – May 15 | Phase 10: Paper writeup |
