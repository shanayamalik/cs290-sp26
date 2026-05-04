# Autonomous Driving with Multi-Agent Lane Merging

**Shanaya Malik, Nathan McNaughton**

ELENG 290: Learning-Enabled Multi-Agent Systems — Spring 2026

Extension of Sadigh et al. (2016) to multi-vehicle lane merging with 3+ agents. Each human driver is assigned a predefined reward archetype (cautious / normal / aggressive). The pipeline has three stages: iterative best-response MPC expert → behavioral cloning → PPO fine-tuning.

Driver type parameters (in `src/driver_types.py`) are hand-tuned based on empirically observed ranges from Treiber et al. (2000) and map onto the Social Value Orientation taxonomy from Schwarting et al. (2019). They are not learned from data.

**Progress:** Phases 1–9 complete. Pipeline: MPC expert → behavioral cloning → PPO fine-tuning → baseline comparison → final evaluation.
See [implementation_guide.md](implementation_guide.md) for the full step-by-step implementation plan.
See [nathan.md](nathan.md) for Phase 7 details and handoff notes.

---

## Setup

**Requirements:** Python 3.10+, no GPU needed.

```bash
# 1. Clone the repo
git clone https://github.com/shanayamalik/cs290-sp26.git
cd cs290-sp26

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

To verify the environment is working:

```bash
python3 -c "import highway_env, gymnasium, torch, stable_baselines3; print('All dependencies OK')"
```

Then run the environment sanity check (a render window with cars should appear, and the terminal will print the observation shape and vehicle states):

```bash
python3 src/merge_env.py
```

> **macOS note:** highway-env's render window requires a display. If you're on a headless machine, set `render_mode=None` in `merge_env.py` or run with `python3 -c "..."` style invocations.

> **highway-env patch required:** The `merge-v0` env raises a `ValueError` when receiving a continuous action array. After installing dependencies, apply this one-line patch:
> ```bash
> python3 -c "
> import highway_env, pathlib
> f = pathlib.Path(highway_env.__file__).parent / 'envs/merge_env.py'
> txt = f.read_text()
> txt = txt.replace('action in [0, 2]', 'action in [0, 2] if isinstance(action, (int, float)) else False')
> f.write_text(txt)
> print('Patch applied.')
> "
> ```

---

## Reproducing Phase 4 results (MPC expert)

After setup and the patch above, run a single MPC expert episode:

```bash
python3 src/run_mpc_episode.py
```

**Expected output:** ego stays at y≈4.0 (main highway lane) throughout, speed holds 15–30 m/s, episode completes without crash. MPC call time is printed for the first 5 steps (~12 ms each).

## Reproducing Phase 5 results (expert dataset generation)

```bash
# Quick sanity check across the four traffic mixtures
python3 src/generate_data.py --episodes 5 --all-mixes

# Full datasets used for BC (400 episodes per mixture)
python3 src/generate_data.py --episodes 400 --all-mixes
```

Outputs are saved under `data/`:

| mixture | records | clean records | crashes / episodes | crash rate |
| --- | ---: | ---: | ---: | ---: |
| all_normal | 18,629 | 18,519 | 10 / 400 | 2.5% |
| default_mix | 18,346 | 18,248 | 11 / 400 | 2.8% |
| cautious_heavy | 18,630 | 18,524 | 12 / 400 | 3.0% |
| aggressive_heavy | 18,523 | 18,460 | 8 / 400 | 2.0% |

Total clean transitions available for BC: 73,751.

To run the best-response prediction tests:

```bash
python3 src/test_best_response.py
# Saves plots to plots/test_single_follower.png and plots/test_chain_propagation.png
```

## Reproducing Phase 6 results (behavioral cloning)

```bash
# Train on each driver mix (ablation)
python3 src/train_policy.py --dataset all_normal
python3 src/train_policy.py --dataset default_mix
python3 src/train_policy.py --dataset cautious_heavy
python3 src/train_policy.py --dataset aggressive_heavy

# Train on all four combined (PPO warm-start)
python3 src/train_policy.py --dataset all

# Evaluate a model (20 rollout episodes)
python3 src/eval_policy.py --model models/bc_policy_all.pt --episodes 20 --traffic-mix default_mix --save-plot --no-mpc-baseline

# Cross-evaluate the four single-mixture models
python3 src/cross_eval_bc.py --episodes 50
```

Outputs: `models/bc_policy_{dataset}.pt` (local weights), `models/bc_policy_{dataset}.npz` (normalization stats), and `plots/bc_policy_{dataset}_loss.csv/png` (loss curve). Both `.pt` and `.npz` are required at rollout time. The `.pt` weights are local training artifacts and may need to be regenerated before evaluation.

**Phase 6 results (400-ep clean datasets, 50-ep cross-eval, May 2):**

Model MSE on validation set (all beat mean-action baseline by ~30%):

- all_normal: MSE 0.054 (baseline 0.079)
- default_mix: MSE 0.061 (baseline 0.092)
- cautious_heavy: MSE 0.063 (baseline 0.093)
- aggressive_heavy: MSE 0.061 (baseline 0.092)
- all combined: MSE 0.060 (baseline 0.089)

**4×4 cross-evaluation (50 episodes per cell):**

| trained_on \ tested_on | all_normal | default_mix | cautious_heavy | aggressive_heavy |
| --- | --- | --- | --- | --- |
| all_normal | 0% crash, R=30.5, v=0.5 | 0% crash, R=30.4, v=0.5 | 0% crash, R=30.5, v=0.5 | 0% crash, R=30.5, v=0.5 |
| default_mix | 18% crash, R=26.9, v=7.0 | 12% crash, R=28.9, v=5.6 | 12% crash, R=28.6, v=5.3 | 26% crash, R=24.3, v=8.5 |
| cautious_heavy | 10% crash, R=28.5, v=4.0 | 10% crash, R=28.4, v=3.8 | 20% crash, R=25.8, v=7.4 | 10% crash, R=28.4, v=3.9 |
| aggressive_heavy | 72% crash, R=10.5, v=22.5 | 46% crash, R=19.2, v=17.0 | 62% crash, R=13.8, v=20.3 | 52% crash, R=17.2, v=18.3 |

All crashes verified as spawn collisions at step ≤ 3. Zero policy-induced crashes. `default_mix` model chosen as PPO warm-start (best crash/speed tradeoff). See `nathan.md` for full bug diagnosis and fix details.

## Reproducing Phase 7 results (PPO fine-tuning)

Final model: `models/ppo_500k_v2_merge.zip` (not tracked by git — transfer separately).

```bash
# Train from scratch (90 min on Apple M2)
python3 src/rl_finetune.py --timesteps 500000 --eval-episodes 50 --traffic-mix default_mix --out models/ppo_500k_v2_merge

# Verify final model (should print 18/20 MERGE OK)
python3 src/diagnose_v3.py models/ppo_500k_v2_merge.zip
```

## Reproducing Phase 8–9 results (baseline + final evaluation)

```bash
# Run baseline standalone (independent 2-agent MPC)
python3 src/baseline.py --episodes 50 --traffic-mix default_mix --seed 0

# Run full 4-method evaluation on one traffic mix
python3 src/evaluate.py --episodes 50 --traffic-mix default_mix --seed 0

# Run all 4 traffic mixes
for mix in default_mix all_normal cautious_heavy aggressive_heavy; do
  python3 src/evaluate.py --episodes 50 --traffic-mix $mix --seed 0 --output diagnostics/final_eval_${mix}.csv
done
```

**Final results (50 episodes, default_mix):**

| Method | Merge Success | Crash | Mean Speed | Merge Step | Min TTC |
|--------|-------------|-------|------------|------------|--------|
| BC | 0% | 10% | 2.93 m/s | — | 17.21s |
| PPO (ours) | **96%** | **0%** | **19.56 m/s** | **27.6** | 15.11s |
| Baseline (indep. MPC) | 90% | 2% | 6.56 m/s | 48.7 | 0.60s |
| Full MPC | 92% | 2% | 7.89 m/s | 46.3 | 0.48s |

Results CSVs saved to `diagnostics/`.
