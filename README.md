# Autonomous Driving with Multi-Agent Lane Merging

**Shanaya Malik, Nathan McNaughton**

ELENG 290: Learning-Enabled Multi-Agent Systems — Spring 2026

Extension of Sadigh et al. (2016) to multi-vehicle lane merging with 3+ agents. Each human driver is assigned a predefined reward archetype (cautious / normal / aggressive). The pipeline has three stages: iterative best-response MPC expert → behavioral cloning → PPO fine-tuning.

Driver type parameters (in `src/driver_types.py`) are hand-tuned based on empirically observed ranges from Treiber et al. (2000) and map onto the Social Value Orientation taxonomy from Schwarting et al. (2019). They are not learned from data.

**Progress:** Phases 1–5 complete (environment setup, driver types, reward functions, MPC expert, expert dataset generation). Phase 6 behavioral cloning is implemented and under sanity-check evaluation before PPO fine-tuning.
See [implementation_guide.md](implementation_guide.md) for the full step-by-step implementation plan.
See [report_notes.md](report_notes.md) for things to address in the final paper writeup.

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
| all_normal | 18,718 | 18,616 | 10 / 400 | 2.5% |
| default_mix | 18,983 | 18,913 | 8 / 400 | 2.0% |
| cautious_heavy | 19,003 | 18,933 | 10 / 400 | 2.5% |
| aggressive_heavy | 19,060 | 18,980 | 7 / 400 | 1.8% |

Total clean transitions available for BC: 75,442.

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

**Phase 6 ablation summary (100-episode rollout, seed=0):**

- all_normal: val loss 0.11195, crash rate 41%, mean steps 30.3
- default_mix: val loss 0.11245, crash rate 12%, mean steps 44.2
- cautious_heavy: val loss 0.11508, crash rate 26%, mean steps 37.5
- aggressive_heavy: val loss 0.11825, crash rate 37%, mean steps 32.2
- all combined: val loss 0.11194, crash rate 14%, mean steps 43.3

All crashes at rollout occur at step 2 (spawn collisions, verified by crash-step breakdown). The BC policy causes 0 self-crashes but never completes the merge — it learned safe deceleration from expert demonstrations but not goal-directed forward commitment. This is the expected BC limitation (distribution shift); PPO fine-tuning is the fix.

**2x2 cross-evaluation (50 episodes each, seed=0) — crash rate / mean steps:**

- all_normal model + normal traffic: 34% / 33.7 steps
- all_normal model + aggressive traffic: 44% / 27.9 steps
- aggressive_heavy model + normal traffic: 34% / 32.7 steps
- aggressive_heavy model + aggressive traffic: 28% / 33.7 steps

All cross-eval crashes also at step 2 (verified). Crash rate variation across traffic conditions reflects spawn geometry differences, not policy failure. BC causes 0 policy-induced crashes regardless of traffic condition.
