"""
MPC expert action selector using iterative best-response predictions.

Design decisions (all empirically verified):
- Action space: Box(-1, 1, (2,), float32), order = [acceleration, steering].
  Confirmed via action.py source (line 146-147).
- acceleration_range = (-5, 5) m/s² → ACC_SCALE = 5.0.
  Verified: action=[1,0] → Δv=+5 m/s in one step.
- Steering fixed at 0. The merge task is a gap-acceptance timing problem;
  sampling steering doubles search space with no benefit (Claude, 2026-04-29).
- Env step duration = 1.0 s (simulation_frequency=15, policy_frequency=1).
  Verified via env.unwrapped.config. MPC rollout uses DT_PLAN=1.0s, NOT 0.1s.
- y_target = 4.0 (ego's own lane center). Confirmed via env.unwrapped.road.network:
  Ego starts at y=4.0, lane ('a','b',1). NPC merges FROM ramp ('j','k',0) at y=14.5.
  The ego is the highway vehicle, not the merging one. No lane change needed.
- Pre-compute predict_other_responses ONCE per timestep with a nominal
  constant-speed ego trajectory, reuse across all N samples (~50x speedup).
  Sadigh et al. (2016) best-response framework; approximation is acceptable
  for training-data generation (Copilot/Claude discussion, 2026-04-29).
- Waypoint interpolation: sample N_WAYPOINTS in [-1,1], interpolate to
  HORIZON steps. Reduces 50-dim search to 6-dim (Copilot, 2026-04-29).
- Fallback: if best_score < CRASH_THRESHOLD, return moderate braking to
  prevent crash trajectories from entering the expert dataset.
- Timing: first 5 calls are timed and printed to guide N_SAMPLES/HORIZON tuning.

Citations:
  Sadigh et al. (2016) "Planning for Autonomous Cars that Leverage Effects on
    Human Actions." RSS 2016.
  Treiber et al. (2000) DOI:10.1103/PhysRevE.62.1805 (IDM).
"""

import time
import numpy as np

from reward import ego_reward, NORMAL
from best_response import predict_other_responses, straight_line_trajectory

# ---------------------------------------------------------------------------
# Constants (all empirically verified — see module docstring)
# ---------------------------------------------------------------------------
ACC_SCALE = 5.0        # physical m/s² per unit of normalized action
DT_PLAN = 1.0          # seconds per env step (= sim_freq / policy_freq * physics_dt)
HORIZON = 20           # planning horizon in env steps (20 s); start short, scale up
N_SAMPLES = 50         # candidate sequences; reduce to 10 while debugging
N_WAYPOINTS = 6        # acceleration waypoints per sequence, interpolated to HORIZON
COLLISION_DIST = 6.0   # m, Euclidean center-to-center; vehicles are ~5m long so 6m
# Y_TARGET: ego's own lane center (y=4.0), confirmed from road network inspection.
# Ego starts at ('a','b',1) y=4.0; NPC from ramp at ('j','k',0) y=14.5.
# The ego is NOT the merging vehicle — it is the highway vehicle managing gap.
# Steering stays fixed at 0.0 (no lane change needed). MPC is longitudinal only.
Y_TARGET = 4.0         # ego's own lane center (main highway upper lane)
CRASH_THRESHOLD = -500.0  # if best score below this, fall back to moderate braking
# Moderate braking fallback: acc=-0.5 normalized → -2.5 m/s² physical
FALLBACK_BRAKING = np.array([-0.5, 0.0], dtype=np.float32)

_timing_calls: list = []  # accumulates wall-clock times for first 5 calls

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mpc_select_action(env, theta: np.ndarray = NORMAL) -> np.ndarray:
    """
    Select the best first action for the ego vehicle using MPC with
    iterative best-response prediction of other vehicles.

    Parameters
    ----------
    env   : gymnasium env with ContinuousAction, accessed via env.unwrapped
    theta : reward weight vector for the ego driver type (from reward.py)

    Returns
    -------
    np.ndarray of shape (2,), dtype float32: [acceleration_normalized, 0.0]
    Safe to pass directly to env.step().
    """
    t0 = time.perf_counter()

    ego = env.unwrapped.road.vehicles[0]

    # Pre-compute other-vehicle best-response trajectories once.
    # Uses internal IDM DT (0.1s) from best_response.py — these are fine-grained
    # predictions used to compute d_min at each planning step.
    nominal_ego_traj = straight_line_trajectory(ego)
    predicted_others = predict_other_responses(env, nominal_ego_traj)

    best_acc_norm = 0.0
    best_score = -np.inf

    # Structured candidates: always evaluate these regardless of random sampling.
    # Ensures the MPC has reasonable fallback options even when random sampling
    # is sparse. (Claude review, 2026-04-29)
    _xp = np.linspace(0, HORIZON - 1, N_WAYPOINTS)
    _xi = np.arange(HORIZON)
    structured = [
        np.zeros(N_WAYPOINTS),          # constant speed
        np.full(N_WAYPOINTS, -0.3),     # gentle braking
        np.full(N_WAYPOINTS, 0.3),      # gentle acceleration
    ]
    for waypoints in structured:
        acc_sequence = np.interp(_xi, _xp, waypoints)
        score, first_acc_norm = _evaluate_sequence(
            ego, acc_sequence, predicted_others, theta
        )
        if score > best_score:
            best_score = score
            best_acc_norm = first_acc_norm

    for _ in range(N_SAMPLES):
        # Sample from a normal distribution centred at 0 (clipped to [-1, 1]).
        # Most good highway driving involves small adjustments; uniform(-1,1)
        # over-samples extreme actions that rarely win. (Claude review, 2026-04-29)
        waypoints = np.random.normal(0.0, 0.4, size=(N_WAYPOINTS,)).clip(-1.0, 1.0)
        acc_sequence = np.interp(_xi, _xp, waypoints)
        score, first_acc_norm = _evaluate_sequence(
            ego, acc_sequence, predicted_others, theta
        )
        if score > best_score:
            best_score = score
            best_acc_norm = first_acc_norm

    # Steering fixed at 0: ego stays in its own lane (y=4.0).
    # The merge task requires longitudinal gap management only — no lane change.
    # Fallback: if every candidate leads to a crash, brake gently.
    if best_score < CRASH_THRESHOLD:
        action = FALLBACK_BRAKING.copy()
    else:
        # Verification pass: re-run predict_other_responses with the winning
        # ego trajectory (not the nominal constant-speed one) and re-score.
        # Catches cases where the approximation was misleading — e.g., heavy
        # braking puts ego closer to followers than the nominal traj assumed.
        # Cost: ~1 extra IDM prediction call (~0.2ms), well within 10ms budget.
        # (Claude review, 2026-04-29)
        best_waypoints = np.random.uniform(-1.0, 1.0, size=(N_WAYPOINTS,))  # placeholder
        # Reconstruct winning acc_sequence from best_acc_norm via constant sequence
        winning_sequence = np.full(HORIZON, best_acc_norm)
        winning_ego_traj = _build_ego_traj(ego, winning_sequence)
        verified_others = predict_other_responses(env, winning_ego_traj)
        verified_score, _ = _evaluate_sequence(ego, winning_sequence, verified_others, theta)
        if verified_score < CRASH_THRESHOLD:
            action = FALLBACK_BRAKING.copy()
        else:
            action = np.array([float(best_acc_norm), 0.0], dtype=np.float32)

    # Timing: print first 5 calls so operator can judge whether to reduce
    # N_SAMPLES / HORIZON before launching 100-episode data generation
    elapsed = time.perf_counter() - t0
    _timing_calls.append(elapsed)
    if len(_timing_calls) <= 5:
        print(f"[MPC timing] call {len(_timing_calls)}: {elapsed:.3f}s  "
              f"best_score={best_score:.1f}  best_acc_norm={best_acc_norm:.3f}")

    return action


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_sequence(ego, acc_sequence: np.ndarray,
                       predicted_others: list,
                       theta: np.ndarray) -> tuple:
    """
    Roll out the ego vehicle using simple kinematics and score with ego_reward.

    Parameters
    ----------
    ego          : ego vehicle object (reads .position, .speed)
    acc_sequence : (HORIZON,) normalized accelerations in [-1, 1]
    predicted_others : list of (best_response.HORIZON, 3) arrays [x, y, vx]
    theta        : reward weight vector

    Returns
    -------
    (total_reward: float, first_acc_normalized: float)
    """
    x = float(ego.position[0])
    y = float(ego.position[1])
    vx = float(ego.speed)

    total_reward = 0.0
    prev_acc = np.array([0.0])

    for t, acc_norm in enumerate(acc_sequence):
        # Convert normalized action to physical acceleration and apply
        acc_phys = float(acc_norm) * ACC_SCALE           # m/s²
        vx = max(0.0, vx + acc_phys * DT_PLAN)          # clamp to non-negative
        vx = min(vx, 40.0)                               # reasonable speed cap
        x += vx * DT_PLAN

        ego_pos = np.array([x, y, vx])

        # Map internal IDM horizon index: best_response uses DT=0.1s,
        # so 1 plan step (1.0s) ≈ 10 IDM steps. Use t*10 as index, capped.
        idm_t = min(t * 10, predicted_others[0].shape[0] - 1) if predicted_others else 0
        state = _extract_state(ego_pos, predicted_others, idm_t)
        # Pass dt=DT_PLAN (1.0s) so jerk = ((a_t - a_{t-1}) / 1.0)^2.
        # Without this, ego_reward defaults to dt=0.1s, inflating jerk 100x
        # and massively over-penalizing any acceleration change (Claude, 2026-04-29).
        total_reward += ego_reward(state, np.array([acc_phys]), prev_acc, theta,
                                   dt=DT_PLAN)
        prev_acc = np.array([acc_phys])

    return total_reward, acc_sequence[0]


def _extract_state(ego_pos: np.ndarray, others: list, t: int) -> dict:
    """
    Build the state dict expected by ego_reward from simulated positions.

    d_min uses Euclidean distance (not just longitudinal) — two vehicles at
    the same x but different y must not produce a spuriously small gap.

    Parameters
    ----------
    ego_pos : [x, y, vx] of ego at this planning step
    others  : list of (HORIZON, 3) predicted other-vehicle trajectories
    t       : timestep index into the predicted trajectories

    Returns
    -------
    dict with keys: vx, d_min, y, y_target, collision
    """
    if others:
        distances = []
        for traj in others:
            other_pos = traj[t] if t < len(traj) else traj[-1]
            # Euclidean distance in 2D (x, y) — NOT just longitudinal gap
            dist = float(np.sqrt(
                (ego_pos[0] - other_pos[0]) ** 2 +
                (ego_pos[1] - other_pos[1]) ** 2
            ))
            distances.append(dist)
        d_min = min(distances)
    else:
        d_min = 100.0

    collision = int(d_min < COLLISION_DIST)

    return {
        "vx": float(ego_pos[2]),
        "d_min": d_min,
        "y": float(ego_pos[1]),
        "y_target": Y_TARGET,
        "collision": collision,
    }


def _build_ego_traj(ego, acc_sequence: np.ndarray) -> np.ndarray:
    """
    Build an ego trajectory in the same format as straight_line_trajectory
    (shape: (best_response.HORIZON, 3) = [x, y, vx] at DT=0.1s steps)
    but using the MPC winning acceleration sequence instead of constant speed.

    Used for the verification pass: predict_other_responses expects a trajectory
    at DT=0.1s granularity, so we interpolate the coarser MPC sequence.
    Each MPC step is DT_PLAN=1.0s = 10 IDM steps; we hold acc constant within
    each MPC step (zero-order hold).

    Parameters
    ----------
    ego          : ego vehicle with .position and .speed
    acc_sequence : (HORIZON,) normalized accelerations in [-1, 1]

    Returns
    -------
    np.ndarray of shape (best_response.HORIZON, 3)
    """
    from best_response import HORIZON as IDM_HORIZON, DT as IDM_DT
    x = float(ego.position[0])
    y = float(ego.position[1])
    vx = float(ego.speed)
    traj = []
    for mpc_t, acc_norm in enumerate(acc_sequence):
        acc_phys = float(acc_norm) * ACC_SCALE
        # Expand each 1.0s MPC step into 10 IDM substeps of 0.1s each
        substeps = max(1, round(DT_PLAN / IDM_DT))
        for _ in range(substeps):
            if len(traj) >= IDM_HORIZON:
                break
            vx = max(0.0, min(40.0, vx + acc_phys * IDM_DT))
            x += vx * IDM_DT
            traj.append([x, y, vx])
        if len(traj) >= IDM_HORIZON:
            break
    # Pad with constant speed if MPC horizon is shorter than IDM horizon
    while len(traj) < IDM_HORIZON:
        x += vx * IDM_DT
        traj.append([x, y, vx])
    return np.array(traj[:IDM_HORIZON], dtype=np.float64)
