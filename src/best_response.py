"""
Iterative best-response trajectory prediction for non-ego vehicles.

Each non-ego vehicle is modelled as an IDM follower (Treiber et al. 2000,
DOI:10.1103/PhysRevE.62.1805). Given a nominal ego trajectory, we iterate
K times updating every vehicle's predicted trajectory given the others' current
predictions (Sadigh et al. 2016). The last iterate is always returned as a
fallback — non-convergence with aggressive drivers is expected and is documented
in the paper's methods section.

Action-space note: this module works entirely in physical units (m, m/s, m/s²).
The ego's *env* actions are normalised to [-1, 1] per Box(-1,1,(2,),float32),
but the trajectories stored here are raw positions and speeds, not env actions.
"""

import numpy as np

DT = 0.1                    # seconds per timestep (matches highway-env default)
HORIZON = 50                # planning horizon: 5 s at 0.1 s/step
MAX_ITER = 4                # max best-response iterations
EPSILON = 0.1               # convergence: per-step max diff in m or m/s
LANE_WIDTH_THRESHOLD = 4.0  # m; vehicles farther apart laterally are ignored by IDM
VEHICLE_LENGTH = 4.5        # m; subtracted from raw gap to get bumper-to-bumper gap


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def straight_line_trajectory(vehicle, horizon: int = HORIZON, dt: float = DT) -> np.ndarray:
    """
    Constant-speed, straight-line prediction for a vehicle.

    Parameters
    ----------
    vehicle : object with .position (array-like [x, y]) and .speed (float)
    horizon : number of timesteps to predict
    dt      : seconds per step

    Returns
    -------
    np.ndarray of shape (horizon, 3): columns are [x, y, vx]
    """
    x, y, vx = float(vehicle.position[0]), float(vehicle.position[1]), float(vehicle.speed)
    traj = []
    for _ in range(horizon):
        x += vx * dt
        traj.append([x, y, vx])
    return np.array(traj, dtype=np.float64)


def idm_acceleration(speed: float, target_speed: float, gap: float,
                     time_wanted: float, a_max: float = 1.5, s0: float = 2.0,
                     delta: int = 4) -> float:
    """
    Intelligent Driver Model acceleration (Treiber et al. 2000).

    Parameters
    ----------
    speed        : current vehicle speed (m/s)
    target_speed : desired free-flow speed (m/s)
    gap          : bumper-to-bumper distance to lead vehicle (m)
    time_wanted  : desired time headway T (s)
    a_max        : maximum acceleration (m/s²)
    s0           : minimum jam distance (m)
    delta        : free-road exponent (dimensionless)

    Returns
    -------
    Acceleration clipped to [-6, 4] m/s².
    """
    s_star = s0 + max(0.0, speed * time_wanted)
    acc = a_max * (
        1.0
        - (speed / max(target_speed, 0.1)) ** delta
        - (s_star / max(gap, 0.5)) ** 2
    )
    return float(np.clip(acc, -6.0, 4.0))


def idm_predict(vehicle, ego_trajectory: np.ndarray,
                all_predicted: list, own_idx: int) -> np.ndarray:
    """
    Predict a single non-ego vehicle's trajectory via IDM, accounting for the
    ego and all other vehicles' current predicted trajectories.

    Bug fixes vs. naive version
    ---------------------------
    * Positive-gap filter : only vehicles strictly ahead (other_x > x) are
      considered as a lead vehicle.
    * Lane check          : vehicles with lateral separation > LANE_WIDTH_THRESHOLD
      are ignored (cross-lane vehicles should not trigger IDM braking).

    Parameters
    ----------
    vehicle        : the vehicle to predict (needs .position, .speed,
                     optionally .TIME_WANTED and .target_speed)
    ego_trajectory : (HORIZON, 3) array [x, y, vx] for the ego vehicle
    all_predicted  : list of (HORIZON, 3) arrays, one per non-ego vehicle
    own_idx        : index of this vehicle in all_predicted (excluded from
                     lead-vehicle search to avoid self-interaction)

    Returns
    -------
    np.ndarray of shape (HORIZON, 3): predicted [x, y, vx] trajectory.
    """
    x = float(vehicle.position[0])
    y = float(vehicle.position[1])
    vx = float(vehicle.speed)
    time_wanted = float(getattr(vehicle, "TIME_WANTED", 1.5))
    target_speed = float(getattr(vehicle, "target_speed", 30.0))
    # Per-vehicle IDM params if available (driver_types sets these)
    a_max = float(getattr(vehicle, "COMFORT_ACC_MAX", 1.5))
    s0 = float(getattr(vehicle, "DISTANCE_WANTED", 2.0))
    delta = int(getattr(vehicle, "DELTA", 4))

    traj = []
    for t in range(HORIZON):
        candidate_gaps = []

        # --- ego as potential lead vehicle ---
        ego_pos = ego_trajectory[t] if t < len(ego_trajectory) else ego_trajectory[-1]
        if ego_pos[0] > x and abs(ego_pos[1] - y) < LANE_WIDTH_THRESHOLD:
            candidate_gaps.append(ego_pos[0] - x)

        # --- other non-ego vehicles as potential lead vehicles ---
        for j, other_traj in enumerate(all_predicted):
            if j == own_idx:
                continue
            other_pos = other_traj[t] if t < len(other_traj) else other_traj[-1]
            if other_pos[0] > x and abs(other_pos[1] - y) < LANE_WIDTH_THRESHOLD:
                candidate_gaps.append(other_pos[0] - x)

        # Bumper-to-bumper gap; 100 m if no lead vehicle found
        raw_gap = min(candidate_gaps) if candidate_gaps else 100.0
        gap = max(raw_gap - VEHICLE_LENGTH, 0.5)

        acc = idm_acceleration(vx, target_speed, gap, time_wanted, a_max, s0, delta)
        vx = max(0.0, vx + acc * DT)
        x += vx * DT
        traj.append([x, y, vx])

    return np.array(traj, dtype=np.float64)


# ---------------------------------------------------------------------------
# Iterative best-response
# ---------------------------------------------------------------------------

def predict_other_responses(env, ego_trajectory: np.ndarray,
                            max_iter: int = MAX_ITER) -> list:
    """
    Given a nominal ego trajectory, iteratively predict non-ego vehicle
    trajectories using best-response dynamics (Sadigh et al. 2016).

    Design decision: call this ONCE per MPC timestep with a constant-speed
    nominal ego trajectory, then reuse the result across all N candidate
    sequences. This gives a ~50x speedup vs. calling per sample, with
    acceptable accuracy loss for training-data generation purposes.

    Convergence: per-step maximum absolute difference across all vehicles,
    not the full trajectory L2 norm (which scales with horizon length).

    Parameters
    ----------
    env            : gymnasium env (uses env.unwrapped.road.vehicles)
    ego_trajectory : (HORIZON, 3) nominal ego trajectory [x, y, vx]
    max_iter       : maximum best-response iterations

    Returns
    -------
    List of (HORIZON, 3) arrays, one per non-ego vehicle (vehicles[1:]).
    The last iterate is always returned regardless of convergence.
    """
    non_ego = env.unwrapped.road.vehicles[1:]
    predicted = [straight_line_trajectory(v) for v in non_ego]

    for iteration in range(max_iter):
        prev = [p.copy() for p in predicted]
        for i, vehicle in enumerate(non_ego):
            predicted[i] = idm_predict(vehicle, ego_trajectory, predicted, i)

        # Per-step max diff convergence check
        diffs = [np.max(np.abs(predicted[i] - prev[i])) for i in range(len(non_ego))]
        if max(diffs) < EPSILON:
            break  # converged; last iterate is already the result

    return predicted
