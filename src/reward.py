"""
Feature-based reward functions for the ego vehicle.

r_i = theta_i^T f(s_t, u_t)  -- six features, three weight vectors.

Features follow Sadigh et al. (2016) RSS (forward_progress, proximity,
lane_deviation) extended with smoothness/jerk for comfort per Bae et al.
(2020) IROS. Weight vectors encode the SVO taxonomy from Schwarting et al.
(2019) PNAS 116(50) DOI:10.1073/pnas.1820676116.

References:
  Sadigh et al. (2016) RSS
  Schwarting et al. (2019) PNAS DOI:10.1073/pnas.1820676116
  Bae et al. (2020) IROS
  Leurent (2019) highway-env
  Khatib (1986) IJRR 5(1)
"""

import numpy as np

# Weight vectors: [forward_progress, proximity_penalty, smoothness, jerk, collision, lane_deviation]
#
# forward_progress (0.2 / 0.5 / 0.9)
#   Feature = vx/v_desired in [0,1]. 0.5 is midpoint reference for normal.
#   Ordering encodes SVO angle phi (Schwarting 2019): prosocial -> lower
#   self-weight, egoistic -> higher. Only positive weights; set the scale.
#
# proximity_penalty (-2.0 / -1.0 / -0.5)
#   Feature = -1/max(d_min^2, 0.1). Repulsive potential growing as 1/d^2
#   (Khatib 1986 IJRR). 4:2:1 ratio -> cautious feels 4x proximity cost of
#   aggressive at any gap. Aggressive raised from -0.3 to -0.5 to prevent
#   tailgating from collapsing all behavior into the collision-penalty regime.
#
# smoothness (-0.5 / -0.3 / -0.1)
#   Feature = -a^2. Comfort objective, Bae et al. (2020) Eq. 4. Secondary
#   term -- at a=3 m/s^2 contributes only -4.5/-2.7/-0.9. Tune after MPC.
#
# jerk (-0.3 / -0.2 / -0.1)
#   Feature = -((a_t - a_{t-1})/dt)^2. Rate of acceleration change, also
#   Bae et al. (2020). Smaller than smoothness; second-order comfort term.
#
# collision (-1000.0 / -1000.0 / -1000.0)
#   Feature = -1 if collision else 0. -1000 chosen so collision cost exceeds
#   max cumulative per-step reward at any horizon: 50 steps * max weight 0.9
#   = 45 << 1000. No per-step gain can justify a crash (Leurent 2019,
#   Sadigh et al. 2016). Identical across types -- all drivers avoid crashes.
#
# lane_deviation (-0.5 / -0.3 / -0.1)
#   Feature = -(y - y_target)^2. At 1 m lateral error, feature = -1. Cautious
#   centers strictly; aggressive tolerates drift to exploit gaps. IMPORTANT:
#   y_target must be the *target* lane center (lane being merged into), not
#   current lane -- using current lane would penalize the merge maneuver itself.

# fmt: off
#                [fwd_prog, proximity, smooth,  jerk,   collision, lane_dev]
CAUTIOUS   = np.array([0.2,  -2.0,    -0.5,   -0.3,   -1000.0,   -0.5])
NORMAL     = np.array([0.5,  -1.0,    -0.3,   -0.2,   -1000.0,   -0.3])
AGGRESSIVE = np.array([0.9,  -0.5,    -0.1,   -0.1,   -1000.0,   -0.1])
# fmt: on

THETA = {
    "cautious": CAUTIOUS,
    "normal": NORMAL,
    "aggressive": AGGRESSIVE,
}


def compute_features(
    state: dict,
    action,
    prev_action,
    dt: float = 0.1,
):
    """
    Compute the six reward features for one timestep.

    Args:
        state: dict with keys:
            vx        -- ego longitudinal speed (m/s)
            d_min     -- distance to nearest vehicle (m)
            y         -- ego lateral position (m)
            y_target  -- lateral center of the *target* lane (m), NOT the
                         current lane. During a merge set to the center of
                         the lane being merged into; using current lane would
                         penalize the merge maneuver itself.
            collision -- (optional bool) True if ego is in collision
        action:      shape (1,) -- [acceleration command]
        prev_action: shape (1,) -- previous [acceleration command]
        dt:          timestep duration (s)

    Returns:
        f: np.ndarray of shape (6,)
    """
    v_desired = 30.0  # m/s -- highway target speed

    forward_progress = state["vx"] / v_desired
    proximity_penalty = -1.0 / max(state["d_min"] ** 2, 0.1)
    smoothness = -(action[0] ** 2)
    jerk = -((action[0] - prev_action[0]) / dt) ** 2
    collision = -1.0 if state.get("collision", False) else 0.0
    lane_deviation = -((state["y"] - state["y_target"]) ** 2)

    return np.array([forward_progress, proximity_penalty, smoothness, jerk, collision, lane_deviation])


def ego_reward(
    state: dict,
    action,
    prev_action,
    theta,
    dt: float = 0.1,
) -> float:
    """
    Scalar reward for the ego vehicle given weight vector theta.

    Returns: float r = theta^T f
    """
    f = compute_features(state, action, prev_action, dt)
    return float(theta @ f)
