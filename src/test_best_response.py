"""
Isolated tests for best_response.py.

Two scenarios:
  1. Single follower — does it brake when the leader decelerates?
  2. Three-vehicle chain — does braking propagate backward through the chain?

Run from project root:
    python3 src/test_best_response.py

Saves plots to:
    plots/test_single_follower.png
    plots/test_chain_propagation.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from best_response import (
    straight_line_trajectory, idm_predict, idm_acceleration,
    DT, HORIZON
)

os.makedirs("plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Mock vehicle — mimics the interface best_response.py expects
# ---------------------------------------------------------------------------
class MockVehicle:
    def __init__(self, x, y, vx, time_wanted=1.5, target_speed=30.0):
        self.position = np.array([x, y], dtype=float)
        self.speed = float(vx)
        self.TIME_WANTED = time_wanted
        self.target_speed = target_speed
        self.COMFORT_ACC_MAX = 3.0
        self.DISTANCE_WANTED = 2.0
        self.DELTA = 4


# ---------------------------------------------------------------------------
# Helper: simulate a decelerating leader trajectory manually
# ---------------------------------------------------------------------------
def decelerating_trajectory(x0, y0, vx0, decel, horizon=HORIZON, dt=DT):
    """Leader brakes at `decel` m/s² from t=0."""
    x, vx = x0, vx0
    traj = []
    for _ in range(horizon):
        vx = max(0.0, vx + decel * dt)
        x += vx * dt
        traj.append([x, y0, vx])
    return np.array(traj, dtype=np.float64)


# ---------------------------------------------------------------------------
# Scenario 1: Single follower
# ---------------------------------------------------------------------------
def test_single_follower():
    print("=" * 60)
    print("Scenario 1: Single follower brakes when leader decelerates")
    print("=" * 60)

    # Leader starts 30 m ahead, brakes hard at -4 m/s²
    leader_traj = decelerating_trajectory(x0=100.0, y0=0.0, vx0=30.0, decel=-4.0)
    # Baseline: leader at constant speed
    leader_const = decelerating_trajectory(x0=100.0, y0=0.0, vx0=30.0, decel=0.0)

    follower = MockVehicle(x=70.0, y=0.0, vx=30.0)

    # Predict follower response to braking leader vs. constant-speed leader
    follower_braking = idm_predict(follower, leader_traj,   all_predicted=[], own_idx=0)
    follower_const   = idm_predict(follower, leader_const,  all_predicted=[], own_idx=0)

    time = np.arange(HORIZON) * DT

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].set_title("Scenario 1: Single Follower Response to Leader Deceleration")
    axes[0].plot(time, leader_traj[:, 2],   "r-",  lw=2, label="Leader speed (braking)")
    axes[0].plot(time, leader_const[:, 2],  "r--", lw=1, label="Leader speed (constant)")
    axes[0].plot(time, follower_braking[:, 2], "b-",  lw=2, label="Follower speed (braking leader)")
    axes[0].plot(time, follower_const[:, 2],   "b--", lw=1, label="Follower speed (constant leader)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, leader_traj[:, 0],      "r-",  lw=2, label="Leader position (braking)")
    axes[1].plot(time, leader_const[:, 0],     "r--", lw=1, label="Leader position (constant)")
    axes[1].plot(time, follower_braking[:, 0], "b-",  lw=2, label="Follower position (braking leader)")
    axes[1].plot(time, follower_const[:, 0],   "b--", lw=1, label="Follower position (constant leader)")
    axes[1].set_ylabel("Position (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/test_single_follower.png", dpi=150)
    print("Saved: plots/test_single_follower.png")

    # Sanity checks
    final_speed_braking = follower_braking[-1, 2]
    final_speed_const   = follower_const[-1, 2]
    print(f"  Follower final speed (braking leader):  {final_speed_braking:.2f} m/s")
    print(f"  Follower final speed (constant leader): {final_speed_const:.2f} m/s")
    assert final_speed_braking < final_speed_const, \
        "FAIL: follower should be slower when leader brakes"
    print("  PASS: follower slows down when leader brakes\n")


# ---------------------------------------------------------------------------
# Scenario 2: Three-vehicle chain — iterative propagation
# ---------------------------------------------------------------------------
def test_chain_propagation():
    print("=" * 60)
    print("Scenario 2: Three-vehicle chain — iterative best-response propagation")
    print("=" * 60)

    # V1 (lead/ego): x=200, decelerates at -3 m/s²
    # V2: x=155 (45 m gap from V1)
    # V3: x=110 (45 m gap from V2)
    # All start at vx=30 m/s, same lane (y=0)

    ego_traj = decelerating_trajectory(x0=200.0, y0=0.0, vx0=30.0, decel=-3.0)

    v2 = MockVehicle(x=155.0, y=0.0, vx=30.0)
    v3 = MockVehicle(x=110.0, y=0.0, vx=30.0)
    vehicles = [v2, v3]

    # --- Run iterative best-response ---
    from best_response import MAX_ITER, EPSILON
    predicted = [straight_line_trajectory(v) for v in vehicles]

    iter_speeds = {0: [p[:, 2].copy() for p in predicted]}

    for iteration in range(MAX_ITER):
        prev = [p.copy() for p in predicted]
        for i, vehicle in enumerate(vehicles):
            predicted[i] = idm_predict(vehicle, ego_traj, predicted, i)
        diffs = [np.max(np.abs(predicted[i] - prev[i])) for i in range(len(vehicles))]
        iter_speeds[iteration + 1] = [p[:, 2].copy() for p in predicted]
        print(f"  Iteration {iteration + 1}: max per-step diff = {max(diffs):.4f} m/s  "
              f"({'converged' if max(diffs) < EPSILON else 'not converged'})")
        if max(diffs) < EPSILON:
            break

    time = np.arange(HORIZON) * DT

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].set_title("Scenario 2: Chain Propagation — Speed Profiles")
    axes[0].plot(time, ego_traj[:, 2], "k-", lw=2.5, label="V1 (lead, braking at -3 m/s²)")
    axes[0].plot(time, predicted[0][:, 2], "b-", lw=2, label="V2 (responds to V1)")
    axes[0].plot(time, predicted[1][:, 2], "g-", lw=2, label="V3 (responds to V2)")
    axes[0].plot(time, iter_speeds[0][0], "b--", lw=1, alpha=0.5, label="V2 init (straight-line)")
    axes[0].plot(time, iter_speeds[0][1], "g--", lw=1, alpha=0.5, label="V3 init (straight-line)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Position Profiles")
    axes[1].plot(time, ego_traj[:, 0], "k-", lw=2.5, label="V1 (lead)")
    axes[1].plot(time, predicted[0][:, 0], "b-", lw=2, label="V2")
    axes[1].plot(time, predicted[1][:, 0], "g-", lw=2, label="V3")
    axes[1].set_ylabel("Position (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/test_chain_propagation.png", dpi=150)
    print("Saved: plots/test_chain_propagation.png")

    # Sanity checks
    v2_final = predicted[0][-1, 2]
    v3_final = predicted[1][-1, 2]
    ego_final = ego_traj[-1, 2]
    print(f"\n  Final speeds — V1: {ego_final:.2f}  V2: {v2_final:.2f}  V3: {v3_final:.2f} m/s")
    assert v2_final < 30.0, "FAIL: V2 should have slowed down"
    assert v3_final < 30.0, "FAIL: V3 should have slowed down (chain propagation)"
    print("  PASS: braking propagates from V1 → V2 → V3\n")


# ---------------------------------------------------------------------------
# Run both scenarios
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_single_follower()
    test_chain_propagation()
    print("All tests passed. Check plots/ for visualisations.")
