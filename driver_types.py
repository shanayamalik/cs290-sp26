"""
Driver type definitions for non-ego vehicles in the merge environment.

IDM parameters are hand-tuned based on empirical ranges from:
  - Treiber et al. (2000), Physical Review E 62(2) — IDM parameter ranges
  - Schwarting et al. (2019), PNAS 116(50) — SVO taxonomy (cautious ~
    prosocial, normal ~ individualistic, aggressive ~ competitive)

Parameters are not learned from data. Driver types are assumed known at
planning time; online estimation is left as future work.
"""

from highway_env.vehicle.behavior import IDMVehicle


def make_cautious(vehicle: IDMVehicle) -> None:
    """
    Cautious driver: large headway, conservative acceleration, prioritizes safety.

    Corresponds to a prosocial SVO (Schwarting et al., phi ~ pi/4).
    TIME_WANTED of 2.5 s and DISTANCE_WANTED of 10 m are at the high end of
    empirically observed values in the IDM literature (Treiber et al., 2000).
    """
    vehicle.COMFORT_ACC_MAX = 2.0   # m/s^2 — gentle acceleration
    vehicle.COMFORT_ACC_MIN = -3.0  # m/s^2 — moderate braking
    vehicle.TIME_WANTED = 2.5       # s — large desired time headway
    vehicle.DISTANCE_WANTED = 10.0  # m — large minimum gap
    vehicle.DELTA = 4               # IDM velocity exponent (standard value)


def make_normal(vehicle: IDMVehicle) -> None:
    """
    Normal driver: balanced safety and efficiency.

    Corresponds to an individualistic SVO (Schwarting et al., phi ~ 0).
    Parameters match highway-env's built-in IDMVehicle defaults, which are
    calibrated to typical observed highway driving behavior.
    """
    vehicle.COMFORT_ACC_MAX = 3.0   # m/s^2
    vehicle.COMFORT_ACC_MIN = -5.0  # m/s^2
    vehicle.TIME_WANTED = 1.5       # s — typical observed highway headway
    vehicle.DISTANCE_WANTED = 5.0   # m
    vehicle.DELTA = 4


def make_aggressive(vehicle: IDMVehicle) -> None:
    """
    Aggressive driver: short headway, high speed priority, accepts small gaps.

    Corresponds to a competitive/egoistic SVO (Schwarting et al., phi ~ -pi/4).
    TIME_WANTED is kept >= 1.0 s to avoid physically unrealistic collisions
    during early testing; this can be tightened once the MPC expert is stable.
    DISTANCE_WANTED of 2.0 m represents the physical minimum gap (roughly
    half a vehicle length).
    """
    vehicle.COMFORT_ACC_MAX = 4.0   # m/s^2 — strong acceleration
    vehicle.COMFORT_ACC_MIN = -6.0  # m/s^2 — hard braking capability
    vehicle.TIME_WANTED = 1.0       # s — short headway; keep >= 1.0 early on
    vehicle.DISTANCE_WANTED = 2.0   # m — minimum physical gap
    vehicle.DELTA = 4
