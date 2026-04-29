"""
Driver archetypes for non-ego vehicles based on the Intelligent Driver Model (IDM).

Reference IDM values (Treiber et al., 2000, Physical Review E 62(2)):
  DOI: 10.1103/PhysRevE.62.1805 | arXiv: cond-mat/0002177
  Standard parameters: T = 1.5 s, s0 = 2 m, delta = 4.
  (Note: Treiber's original empirical fits use a ≈ 0.73 m/s², b ≈ 1.67 m/s²
   for real-world data; we use values appropriate for highway simulation.)

SVO taxonomy for cautious / normal / aggressive mapping:
  Schwarting et al. (2019), PNAS 116(50) — prosocial, individualistic, competitive.
  DOI: 10.1073/pnas.1820676116

All parameters are hand-tuned, not learned from data. Normal driver uses
Treiber reference values for T and s0. Cautious and aggressive are scaled
from there. Driver types are assumed known at planning time; online
estimation is left as future work.
"""

from highway_env.vehicle.behavior import IDMVehicle


def make_cautious(vehicle: IDMVehicle) -> None:
    """
    Cautious driver:
    Larger time headway and larger standstill gap than standard IDM.
    Lower acceleration and gentler braking to create conservative behavior.

    Literature basis:
    - IDM uses desired time headway T, minimum gap s0, acceleration a,
      comfortable braking b, and exponent delta.
    - Standard IDM commonly uses T ≈ 1.5 s, s0 ≈ 2 m, delta = 4.
    - We hand-tune cautious behavior by increasing T and s0.

    Corresponds to a prosocial SVO (Schwarting et al., phi ~ pi/4).
    """
    vehicle.COMFORT_ACC_MAX = 1.5   # m/s^2 — gentle acceleration
    vehicle.COMFORT_ACC_MIN = -2.0  # m/s^2 — moderate braking
    vehicle.TIME_WANTED = 2.5       # s — large desired time headway
    vehicle.DISTANCE_WANTED = 8.0   # m — large minimum gap
    vehicle.DELTA = 4               # IDM velocity exponent (standard value)


def make_normal(vehicle: IDMVehicle) -> None:
    """
    Normal driver:
    Uses standard/reference IDM-style values.

    Literature basis:
    - Treiber-style IDM reference values commonly use:
      T ≈ 1.5 s, s0 ≈ 2 m, delta = 4.

    Corresponds to an individualistic SVO (Schwarting et al., phi ~ 0).
    """
    vehicle.COMFORT_ACC_MAX = 1.5   # m/s^2
    vehicle.COMFORT_ACC_MIN = -2.0  # m/s^2
    vehicle.TIME_WANTED = 1.5       # s — reference IDM time headway
    vehicle.DISTANCE_WANTED = 2.0   # m — reference IDM minimum gap
    vehicle.DELTA = 4


def make_aggressive(vehicle: IDMVehicle) -> None:
    """
    Aggressive driver:
    Smaller time headway and smaller minimum gap than standard IDM.
    Higher acceleration and stronger braking to create assertive behavior.

    We keep TIME_WANTED >= 1.0 s for simulation stability.

    Corresponds to a competitive/egoistic SVO (Schwarting et al., phi ~ -pi/4).
    """
    vehicle.COMFORT_ACC_MAX = 2.5   # m/s^2 — strong acceleration
    vehicle.COMFORT_ACC_MIN = -4.0  # m/s^2 — hard braking capability
    vehicle.TIME_WANTED = 1.0       # s — short headway; keep >= 1.0 for stability
    vehicle.DISTANCE_WANTED = 1.0   # m — tight minimum gap
    vehicle.DELTA = 4
