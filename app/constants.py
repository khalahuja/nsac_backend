"""Central store of physical and lunar constants.

All values are SI unless the name says otherwise. Keeping them here means the
science modules never hard-code a magic number, and swapping in a better value
(e.g. a body-specific albedo map) touches exactly one file.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Universal physical constants
# --------------------------------------------------------------------------

#: Stefan-Boltzmann constant sigma [W m^-2 K^-4] (CODATA 2018).
STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8

#: Astronomical unit [km] (IAU 2012 definition).
ASTRONOMICAL_UNIT_KM: Final[float] = 149_597_870.7

#: Total solar irradiance at exactly 1 AU [W m^-2].
#: Used as S0 in  S = S0 * (1 AU / d)^2.
SOLAR_CONSTANT_1AU_W_M2: Final[float] = 1361.0

# --------------------------------------------------------------------------
# Lunar body constants
# --------------------------------------------------------------------------

#: IAU mean lunar radius [km]. The Moon is treated as a sphere of this radius
#: in the MVP; a DEM later supplies radius = MOON_RADIUS_KM + elevation.
MOON_RADIUS_KM: Final[float] = 1737.4
MOON_RADIUS_M: Final[float] = MOON_RADIUS_KM * 1000.0

#: Mean synodic month [days] - the lunar "day" as seen from a surface point.
SYNODIC_MONTH_DAYS: Final[float] = 29.530588853
SYNODIC_MONTH_HOURS: Final[float] = SYNODIC_MONTH_DAYS * 24.0

# --------------------------------------------------------------------------
# Regolith thermophysical defaults
# --------------------------------------------------------------------------
# These are broad global averages. They are NOT a spatially resolved albedo or
# emissivity map; any result derived from them is an approximation.

#: Bolometric Bond albedo of average lunar regolith (dimensionless).
DEFAULT_ALBEDO: Final[float] = 0.12

#: Thermal infrared emissivity of lunar regolith (dimensionless).
DEFAULT_EMISSIVITY: Final[float] = 0.95

#: Night-side temperature floor [K].
#:
#: Instantaneous radiative equilibrium would give T -> 0 K in darkness, which is
#: unphysical: the regolith stores heat during the day and releases it slowly,
#: and there is a small endogenic heat flux. Diviner observes equatorial
#: pre-dawn lows near 90-100 K, so we model a background flux that asymptotes
#: to this value. See thermal.py for how it enters the energy balance.
NIGHT_TEMPERATURE_FLOOR_K: Final[float] = 100.0

#: Absolute zero offset for K <-> degC conversion.
KELVIN_OFFSET: Final[float] = 273.15

# --------------------------------------------------------------------------
# Numerical tolerances
# --------------------------------------------------------------------------

#: Below this |cos(incidence)| the Sun is treated as exactly on the horizon.
#: Guards against sign noise from floating-point dot products.
HORIZON_EPSILON: Final[float] = 1e-12
