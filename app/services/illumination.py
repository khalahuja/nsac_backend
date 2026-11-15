"""Solar illumination geometry at a point on the lunar surface.

Given the Sun's direction (from :mod:`app.services.astronomy`) and the local
surface normal (from :mod:`app.services.terrain`) this module produces the
angles and fluxes a mission planner cares about.

Angle definitions
-----------------

**Solar incidence angle** ``theta`` - between the *surface normal* and the Sun:

.. math:: \\cos\\theta = \\mathbf{N}\\cdot\\mathbf{S}

**Solar elevation** ``h`` - between the Sun and the *local horizontal plane*,
where horizontal is defined by the spherical radial direction:

.. math:: h = 90^\\circ - \\arccos(\\mathbf{U}\\cdot\\mathbf{S})

On a perfect sphere ``N == U`` so ``h = 90 - theta`` exactly, as specified. On
sloped terrain the two differ, which is precisely why a north-facing slope can
be lit while a south-facing one beside it is not.

**Solar azimuth** ``A`` - compass bearing of the Sun, clockwise from North:

.. math:: A = \\operatorname{atan2}(\\mathbf{S}\\cdot\\hat{e},\\;
                                    \\mathbf{S}\\cdot\\hat{n}) \\bmod 360

so 0 = North, 90 = East, 180 = South, 270 = West.

Irradiance
----------

Direct-beam irradiance on the tilted surface:

.. math:: Q = S_0 \\left(\\frac{1\\,\\text{AU}}{d}\\right)^2 \\cos\\theta

with ``S0 = 1361 W/m^2`` at 1 AU and ``d`` the true Sun-Moon distance supplied
by the astronomy provider. The Moon has no atmosphere, so there is no
attenuation term and no diffuse component (scattered light from nearby terrain
and earthshine are both neglected - see the accuracy note in the API response).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ..constants import (
    ASTRONOMICAL_UNIT_KM,
    HORIZON_EPSILON,
    SOLAR_CONSTANT_1AU_W_M2,
)
from .astronomy import AstronomyProvider
from .geometry import Vector3, angle_between_deg, local_enu_basis, normalize
from .terrain import TerrainProvider


@dataclass(frozen=True)
class IlluminationResult:
    """Solar geometry and flux at one coordinate and instant.

    Attributes:
        elevation_deg: Sun angle above the local horizontal plane. Negative
            when the Sun is below the geometric horizon.
        azimuth_deg: Sun bearing in [0, 360), clockwise from North.
        incidence_angle_deg: Angle between the surface normal and the Sun.
        cos_incidence: ``max(0, N . S)`` - the geometric flux factor.
        illuminated: ``True`` when the Sun is above the horizon *and* not
            blocked by terrain.
        above_horizon: ``True`` when the Sun is geometrically above the local
            horizontal plane, ignoring terrain.
        terrain_shadowed: ``True``/``False`` from the DEM horizon trace, or
            ``None`` when no DEM is loaded (unknown, not "no").
        solar_irradiance_w_m2: Direct-beam irradiance on the surface.
        solar_constant_at_moon_w_m2: Irradiance on a surface facing the Sun
            squarely, at the current Sun-Moon distance.
        sun_distance_km: Sun-Moon centre distance used for the inverse square.
        sun_vector_body: Unit vector to the Sun in the lunar body-fixed frame.
        surface_normal_body: Unit surface normal used for the incidence angle.
    """

    elevation_deg: float
    azimuth_deg: float
    incidence_angle_deg: float
    cos_incidence: float
    illuminated: bool
    above_horizon: bool
    terrain_shadowed: bool | None
    solar_irradiance_w_m2: float
    horizontal_irradiance_w_m2: float
    solar_constant_at_moon_w_m2: float
    sun_distance_km: float
    sun_vector_body: Vector3
    surface_normal_body: Vector3


def solar_constant_at_distance(distance_km: float) -> float:
    """Total solar irradiance at ``distance_km`` from the Sun.

    .. math:: S = S_0 \\left(\\frac{1\\,\\text{AU}}{d}\\right)^2

    Args:
        distance_km: Distance from the Sun's centre in kilometres.

    Returns:
        Irradiance on a surface normal to the beam [W/m^2].
    """
    if distance_km <= 0.0:
        raise ValueError("Sun distance must be positive")
    return SOLAR_CONSTANT_1AU_W_M2 * (ASTRONOMICAL_UNIT_KM / distance_km) ** 2


def solar_incidence_angle_deg(surface_normal: Vector3, sun_direction: Vector3) -> float:
    """Angle between the surface normal and the Sun direction [deg].

    Both arguments are normalised defensively so callers may pass unnormalised
    vectors. ``0 deg`` means the Sun is directly overhead relative to the
    surface; ``>= 90 deg`` means it is below the surface's own horizon.

    Computed via :func:`~app.services.geometry.angle_between_deg`, which is
    numerically stable at the sub-solar point where ``acos`` would not be.
    """
    return angle_between_deg(surface_normal, sun_direction)


def solar_azimuth_deg(
    latitude_deg: float, longitude_deg: float, sun_direction: Vector3
) -> float:
    """Compass bearing of the Sun in [0, 360), clockwise from North.

    Degenerate at the exact poles, where "North" has no unique meaning; the ENU
    basis there is aligned with the prime meridian, so the returned value is
    the bearing relative to that reference direction.
    """
    east, north, _up = local_enu_basis(latitude_deg, longitude_deg)
    sun = normalize(sun_direction)
    azimuth = math.degrees(math.atan2(float(np.dot(sun, east)), float(np.dot(sun, north))))
    return azimuth % 360.0


def solar_elevation_deg(
    latitude_deg: float, longitude_deg: float, sun_direction: Vector3
) -> float:
    """Sun elevation above the local horizontal plane [deg], sign-preserving."""
    _east, _north, up = local_enu_basis(latitude_deg, longitude_deg)
    sun = normalize(sun_direction)
    sine = float(np.clip(np.dot(sun, up), -1.0, 1.0))
    return math.degrees(math.asin(sine))


def compute_illumination(
    *,
    latitude_deg: float,
    longitude_deg: float,
    sun_direction_body: Vector3,
    sun_distance_km: float,
    surface_normal_body: Vector3,
    terrain_shadowed: bool | None = None,
) -> IlluminationResult:
    """Assemble the full solar geometry for one coordinate.

    Args:
        latitude_deg: Latitude of the point, north positive.
        longitude_deg: Longitude of the point, east positive.
        sun_direction_body: Unit vector to the Sun, body-fixed frame.
        sun_distance_km: Sun-Moon centre distance, for the inverse-square law.
        surface_normal_body: Unit surface normal (spherical or terrain-tilted).
        terrain_shadowed: Result of a DEM horizon trace, or ``None`` if unknown.

    Returns:
        A populated :class:`IlluminationResult`.
    """
    sun = normalize(sun_direction_body)
    normal = normalize(surface_normal_body)

    incidence_deg = solar_incidence_angle_deg(normal, sun)
    elevation = solar_elevation_deg(latitude_deg, longitude_deg, sun)
    azimuth = solar_azimuth_deg(latitude_deg, longitude_deg, sun)

    # cos(theta) = N . S. Values at or below zero mean the Sun has set with
    # respect to this surface element, so no direct beam reaches it.
    cos_incidence = float(np.dot(normal, sun))
    if cos_incidence <= HORIZON_EPSILON:
        cos_incidence = 0.0

    horizontal_cosine = max(0.0, math.sin(math.radians(elevation)))
    above_horizon = elevation > 0.0
    illuminated = above_horizon and cos_incidence > 0.0 and terrain_shadowed is not True

    solar_constant = solar_constant_at_distance(sun_distance_km)
    irradiance = solar_constant * cos_incidence if illuminated else 0.0
    horizontal_irradiance = solar_constant * horizontal_cosine if above_horizon and terrain_shadowed is not True else 0.0

    return IlluminationResult(
        elevation_deg=elevation,
        azimuth_deg=azimuth,
        incidence_angle_deg=incidence_deg,
        cos_incidence=cos_incidence,
        illuminated=illuminated,
        above_horizon=above_horizon,
        terrain_shadowed=terrain_shadowed,
        solar_irradiance_w_m2=irradiance,
        horizontal_irradiance_w_m2=horizontal_irradiance,
        solar_constant_at_moon_w_m2=solar_constant,
        sun_distance_km=sun_distance_km,
        sun_vector_body=sun,
        surface_normal_body=normal,
    )


def evaluate_illumination(
    *,
    latitude_deg: float,
    longitude_deg: float,
    moment: datetime,
    astronomy: AstronomyProvider,
    terrain: TerrainProvider,
) -> IlluminationResult:
    """Convenience wrapper: query both providers, then compute illumination."""
    sun_direction, sun_distance = astronomy.get_sun_vector_from_moon(moment)
    surface_normal = terrain.get_surface_normal(latitude_deg, longitude_deg)

    # Terrain shadowing needs the azimuth/elevation first, so do a cheap
    # geometric pass before the (potentially expensive) DEM ray-march.
    provisional_elevation = solar_elevation_deg(latitude_deg, longitude_deg, sun_direction)
    provisional_azimuth = solar_azimuth_deg(latitude_deg, longitude_deg, sun_direction)
    shadowed = terrain.is_terrain_shadowed(
        latitude_deg, longitude_deg, provisional_azimuth, provisional_elevation
    )

    return compute_illumination(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        sun_direction_body=sun_direction,
        sun_distance_km=sun_distance,
        surface_normal_body=surface_normal,
        terrain_shadowed=shadowed,
    )


# ---------------------------------------------------------------------------
# Sunrise / sunset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SunEvents:
    """Next sunrise and sunset for a surface point.

    Attributes:
        next_sunrise_utc: Next time the Sun crosses above the horizon, or
            ``None`` if no crossing occurs inside the search window.
        next_sunset_utc: Next time the Sun crosses below the horizon, or
            ``None``.
        search_window_hours: How far ahead the search ran.
        note: Explanation when either value is ``None`` (e.g. polar night).
    """

    next_sunrise_utc: datetime | None
    next_sunset_utc: datetime | None
    search_window_hours: float
    note: str | None


def find_sun_events(
    *,
    latitude_deg: float,
    longitude_deg: float,
    moment: datetime,
    astronomy: AstronomyProvider,
    search_window_hours: float = 30.0 * 24.0,
    coarse_step_hours: float = 2.0,
    refine_iterations: int = 40,
) -> SunEvents:
    """Locate the next horizon crossings by scan-then-bisect.

    A lunar day lasts ~29.53 Earth days, so the default window is one full
    synodic month plus margin - enough to find a crossing anywhere except
    permanently shadowed or permanently lit polar terrain.

    The geometric (spherical) horizon is used: ``U . S = 0``. Terrain horizons
    are deliberately excluded here because a DEM ray-march at every bisection
    step would be prohibitively slow; when terrain shadowing is active the true
    local sunrise is *later* than the value reported here.

    Args:
        search_window_hours: How far into the future to look.
        coarse_step_hours: Scan interval used to bracket a sign change.
        refine_iterations: Bisection steps; 40 gives sub-second precision over
            a 2-hour bracket.

    Returns:
        A :class:`SunEvents` with ``None`` for any crossing not found.
    """

    def sun_altitude(at_time: datetime) -> float:
        sun_direction, _distance = astronomy.get_sun_vector_from_moon(at_time)
        return solar_elevation_deg(latitude_deg, longitude_deg, sun_direction)

    def bisect(low_time: datetime, high_time: datetime) -> datetime:
        """Refine a bracketed horizon crossing to high precision."""
        low, high = low_time, high_time
        low_value = sun_altitude(low)
        for _ in range(refine_iterations):
            middle = low + (high - low) / 2
            middle_value = sun_altitude(middle)
            if (low_value < 0.0) == (middle_value < 0.0):
                low, low_value = middle, middle_value
            else:
                high = middle
        return low + (high - low) / 2

    sunrise: datetime | None = None
    sunset: datetime | None = None

    step = timedelta(hours=coarse_step_hours)
    steps = int(search_window_hours / coarse_step_hours)

    previous_time = moment
    previous_altitude = sun_altitude(previous_time)

    for index in range(1, steps + 1):
        current_time = moment + step * index
        current_altitude = sun_altitude(current_time)

        rising = previous_altitude < 0.0 <= current_altitude
        setting = previous_altitude >= 0.0 > current_altitude

        if rising and sunrise is None:
            sunrise = bisect(previous_time, current_time)
        elif setting and sunset is None:
            sunset = bisect(previous_time, current_time)

        if sunrise is not None and sunset is not None:
            break

        previous_time, previous_altitude = current_time, current_altitude

    note: str | None = None
    if sunrise is None and sunset is None:
        state = "continuously sunlit" if previous_altitude >= 0.0 else "in continuous darkness"
        note = (
            f"No horizon crossing within {search_window_hours:.0f} h - this point is "
            f"{state} over the search window (typical of high-latitude terrain)."
        )
    elif sunrise is None:
        note = f"No sunrise within {search_window_hours:.0f} h of the selected time."
    elif sunset is None:
        note = f"No sunset within {search_window_hours:.0f} h of the selected time."

    return SunEvents(
        next_sunrise_utc=sunrise,
        next_sunset_utc=sunset,
        search_window_hours=search_window_hours,
        note=note,
    )


__all__ = [
    "IlluminationResult",
    "SunEvents",
    "compute_illumination",
    "evaluate_illumination",
    "find_sun_events",
    "solar_azimuth_deg",
    "solar_constant_at_distance",
    "solar_elevation_deg",
    "solar_incidence_angle_deg",
]
