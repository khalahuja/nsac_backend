"""Local DEM screens for skyline, shadow maps and landing footprint.

These products only report terrain-derived values when a measured raster is
loaded. An absent DEM is represented by nulls rather than a flat safe site.
"""

from __future__ import annotations

import math
from datetime import datetime

from ..services.providers import ScienceProviders
from .positions import body_position

MOON_RADIUS_M = 1_737_400.0


def _offset(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    distance = math.hypot(east_m, north_m)
    if distance == 0:
        return lat, lon
    latitude = math.radians(lat)
    longitude = math.radians(lon)
    bearing = math.atan2(east_m, north_m)
    central_angle = distance / MOON_RADIUS_M
    destination_latitude = math.asin(
        math.sin(latitude) * math.cos(central_angle)
        + math.cos(latitude) * math.sin(central_angle) * math.cos(bearing)
    )
    destination_longitude = longitude + math.atan2(
        math.sin(bearing) * math.sin(central_angle) * math.cos(latitude),
        math.cos(central_angle) - math.sin(latitude) * math.sin(destination_latitude),
    )
    return math.degrees(destination_latitude), ((math.degrees(destination_longitude) + 180) % 360) - 180


def horizon_or_none(terrain, lat: float, lon: float, azimuths: list[float]) -> list[float | None] | None:
    """Long-range (150 km) horizon from a polar DEM, when the provider supports it."""
    profile = getattr(terrain, "horizon_profile", None)
    if profile is None or not terrain.is_measured:
        return None
    return profile(lat, lon, azimuths)


def _blocked(skyline: list[dict], azimuth: float, elevation: float) -> bool | None:
    """True when the body is above the geometric horizon but below the terrain skyline."""
    known = [p for p in skyline if p["elevation_deg"] is not None]
    if not known:
        return None
    nearest = min(known, key=lambda p: abs(((p["azimuth_deg"] - azimuth + 180) % 360) - 180))
    return elevation <= nearest["elevation_deg"]


def compute_surface_screen(lat: float, lon: float, moment: datetime, radius_m: float, providers: ScienceProviders) -> dict:
    terrain = providers.terrain
    geometry = providers.astronomy.get_geometry(moment)
    sun = body_position(lat, lon, geometry.sun_direction_body, geometry.sun_distance_km)
    earth = body_position(lat, lon, geometry.earth_direction_body, geometry.earth_distance_km)
    if not terrain.is_measured:
        return {
            "available": False, "terrain_source": terrain.source, "astronomy_source": geometry.source,
            "reason": "Install and enable a lunar DEM to calculate terrain skyline, moving shadows and footprint hazards.",
            "sun": {"azimuth_deg": sun.azimuth_deg, "elevation_deg": sun.elevation_deg},
            "earth": {"azimuth_deg": earth.azimuth_deg, "elevation_deg": earth.elevation_deg},
            "skyline": [], "grid": [], "safe_fraction": None, "collision_count": None,
            "sunlit_fraction": None, "coverage_fraction": None,
        }
    base = terrain.get_elevation(lat, lon)
    skyline = []
    profile = horizon_or_none(terrain, lat, lon, list(range(0, 360, 2)))
    if profile is not None:
        skyline = [{"azimuth_deg": az, "elevation_deg": el} for az, el in zip(range(0, 360, 2), profile)]
    for azimuth in ([] if profile is not None else range(0, 360, 10)):
        radians = math.radians(azimuth)
        highest = -90.0
        valid = 0
        for step in range(1, 25):
            distance = 20000 * step / 24
            sample_lat, sample_lon = _offset(lat, lon, distance * math.sin(radians), distance * math.cos(radians))
            sample = terrain.get_elevation(sample_lat, sample_lon)
            if sample is None or base is None:
                continue
            valid += 1
            rise = sample - base - distance * distance / (2 * MOON_RADIUS_M)
            highest = max(highest, math.degrees(math.atan2(rise, distance)))
        skyline.append({"azimuth_deg": azimuth, "elevation_deg": highest if valid else None})
    grid = []
    safe = collision_count = measured = known_light = sunlit_count = footprint_cells = 0
    for y in range(-6, 7):
        for x in range(-6, 7):
            east_m, north_m = x * radius_m / 6, y * radius_m / 6
            sample_lat, sample_lon = _offset(lat, lon, east_m, north_m)
            terrain_sample = terrain.sample(sample_lat, sample_lon)
            slope = terrain_sample.slope_deg
            shadow = terrain.is_terrain_shadowed(sample_lat, sample_lon, sun.azimuth_deg, sun.elevation_deg) if terrain.supports_shadows else None
            sunlit = False if sun.elevation_deg <= 0 else (not shadow if shadow is not None else None)
            inside = math.hypot(east_m, north_m) <= radius_m
            footprint_cells += inside
            hazard = slope is not None and slope > 12
            if inside and slope is not None:
                measured += 1
                safe += not hazard
                collision_count += hazard
            if inside and sunlit is not None:
                known_light += 1
                sunlit_count += sunlit
            grid.append({"x": x, "y": y, "lat": sample_lat, "lon": sample_lon,
                         "elevation_m": terrain_sample.elevation_m, "slope_deg": slope,
                         "shadowed": shadow, "sunlit": sunlit, "hazard": hazard if slope is not None else None,
                         "inside_footprint": inside})
    return {
        "available": True, "terrain_source": terrain.source, "astronomy_source": geometry.source,
        "reason": None, "sun": {"azimuth_deg": sun.azimuth_deg, "elevation_deg": sun.elevation_deg},
        "earth": {"azimuth_deg": earth.azimuth_deg, "elevation_deg": earth.elevation_deg},
        "skyline": skyline, "grid": grid,
        "sun_blocked_by_terrain": _blocked(skyline, sun.azimuth_deg, sun.elevation_deg),
        "earth_blocked_by_terrain": _blocked(skyline, earth.azimuth_deg, earth.elevation_deg),
        "safe_fraction": safe / measured if measured == footprint_cells and measured else None,
        "coverage_fraction": measured / footprint_cells if footprint_cells else None,
        "collision_count": collision_count if measured else None,
        "sunlit_fraction": sunlit_count / known_light if known_light else None,
        "shadow_model_available": terrain.supports_shadows,
        "limitations": "Slope threshold is 12 degrees; safe fraction is withheld when any footprint cell is unmeasured. Unresolved rocks and descent dynamics are not represented. Shadows are evaluated at the requested instant.",
    }
