"""Topocentric body-centre positions over a spherical local horizon."""
from __future__ import annotations

import numpy as np
from math import asin, degrees

from ..constants import MOON_RADIUS_KM
from ..models.schemas import BodyPosition, HorizonPositions
from .geometry import angle_between_deg, lat_lon_to_cartesian, normalize
from .illumination import solar_azimuth_deg, solar_elevation_deg


def topocentric_vector(latitude: float, longitude: float, direction, distance_km: float, elevation_m: float = 0):
    # Earth parallax matters near the poles: observer is on the surface, not
    # at the Moon's centre. Provider directions and this offset are body-fixed.
    observer = lat_lon_to_cartesian(latitude, longitude, MOON_RADIUS_KM + elevation_m / 1000)
    return normalize(direction) * distance_km - observer


def body_position(latitude: float, longitude: float, direction, distance_km: float, elevation_m: float = 0) -> BodyPosition:
    vector = topocentric_vector(latitude, longitude, direction, distance_km, elevation_m)
    elevation = solar_elevation_deg(latitude, longitude, vector)
    return BodyPosition(elevation_deg=elevation,
                        azimuth_deg=solar_azimuth_deg(latitude, longitude, vector),
                        distance_km=float(np.linalg.norm(vector)),
                        above_horizon=elevation > 0)


def earth_overlaps_solar_disk(latitude: float, longitude: float,
                              sun_direction, sun_distance_km: float,
                              earth_direction, earth_distance_km: float) -> bool:
    """Whether any of Earth's apparent disk covers the Sun for a surface observer."""
    sun_vector = topocentric_vector(latitude, longitude, sun_direction, sun_distance_km)
    earth_vector = topocentric_vector(latitude, longitude, earth_direction, earth_distance_km)
    separation = angle_between_deg(sun_vector, earth_vector)
    earth_radius = degrees(asin(min(1.0, 6378.137 / float(np.linalg.norm(earth_vector)))))
    sun_radius = degrees(asin(min(1.0, 695700.0 / float(np.linalg.norm(sun_vector)))))
    return separation <= earth_radius + sun_radius


def horizon_positions(latitude: float, longitude: float, geometry, elevation_m: float = 0) -> HorizonPositions:
    return HorizonPositions(
        sun=body_position(latitude, longitude, geometry.sun_direction_body, geometry.sun_distance_km, elevation_m),
        earth=body_position(latitude, longitude, geometry.earth_direction_body, geometry.earth_distance_km, elevation_m),
    )
