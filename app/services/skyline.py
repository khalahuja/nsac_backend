"""Reusable terrain visibility masks, sampled once for a complete forecast."""
from __future__ import annotations
import math
from functools import lru_cache

@lru_cache(maxsize=128)
def terrain_skyline(terrain, latitude: float, longitude: float, height_m: float = 0.0):
    if not terrain.is_measured or not terrain.supports_shadows:
        return None
    profile = getattr(terrain, "horizon_profile", None)
    if profile is None:
        return None
    values = profile(latitude, longitude, list(range(360)), observer_height_m=height_m)
    if values is None or len(values) != 360 or any(v is None or not math.isfinite(v) for v in values):
        return None
    return tuple(float(v) for v in values)

def horizon_at(skyline, azimuth: float) -> float:
    if skyline is None:
        return 0.0
    az = azimuth % 360
    index = int(az)
    fraction = az - index
    return skyline[index] * (1 - fraction) + skyline[(index + 1) % 360] * fraction
