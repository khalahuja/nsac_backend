"""Computed facts about the selected site, used to ground the assistant's answers.

The assistant may only quote numbers that appear here, in the user's messages,
or in a calculation the website runs. Everything below comes from the
installed providers (JPL SPICE ephemeris, NASA LOLA terrain when it covers the
site); nothing is estimated by the language model.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from .positions import body_position

_STEP_HOURS = 6
_DAYS = 30


def _classify(year_min: float, year_max: float, month_max: float) -> str:
    """Classified over a full year: libration moves Earth by up to ~8° from month to month."""
    if year_max <= 0:
        return "far side: Earth stays below the horizon all year, so direct-to-Earth contact is impossible on any date"
    if year_min > 0:
        return "near side: Earth is always above the horizon here"
    if month_max <= 0:
        return ("limb region: Earth does not rise in the 30-day period shown but does rise in other months "
                f"(up to {year_max:.1f}° over the year); run a search for the requested dates before ruling it out")
    return "limb region: Earth rises and sets as the Moon librates"


@lru_cache(maxsize=256)
def _facts_cached(lat: float, lon: float, day: str, providers_id: int, providers: Any) -> dict:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    from .surface_screen import horizon_or_none  # noqa: PLC0415 - avoid import cycle

    terrain = providers.terrain
    horizon = horizon_or_none(terrain, lat, lon, list(range(360)))
    skyline = horizon is not None and all(h is not None for h in horizon)
    sky = (lambda az: horizon[int(round(az)) % 360]) if skyline else (lambda az: 0.0)
    earth_el, sun_up, earth_up, n = [], 0, 0, 0
    sun_el = []
    for i in range(_DAYS * 24 // _STEP_HOURS + 1):
        g = providers.astronomy.get_geometry(start + timedelta(hours=i * _STEP_HOURS))
        s = body_position(lat, lon, g.sun_direction_body, g.sun_distance_km)
        e = body_position(lat, lon, g.earth_direction_body, g.earth_distance_km)
        earth_el.append(e.elevation_deg)
        sun_el.append(s.elevation_deg)
        sun_up += s.elevation_deg > sky(s.azimuth_deg)
        earth_up += e.elevation_deg > sky(e.azimuth_deg)
        n += 1
    # A full year at 12-hourly steps covers the monthly libration cycle.
    year_el = []
    for i in range(365 * 2):
        g = providers.astronomy.get_geometry(start + timedelta(hours=12 * i))
        year_el.append(body_position(lat, lon, g.earth_direction_body, g.earth_distance_km).elevation_deg)
    facts: dict[str, Any] = {
        "latitude_deg": round(lat, 4),
        "longitude_deg": round(lon, 4),
        "period": f"{start:%Y-%m-%d} + {_DAYS} days, {_STEP_HOURS}-hourly",
        "side_of_moon": _classify(min(year_el), max(year_el), max(earth_el)),
        "earth_elevation_max_over_year_deg": round(max(year_el), 1),
        "earth_elevation_min_over_year_deg": round(min(year_el), 1),
        "earth_elevation_min_deg": round(min(earth_el), 1),
        "earth_elevation_max_deg": round(max(earth_el), 1),
        "sun_elevation_max_deg": round(max(sun_el), 1),
        "sun_up_pct": round(100 * sun_up / n, 1),
        "earth_up_pct": round(100 * earth_up / n, 1),
        "horizon": "measured terrain skyline" if skyline else "smooth horizon (no terrain data for this site)",
        "astronomy_source": getattr(providers.astronomy, "source", "unknown"),
    }
    if terrain.is_measured:
        elevation = terrain.get_elevation(lat, lon)
        slope = terrain.get_slope(lat, lon)
        if elevation is not None:
            facts["terrain_elevation_m"] = round(elevation)
            facts["terrain_slope_deg"] = round(slope, 1) if slope is not None else None
            facts["terrain_source"] = terrain.source
        else:
            facts["terrain"] = f"not covered by the installed DEM ({terrain.source}, 80°S to the pole)"
    else:
        facts["terrain"] = "no DEM installed"
    if skyline:
        facts["highest_ridge_deg"] = round(max(horizon), 1)
    return facts


def site_facts(lat: float, lon: float, start: datetime, providers: Any) -> dict | None:
    """Facts for a site over the 30 days from ``start``; ``None`` if providers are unavailable."""
    if providers is None:
        return None
    try:
        day = start.astimezone(timezone.utc).strftime("%Y-%m-%d")
        return _facts_cached(round(lat, 3), round(lon, 3), day, id(providers), providers)
    except Exception:  # noqa: BLE001 - grounding is best-effort; the chat still works
        return None
