"""Bounded, per-case comparisons against JPL Horizons observer quantity 4."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .horizons import HorizonsUnavailable, observer_angles
from .positions import body_position


def validate_ephemeris(sites: list, times: list[datetime], providers) -> dict:
    cases = [(site, time) for site in sites for time in times]

    def run(case):
        site, time = case
        moment = time.replace(tzinfo=timezone.utc) if time.tzinfo is None else time.astimezone(timezone.utc)
        geometry = providers.astronomy.get_geometry(moment)
        rows = []
        for body, target, direction, distance in (
            ("sun", "10", geometry.sun_direction_body, geometry.sun_distance_km),
            ("earth", "399", geometry.earth_direction_body, geometry.earth_distance_km),
        ):
            model = body_position(site.latitude, site.longitude, direction, distance)
            try:
                reference = observer_angles(site.latitude, site.longitude, moment, target)
                azimuth_delta = (model.azimuth_deg - reference["azimuth_deg"] + 180) % 360 - 180
                elevation_delta = model.elevation_deg - reference["elevation_deg"]
                rows.append({"body": body, "model": {"azimuth_deg": model.azimuth_deg, "elevation_deg": model.elevation_deg},
                             "reference": reference, "azimuth_delta_deg": azimuth_delta, "elevation_delta_deg": elevation_delta, "error": None})
            except HorizonsUnavailable as error:
                rows.append({"body": body, "model": {"azimuth_deg": model.azimuth_deg, "elevation_deg": model.elevation_deg},
                             "reference": None, "azimuth_delta_deg": None, "elevation_delta_deg": None, "error": str(error)})
        return {"latitude": site.latitude, "longitude": site.longitude, "datetime_utc": moment.isoformat(), "bodies": rows}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, cases))
    valid = [body for case in results for body in case["bodies"] if body["error"] is None]
    return {
        "astronomy_source": providers.astronomy.source, "reference": "JPL Horizons observer quantity 4",
        "case_count": len(cases), "successful_body_checks": len(valid), "results": results,
        "max_abs_elevation_delta_deg": max((abs(row["elevation_delta_deg"]) for row in valid), default=None),
        "notes": [
            "Azimuth differences wrap to [-180, 180); azimuth is ill-conditioned at zenith.",
            "Both calculations use a zero-height lunar observer and body centres. Terrain is excluded from this geometry comparison.",
            "Horizons apparent angles and the local provider may use different aberration, orientation and ephemeris conventions; differences do not alone establish an accuracy bound.",
            "Unavailable reference queries remain missing; no synthetic reference angles are substituted.",
        ],
    }
