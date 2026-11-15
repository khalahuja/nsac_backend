"""Rank south-pole candidate sites for landing safety, water access, sunlight and Earth contact.

Every number comes from installed data or published catalogues:
- slope safety from the NASA LOLA polar DEM (surface screen of the landing footprint),
- sunlight and Earth visibility against the measured skyline with JPL ephemerides,
- water evidence from the craters where Moon Mineralogy Mapper detected exposed
  surface ice (Li et al. 2018, PNAS 115:8907), using USGS/IAU centres and diameters.

The scores are transparent screening ranks, not probabilities of success or of ice.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .mission_swarm import CANDIDATE_SITES, MAX_SITE_SLOPE_DEG, _site_metrics
from .mission_windows import WindowSearchInputs, search_mission_windows
from .mission_swarm import MissionInputs
from .science_windows import Criteria, matching_intervals
from .forecast import compute_forecast
from .surface_screen import horizon_or_none
from .providers import ScienceProviders
from .surface_screen import compute_surface_screen

MOON_RADIUS_KM = 1737.4

#: South-polar craters with M3 surface-ice detections (Li et al. 2018), USGS/IAU centre and diameter.
ICE_CRATERS: list[tuple[str, float, float, float]] = [
    ("Shackleton", -89.67, 129.78, 20.92),
    ("Haworth", -87.45, -5.17, 51.42),
    ("Shoemaker", -88.14, 45.91, 51.82),
    ("Faustini", -87.18, 84.31, 42.48),
    ("de Gerlache", -88.48, -88.34, 32.71),
    ("Sverdrup", -88.32, -153.39, 32.83),
]


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return MOON_RADIUS_KM * math.acos(max(-1.0, min(1.0, c)))


def nearest_ice_crater(lat: float, lon: float) -> tuple[str, float]:
    """Nearest ice-detection crater and the distance (km) to its rim (0 when inside)."""
    best = min(ICE_CRATERS, key=lambda c: _distance_km(lat, lon, c[1], c[2]) - c[3] / 2)
    return best[0], max(0.0, _distance_km(lat, lon, best[1], best[2]) - best[3] / 2)


class SiteRankingRequest(BaseModel):
    start_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_utc: datetime | None = None
    min_sunlight_days: float = Field(default=0, ge=0, le=30)
    priority: Literal["overall", "water"] = "overall"
    criteria: Criteria | None = None
    max_slope_deg: float | None = Field(default=None, ge=0, le=90)
    min_safe_fraction: float | None = Field(default=None, ge=0, le=1)
    max_ice_distance_km: float | None = Field(default=None, ge=0)
    landing_radius_m: float = Field(default=500, ge=50, le=5000)
    #: Extra user points to rank alongside the catalogue.
    extra_sites: list[tuple[str, float, float]] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def window(self):
        if self.start_utc.tzinfo is None:
            self.start_utc = self.start_utc.replace(tzinfo=timezone.utc)
        if self.end_utc is None:
            self.end_utc = self.start_utc + timedelta(days=60)
        if self.end_utc.tzinfo is None:
            self.end_utc = self.end_utc.replace(tzinfo=timezone.utc)
        if not timedelta(days=1) <= self.end_utc - self.start_utc <= timedelta(days=400):
            raise ValueError("Use a ranking period between 1 and 400 days")
        if self.min_sunlight_days and (self.min_sunlight_days < 1 or self.end_utc - self.start_utc > timedelta(days=366)):
            raise ValueError("Continuous sunlight requires 1 to 30 days and a search period of at most 366 days")
        return self


class RankedSite(BaseModel):
    name: str
    latitude: float
    longitude: float
    slope_deg: float | None
    safe_footprint_fraction: float | None
    sunlit_pct: float
    earth_visible_pct: float
    nearest_ice_crater: str
    ice_crater_distance_km: float
    landing_score: float
    water_score: float
    overall_score: float
    continuous_sunlight_days: float | None = None
    sunlight_start_utc: datetime | None = None
    sunlight_end_utc: datetime | None = None


class SiteRankingResult(BaseModel):
    sites: list[RankedSite]
    best_landing: str | None
    best_water: str | None
    best_overall: str | None
    terrain_source: str
    astronomy_source: str
    method: list[str]


def rank_sites(request: SiteRankingRequest, providers: ScienceProviders) -> SiteRankingResult:
    ranked: list[RankedSite] = []
    sunlight_notes: list[str] = []
    for name, lat, lon in CANDIDATE_SITES + [tuple(s) for s in request.extra_sites]:
        sunlight = None
        if request.min_sunlight_days > 0 and request.criteria is None:
            search = search_mission_windows(WindowSearchInputs(
                inputs=MissionInputs(departure_utc=request.start_utc, target_latitude_deg=lat, target_longitude_deg=lon),
                landing_start_utc=request.start_utc, landing_end_utc=request.end_utc,
                min_continuous_sunlight_days=request.min_sunlight_days,
            ), providers, run_best_plan=False)
            sunlight_notes.extend(search.limitations[:3])
            if not search.candidates:
                continue
            sunlight = max(search.candidates, key=lambda c: c.sunlight_interval_end_utc - c.sunlight_interval_start_utc)
        metrics = _site_metrics(lat, lon, request.start_utc, request.end_utc, providers)
        screen = compute_surface_screen(lat, lon, request.start_utc, request.landing_radius_m, providers)
        safe = screen.get("safe_fraction")
        slope = metrics["slope_deg"]
        ice_name, ice_km = nearest_ice_crater(lat, lon)
        if request.max_slope_deg is not None and (slope is None or slope > request.max_slope_deg):
            continue
        if request.min_safe_fraction is not None and (safe is None or safe < request.min_safe_fraction):
            continue
        if request.max_ice_distance_km is not None and ice_km > request.max_ice_distance_km:
            continue
        if request.criteria is not None:
            criteria = request.criteria.model_copy(update={
                "requireSun": request.criteria.requireSun or request.min_sunlight_days > 0,
                "minHours": max(request.criteria.minHours, request.min_sunlight_days * 24),
            })
            forecast = compute_forecast(latitude_deg=lat, longitude_deg=lon, start=request.start_utc,
                end=request.end_utc, step_hours=1, max_samples=10000, providers=providers)
            skyline = horizon_or_none(providers.terrain, lat, lon, list(range(360)))
            adjusted = []
            for sample in forecast.samples:
                def clear(azimuth, elevation):
                    if skyline is None:
                        return elevation > 0
                    horizon = skyline[int(round(azimuth)) % 360]
                    return horizon is not None and elevation > horizon
                adjusted.append(sample.model_copy(update={
                    "illuminated": sample.illuminated and clear(sample.solar_azimuth_deg, sample.solar_elevation_deg),
                    "earth_visible": sample.earth_visible and clear(sample.earth_azimuth_deg, sample.earth_elevation_deg),
                }))
            intervals = matching_intervals(adjusted, criteria)
            if not intervals:
                continue
            from types import SimpleNamespace
            first, last = max(intervals, key=lambda interval: interval[1] - interval[0])
            sunlight = SimpleNamespace(sunlight_interval_start_utc=first, sunlight_interval_end_utc=last)
            sunlight_notes.extend([
                "All environmental filters apply within the displayed interval at 1-hour sampling; Earth duration is continuous within that interval.",
                "Temperature and irradiance are reduced-order model estimates; local terrain is applied to Sun/Earth visibility, not a terrain-resolved thermal model.",
                "Short shadows or eclipses between hourly samples may be missed.",
            ])
        # Landing: share of the footprint within the slope limit (0 if the centre is too steep).
        slope_ok = slope is None or slope <= MAX_SITE_SLOPE_DEG
        landing = (safe if safe is not None else (1.0 if slope_ok else 0.0)) * (1.0 if slope_ok else 0.0)
        # Water: 1 at the rim of an ice-detection crater, falling to 0.5 at 10 km and 0 by ~40 km.
        water = math.exp(-ice_km / 14.4)
        overall = 0.4 * landing + 0.3 * water + 0.2 * metrics["sunlit_pct"] / 100 + 0.1 * metrics["earth_visible_pct"] / 100
        ranked.append(RankedSite(
            name=name, latitude=lat, longitude=lon, slope_deg=slope, safe_footprint_fraction=safe,
            sunlit_pct=metrics["sunlit_pct"], earth_visible_pct=metrics["earth_visible_pct"],
            nearest_ice_crater=ice_name, ice_crater_distance_km=ice_km,
            landing_score=landing, water_score=water, overall_score=overall,
            continuous_sunlight_days=((sunlight.sunlight_interval_end_utc - sunlight.sunlight_interval_start_utc).total_seconds() / 86400 if sunlight else None),
            sunlight_start_utc=sunlight.sunlight_interval_start_utc if sunlight else None,
            sunlight_end_utc=sunlight.sunlight_interval_end_utc if sunlight else None,
        ))
    ranked.sort(key=lambda s: (s.water_score, s.overall_score) if request.priority == "water" else (s.overall_score,), reverse=True)
    return SiteRankingResult(
        sites=ranked,
        best_landing=max(ranked, key=lambda s: (s.landing_score, -(s.slope_deg or 0))).name if ranked else None,
        best_water=max(ranked, key=lambda s: s.water_score).name if ranked else None,
        best_overall=max(ranked, key=lambda s: s.overall_score).name if ranked else None,
        terrain_source=providers.terrain.source,
        astronomy_source=providers.astronomy.source,
        method=[
            "Environmental filters: " + request.criteria.model_dump_json() if request.criteria else "No additional environmental filters.",
            f"Site limits: maximum slope {request.max_slope_deg}, minimum safe footprint fraction {request.min_safe_fraction}, maximum ice distance {request.max_ice_distance_km} km. Missing measurements cannot pass a requested limit.",
            f"Search period: {request.start_utc.isoformat()} to {request.end_utc.isoformat()}.",
            f"Minimum continuous sunlight: {request.min_sunlight_days:g} days. Only qualifying sites are listed; interval boundaries are sampled hourly." if request.min_sunlight_days else "No minimum continuous-sunlight filter requested.",
            "Sites ordered by ice-access score, then overall score." if request.priority == "water" else "Sites ordered by overall score.",
            f"Landing score = share of a {request.landing_radius_m:.0f} m-radius footprint with DEM slope ≤ 12° (0 if the centre exceeds {MAX_SITE_SLOPE_DEG:.0f}°). Rocks smaller than the 80 m DEM pixel are not seen.",
            "Water score = exp(−d/14.4 km), d = distance to the rim of the nearest crater with M3 surface-ice detections (Li et al. 2018). It ranks access to known ice, not the chance that ice is present.",
            "Sunlight and Earth visibility: 6-hourly over the period, against the measured skyline (2 m height).",
            "Overall = 0.4·landing + 0.3·water + 0.2·sunlight + 0.1·Earth. Weights are a transparent choice; none of these scores is a probability.",
        ] + list(dict.fromkeys(sunlight_notes)),
    )
