"""Bounded hourly surface-operations screening, with explicit model provenance."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from math import cos, sin, isfinite, atan2, hypot, radians, degrees
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .positions import body_position, earth_overlaps_solar_disk
from .surface_screen import horizon_or_none

class MissionRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mission_hours: int = Field(default=168, ge=1, le=720)
    min_sunlight_pct: float = Field(default=70, ge=0, le=100)
    max_dark_hours: float = Field(default=48, ge=0, le=720)
    min_earth_pct: float = Field(default=30, ge=0, le=100)
    max_outage_hours: float = Field(default=72, ge=0, le=720)
    min_both_hours: float = Field(default=24, ge=0, le=720)
    min_earth_elevation_deg: float = Field(default=0, ge=0, le=30)
    @model_validator(mode="after")
    def possible_overlap(self):
        if self.min_both_hours > self.mission_hours:
            raise ValueError("Required overlap cannot exceed the mission duration")
        return self

class OperationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    landing_start_utc: datetime
    landing_end_utc: datetime
    nearby_radius_km: float = Field(default=10, ge=0.1, le=50)
    requirements: MissionRequirements = Field(default_factory=MissionRequirements)
    @model_validator(mode="after")
    def bounded(self):
        a, b = self.landing_start_utc, self.landing_end_utc
        if a.utcoffset() is None or b.utcoffset() is None:
            raise ValueError("Landing times need explicit UTC offsets")
        if not timedelta(0) < b - a <= timedelta(days=90):
            raise ValueError("Use an ordered landing search of at most 90 days")
        if a.year < 1900 or b.year > 2100:
            raise ValueError("Use dates from 1900 through 2100")
        return self

def summarize_intervals(times, sunlight, contact, requirements):
    """Positive endpoints are required; sub-hour interruptions remain unresolved."""
    sun = [a and b for a, b in zip(sunlight, sunlight[1:])]
    earth = [a and b for a, b in zip(contact, contact[1:])]
    both = [a and b for a, b in zip(sun, earth)]
    def longest_off(values):
        longest = run = 0
        for available in values:
            run = 0 if available else run + 1
            longest = max(longest, run)
        return longest
    n = len(sun)
    metrics = dict(sunlight_pct=100 * sum(sun) / n, earth_pct=100 * sum(earth) / n,
                   sunlight_hours=sum(sun), earth_hours=sum(earth), both_hours=sum(both),
                   longest_dark_hours=longest_off(sun), longest_outage_hours=longest_off(earth))
    checks = [
        dict(label="Sunlight coverage", actual=metrics['sunlight_pct'], limit=requirements.min_sunlight_pct, unit="%", operator=">="),
        dict(label="Longest darkness", actual=metrics['longest_dark_hours'], limit=requirements.max_dark_hours, unit="h", operator="<="),
        dict(label="Earth visibility", actual=metrics['earth_pct'], limit=requirements.min_earth_pct, unit="%", operator=">="),
        dict(label="Longest contact outage", actual=metrics['longest_outage_hours'], limit=requirements.max_outage_hours, unit="h", operator="<="),
        dict(label="Sunlight and contact together", actual=metrics['both_hours'], limit=requirements.min_both_hours, unit="h", operator=">="),
    ]
    for check in checks:
        check['passed'] = check['actual'] + 1e-9 >= check['limit'] if check['operator'] == '>=' else check['actual'] <= check['limit'] + 1e-9
    tracks = []
    for key, values in [('sunlight', sun), ('contact', earth), ('both', both)]:
        segments = []
        for i, available in enumerate(values):
            if segments and segments[-1]['available'] == available:
                segments[-1]['end_utc'] = times[i + 1].isoformat()
                segments[-1]['hours'] += 1
            else:
                segments.append(dict(start_utc=times[i].isoformat(), end_utc=times[i + 1].isoformat(), hours=1, available=available))
        tracks.append(dict(key=key, segments=segments))
    score = (metrics['sunlight_pct'] + metrics['earth_pct'] + 100 * sum(both) / n) / 3
    return dict(landing_utc=times[0].isoformat(), end_utc=times[-1].isoformat(), metrics=metrics,
                checks=checks, feasible=all(c['passed'] for c in checks), score=score, tracks=tracks)

def _site_states(latitude, longitude, geometries, providers, min_earth_elevation):
    horizon = horizon_or_none(providers.terrain, latitude, longitude, list(range(0, 360, 5)))
    verified = horizon is not None and len(horizon) == 72 and all(h is not None and isfinite(h) for h in horizon)
    elevation = providers.terrain.get_elevation(latitude, longitude) if verified else 0
    if elevation is None:
        elevation = 0
    def skyline(azimuth):
        return max(0, horizon[int((azimuth + 2.5) % 360 // 5)]) if verified else 0
    sun, earth = [], []
    eclipse_screened = providers.astronomy.source == 'jpl-spice'
    for geometry in geometries:
        s = body_position(latitude, longitude, geometry.sun_direction_body, geometry.sun_distance_km, elevation)
        e = body_position(latitude, longitude, geometry.earth_direction_body, geometry.earth_distance_km, elevation)
        overlap = eclipse_screened and earth_overlaps_solar_disk(latitude, longitude, geometry.sun_direction_body,
                   geometry.sun_distance_km, geometry.earth_direction_body, geometry.earth_distance_km)
        sun.append(s.elevation_deg > skyline(s.azimuth_deg) and not overlap)
        earth.append(e.elevation_deg > max(skyline(e.azimuth_deg), min_earth_elevation))
    return sun, earth, verified

def nearby_point(latitude, longitude, distance_km, bearing_deg):
    """Great-circle offset in a local vector frame, including at either pole."""
    lat, lon, bearing = radians(latitude), radians(longitude), radians(bearing_deg)
    distance = distance_km / 1737.4
    up = (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
    north = (-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))
    east = (-sin(lon), cos(lon), 0)
    x, y, z = [up[i] * cos(distance) + (north[i] * cos(bearing) + east[i] * sin(bearing)) * sin(distance) for i in range(3)]
    return degrees(atan2(z, hypot(x, y))), ((degrees(atan2(y, x)) + 180) % 360) - 180


def plan_operations(request: OperationsRequest, providers):
    start = request.landing_start_utc.astimezone(timezone.utc)
    end = request.landing_end_utc.astimezone(timezone.utc)
    hours = request.requirements.mission_hours
    landing_count = int((end - start).total_seconds() // 3600) + 1
    times = [start + timedelta(hours=i) for i in range(landing_count + hours)]
    geometries = [providers.astronomy.get_geometry(t) for t in times]
    sunlight, contact, terrain_verified = _site_states(request.latitude, request.longitude, geometries,
                                         providers, request.requirements.min_earth_elevation_deg)
    candidates = []
    for i in range(landing_count):
        candidate = summarize_intervals(times[i:i + hours + 1], sunlight[i:i + hours + 1], contact[i:i + hours + 1], request.requirements)
        candidate['_index'] = i
        candidates.append(candidate)
    feasible_count = sum(c['feasible'] for c in candidates)
    candidates.sort(key=lambda c: (not c['feasible'], -sum(x['passed'] for x in c['checks']), -c['score'], c['landing_utc']))
    best = candidates[0]
    index = best['_index']
    stay_times = times[index:index + hours + 1]
    stay_geometry = geometries[index:index + hours + 1]
    nearby = []
    for bearing, name in zip(range(0, 360, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']):
        lat, lon = nearby_point(request.latitude, request.longitude, request.nearby_radius_km, bearing)
        s, e, verified = _site_states(lat, lon, stay_geometry, providers, request.requirements.min_earth_elevation_deg)
        alternative = summarize_intervals(stay_times, s, e, request.requirements)
        nearby.append(dict(name=f'{name} alternative', bearing_deg=bearing, latitude=lat, longitude=lon,
                           distance_km=request.nearby_radius_km, terrain_verified=verified,
                           score_delta=alternative['score'] - best['score'], **alternative))
    for candidate in candidates:
        candidate.pop('_index', None)
    return dict(request=request.model_dump(mode='json'), candidates=candidates[:12], best=best,
                candidate_count=landing_count, feasible_count=feasible_count, nearby=nearby,
                astronomy_source=providers.astronomy.source, terrain_source=providers.terrain.source,
                terrain_verified=terrain_verified, sampling_hours=1,
                limitations=[
                    'Hourly landing grid and endpoint screening; events and interruptions within an hour may be missed.',
                    'Visibility is a body-centre geometric screen, not a power budget, radio link budget or ground-station schedule.',
                    'Coverage score is the equal-weight mean of sunlight, Earth visibility and simultaneous coverage; it is not a probability.',
                    'Nearby points use the recommended date and the same requirements. Landing safety is not evaluated.',
                    'Terrain skyline is sampled every 5 degrees when a complete DEM horizon is available; otherwise a spherical horizon is used.',
                    'Terrain visibility uses the existing DEM skyline observer convention; equipment height is not configurable here.',
                    'Conservative eclipse-overlap rejection is enabled only with JPL SPICE; approximate-mode eclipses are not screened.',
                ])
