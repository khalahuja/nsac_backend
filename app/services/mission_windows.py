"""Geometric sunlight screening for a launch-to-landing mission graph.

These dates are *not* launch opportunities. The transfer graph assumes a fixed
Moon distance and does not target the Moon's position at arrival.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import ceil, pi, sqrt

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .mission_swarm import MU_E, MU_M, MOON_DISTANCE, R_E, R_M, MissionInputs, MissionPlan, plan_mission, flight_duration_days
from .positions import body_position, earth_overlaps_solar_disk
from .providers import ScienceProviders
from .surface_screen import horizon_or_none


class WindowSearchInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: MissionInputs
    landing_start_utc: datetime
    landing_end_utc: datetime
    min_continuous_sunlight_days: float = Field(default=8, ge=1 / 24, le=30)
    min_continuous_earth_hours: float = Field(default=0, ge=0, le=720)

    @model_validator(mode="after")
    def valid_range(self):
        start, end = self.landing_start_utc, self.landing_end_utc
        if start.tzinfo is None or end.tzinfo is None or start.utcoffset() is None or end.utcoffset() is None:
            raise ValueError("Landing date range must include a UTC offset")
        if end <= start or end - start > timedelta(days=366):
            raise ValueError("Landing date range must be positive and at most 366 days")
        return self


class WindowCandidate(BaseModel):
    sunlight_interval_start_utc: datetime
    sunlight_interval_end_utc: datetime
    estimated_launch_utc: datetime
    estimated_landing_utc: datetime
    continuous_sunlight_days_screened: float
    minimum_sun_elevation_deg: float
    earth_contact_hours_first_8_days: float
    earth_contact_hours_during_required_stay: float
    earth_contact_hours_in_sunlight_interval: float
    continuous_earth_hours_screened: float = 0
    phase_targeted: bool = False
    terrain_verified: bool = False
    eclipse_screened: bool = False


class WindowSearchResult(BaseModel):
    candidates: list[WindowCandidate]
    best_plan: MissionPlan | None
    astronomy_source: str
    landing_start_utc: datetime
    landing_end_utc: datetime
    sampling_hours: float = 1
    required_sunlight_days: float
    required_earth_hours: float = 0
    total_screened_sunlight_hours: float
    minimum_earth_elevation_deg: float
    maximum_earth_elevation_deg: float
    recommended_landing_utc: datetime | None = None
    geometric_only: bool = True
    #: Horizon used for every candidate (also reported when none are found).
    terrain_verified: bool = False
    #: Share of hourly samples with the Sun above the horizon at this site.
    site_sunlit_pct: float | None = None
    #: Highest point of the terrain horizon around the site (deg), when measured.
    max_horizon_deg: float | None = None
    limitations: list[str]


def _transfer_and_orbit_days(inputs: MissionInputs, providers=None) -> float:
    return flight_duration_days(inputs, providers)


def search_mission_windows(search: WindowSearchInputs, providers: ScienceProviders, run_best_plan: bool = True) -> WindowSearchResult:
    """Rank one-hour landing epochs by the weakest Sun elevation over the stay.

    An hour-grid positive horizon is a *sampled* condition. For conservative
    reporting, we never extend its duration beyond the last positive sample.
    """
    start = search.landing_start_utc.astimezone(timezone.utc)
    end = search.landing_end_utc.astimezone(timezone.utc)
    required_hours = ceil(search.min_continuous_sunlight_days * 24)
    required_earth_hours = ceil(search.min_continuous_earth_hours)
    step = timedelta(hours=1)
    count = int((end - start) // step) + 1
    times = [start + i * step for i in range(count)]
    lat, lon = search.inputs.target_latitude_deg, search.inputs.target_longitude_deg
    # Real terrain horizon (LOLA polar DEM, 2 m above ground) when installed:
    # the Sun counts as visible only above the local skyline.
    horizon = horizon_or_none(providers.terrain, lat, lon, list(range(360)))
    terrain_verified = horizon is not None and all(h is not None for h in horizon)
    sun_elevations: list[float] = []
    sunlit: list[bool] = []
    earth_visible: list[bool] = []
    earth_elevations: list[float] = []
    eclipse_screened = providers.astronomy.source == "jpl-spice"
    for moment in times:
        direction, distance = providers.astronomy.get_sun_vector_from_moon(moment)
        sun = body_position(lat, lon, direction, distance)
        earth_direction, earth_distance = providers.astronomy.get_earth_vector_from_moon(moment)
        earth = body_position(lat, lon, earth_direction, earth_distance)
        earth_skyline = horizon[int(round(earth.azimuth_deg)) % 360] if terrain_verified else 0.0
        earth_visible.append(earth.elevation_deg > earth_skyline)
        earth_elevations.append(earth.elevation_deg)
        # Trace the Sun's exact azimuth rather than rounding to a 1-degree
        # skyline cell; that approximation can miss short low-angle shadows.
        local_horizon = horizon_or_none(providers.terrain, lat, lon, [sun.azimuth_deg]) if terrain_verified else None
        margin = sun.elevation_deg - local_horizon[0] if local_horizon and local_horizon[0] is not None else sun.elevation_deg
        clear = margin > 0 and (not terrain_verified or local_horizon is not None and local_horizon[0] is not None)
        if clear and eclipse_screened:
            clear = not earth_overlaps_solar_disk(lat, lon, direction, distance, earth_direction, earth_distance)
        sun_elevations.append(margin)
        sunlit.append(clear)

    # Report every qualifying continuous interval, rather than selecting five
    # widely separated dates. Choose one landing epoch per interval by its
    # weakest Sun margin over the requested stay.
    transfer_offset = timedelta(days=_transfer_and_orbit_days(search.inputs, providers))
    intervals: list[tuple[int, int]] = []
    first: int | None = None
    for index, clear in enumerate(sunlit):
        if clear and first is None:
            first = index
        if first is not None and (not clear or index == count - 1):
            last = index if clear else index - 1
            if last - first >= required_hours:
                intervals.append((first, last))
            first = None

    candidates: list[WindowCandidate] = []
    for first, last in intervals:
        eligible = []
        for landing in range(first, last - required_hours + 1):
            earth_end = landing + required_earth_hours
            if required_earth_hours and (earth_end >= count or not all(earth_visible[landing:earth_end + 1])):
                continue
            eligible.append((min(sun_elevations[landing:landing + required_hours + 1]), landing))
        if not eligible:
            continue
        minimum, i = max(eligible, key=lambda item: (item[0], -item[1]))
        contact = sum(earth_visible[i:i + min(8 * 24, count - i)])
        candidates.append(WindowCandidate(
            sunlight_interval_start_utc=times[first],
            sunlight_interval_end_utc=times[last],
            estimated_launch_utc=times[i] - transfer_offset,
            estimated_landing_utc=times[i],
            continuous_sunlight_days_screened=(last - i) / 24,
            minimum_sun_elevation_deg=minimum,
            earth_contact_hours_first_8_days=contact,
            earth_contact_hours_during_required_stay=sum(earth_visible[i:i + required_hours]),
            earth_contact_hours_in_sunlight_interval=sum(earth_visible[first:last]),
            continuous_earth_hours_screened=search.min_continuous_earth_hours,
            terrain_verified=terrain_verified,
            eclipse_screened=eclipse_screened,
        ))

    # Ranking identifies a recommendation but must not truncate the full list.
    recommended = max(candidates, key=lambda c: (
        c.continuous_earth_hours_screened >= search.min_continuous_earth_hours,
        c.minimum_sun_elevation_deg,
        c.continuous_sunlight_days_screened,
    )) if candidates else None

    best_plan = None
    if recommended and run_best_plan:
        best_inputs = search.inputs.model_copy(update={"departure_utc": recommended.estimated_launch_utc})
        best_plan = plan_mission(best_inputs, providers)
    limitations = [
        ("Sunlight and Earth contact are sampled hourly against the measured terrain skyline "
         f"({providers.terrain.source}, 2 m above ground, out to 150 km); 'minimum Sun elevation' is the margin above that skyline."
         if terrain_verified else
         ("Sunlight is sampled hourly against a spherical horizon; local terrain shadows and skyline are not evaluated "
          + ("(install the LOLA polar DEM)." if lat <= -80 else "(the installed LOLA DEM covers only 80°S to the pole)."))),
        ("Earth occultations of the Sun are screened with JPL SPICE geometry at hourly samples."
         if eclipse_screened else "Solar eclipses by Earth are not evaluated, so continuous direct sunlight is not verified."),
        "Short shadows or eclipses between hourly samples may be missed; these dates are screening estimates.",
        "The list contains one screened landing epoch for each qualifying continuous sunlight interval; interval edges are only known to the hourly sampling resolution.",
        "Departure times are fixed-offset back-calculations from a two-body transfer, configured lunar orbits and ideal descent burn, not validated launch dates; Moon phase and launch geometry are not targeted.",
        "Earth contact hours count sampled above-horizon hours within the stated stay or sunlight interval, not maximum possible contact or ground-station availability.",
        "A geometric Sun above the horizon does not guarantee adequate solar power or a safe landing site.",
    ]
    if abs(lat) == 90:
        limitations.append("At exactly 90 degrees latitude, all longitudes meet at the same pole; 65 degrees W does not select a distinct landing footprint. Provide a surveyed off-pole coordinate for terrain verification.")
    return WindowSearchResult(
        terrain_verified=terrain_verified,
        site_sunlit_pct=100 * sum(1 for e in sun_elevations if e > 0) / len(sun_elevations) if sun_elevations else None,
        max_horizon_deg=max(horizon) if terrain_verified else None,
        candidates=candidates, best_plan=best_plan,
        astronomy_source=providers.astronomy.source,
        landing_start_utc=start, landing_end_utc=end,
        required_sunlight_days=search.min_continuous_sunlight_days,
        required_earth_hours=search.min_continuous_earth_hours,
        # Trapezoidal tally: each transition lies halfway between its hourly
        # samples for this estimate. Individual interval endpoints remain
        # conservative sampled timestamps above.
        total_screened_sunlight_hours=sum((int(a) + int(b)) / 2 for a, b in zip(sunlit, sunlit[1:])),
        minimum_earth_elevation_deg=min(earth_elevations),
        maximum_earth_elevation_deg=max(earth_elevations),
        recommended_landing_utc=recommended.estimated_landing_utc if recommended else None,
        geometric_only=not terrain_verified,
        limitations=limitations,
    )
