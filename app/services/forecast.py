"""Explicit UTC-range lunar environment forecasts.

The temperature calculation is a reduced-order thermal inertia model. It
relaxes the surface toward radiative equilibrium with separate heating and
cooling time constants, after a two-lunar-day spin-up. This gives continuous
night-time cooling and sunrise warming without pretending to be the full
depth-resolved Diviner-calibrated conduction model planned for production.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from ..constants import SYNODIC_MONTH_HOURS
from ..models.schemas import (
    Coordinate,
    FidelityInfo,
    ForecastResponse,
    SourceInfo,
    TimelineSample,
    VisibilityWindow,
)
from .environment import build_fidelity_info, build_source_info
from .illumination import IlluminationResult, compute_illumination
from .providers import ScienceProviders
from .thermal import estimate_temperature
from .skyline import terrain_skyline, horizon_at
from .positions import body_position, earth_overlaps_solar_disk, topocentric_vector

FORECAST_MODEL_NAME = "reduced-order-regolith-thermal-inertia-v1"
FORECAST_MODEL_NOTES = (
    "Model-based estimate. Surface temperature is relaxed toward radiative "
    "equilibrium after a two-synodic-month spin-up (10 h heating and 55 h "
    "cooling response constants). Response constants are uncalibrated assumptions. "
    "A fixed 30-minute integration step is independent of output sampling. "
    "A 1-degree terrain skyline is used when available, interpolated in azimuth; "
    "otherwise visibility uses a spherical horizon. Body centres define crossings. "
    "It does not solve the depth-resolved "
    "heat equation and is not a future NASA temperature measurement."
)


class ForecastError(ValueError):
    """Raised when a requested prediction range is invalid."""


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _time_grid(start: datetime, end: datetime, step_hours: float) -> list[datetime]:
    if end < start:
        raise ForecastError("end_datetime_utc must be after start_datetime_utc")
    if step_hours <= 0:
        raise ForecastError("step_hours must be positive")

    step = timedelta(hours=step_hours)
    values: list[datetime] = []
    moment = start
    while moment <= end:
        values.append(moment)
        moment += step
    if values[-1] != end:
        values.append(end)
    return values


def _illumination_at(
    moment: datetime,
    latitude_deg: float,
    longitude_deg: float,
    providers: ScienceProviders,
    surface_normal: object,
    skyline=None,
) -> IlluminationResult:
    sun_direction, sun_distance = providers.astronomy.get_sun_vector_from_moon(moment)
    sun = body_position(latitude_deg, longitude_deg, sun_direction, sun_distance)
    shadowed = sun.elevation_deg <= horizon_at(skyline, sun.azimuth_deg) if skyline is not None else None
    return compute_illumination(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        sun_direction_body=topocentric_vector(latitude_deg, longitude_deg, sun_direction, sun_distance),
        sun_distance_km=sun_distance,
        surface_normal_body=surface_normal,  # type: ignore[arg-type]
        terrain_shadowed=shadowed,
    )


def _advance_temperature(
    temperature_k: float,
    target_k: float,
    hours: float,
    illuminated: bool,
) -> float:
    time_constant_hours = 10.0 if illuminated else 55.0
    fraction = 1.0 - math.exp(-hours / time_constant_hours)
    return temperature_k + (target_k - temperature_k) * fraction


def compute_forecast(
    *,
    latitude_deg: float,
    longitude_deg: float,
    start: datetime,
    end: datetime,
    step_hours: float,
    providers: ScienceProviders,
    max_samples: int = 1200,
    min_earth_elevation_deg: float = 0,
) -> ForecastResponse:
    """Compute one prediction row per requested UTC sample."""
    start = _utc(start)
    end = _utc(end)
    if not math.isfinite(step_hours) or step_hours <= 0:
        raise ForecastError("step_hours must be finite and positive")
    if end < start:
        raise ForecastError("end_datetime_utc must be after start_datetime_utc")
    requested_count = math.ceil((end - start).total_seconds() / (step_hours * 3600)) + 1
    if requested_count > max_samples:
        raise ForecastError(f"Requested {requested_count} samples, which exceeds the limit of {max_samples}.")
    times = _time_grid(start, end, step_hours)
    if len(times) > max_samples:
        raise ForecastError(
            f"Requested {len(times)} samples, which exceeds the limit of {max_samples}. "
            "Increase the sampling interval or shorten the prediction period."
        )

    surface_normal = providers.terrain.get_surface_normal(latitude_deg, longitude_deg)
    skyline = terrain_skyline(providers.terrain, latitude_deg, longitude_deg)
    internal_step_hours = 0.5

    # Spin up the low-order state so the first requested row does not depend on
    # an arbitrary initial temperature.
    temperature_k = 100.0
    spin_start = start - timedelta(hours=2.0 * SYNODIC_MONTH_HOURS)
    moment = spin_start
    while moment < start:
        next_moment = min(moment + timedelta(hours=internal_step_hours), start)
        illumination = _illumination_at(
            next_moment, latitude_deg, longitude_deg, providers, surface_normal, skyline
        )
        target = estimate_temperature(
            illumination.solar_irradiance_w_m2,
            albedo=providers.albedo,
            emissivity=providers.emissivity,
        ).temperature_k
        temperature_k = _advance_temperature(
            temperature_k,
            target,
            (next_moment - moment).total_seconds() / 3600.0,
            illumination.illuminated,
        )
        moment = next_moment

    samples: list[TimelineSample] = []
    cursor = start
    integration_temperature_k = temperature_k
    for sample_time in times:
        # Advance only on the fixed internal grid. Output rows inside a grid
        # cell evaluate a partial response without changing the integrator state.
        while cursor + timedelta(hours=internal_step_hours) <= sample_time:
            next_moment = cursor + timedelta(hours=internal_step_hours)
            step_illumination = _illumination_at(
                next_moment, latitude_deg, longitude_deg, providers, surface_normal, skyline
            )
            target = estimate_temperature(
                step_illumination.solar_irradiance_w_m2,
                albedo=providers.albedo, emissivity=providers.emissivity,
            ).temperature_k
            integration_temperature_k = _advance_temperature(
                integration_temperature_k, target, internal_step_hours, step_illumination.illuminated,
            )
            cursor = next_moment
        temperature_k = integration_temperature_k
        if sample_time > cursor:
            right = cursor + timedelta(hours=internal_step_hours)
            partial_illumination = _illumination_at(
                right, latitude_deg, longitude_deg, providers, surface_normal, skyline
            )
            target = estimate_temperature(
                partial_illumination.solar_irradiance_w_m2,
                albedo=providers.albedo, emissivity=providers.emissivity,
            ).temperature_k
            temperature_k = _advance_temperature(
                integration_temperature_k, target, (sample_time - cursor).total_seconds() / 3600,
                partial_illumination.illuminated,
            )

        illumination = _illumination_at(
            sample_time, latitude_deg, longitude_deg, providers, surface_normal, skyline
        )
        earth_direction, earth_distance = providers.astronomy.get_earth_vector_from_moon(sample_time)
        earth = body_position(latitude_deg, longitude_deg, earth_direction, earth_distance)
        sun_direction, sun_distance = providers.astronomy.get_sun_vector_from_moon(sample_time)
        solar_earth_overlap = earth_overlaps_solar_disk(
            latitude_deg, longitude_deg, sun_direction, sun_distance, earth_direction, earth_distance
        )
        samples.append(
            TimelineSample(
                datetime_utc=sample_time,
                temperature_k=temperature_k,
                temperature_c=temperature_k - 273.15,
                solar_irradiance_w_m2=illumination.solar_irradiance_w_m2,
                horizontal_irradiance_w_m2=illumination.horizontal_irradiance_w_m2,
                solar_elevation_deg=illumination.elevation_deg,
                solar_azimuth_deg=illumination.azimuth_deg,
                incidence_angle_deg=illumination.incidence_angle_deg,
                illuminated=illumination.illuminated,
                solar_earth_overlap=solar_earth_overlap,
                earth_elevation_deg=earth.elevation_deg,
                earth_azimuth_deg=earth.azimuth_deg,
                earth_distance_km=earth.distance_km,
                earth_visible=earth.elevation_deg > max(min_earth_elevation_deg, horizon_at(skyline, earth.azimuth_deg)),
                terrain_shadowed=illumination.terrain_shadowed,
                solar_horizon_margin_deg=min(illumination.elevation_deg - max(0.0, horizon_at(skyline, illumination.azimuth_deg)), 90.0 - illumination.incidence_angle_deg),
                earth_horizon_margin_deg=earth.elevation_deg - max(min_earth_elevation_deg, horizon_at(skyline, earth.azimuth_deg)),
                sun_distance_km=illumination.sun_distance_km,
            )
        )

    source = build_source_info(providers).model_copy(
        update={"temperature": FORECAST_MODEL_NAME}
    )
    fidelity = build_fidelity_info(providers).model_copy(
        update={"temperature": "Experimental (thermal inertia response model)"}
    )
    sun_windows = visibility_windows(samples, 'solar_horizon_margin_deg', 0)
    earth_windows = visibility_windows(samples, 'earth_horizon_margin_deg', 0)
    longest_darkness, boundary_limited = darkness_summary(start, end, sun_windows)
    return ForecastResponse(
        coordinate=Coordinate(latitude=latitude_deg, longitude=longitude_deg),
        start_datetime_utc=start,
        end_datetime_utc=end,
        step_hours=step_hours,
        sample_count=len(samples),
        samples=samples,
        sources=SourceInfo(**source.model_dump()),
        fidelity=FidelityInfo(**fidelity.model_dump()),
        model_name=FORECAST_MODEL_NAME,
        model_notes=FORECAST_MODEL_NOTES,
        earth_visibility_windows=earth_windows,
        earth_visible_hours=sum(window.duration_hours for window in earth_windows),
        terrain_visibility_available=skyline is not None,
        sun_visibility_windows=sun_windows,
        sunlit_hours=sum(window.duration_hours for window in sun_windows),
        longest_darkness_hours=longest_darkness,
        darkness_boundary_limited=boundary_limited,
        position_notes=("Visibility uses an interpolated 1-degree measured terrain skyline." if skyline is not None else "Visibility uses a spherical horizon; terrain coverage is unavailable or disabled.") + " Body centres define visibility; finite disks, ground-station availability, antenna patterns and link budgets are not included. Earth/Sun disk overlap is flagged but does not attenuate power.",
        sun_above_horizon_hours=sum(window.duration_hours for window in visibility_windows(samples, 'solar_elevation_deg', 0)),
        horizontal_energy_wh_m2=sum((b.datetime_utc - a.datetime_utc).total_seconds() / 3600 * (a.horizontal_irradiance_w_m2 + b.horizontal_irradiance_w_m2) / 2 for a, b in zip(samples, samples[1:])),
        min_earth_elevation_deg=min_earth_elevation_deg,
    )


def darkness_summary(start: datetime, end: datetime, windows: list[VisibilityWindow]) -> tuple[float, bool]:
    """Largest complement interval; boundary-touching durations are lower bounds."""
    cursor = start
    intervals = []
    for window in windows:
        if window.start_utc > cursor:
            intervals.append((cursor, window.start_utc))
        cursor = max(cursor, window.end_utc)
    if cursor < end:
        intervals.append((cursor, end))
    if not intervals:
        return 0.0, False
    a, b = max(intervals, key=lambda pair: pair[1] - pair[0])
    return (b - a).total_seconds() / 3600, a == start or b == end


def visibility_windows(samples: list[TimelineSample], field: str, threshold: float) -> list[VisibilityWindow]:
    """Linear crossing estimates, clipped to the sampled period (not exact events)."""
    windows: list[VisibilityWindow] = []
    active_start = None
    for a, b in zip(samples, samples[1:]):
        va, vb = getattr(a, field) - threshold, getattr(b, field) - threshold
        start, end = a.datetime_utc, b.datetime_utc
        if va <= 0 and vb <= 0:
            continue
        if (va > 0) != (vb > 0):
            crossing = start + (end - start) * (-va / (vb - va))
            if va <= 0:
                start = crossing
            else:
                end = crossing
        if active_start is None:
            active_start = start
        if vb <= 0 or b is samples[-1]:
            windows.append(VisibilityWindow(start_utc=active_start, end_utc=end, duration_hours=(end - active_start).total_seconds() / 3600))
            active_start = None
    return windows


__all__ = [
    "FORECAST_MODEL_NAME",
    "FORECAST_MODEL_NOTES",
    "ForecastError",
    "compute_forecast",
]
