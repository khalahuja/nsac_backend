"""Environmental timeline: sample the solar/thermal state over a time window.

Used to drive the temperature and irradiance charts. A window of one synodic
month (~709 h) at a 3 h step is 237 samples, which the analytic ephemeris
evaluates in a few tens of milliseconds.

Sun events are deliberately *not* computed per sample - that search is the
expensive part of the environment endpoint and would dominate the runtime here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..constants import SYNODIC_MONTH_HOURS
from ..models.schemas import (
    Coordinate,
    TimelineResponse,
    TimelineSample,
)
from .environment import build_fidelity_info, build_source_info
from .illumination import compute_illumination
from .positions import body_position
from .providers import ScienceProviders
from .thermal import estimate_temperature

logger = logging.getLogger(__name__)


class TimelineError(ValueError):
    """Raised when a timeline request cannot be satisfied."""


def build_time_grid(
    center: datetime, hours_before: float, hours_after: float, step_hours: float
) -> list[datetime]:
    """Build the sample times, inclusive of both ends and of the centre.

    The centre is guaranteed to be present even when it does not fall on a step
    boundary, so the chart always has a marker at the user's selected instant.
    """
    if step_hours <= 0.0:
        raise TimelineError("step_hours must be positive")
    if hours_before == 0.0 and hours_after == 0.0:
        return [center]

    if center.tzinfo is None:
        center = center.replace(tzinfo=timezone.utc)

    start = center - timedelta(hours=hours_before)
    total_hours = hours_before + hours_after
    step_count = int(np.floor(total_hours / step_hours))

    times = [start + timedelta(hours=step_hours * index) for index in range(step_count + 1)]
    if times[-1] < center + timedelta(hours=hours_after):
        times.append(center + timedelta(hours=hours_after))
    if center not in times:
        times.append(center)
    return sorted(set(times))


def compute_timeline(
    *,
    latitude_deg: float,
    longitude_deg: float,
    center: datetime,
    hours_before: float,
    hours_after: float,
    step_hours: float,
    providers: ScienceProviders,
    max_samples: int = 1200,
) -> TimelineResponse:
    """Evaluate solar geometry and temperature across a time window.

    Args:
        latitude_deg: Latitude, north positive.
        longitude_deg: Longitude, east positive.
        center: Centre of the window (UTC).
        hours_before: Extent before the centre.
        hours_after: Extent after the centre.
        step_hours: Sampling interval.
        providers: Active science providers.
        max_samples: Hard cap; exceeding it raises rather than silently
            coarsening the grid, so the client knows what it asked for.

    Returns:
        A :class:`TimelineResponse` with one sample per timestep.

    Raises:
        TimelineError: if the requested grid exceeds ``max_samples``.
    """
    times = build_time_grid(center, hours_before, hours_after, step_hours)
    if len(times) > max_samples:
        raise TimelineError(
            f"Requested {len(times)} samples, which exceeds the limit of {max_samples}. "
            f"Increase step_hours (currently {step_hours} h) or shorten the window."
        )

    astronomy = providers.astronomy
    terrain = providers.terrain

    # The surface normal is time-invariant, so resolve it once rather than per
    # sample - this is the single biggest saving for DEM-backed terrain.
    surface_normal = terrain.get_surface_normal(latitude_deg, longitude_deg)

    samples: list[TimelineSample] = []
    for moment in times:
        sun_direction, sun_distance = astronomy.get_sun_vector_from_moon(moment)
        illumination = compute_illumination(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            sun_direction_body=sun_direction,
            sun_distance_km=sun_distance,
            surface_normal_body=surface_normal,
            # Per-sample DEM ray-marching is too slow for a timeline; the
            # environment endpoint reports the authoritative shadow state.
            terrain_shadowed=None,
        )
        earth_direction, earth_distance = astronomy.get_earth_vector_from_moon(moment)
        earth = body_position(latitude_deg, longitude_deg, earth_direction, earth_distance)
        thermal = estimate_temperature(
            illumination.solar_irradiance_w_m2,
            albedo=providers.albedo,
            emissivity=providers.emissivity,
        )
        samples.append(
            TimelineSample(
                datetime_utc=moment,
                temperature_k=thermal.temperature_k,
                temperature_c=thermal.temperature_c,
                solar_irradiance_w_m2=illumination.solar_irradiance_w_m2,
                horizontal_irradiance_w_m2=illumination.horizontal_irradiance_w_m2,
                solar_elevation_deg=illumination.elevation_deg,
                solar_azimuth_deg=illumination.azimuth_deg,
                incidence_angle_deg=illumination.incidence_angle_deg,
                illuminated=illumination.illuminated,
                earth_elevation_deg=earth.elevation_deg,
                earth_azimuth_deg=earth.azimuth_deg,
                earth_distance_km=earth.distance_km,
                earth_visible=earth.elevation_deg > 0.0,
                sun_distance_km=sun_distance,
            )
        )

    return TimelineResponse(
        coordinate=Coordinate(latitude=latitude_deg, longitude=longitude_deg),
        center_datetime_utc=center,
        step_hours=step_hours,
        sample_count=len(samples),
        samples=samples,
        sources=build_source_info(providers),
        fidelity=build_fidelity_info(providers),
        synodic_month_hours=SYNODIC_MONTH_HOURS,
    )


def timeline_to_dataframe(response: TimelineResponse) -> pd.DataFrame:
    """Convert a timeline to a pandas ``DataFrame``.

    Not used by the HTTP API - the browser receives compact JSON - but handy
    for notebook analysis and for the statistics helper below.
    """
    return pd.DataFrame(
        [
            {
                "datetime_utc": sample.datetime_utc,
                "temperature_k": sample.temperature_k,
                "temperature_c": sample.temperature_c,
                "solar_irradiance_w_m2": sample.solar_irradiance_w_m2,
                "solar_elevation_deg": sample.solar_elevation_deg,
                "illuminated": sample.illuminated,
            }
            for sample in response.samples
        ]
    ).set_index("datetime_utc")


def summarise_timeline(response: TimelineResponse) -> dict[str, float]:
    """Min/max/mean temperature and illuminated fraction over the window."""
    frame = timeline_to_dataframe(response)
    if frame.empty:
        return {}
    return {
        "temperature_min_k": float(frame["temperature_k"].min()),
        "temperature_max_k": float(frame["temperature_k"].max()),
        "temperature_mean_k": float(frame["temperature_k"].mean()),
        "peak_irradiance_w_m2": float(frame["solar_irradiance_w_m2"].max()),
        "illuminated_fraction": float(frame["illuminated"].mean()),
    }


__all__ = [
    "TimelineError",
    "build_time_grid",
    "compute_timeline",
    "summarise_timeline",
    "timeline_to_dataframe",
]
