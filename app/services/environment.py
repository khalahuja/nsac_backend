"""Orchestration: turn a (coordinate, instant) pair into a full analysis.

This is the single place where astronomy, terrain, illumination, thermal and
Diviner results are composed. Routes stay thin; the science modules stay
independent and individually testable.

Caching
-------

Analyses are memoised on a quantised key (1e-4 deg ~= 3 m on the lunar surface,
1 s in time). A user dragging across the Moon generates many near-identical
requests, and the sunrise/sunset search is the expensive part. The cache is
cleared whenever the provider bundle changes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache

from ..constants import SYNODIC_MONTH_HOURS
from ..models.schemas import (
    Coordinate,
    DivinerObservationInfo,
    EnvironmentResponse,
    FidelityInfo,
    SolarInfo,
    SourceInfo,
    SunEventsInfo,
    TerrainInfo,
    ThermalInfo,
)
from .astronomy import SimpleAstronomyProvider
from .illumination import compute_illumination, find_sun_events
from .providers import ScienceProviders
from .terrain import FlatTerrainProvider
from .thermal import estimate_temperature, temperature_confidence_label

logger = logging.getLogger(__name__)


class ScienceComputationError(RuntimeError):
    """Raised when a scientific calculation fails for a valid input."""


def _quantise(value: float, places: int = 4) -> float:
    """Round a coordinate for cache-key stability (1e-4 deg ~ 3 m)."""
    return round(value, places)


def analyse_environment(
    *,
    latitude_deg: float,
    longitude_deg: float,
    moment: datetime,
    providers: ScienceProviders,
    include_sun_events: bool = True,
) -> EnvironmentResponse:
    """Compute the complete environmental state at a point and time.

    Args:
        latitude_deg: Latitude, north positive, in [-90, 90].
        longitude_deg: Longitude, east positive, in [-180, 180].
        moment: Instant of interest (UTC).
        providers: Active science providers.
        include_sun_events: Whether to search for the next sunrise/sunset.

    Returns:
        A fully populated :class:`EnvironmentResponse`, including provenance.

    Raises:
        ScienceComputationError: if any underlying model fails.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    try:
        return _analyse_cached(
            _quantise(latitude_deg),
            _quantise(longitude_deg),
            moment.replace(microsecond=0),
            include_sun_events,
            providers,
        )
    except ScienceComputationError:
        raise
    except Exception as error:  # noqa: BLE001 - wrapped for the API layer
        logger.exception("Environment analysis failed")
        raise ScienceComputationError(
            f"Scientific calculation failed at ({latitude_deg}, {longitude_deg}): {error}"
        ) from error


@lru_cache(maxsize=2048)
def _analyse_cached(
    latitude_deg: float,
    longitude_deg: float,
    moment: datetime,
    include_sun_events: bool,
    providers: ScienceProviders,
) -> EnvironmentResponse:
    """Memoised core of :func:`analyse_environment`.

    ``ScienceProviders`` is a frozen dataclass and therefore hashable, so
    swapping providers naturally partitions the cache instead of returning
    stale results from a previous configuration.
    """
    astronomy = providers.astronomy
    terrain = providers.terrain

    # --- Geometry ------------------------------------------------------
    geometry = astronomy.get_geometry(moment)

    # --- Terrain -------------------------------------------------------
    terrain_sample = terrain.sample(latitude_deg, longitude_deg)
    from .positions import horizon_positions, topocentric_vector
    positions = horizon_positions(latitude_deg, longitude_deg, geometry, terrain_sample.elevation_m or 0)
    sun_topocentric = topocentric_vector(latitude_deg, longitude_deg, geometry.sun_direction_body, geometry.sun_distance_km, terrain_sample.elevation_m or 0)

    # --- Illumination --------------------------------------------------
    from .illumination import solar_azimuth_deg, solar_elevation_deg  # noqa: PLC0415

    provisional_elevation = solar_elevation_deg(
        latitude_deg, longitude_deg, sun_topocentric
    )
    provisional_azimuth = solar_azimuth_deg(
        latitude_deg, longitude_deg, sun_topocentric
    )
    shadowed = terrain.is_terrain_shadowed(
        latitude_deg, longitude_deg, provisional_azimuth, provisional_elevation
    )

    illumination = compute_illumination(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        sun_direction_body=sun_topocentric,
        sun_distance_km=geometry.sun_distance_km,
        surface_normal_body=terrain_sample.surface_normal,
        terrain_shadowed=shadowed,
    )

    # --- Thermal -------------------------------------------------------
    thermal = estimate_temperature(
        illumination.solar_irradiance_w_m2,
        albedo=providers.albedo,
        emissivity=providers.emissivity,
    )
    confidence = temperature_confidence_label(
        has_terrain=terrain_sample.is_measured,
        has_diviner=providers.diviner.is_available,
    )

    # --- Diviner (measured, optional) ----------------------------------
    observation = providers.diviner.get_observed_temperature(
        latitude_deg, longitude_deg, moment
    )
    diviner_info = (
        DivinerObservationInfo(
            temperature_k=observation.temperature_k,
            local_time_hours=observation.local_time_hours,
            dataset=observation.dataset,
            observation_datetime_utc=observation.observation_datetime_utc,
            comparison_kind=observation.comparison_kind, source_url=observation.source_url,
            reference_period=observation.reference_period, season=observation.season, distance_m=observation.distance_m,
        )
        if observation is not None
        else None
    )

    # --- Sunrise / sunset ----------------------------------------------
    sun_events_info: SunEventsInfo | None = None
    if include_sun_events:
        events = find_sun_events(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            moment=moment,
            astronomy=astronomy,
        )
        sun_events_info = SunEventsInfo(
            next_sunrise_utc=events.next_sunrise_utc,
            next_sunset_utc=events.next_sunset_utc,
            search_window_hours=events.search_window_hours,
            note=events.note,
        )

    return EnvironmentResponse(
        coordinate=Coordinate(latitude=latitude_deg, longitude=longitude_deg),
        datetime_utc=moment,
        positions=positions,
        terrain=TerrainInfo(
            elevation_m=terrain_sample.elevation_m,
            slope_deg=terrain_sample.slope_deg,
            aspect_deg=terrain_sample.aspect_deg,
            source=terrain_sample.source,
            is_measured=terrain_sample.is_measured,
        ),
        solar=SolarInfo(
            elevation_deg=illumination.elevation_deg,
            azimuth_deg=illumination.azimuth_deg,
            incidence_angle_deg=illumination.incidence_angle_deg,
            illuminated=illumination.illuminated,
            above_horizon=illumination.above_horizon,
            terrain_shadowed=illumination.terrain_shadowed,
            solar_irradiance_w_m2=illumination.solar_irradiance_w_m2,
            horizontal_irradiance_w_m2=illumination.horizontal_irradiance_w_m2,
            solar_constant_at_moon_w_m2=illumination.solar_constant_at_moon_w_m2,
            sun_distance_km=illumination.sun_distance_km,
            sun_vector_body=[float(component) for component in illumination.sun_vector_body],
            sun_vector_j2000=[
                float(component) for component in geometry.sun_direction_j2000
            ],
            earth_vector_body=[
                float(component) for component in geometry.earth_direction_body
            ],
            earth_distance_km=geometry.earth_distance_km,
            subsolar_latitude_deg=geometry.subsolar_latitude_deg,
            subsolar_longitude_deg=geometry.subsolar_longitude_deg,
        ),
        thermal=ThermalInfo(
            temperature_k=thermal.temperature_k,
            temperature_c=thermal.temperature_c,
            model=thermal.model,
            confidence=confidence,  # type: ignore[arg-type]
            albedo=thermal.albedo,
            emissivity=thermal.emissivity,
            absorbed_flux_w_m2=thermal.absorbed_flux_w_m2,
            is_measured=thermal.is_measured,
            notes=thermal.notes,
        ),
        diviner_observation=diviner_info,
        sun_events=sun_events_info,
        sources=build_source_info(providers),
        fidelity=build_fidelity_info(providers),
        accuracy_notes=build_accuracy_notes(providers, thermal_notes=thermal.notes),
    )


def clear_environment_cache() -> None:
    """Drop every memoised analysis (used on reconfiguration and in tests)."""
    _analyse_cached.cache_clear()


def environment_cache_info() -> str:
    """Return the LRU cache statistics as a string, for ``/api/status``."""
    return str(_analyse_cached.cache_info())


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def build_source_info(providers: ScienceProviders) -> SourceInfo:
    """Machine-readable provenance map."""
    described = providers.describe()
    return SourceInfo(
        astronomy=described["astronomy"],
        terrain=described["terrain"],
        temperature=described["temperature"],
        diviner=described["diviner"],
    )


def build_fidelity_info(providers: ScienceProviders) -> FidelityInfo:
    """Human-readable fidelity labels for the UI panel."""
    astronomy_is_simple = isinstance(providers.astronomy, SimpleAstronomyProvider)
    terrain_is_flat = isinstance(providers.terrain, FlatTerrainProvider)

    astronomy_label = "Approximate (analytic model)" if astronomy_is_simple else "JPL SPICE"
    terrain_label = (
        "Placeholder (perfect sphere)" if terrain_is_flat else f"DEM: {providers.terrain.source}"
    )
    # Loading observations does not fit or calibrate the temperature model.
    temperature_label = "Experimental (simplified radiative model)"

    if astronomy_is_simple and terrain_is_flat:
        overall = (
            "All results are model approximations. No NASA dataset is currently "
            "installed - install SPICE kernels and a LOLA/SLDEM raster to improve them."
        )
    elif terrain_is_flat:
        overall = (
            "Astronomy uses JPL SPICE, but terrain is still a perfect sphere: "
            "elevation, slope and terrain shadowing are not measured."
        )
    elif astronomy_is_simple:
        overall = (
            "Terrain comes from a real DEM, but the ephemeris is the analytic "
            "approximation. Enable SPICE for precise Sun geometry."
        )
    else:
        overall = (
            "Astronomy and terrain use installed NASA datasets. The thermal model "
            "remains a simplified radiative estimate."
        )

    return FidelityInfo(
        astronomy=astronomy_label,
        terrain=terrain_label,
        temperature=temperature_label,
        overall_note=overall,
    )


def build_accuracy_notes(providers: ScienceProviders, *, thermal_notes: str) -> list[str]:
    """Assemble the caveats that apply to the active provider set."""
    notes = [providers.astronomy.accuracy_note, thermal_notes]

    if isinstance(providers.terrain, FlatTerrainProvider):
        notes.append(
            "Terrain is the 1737.4 km reference sphere: elevation and slope are zero "
            "by definition, not by measurement, and terrain shadowing is unknown "
            "(reported as null)."
        )
    if not providers.diviner.is_available:
        notes.append(
            "No LRO Diviner dataset is installed, so no observed temperature is "
            "available for comparison."
        )
    return notes


SYNODIC_MONTH_HOURS_EXPORT = SYNODIC_MONTH_HOURS

__all__ = [
    "ScienceComputationError",
    "analyse_environment",
    "build_accuracy_notes",
    "build_fidelity_info",
    "build_source_info",
    "clear_environment_cache",
    "environment_cache_info",
]
