"""Pydantic request/response models.

These double as the OpenAPI schema shown at ``/docs``, so every field carries a
description and, where useful, an example. Validation limits (latitude range,
timeline length) live here rather than in the route handlers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

LatitudeField = Annotated[
    float, Field(ge=-90.0, le=90.0, description="Planetocentric latitude, north positive [deg]")
]
LongitudeField = Annotated[
    float, Field(ge=-180.0, le=180.0, description="Longitude, east positive [deg]")
]


def _ensure_utc(value: datetime) -> datetime:
    """Normalise any incoming datetime to timezone-aware UTC.

    Naive datetimes are *assumed* to be UTC, matching the documented API
    contract that all times are UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class EnvironmentRequest(BaseModel):
    """Input for ``POST /api/environment``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "latitude": -89.4,
                "longitude": 137.3,
                "datetime_utc": "2026-09-26T12:00:00Z",
            }
        }
    )

    latitude: LatitudeField
    longitude: LongitudeField
    datetime_utc: datetime = Field(
        description="Instant of interest, UTC. ISO 8601; a trailing 'Z' is accepted."
    )
    include_sun_events: bool = Field(
        default=True,
        description=(
            "Search forward for the next sunrise/sunset. Adds ~100 ms; disable "
            "for bulk queries."
        ),
    )

    @field_validator("datetime_utc")
    @classmethod
    def normalise_datetime(cls, value: datetime) -> datetime:
        return _ensure_utc(value)


class TimelineRequest(BaseModel):
    """Input for ``POST /api/timeline``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "latitude": -89.4,
                "longitude": 137.3,
                "center_datetime_utc": "2026-09-26T12:00:00Z",
                "hours_before": 72,
                "hours_after": 72,
                "step_hours": 3,
            }
        }
    )

    latitude: LatitudeField
    longitude: LongitudeField
    center_datetime_utc: datetime = Field(description="Centre of the time window, UTC.")
    hours_before: float = Field(
        default=72.0,
        ge=0.0,
        le=1440.0,
        description="Hours to extend before the centre (max 1440 h = 60 days).",
    )
    hours_after: float = Field(
        default=72.0, ge=0.0, le=1440.0, description="Hours to extend after the centre."
    )
    step_hours: float = Field(
        default=3.0,
        gt=0.0,
        le=24.0,
        description="Sampling interval in hours.",
    )

    @field_validator("center_datetime_utc")
    @classmethod
    def normalise_datetime(cls, value: datetime) -> datetime:
        return _ensure_utc(value)


class ForecastRequest(BaseModel):
    """Input shared by ``POST /api/forecast`` and the XLSX export route."""

    latitude: LatitudeField
    longitude: LongitudeField
    min_earth_elevation_deg: float = Field(default=0, ge=0, le=90, description="Minimum topocentric Earth-centre elevation for geometric DTE visibility.")
    start_datetime_utc: datetime = Field(description="First forecast sample, UTC.")
    end_datetime_utc: datetime = Field(description="Last forecast sample, UTC.")
    step_hours: float = Field(
        default=1.0,
        gt=0.0,
        le=24.0,
        description="Sampling interval in hours.",
    )
    display_timezone: str = Field(
        default="UTC",
        max_length=64,
        description="IANA time zone used for local-time display in exports.",
    )

    @field_validator("start_datetime_utc", "end_datetime_utc")
    @classmethod
    def normalise_forecast_datetime(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_validator("display_timezone")
    @classmethod
    def validate_display_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("display_timezone must be a valid IANA time zone") from error
        return value


# ---------------------------------------------------------------------------
# Response components
# ---------------------------------------------------------------------------


class Coordinate(BaseModel):
    """A selenographic surface coordinate."""

    latitude: float = Field(description="Latitude, north positive [deg]")
    longitude: float = Field(description="Longitude, east positive [deg]")


class TerrainInfo(BaseModel):
    """Local topography. ``null`` fields mean 'not measured', never 'zero'."""

    elevation_m: float | None = Field(
        description="Elevation above the 1737.4 km reference sphere [m]; null if unmeasured."
    )
    slope_deg: float | None = Field(description="Terrain slope [deg]; null if unmeasured.")
    aspect_deg: float | None = Field(
        description="Downhill azimuth [deg clockwise from North]; null if undefined."
    )
    source: str = Field(description="Dataset or model that produced these values.")
    is_measured: bool = Field(
        description="True only when values come from a real DEM, not the reference sphere."
    )


class SolarInfo(BaseModel):
    """Solar geometry and direct-beam irradiance at the point."""

    elevation_deg: float = Field(description="Sun elevation above local horizontal [deg].")
    azimuth_deg: float = Field(description="Sun azimuth [deg]; 0=N, 90=E, 180=S, 270=W.")
    incidence_angle_deg: float = Field(
        description="Angle between the surface normal and the Sun [deg]."
    )
    illuminated: bool = Field(
        description="Sun above the horizon AND not blocked by terrain."
    )
    above_horizon: bool = Field(
        description="Sun above the local horizontal plane, ignoring terrain."
    )
    terrain_shadowed: bool | None = Field(
        description="Terrain horizon blocks the Sun; null when no DEM is loaded."
    )
    solar_irradiance_w_m2: float = Field(
        description="Direct-beam irradiance on the (possibly tilted) surface [W/m^2]."
    )
    horizontal_irradiance_w_m2: float = Field(
        description="Direct irradiance on the local horizontal plane [W/m^2]."
    )
    solar_constant_at_moon_w_m2: float = Field(
        description="Irradiance on a Sun-facing surface at the current distance [W/m^2]."
    )
    sun_distance_km: float = Field(description="Sun-Moon centre distance [km].")
    sun_vector_body: list[float] = Field(
        description="Unit vector to the Sun in the lunar body-fixed (IAU_MOON) frame."
    )
    sun_vector_j2000: list[float] = Field(
        description="Unit vector to the Sun in the J2000 inertial frame."
    )
    earth_vector_body: list[float] = Field(
        description="Unit vector to Earth in the lunar body-fixed frame."
    )
    earth_distance_km: float = Field(description="Earth-Moon centre distance [km].")
    subsolar_latitude_deg: float = Field(description="Latitude of the sub-solar point [deg].")
    subsolar_longitude_deg: float = Field(
        description="Longitude of the sub-solar point [deg]."
    )


class ThermalInfo(BaseModel):
    """Modelled surface temperature. Never a measurement."""

    temperature_k: float = Field(description="Estimated surface temperature [K].")
    temperature_c: float = Field(description="Estimated surface temperature [deg C].")
    model: str = Field(description="Thermal model identifier.")
    confidence: Literal["experimental", "experimental-with-terrain", "calibrated"] = Field(
        description="Qualitative reliability of the estimate."
    )
    albedo: float = Field(description="Bond albedo used [dimensionless].")
    emissivity: float = Field(description="Thermal emissivity used [dimensionless].")
    absorbed_flux_w_m2: float = Field(description="(1 - albedo) * irradiance [W/m^2].")
    is_measured: bool = Field(description="Always false for modelled temperatures.")
    notes: str = Field(description="Plain-language statement of model limitations.")


class DivinerObservationInfo(BaseModel):
    """A real LRO Diviner measurement, when one is installed locally."""

    temperature_k: float
    local_time_hours: float | None
    dataset: str
    observation_datetime_utc: datetime | None
    comparison_kind: str = "historical-local-time-climatology"
    source_url: str = ""
    reference_period: str = ""
    season: str = "all"
    distance_m: float | None = None


class SunEventsInfo(BaseModel):
    """Next horizon crossings at the point."""

    next_sunrise_utc: datetime | None = Field(
        description="Next time the Sun rises above the geometric horizon."
    )
    next_sunset_utc: datetime | None = Field(
        description="Next time the Sun sets below the geometric horizon."
    )
    search_window_hours: float = Field(description="How far ahead the search ran [h].")
    note: str | None = Field(
        description="Explanation when a crossing was not found (e.g. polar night)."
    )


class SourceInfo(BaseModel):
    """Which model or dataset produced each part of the answer."""

    astronomy: str
    terrain: str
    temperature: str
    diviner: str


class FidelityInfo(BaseModel):
    """Human-readable fidelity labels for the UI's model-fidelity panel."""

    astronomy: str = Field(description="e.g. 'Approximate' or 'JPL SPICE'")
    terrain: str = Field(description="e.g. 'Placeholder (spherical)' or 'SLDEM2015'")
    temperature: str = Field(description="e.g. 'Experimental'")
    overall_note: str = Field(description="One-line summary of the weakest link.")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class BodyPosition(BaseModel):
    azimuth_deg: float
    elevation_deg: float
    distance_km: float
    above_horizon: bool


class HorizonPositions(BaseModel):
    sun: BodyPosition
    earth: BodyPosition
    reference: str = "Topocentric body centres; azimuth clockwise from north. Spherical horizon; terrain and ground stations not evaluated."


class VisibilityWindow(BaseModel):
    start_utc: datetime
    end_utc: datetime
    duration_hours: float


class EnvironmentResponse(BaseModel):
    """Full environmental analysis for one coordinate and instant."""

    coordinate: Coordinate
    datetime_utc: datetime
    positions: HorizonPositions | None = None
    terrain: TerrainInfo
    solar: SolarInfo
    thermal: ThermalInfo
    diviner_observation: DivinerObservationInfo | None = Field(
        default=None,
        description="Measured Diviner temperature, or null when no dataset is installed.",
    )
    sun_events: SunEventsInfo | None = None
    sources: SourceInfo
    fidelity: FidelityInfo
    accuracy_notes: list[str] = Field(
        default_factory=list,
        description="Model limitations that apply to this specific result.",
    )


class TimelineSample(BaseModel):
    """One timestep of the environmental timeline."""

    datetime_utc: datetime
    temperature_k: float
    temperature_c: float
    solar_irradiance_w_m2: float
    horizontal_irradiance_w_m2: float
    solar_elevation_deg: float
    solar_azimuth_deg: float
    incidence_angle_deg: float
    illuminated: bool
    solar_earth_overlap: bool | None = None
    earth_elevation_deg: float | None = None
    earth_azimuth_deg: float | None = None
    earth_distance_km: float | None = None
    earth_visible: bool | None = None
    sun_distance_km: float | None = None
    terrain_shadowed: bool | None = None
    solar_horizon_margin_deg: float | None = None
    earth_horizon_margin_deg: float | None = None


class TimelineResponse(BaseModel):
    """Result of ``POST /api/timeline``."""

    coordinate: Coordinate
    center_datetime_utc: datetime
    step_hours: float
    sample_count: int
    samples: list[TimelineSample]
    sources: SourceInfo
    fidelity: FidelityInfo
    synodic_month_hours: float = Field(
        description="Length of one lunar day [h], for sizing the time window."
    )


class ForecastResponse(BaseModel):
    """Hourly or sub-daily prediction series over an explicit UTC interval."""

    coordinate: Coordinate
    start_datetime_utc: datetime
    end_datetime_utc: datetime
    step_hours: float
    sample_count: int
    samples: list[TimelineSample]
    sources: SourceInfo
    fidelity: FidelityInfo
    model_name: str
    model_notes: str
    earth_visibility_windows: list[VisibilityWindow] = Field(default_factory=list)
    earth_visible_hours: float = 0
    sun_above_horizon_hours: float = 0
    horizontal_energy_wh_m2: float = 0
    min_earth_elevation_deg: float = 0
    terrain_visibility_available: bool = False
    sun_visibility_windows: list[VisibilityWindow] = Field(default_factory=list)
    sunlit_hours: float = 0
    longest_darkness_hours: float = 0
    darkness_boundary_limited: bool = False
    position_notes: str = "Earth visibility uses topocentric geometry above a spherical horizon. Earth/Sun disk overlap is flagged at each forecast sample; short eclipses between samples can be missed. Terrain masks, ground stations, antenna patterns and link budgets are not included."


class StatusResponse(BaseModel):
    """Result of ``GET /api/status`` - which providers are actually active."""

    status: Literal["ok"]
    api_version: str
    astronomy_source: str
    astronomy_accuracy: str
    terrain_source: str
    terrain_is_measured: bool
    terrain_shadows_enabled: bool
    diviner_available: bool
    spice_requested: bool
    dem_requested: bool
    diviner_requested: bool


class ErrorResponse(BaseModel):
    """Structured error body returned for 4xx/5xx responses."""

    detail: str = Field(description="Human-readable explanation of the failure.")
    error_type: str = Field(description="Machine-readable error category.")


__all__ = [
    "Coordinate",
    "DivinerObservationInfo",
    "EnvironmentRequest",
    "EnvironmentResponse",
    "ErrorResponse",
    "ForecastRequest",
    "ForecastResponse",
    "FidelityInfo",
    "SolarInfo",
    "SourceInfo",
    "StatusResponse",
    "SunEventsInfo",
    "TerrainInfo",
    "ThermalInfo",
    "TimelineRequest",
    "TimelineResponse",
    "TimelineSample",
]
