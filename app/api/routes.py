"""HTTP routes.

Handlers stay thin: validate (via Pydantic), delegate to the service layer,
translate domain exceptions into structured HTTP errors. No science lives here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..models.schemas import (
    EnvironmentRequest,
    EnvironmentResponse,
    ErrorResponse,
    ForecastRequest,
    ForecastResponse,
    StatusResponse,
    TimelineRequest,
    TimelineResponse,
)
from ..services.environment import ScienceComputationError, analyse_environment
from ..services.operations import OperationsRequest, plan_operations
from ..services.forecast import ForecastError, compute_forecast
from ..services.providers import ScienceProviders
from ..services.terrain import FlatTerrainProvider
from ..services.surface_screen import compute_surface_screen, horizon_or_none
from ..services.horizons import HorizonsUnavailable, observer_angles
from ..services.timeline import TimelineError, compute_timeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["lunar-environment"])

#: Curated lunar feature catalogue shipped with the repo. Used by the frontend
#: to label the procedurally generated Moon texture; it is cartographic
#: reference data, not a measurement product.
_CRATERS_PATH = Path(__file__).resolve().parents[2] / "data" / "moon_craters.json"

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse, "description": "Invalid request parameters"},
    500: {"model": ErrorResponse, "description": "Scientific calculation failed"},
}


def get_providers(request: Request) -> ScienceProviders:
    """FastAPI dependency returning the providers built at startup."""
    providers: ScienceProviders | None = getattr(request.app.state, "providers", None)
    if providers is None:  # pragma: no cover - only if startup failed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Science providers are not initialised.",
        )
    return providers


ProvidersDep = Annotated[ScienceProviders, Depends(get_providers)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post("/mission/operations", summary="Rank surface landing dates against mission requirements")
def post_operations(payload: OperationsRequest, providers: ProvidersDep):
    return plan_operations(payload, providers)


class SurfaceScreenRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    datetime_utc: datetime
    radius_m: float = Field(default=500, ge=50, le=5000)


@router.post("/surface/screen", summary="DEM skyline, moving shadow and landing footprint screen")
def post_surface_screen(payload: SurfaceScreenRequest, providers: ProvidersDep) -> dict[str, Any]:
    return compute_surface_screen(payload.latitude, payload.longitude, payload.datetime_utc, payload.radius_m, providers)


class HorizonRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    step_deg: float = Field(default=2, ge=0.5, le=30)


@router.post("/surface/skyline", summary="Terrain horizon around a site from the installed DEM")
def post_surface_skyline(payload: HorizonRequest, providers: ProvidersDep) -> dict[str, Any]:
    """Horizon elevation angle per azimuth, out to 150 km; empty when no polar DEM is loaded."""
    count = int(round(360 / payload.step_deg))
    azimuths = [i * 360 / count for i in range(count)]
    profile = horizon_or_none(providers.terrain, payload.latitude, payload.longitude, azimuths)
    elevation = providers.terrain.get_elevation(payload.latitude, payload.longitude) if providers.terrain.is_measured else None
    return {
        "available": profile is not None,
        "terrain_source": providers.terrain.source,
        "site_elevation_m": elevation,
        "skyline": [] if profile is None else [{"azimuth_deg": a, "elevation_deg": e} for a, e in zip(azimuths, profile)],
    }


@router.post("/surface/horizons", summary="Verify Sun and Earth observer angles with JPL Horizons")
def post_surface_horizons(payload: SurfaceScreenRequest) -> dict[str, Any]:
    try:
        return {
            "sun": observer_angles(payload.latitude, payload.longitude, payload.datetime_utc, "10"),
            "earth": observer_angles(payload.latitude, payload.longitude, payload.datetime_utc, "399"),
        }
    except HorizonsUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post(
    "/environment",
    response_model=EnvironmentResponse,
    responses=ERROR_RESPONSES,
    summary="Full environmental analysis at a lunar coordinate and time",
)
def post_environment(
    payload: EnvironmentRequest, providers: ProvidersDep
) -> EnvironmentResponse:
    """Compute solar geometry, illumination, terrain and temperature.

    Every value carries provenance. Fields that would require a NASA dataset
    that is not installed are returned as ``null`` rather than estimated.
    """
    try:
        return analyse_environment(
            latitude_deg=payload.latitude,
            longitude_deg=payload.longitude,
            moment=payload.datetime_utc,
            providers=providers,
            include_sun_events=payload.include_sun_events,
        )
    except ScienceComputationError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)
        ) from error


@router.post(
    "/timeline",
    response_model=TimelineResponse,
    responses=ERROR_RESPONSES,
    summary="Solar irradiance and temperature over a time window",
)
def post_timeline(
    payload: TimelineRequest, providers: ProvidersDep, settings: SettingsDep
) -> TimelineResponse:
    """Sample the environment around a central instant.

    The window may span up to one full lunar day (~709 h) in either direction.
    Requests that would exceed the sample cap are rejected with a 422 that
    explains how to fix them, rather than silently coarsened.
    """
    try:
        return compute_timeline(
            latitude_deg=payload.latitude,
            longitude_deg=payload.longitude,
            center=payload.center_datetime_utc,
            hours_before=payload.hours_before,
            hours_after=payload.hours_after,
            step_hours=payload.step_hours,
            providers=providers,
            max_samples=settings.max_timeline_samples,
        )
    except TimelineError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except Exception as error:  # noqa: BLE001 - surfaced as a 500 with context
        logger.exception("Timeline computation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Timeline computation failed: {error}",
        ) from error


def _forecast_from_payload(
    payload: ForecastRequest, providers: ScienceProviders, settings: Settings, max_samples: int | None = None
) -> ForecastResponse:
    try:
        return compute_forecast(
            latitude_deg=payload.latitude,
            longitude_deg=payload.longitude,
            start=payload.start_datetime_utc,
            end=payload.end_datetime_utc,
            step_hours=payload.step_hours,
            providers=providers,
            max_samples=max_samples or settings.max_timeline_samples,
            min_earth_elevation_deg=payload.min_earth_elevation_deg,
        )
    except ForecastError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except Exception as error:  # noqa: BLE001
        logger.exception("Forecast computation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast computation failed: {error}",
        ) from error


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    responses=ERROR_RESPONSES,
    summary="Environmental forecast over an explicit UTC date range",
)
def post_forecast(
    payload: ForecastRequest, providers: ProvidersDep, settings: SettingsDep
) -> ForecastResponse:
    return _forecast_from_payload(payload, providers, settings)


@router.post(
    "/export/xlsx",
    responses=ERROR_RESPONSES,
    summary="Download the environmental forecast as an Excel workbook",
)
def post_export_xlsx(
    payload: ForecastRequest, providers: ProvidersDep, settings: SettingsDep
) -> StreamingResponse:
    # Exports match the website's chunked forecasts (up to ~2.3 years hourly).
    forecast = _forecast_from_payload(payload, providers, settings, settings.max_export_samples)
    from ..services.workbook import forecast_workbook
    stream = forecast_workbook(forecast, providers, payload.display_timezone)
    filename = (
        f"lunar-forecast-{payload.latitude:.2f}-{payload.longitude:.2f}-"
        f"{payload.start_datetime_utc:%Y%m%d}.xlsx"
    )
    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Which data providers are actually active",
)
def get_status(providers: ProvidersDep, settings: SettingsDep) -> StatusResponse:
    """Report the live provider selection.

    Lets the UI state plainly whether it is showing NASA-derived values or
    model approximations - including the case where a dataset was *requested*
    but could not be loaded.
    """
    return StatusResponse(
        status="ok",
        api_version=settings.api_version,
        astronomy_source=providers.astronomy.source,
        astronomy_accuracy=providers.astronomy.accuracy_note,
        terrain_source=providers.terrain.source,
        terrain_is_measured=not isinstance(providers.terrain, FlatTerrainProvider),
        terrain_shadows_enabled=providers.terrain.supports_shadows,
        diviner_available=providers.diviner.is_available,
        spice_requested=settings.use_spice,
        dem_requested=settings.use_terrain_dem,
        diviner_requested=settings.use_diviner,
    )


@router.get("/health", summary="Liveness probe")
def get_health() -> dict[str, str]:
    """Trivial liveness check for Docker and load balancers."""
    return {"status": "ok"}


@router.get(
    "/moon/features",
    summary="Curated lunar surface feature catalogue",
)
def get_moon_features() -> list[dict[str, Any]]:
    """Named craters and maria used to annotate the 3D Moon.

    This is a small hand-curated reference list of well-known features with
    published coordinates - not a derived measurement product.
    """
    try:
        with _CRATERS_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logger.warning("Feature catalogue missing at %s", _CRATERS_PATH)
        return []
    except json.JSONDecodeError as error:
        logger.error("Feature catalogue is not valid JSON: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lunar feature catalogue is corrupt.",
        ) from error


__all__ = ["router", "get_providers"]
