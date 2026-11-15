"""NASA data exploration, measured comparisons, and bounded validation jobs."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator

from .routes import ProvidersDep, SettingsDep
from ..models.schemas import Coordinate
from ..services.nasa_catalog import catalog_status, terrain_patch, ProductError
from ..services.nasa_imagery import layer_capabilities
from ..services.nasa_validation import validate_ephemeris
from ..services.water_evidence import load_evidence, validate_evidence, summarize_evidence
from ..services.forecast import compute_forecast
from ..services.hydrogen import hydrogen_summary

router = APIRouter(prefix="/api/nasa", tags=["nasa-data"])


class PointRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class TerrainRequest(PointRequest):
    radius_m: float = Field(default=500, ge=50, le=5000)
    slope_limit_deg: float = Field(default=12, gt=0, le=45)
    uncertainty_factor: float = Field(default=2, ge=0, le=5)


@router.get("/catalog")
def get_catalog(settings: SettingsDep, providers: ProvidersDep):
    try:
        return {**catalog_status(Path(settings.nasa_data_dir)),
                "diviner_available": providers.diviner.is_available,
                "terrain_source": providers.terrain.source}
    except (ProductError, ValueError, OSError) as error:
        raise HTTPException(503, str(error)) from error


@router.get("/imagery/{layer_id}")
def get_imagery(layer_id: str):
    try:
        return layer_capabilities(layer_id)
    except ProductError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        raise HTTPException(503, "NASA imagery metadata is unavailable. Retry when the service is reachable.") from error


@router.post("/terrain")
def post_terrain(payload: TerrainRequest, settings: SettingsDep, providers: ProvidersDep):
    try:
        return terrain_patch(Path(settings.nasa_data_dir), payload.latitude, payload.longitude,
                             payload.radius_m, payload.slope_limit_deg, payload.uncertainty_factor, providers.terrain)
    except (ProductError, ValueError, OSError, ImportError) as error:
        raise HTTPException(503, f"Terrain product unavailable: {error}") from error


class EvidenceRequest(PointRequest):
    feature_collection: dict[str, Any] | None = None


@router.post("/water")
def post_water(payload: EvidenceRequest, settings: SettingsDep):
    try:
        data = validate_evidence(payload.feature_collection) if payload.feature_collection is not None else load_evidence(Path(settings.nasa_data_dir))
        return {**summarize_evidence(data, payload.latitude, payload.longitude),
                "hydrogen": hydrogen_summary(Path(settings.nasa_data_dir), payload.latitude, payload.longitude)}
    except (ProductError, ValueError, TypeError, KeyError, OSError) as error:
        raise HTTPException(422, f"Invalid water evidence: {error}") from error


class ThermalRequest(PointRequest):
    start_datetime_utc: datetime
    end_datetime_utc: datetime
    step_hours: float = Field(default=1, ge=0.25, le=24)

    @model_validator(mode="after")
    def bounded_range(self):
        # Normalize naive dates consistently with the existing API.
        from datetime import timezone
        for field in ("start_datetime_utc", "end_datetime_utc"):
            value = getattr(self, field)
            setattr(self, field, value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc))
        if not timedelta(0) < self.end_datetime_utc - self.start_datetime_utc <= timedelta(days=31):
            raise ValueError("Temperature comparisons require a period longer than zero and at most 31 days.")
        return self


@router.post("/thermal")
def post_thermal(payload: ThermalRequest, providers: ProvidersDep):
    import math
    forecast = compute_forecast(latitude_deg=payload.latitude, longitude_deg=payload.longitude,
                                start=payload.start_datetime_utc, end=payload.end_datetime_utc,
                                step_hours=payload.step_hours, providers=providers, max_samples=3000)
    samples = []
    residuals = []
    for sample in forecast.samples:
        observation = providers.diviner.get_observed_temperature(payload.latitude, payload.longitude, sample.datetime_utc)
        residual = None if observation is None else sample.temperature_k - observation.temperature_k
        if residual is not None:
            residuals.append(residual)
        samples.append({"datetime_utc": sample.datetime_utc, "model_temperature_k": sample.temperature_k,
                        "observation": asdict(observation) if observation is not None else None,
                        "residual_k": residual})
    return {
        "available": providers.diviner.is_available,
        "model_name": forecast.model_name, "terrain_visibility_available": forecast.terrain_visibility_available,
        "source": providers.diviner.source, "samples": samples, "matched_samples": len(residuals),
        "bias_k": sum(residuals)/len(residuals) if residuals else None,
        "rmse_k": math.sqrt(sum(r*r for r in residuals)/len(residuals)) if residuals else None,
        "local_time_profile": providers.diviner.temperature_profile(payload.latitude, payload.longitude),
        "notes": [
            "Diviner values are historical local-time and season averages, not measurements of the forecast date.",
            "Only matching local-time bins and covered native pixels are compared; gaps remain missing.",
            "Bias and RMSE describe matched climatology differences and do not establish independent model calibration.",
        ],
    }


class ValidationRequest(BaseModel):
    sites: list[Coordinate] = Field(min_length=1, max_length=4)
    times: list[datetime] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def bounded_cases(self):
        if len(self.sites) * len(self.times) > 12:
            raise ValueError("At most 12 site/time cases may be checked per report.")
        return self


@router.post("/validate")
def post_validate(payload: ValidationRequest, providers: ProvidersDep):
    return validate_ephemeris(payload.sites, payload.times, providers)
