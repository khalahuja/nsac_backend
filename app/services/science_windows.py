"""Shared environmental constraints and continuous interval matching."""
from datetime import datetime
from math import isfinite
from pydantic import BaseModel, Field, ConfigDict, model_validator

class Criteria(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    minTemperature: float = -273.15
    maxTemperature: float = 200
    minIrradiance: float = Field(default=0, ge=0)
    maxIrradiance: float = Field(default=2000, ge=0)
    minEarthElevation: float = Field(default=0, ge=-90, le=90)
    maxEarthElevation: float = Field(default=90, ge=-90, le=90)
    requireEarth: bool = False
    requireSun: bool = False
    minHours: float = Field(default=0, ge=0, le=1440)
    minEarthHours: float = Field(default=0, ge=0, le=1440)

    @model_validator(mode="after")
    def ordered(self):
        for low, high in [(self.minTemperature, self.maxTemperature), (self.minIrradiance, self.maxIrradiance), (self.minEarthElevation, self.maxEarthElevation)]:
            if low > high:
                raise ValueError("Minimum must not exceed maximum")
        return self


def matching_intervals(samples, criteria: Criteria, step_hours=1):
    """Count elapsed time, break on gaps, and require all filters in the same interval."""
    intervals = []
    first = last = earth_start = None
    longest_earth = 0.0
    def close():
        if first is not None and last is not None:
            hours = (last - first).total_seconds() / 3600
            if hours > 0 and hours >= criteria.minHours and longest_earth >= criteria.minEarthHours:
                intervals.append((first, last))
    for sample in samples:
        time = sample.datetime_utc
        if isinstance(time, str):
            time = datetime.fromisoformat(time.replace("Z", "+00:00"))
        earth = (sample.earth_visible and isfinite(sample.earth_elevation_deg)
                 and criteria.minEarthElevation <= sample.earth_elevation_deg <= criteria.maxEarthElevation)
        passes = (isfinite(sample.temperature_c) and criteria.minTemperature <= sample.temperature_c <= criteria.maxTemperature
                  and isfinite(sample.horizontal_irradiance_w_m2) and criteria.minIrradiance <= sample.horizontal_irradiance_w_m2 <= criteria.maxIrradiance
                  and (not criteria.requireSun or sample.illuminated and not getattr(sample, "solar_earth_overlap", False))
                  and (not criteria.requireEarth or earth))
        gap = (time - last).total_seconds() / 3600 if last is not None else 0
        if not passes or last is not None and (gap <= 0 or gap > step_hours * 1.01):
            close()
            first = last = earth_start = None
            longest_earth = 0.0
        if not passes:
            continue
        if first is None:
            first = time
        last = time
        if earth:
            if earth_start is None:
                earth_start = time
            longest_earth = max(longest_earth, (time - earth_start).total_seconds() / 3600)
        else:
            earth_start = None
    close()
    return intervals
