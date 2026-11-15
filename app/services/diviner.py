"""Measured Diviner temperature products with explicit space/time coverage.

The manifest describes prepared PDS polar cumulative products. Local-time
climatologies are historical averages, never observations of a future instant.
"""
from __future__ import annotations
import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..constants import MOON_RADIUS_M
from .nasa_catalog import local_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DivinerObservation:
    temperature_k: float
    local_time_hours: float | None
    dataset: str
    observation_datetime_utc: datetime | None
    comparison_kind: str = "historical-local-time-climatology"
    source_url: str = ""
    reference_period: str = ""
    season: str = "all"
    distance_m: float | None = None


class DivinerDataError(RuntimeError):
    pass


class DivinerProvider(ABC):
    source: str = "unavailable"
    is_available: bool = False

    @abstractmethod
    def get_observed_temperature(self, latitude_deg: float, longitude_deg: float, moment: datetime) -> DivinerObservation | None:
        pass

    def temperature_profile(self, latitude_deg: float, longitude_deg: float) -> list[dict]:
        return []


class NullDivinerProvider(DivinerProvider):
    def get_observed_temperature(self, latitude_deg: float, longitude_deg: float, moment: datetime) -> None:
        return None


class GriddedDivinerProvider(DivinerProvider):
    is_available = True

    def __init__(self, data_directory: Path, astronomy=None):
        self.astronomy = astronomy
        manifest_path = Path(data_directory) / "manifest.json"
        if not manifest_path.is_file():
            raise DivinerDataError("Diviner manifest.json is missing. See data/nasa/README.md.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1:
            raise DivinerDataError("Unsupported Diviner manifest version.")
        self.products = []
        from scipy.spatial import cKDTree
        for metadata in manifest.get("products", []):
            if metadata.get("kind") != "local_time_climatology" or metadata.get("units") != "K":
                raise DivinerDataError("Diviner products must declare local_time_climatology and Kelvin units.")
            if metadata.get("season") not in {"summer", "winter", "all"}:
                raise DivinerDataError("Diviner products must declare their season.")
            start, end = metadata["local_time_min_hours"], metadata["local_time_max_hours"]
            if not 0 <= start < end <= 24:
                raise DivinerDataError("Diviner local-time bin must lie within [0,24].")
            if not str(metadata.get("source_url", "")).startswith("https://"):
                raise DivinerDataError("Diviner products require an HTTPS source link.")
            path = local_path(Path(data_directory), metadata["file"])
            if not path.is_file():
                continue
            # allow_pickle=False prevents code execution through object arrays.
            with np.load(path, allow_pickle=False) as arrays:
                x, y, t = (np.asarray(arrays[k], dtype=float) for k in ("x_m", "y_m", "temperature_k"))
            if x.ndim != 1 or x.shape != y.shape or x.shape != t.shape:
                raise DivinerDataError("Diviner arrays must have matching one-dimensional shapes.")
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & (t > 0) & (t < 500)
            if not valid.any():
                continue
            self.products.append((metadata, cKDTree(np.column_stack((x[valid], y[valid]))), t[valid]))
        if not self.products:
            raise DivinerDataError("No usable Diviner products are installed.")
        self.source = "PDS Diviner: " + ", ".join(dict.fromkeys(m["dataset"] for m, _, _ in self.products))

    @staticmethod
    def _xy(lat: float, lon: float) -> tuple[float, float]:
        # Native PDS PCP coordinates: normalized polar stereographic x/y,
        # converted to metres during preparation; north-positive y at 0 E.
        rho = 2 * MOON_RADIUS_M * math.tan(math.radians(90 + lat) / 2)
        angle = math.radians(lon)
        return rho * math.sin(angle), rho * math.cos(angle)

    def _sample(self, product, latitude: float, longitude: float):
        metadata, tree, temperatures = product
        if latitude >= 0:
            return None
        distance, index = tree.query(self._xy(latitude, longitude))
        # Never bridge missing pixels: only accept a containing/nearest native cell.
        if distance > float(metadata["resolution_m"]) / math.sqrt(2) + 1e-6:
            return None
        return float(temperatures[index]), float(distance)

    def temperature_profile(self, latitude_deg: float, longitude_deg: float) -> list[dict]:
        result = []
        for product in self.products:
            value = self._sample(product, latitude_deg, longitude_deg)
            if value is None:
                continue
            m = product[0]
            result.append({
                "temperature_k": value[0], "distance_m": value[1],
                "local_time_min_hours": m["local_time_min_hours"], "local_time_max_hours": m["local_time_max_hours"],
                "season": m["season"], "dataset": m["dataset"], "source_url": m["source_url"],
                "reference_period": m["reference_period"], "kind": "historical-local-time-climatology",
            })
        return sorted(result, key=lambda p: (p["season"], p["local_time_min_hours"]))

    def get_observed_temperature(self, latitude_deg: float, longitude_deg: float, moment: datetime) -> DivinerObservation | None:
        if self.astronomy is None:
            return None
        geometry = self.astronomy.get_geometry(moment)
        sun = geometry.sun_direction_body
        subsolar_lon = math.degrees(math.atan2(float(sun[1]), float(sun[0])))
        local_time = (12 + (longitude_deg - subsolar_lon) / 15) % 24
        season = "summer" if sun[2] < 0 else "winter"
        for product in self.products:
            m = product[0]
            if m["season"] not in {"all", season} or not m["local_time_min_hours"] <= local_time < m["local_time_max_hours"]:
                continue
            sample = self._sample(product, latitude_deg, longitude_deg)
            if sample is not None:
                return DivinerObservation(sample[0], local_time, m["dataset"], None,
                                          source_url=m["source_url"], reference_period=m["reference_period"],
                                          season=m["season"], distance_m=sample[1])
        return None


def create_diviner_provider(*, use_diviner: bool, data_directory: Path, astronomy=None) -> DivinerProvider:
    if not use_diviner:
        return NullDivinerProvider()
    try:
        return GriddedDivinerProvider(data_directory, astronomy)
    except (DivinerDataError, ValueError, KeyError, OSError, ImportError) as error:
        logger.warning("Diviner unavailable (%s); comparisons will remain missing.", error)
        return NullDivinerProvider()
