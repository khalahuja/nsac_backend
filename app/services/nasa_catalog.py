"""Local NASA products and a fixed catalogue of verified public map services.

Paths are operator configuration, never client-supplied filesystem paths.
Raster values are read in native units; rendered WMTS colours are never measurements.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..constants import MOON_RADIUS_M

PGDA = "https://pgda.gsfc.nasa.gov/products/78"
TREK = "https://trek.nasa.gov/tiles/Moon/"
LAYERS = {
    "lroc": {
        "title": "LROC WAC global mosaic", "kind": "imagery", "projection": "eq",
        "endpoint": TREK + "EQ/LRO_WAC_Mosaic_Global_303ppd_v02",
        "source_url": "https://trek.nasa.gov/tiles/apidoc/trekAPI.html?body=moon",
        "note": "Approximately 100 m/pixel global imagery. Image shadows record acquisition conditions, not the selected forecast time.",
    },
    "cold": {
        "title": "Diviner winter maximum below 110 K", "kind": "cold_conditions", "projection": "south",
        "endpoint": TREK + "SP/DIVINER_winter_Max_Temp_SP_below110K_240mp_RGB",
        "source_url": "https://pds-geosciences.wustl.edu/missions/lro/diviner.htm",
        "note": "Historical winter-temperature product. Cold conditions are not an ice detection or an annual stability measurement.",
    },
    "lroc-south": {
        "title": "LROC WAC south polar mosaic", "kind": "imagery", "projection": "south",
        "endpoint": TREK + "SP/LRO_WAC_Mosaic_SPole60_100mp",
        "source_url": "https://trek.nasa.gov/moon/",
        "note": "LROC south-polar mosaic, native 100 m imagery; displayed detail is limited by the available service zoom levels. Acquisition shadows are not current illumination.",
    },
}


class ProductError(ValueError):
    pass


def local_path(directory: Path, filename: str) -> Path:
    directory = directory.resolve()
    target = (directory / filename).resolve()
    if not target.is_relative_to(directory):
        raise ProductError("Dataset path must remain inside the configured data directory.")
    return target


def read_catalog(directory: Path) -> dict:
    path = directory / "catalog.json"
    if not path.is_file():
        return {"regions": []}
    if path.stat().st_size > 1_000_000:
        raise ProductError("NASA dataset catalogue exceeds 1 MB.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("regions"), list):
        raise ProductError("NASA dataset catalogue must use version 1 with a regions list.")
    for region in data["regions"]:
        if region.get("height_units") != "m" or region.get("slope_units") != "deg":
            raise ProductError("Regional elevation/uncertainty must be in m and slopes/uncertainty in deg.")
        if region.get("frame") != "MOON_ME":
            raise ProductError("Regional products must declare the MOON_ME lunar frame.")
        if not str(region.get("source_url", "")).startswith("https://"):
            raise ProductError("Regional products require an HTTPS provenance link.")
        for name in ("elevation", "slope", "height_uncertainty", "slope_uncertainty"):
            if region.get(name):
                local_path(directory, region[name])
    return data


class RasterSamples:
    """Windowed sampling with lunar CRS checks and explicit no-data handling."""

    def __init__(self, path: Path):
        import rasterio
        from rasterio.crs import CRS

        self.dataset = rasterio.open(path)
        crs = self.dataset.crs
        if crs is None:
            self.close()
            raise ProductError("Raster has no coordinate reference system.")
        ellipsoid = crs.to_dict()
        radius = ellipsoid.get("R", ellipsoid.get("a"))
        # WKT-derived CRS can expose a/b rather than R.
        if radius is None or abs(float(radius) - MOON_RADIUS_M) > 10:
            self.close()
            raise ProductError("Raster CRS must use the 1737.4 km lunar reference sphere.")
        self.geographic = CRS.from_proj4(f"+proj=longlat +R={MOON_RADIUS_M} +no_defs")

    def close(self):
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def sample(self, points: list[tuple[float, float]]) -> list[float | None]:
        from rasterio.warp import transform

        ds = self.dataset
        lats, lons = zip(*points)
        if ds.crs.is_geographic:
            lons = [lon % 360 if ds.bounds.left >= 0 else (lon + 180) % 360 - 180 for lon in lons]
            xs, ys = list(lons), list(lats)
        else:
            xs, ys = transform(self.geographic, ds.crs, list(lons), list(lats))
        results = []
        coordinates = list(zip(xs, ys))
        for (x, y), value in zip(coordinates, ds.sample(coordinates, indexes=1, masked=True)):
            if not (ds.bounds.left <= x < ds.bounds.right and ds.bounds.bottom < y <= ds.bounds.top):
                results.append(None)
                continue
            raw = value[0]
            if np.ma.is_masked(raw) or not math.isfinite(float(raw)):
                results.append(None)
                continue
            scaled = float(raw) * ds.scales[0] + ds.offsets[0]
            results.append(scaled if math.isfinite(scaled) else None)
        return results


def region_values(directory: Path, region: dict, points: list[tuple[float, float]]) -> dict:
    result = {}
    for field in ("elevation", "slope", "height_uncertainty", "slope_uncertainty"):
        name = region.get(field)
        if not name or not local_path(directory, name).is_file():
            result[field] = [None] * len(points)
            continue
        with RasterSamples(local_path(directory, name)) as raster:
            result[field] = raster.sample(points)
        if field != "elevation":
            result[field] = [v if v is not None and v >= 0 else None for v in result[field]]
    return result


def select_region(directory: Path, lat: float, lon: float) -> dict | None:
    regions = sorted(read_catalog(directory)["regions"], key=lambda r: r.get("resolution_m", float("inf")))
    for region in regions:
        path = local_path(directory, region["elevation"])
        if not path.is_file():
            continue
        with RasterSamples(path) as raster:
            if raster.sample([(lat, lon)])[0] is not None:
                return region
    return None


def terrain_patch(directory: Path, lat: float, lon: float, radius: float, slope_limit: float, uncertainty_factor: float, base_terrain) -> dict:
    from .surface_screen import _offset

    region = select_region(directory, lat, lon)
    size = 25
    points = [_offset(lat, lon, (x - 12) * radius / 12, (12 - y) * radius / 12) for y in range(size) for x in range(size)]
    if region is not None:
        values = region_values(directory, region, points)
        source, resolution = region["title"], region["resolution_m"]
    else:
        values = {
            "elevation": [base_terrain.get_elevation(*p) if base_terrain.is_measured else None for p in points],
            "slope": [base_terrain.get_slope(*p) if base_terrain.is_measured else None for p in points],
            "height_uncertainty": [None] * len(points), "slope_uncertainty": [None] * len(points),
        }
        source = base_terrain.source
        resolution = getattr(base_terrain, "_pixel_m", None)
    cells = []
    for index, (a, b) in enumerate(points):
        slope, error = values["slope"][index], values["slope_uncertainty"][index]
        conservative = None if slope is None or error is None else slope + uncertainty_factor * error
        cells.append({
            "x": index % size, "y": index // size, "lat": a, "lon": b,
            "elevation_m": values["elevation"][index], "slope_deg": slope,
            "height_uncertainty_m": values["height_uncertainty"][index], "slope_uncertainty_deg": error,
            "screen": "unknown" if slope is None else "hazard" if slope > slope_limit else "uncertain" if conservative is None else "within_limit" if conservative <= slope_limit else "uncertain",
        })
    covered = sum(c["slope_deg"] is not None for c in cells)
    return {
        "available": covered > 0, "high_resolution": region is not None, "source": source,
        "resolution_m": resolution, "source_url": region["source_url"] if region else "https://pds-geosciences.wustl.edu/missions/lro/lola.htm",
        "frame": region.get("frame") if region else "MOON_ME", "ephemeris_frame": region.get("ephemeris") if region else None,
        "size": size, "radius_m": radius, "slope_limit_deg": slope_limit, "uncertainty_factor": uncertainty_factor,
        "coverage_fraction": covered / len(cells), "cells": cells,
        "notes": [
            "Slope screen only; unresolved rocks, soil bearing strength and descent dynamics are not evaluated.",
            "An uncertainty margin is slope plus the selected multiple of published RMS slope error; it is not a landing-success probability.",
            "Missing uncertainty is unknown. The display grid samples native pixels and does not increase source resolution.",
            "Regional products use their declared Moon ME ephemeris realization; no DE421-to-DE440 frame registration is applied.",
        ],
    }


def catalog_status(directory: Path) -> dict:
    regions = []
    for item in read_catalog(directory)["regions"]:
        regions.append({key: item.get(key) for key in ("id", "title", "source_url", "resolution_m", "frame", "ephemeris")} | {
            "installed": local_path(directory, item["elevation"]).is_file(),
            "uncertainty_available": all(item.get(k) and local_path(directory, item[k]).is_file() for k in ("height_uncertainty", "slope_uncertainty")),
        })
    return {"regions": regions, "layers": [{"id": key, **layer} for key, layer in LAYERS.items()]}
