"""Regression checks for NASA data coverage, provenance and scientific semantics."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.api.nasa import ValidationRequest
from app.services.diviner import GriddedDivinerProvider
from app.services.forecast import compute_forecast
from app.services.nasa_catalog import ProductError, RasterSamples, local_path, terrain_patch
from app.services.nasa_validation import validate_ephemeris
from app.services.terrain import FlatTerrainProvider
from app.services.water_evidence import validate_evidence, summarize_evidence


def test_terrain_shadows_change_power_thermal_and_contacts(providers):
    class ObstructedTerrain(FlatTerrainProvider):
        is_measured = True
        supports_shadows = True
        source = "test-measured-obstruction"
        def horizon_profile(self, lat, lon, azimuths, observer_height_m=0):
            return [85.0] * len(azimuths)
    start = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    direction, _ = providers.astronomy.get_sun_vector_from_moon(start)
    # Place the observer 30 degrees east of the subsolar point.
    lat = math.degrees(math.asin(float(direction[2])))
    lon = (math.degrees(math.atan2(direction[1], direction[0])) + 30 + 180) % 360 - 180
    args = dict(latitude_deg=lat, longitude_deg=lon, start=start, end=start + timedelta(hours=6), step_hours=1)
    reference = compute_forecast(**args, providers=providers)
    masked = compute_forecast(**args, providers=replace(providers, terrain=ObstructedTerrain()))
    assert any(s.illuminated for s in reference.samples)
    assert masked.terrain_visibility_available
    assert all(s.terrain_shadowed and not s.illuminated for s in masked.samples)
    assert all(s.solar_irradiance_w_m2 == s.horizontal_irradiance_w_m2 == 0 for s in masked.samples)
    assert masked.horizontal_energy_wh_m2 == 0
    assert masked.longest_darkness_hours == 6 and masked.darkness_boundary_limited
    assert masked.samples[-1].temperature_k < reference.samples[-1].temperature_k
    assert all(s.earth_visible == (s.earth_horizon_margin_deg > 0) for s in masked.samples)


def test_missing_skyline_remains_explicit(providers):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    forecast = compute_forecast(latitude_deg=0, longitude_deg=0, start=start, end=start + timedelta(hours=1), step_hours=1, providers=providers)
    assert forecast.terrain_visibility_available is False
    assert all(s.terrain_shadowed is None for s in forecast.samples)


def test_raster_units_coverage_and_uncertainty_are_not_guessed(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    from app.services.terrain import FlatTerrainProvider
    crs = CRS.from_proj4("+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs")
    files = {}
    for field, number in (("elevation", .25), ("slope", 10), ("height_uncertainty", .4), ("slope_uncertainty", 2)):
        path = tmp_path / (field + ".tif")
        grid = np.full((10, 10), number, dtype="float32")
        with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=1, dtype="float32", transform=from_origin(-25, 25, 5, 5), crs=crs, nodata=-9999) as ds:
            ds.write(grid, 1)
        files[field] = path.name
    region = {**files, "id": "test", "title": "Test native units", "resolution_m": 5, "frame": "MOON_ME", "ephemeris": "DE421", "height_units": "m", "slope_units": "deg", "source_url": "https://pgda.gsfc.nasa.gov/products/78"}
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 1, "regions": [region]}))
    with RasterSamples(tmp_path / "elevation.tif") as sampler:
        assert sampler.sample([(-90, 0), (0, 0)]) == [.25, None]
    patch = terrain_patch(tmp_path, -90, 0, 50, 12, 2, FlatTerrainProvider())
    assert patch["high_resolution"] and patch["resolution_m"] == 5
    middle = patch["cells"][312]
    assert middle["elevation_m"] == .25, "Low elevations must not be guessed to be kilometres."
    assert middle["screen"] == "uncertain", "10 + 2*2 exceeds the 12 degree screening limit."
    assert any(c["screen"] == "unknown" for c in patch["cells"]), "Missing coverage cannot be safe terrain."


def test_dataset_paths_cannot_escape_directory(tmp_path):
    with pytest.raises(ProductError):
        local_path(tmp_path, "../secret.tif")
    assert local_path(tmp_path, "valid.tif") == tmp_path / "valid.tif"


def test_diviner_only_matches_native_space_time_and_season(tmp_path):
    lat, lon = -89.0, 0
    x, y = GriddedDivinerProvider._xy(lat, lon)
    np.savez(tmp_path / "native.npz", x_m=[x], y_m=[y], temperature_k=[75])
    metadata = {"file": "native.npz", "dataset": "PDS-test", "units": "K", "kind": "local_time_climatology",
                "season": "summer", "local_time_min_hours": 0, "local_time_max_hours": .25,
                "resolution_m": 240, "source_url": "https://pds-geosciences.wustl.edu/", "reference_period": "2009-2019"}
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 1, "products": [metadata]}))
    sun = np.array([-1.0, 0.0, -.05])
    astronomy = SimpleNamespace(get_geometry=lambda moment: SimpleNamespace(sun_direction_body=sun))
    provider = GriddedDivinerProvider(tmp_path, astronomy)
    time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    observation = provider.get_observed_temperature(lat, lon, time)
    assert observation.temperature_k == 75 and observation.observation_datetime_utc is None
    assert observation.comparison_kind == "historical-local-time-climatology"
    assert provider.get_observed_temperature(lat + 1, lon, time) is None
    assert provider.get_observed_temperature(lat, 10, time) is None
    sun[2] = .05
    assert provider.get_observed_temperature(lat, lon, time) is None
    assert provider.temperature_profile(lat, lon)[0]["temperature_k"] == 75


def _evidence(kind="ice_detection", geometry=None):
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": geometry or {"type": "Point", "coordinates": [179, -89]},
        "properties": {"name": "Sourced feature", "evidence_type": kind, "source_url": "https://example.test/source"},
    }]}


def test_water_evidence_types_do_not_turn_proxies_into_detections():
    data = validate_evidence(_evidence("resource_potential"))
    result = summarize_evidence(data, -89, -179)
    assert result["counts"]["ice_detection"] == 0
    assert result["counts"]["resource_potential"] == 1
    assert result["features"][0]["distance_km"] < 2, "Dateline crossing is a short polar distance."
    with pytest.raises(ProductError):
        validate_evidence(_evidence("probably_ice"))
    bad = _evidence()
    bad["features"][0]["properties"]["source_url"] = "javascript:alert(1)"
    with pytest.raises(ProductError):
        validate_evidence(bad)


def test_water_polygon_holes_and_missing_coverage():
    geometry = {"type": "Polygon", "coordinates": [
        [[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]],
        [[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2],[-.2,-.2]],
    ]}
    data = validate_evidence(_evidence(geometry=geometry))
    assert summarize_evidence(data, .5, .5)["features"][0]["contains_selected_site"]
    hole = summarize_evidence(data, 0, 0)["features"][0]
    assert not hole["contains_selected_site"] and hole["distance_km"] > 0


def test_validation_preserves_partial_jpl_failures_and_wraps_azimuth(monkeypatch, providers):
    from app.models.schemas import Coordinate
    from app.services.horizons import HorizonsUnavailable
    from app.services.positions import body_position
    time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    geometry = providers.astronomy.get_geometry(time)
    sun = body_position(0,0,geometry.sun_direction_body,geometry.sun_distance_km)
    def reference(lat, lon, moment, target):
        if target == "399":
            raise HorizonsUnavailable("Reference offline")
        return {"azimuth_deg": (sun.azimuth_deg + 359) % 360, "elevation_deg": sun.elevation_deg, "source": "mock JPL"}
    monkeypatch.setattr("app.services.nasa_validation.observer_angles", reference)
    report = validate_ephemeris([Coordinate(latitude=0, longitude=0)], [time], providers)
    assert report["successful_body_checks"] == 1
    assert report["results"][0]["bodies"][0]["azimuth_delta_deg"] == pytest.approx(1)
    assert report["results"][0]["bodies"][1]["reference"] is None


def test_nasa_routes_and_input_limits(client):
    assert client.get("/api/nasa/catalog").status_code == 200
    assert client.post("/api/nasa/terrain", json={"latitude": 91, "longitude": 0}).status_code == 422
    result = client.post("/api/nasa/terrain", json={"latitude": 0, "longitude": 0})
    assert result.status_code == 200 and result.json()["available"] is False
    result = client.post("/api/nasa/water", json={"latitude": -89, "longitude": 179, "feature_collection": _evidence()})
    assert result.status_code == 200 and result.json()["features"][0]["distance_km"] == 0
    invalid = client.post("/api/nasa/thermal", json={"latitude": 0, "longitude": 0, "start_datetime_utc": "2026-01-01T00:00:00Z", "end_datetime_utc": "2026-03-01T00:00:00Z"})
    assert invalid.status_code == 422
    too_many = {"sites": [{"latitude":0,"longitude":0}]*4, "times": ["2026-01-01T00:00:00Z"]*4}
    assert client.post("/api/nasa/validate", json=too_many).status_code == 422


def test_native_hydrogen_bins_preserve_units_and_polar_bounds(tmp_path):
    from app.services.hydrogen import hydrogen_summary
    (tmp_path / "hydrogenlow.txt").write_text(
        "# H ppm by weight\n-90,-89,-180,180,146.14\n-89,-87,-180,-135,126.91\n89,90,-180,180,75.0\n"
    )
    polar = hydrogen_summary(tmp_path, -89.4398, -137.17)
    assert polar["selected_value_ppm"] == 146.14
    assert polar["units"] == "ppm by weight"
    assert polar["selected_bounds"] == [-90,-89,-180,180]
    assert len(polar["cells"]) == 2
    assert hydrogen_summary(tmp_path, 90, 180)["selected_value_ppm"] == 75
    assert hydrogen_summary(tmp_path, -88, 0)["selected_value_ppm"] is None


def test_horizons_comparison_preserves_subminute_timestamp(monkeypatch):
    from app.services.horizons import observer_angles
    from urllib.parse import parse_qs, urlparse
    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"result": "$$SOE\n date, 12.0, 34.0\n$$EOE"}).encode()
    def open_url(url, timeout):
        captured.update(parse_qs(urlparse(url).query))
        return Response()
    observer_angles.cache_clear()
    monkeypatch.setattr("app.services.horizons.urlopen", open_url)
    time = datetime(2027,1,1,0,0,37,123456,tzinfo=timezone.utc)
    assert observer_angles(0,0,time,"10")["elevation_deg"] == 34
    assert "00:00:37.123456" in captured["TLIST"][0]
    observer_angles.cache_clear()
