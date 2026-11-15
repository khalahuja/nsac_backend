"""Surface product keeps missing DEM values unknown and reports real samples."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from app.services.surface_screen import _offset, compute_surface_screen
from app.services.horizons import HorizonsUnavailable, _angle_row
import pytest
REFERENCE_TIME = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)


def test_surface_screen_without_dem_is_unknown(client):
    response = client.post('/api/surface/screen', json={
        'latitude': -89.54, 'longitude': 0, 'datetime_utc': REFERENCE_TIME.isoformat(), 'radius_m': 500,
    })
    assert response.status_code == 200
    data = response.json()
    assert data['available'] is False
    assert data['safe_fraction'] is None
    assert data['skyline'] == []
    assert data['grid'] == []
    assert data['sunlit_fraction'] is None
    assert data['coverage_fraction'] is None


def test_surface_screen_renders_measured_grid(providers):
    class SyntheticTerrain:
        source = 'synthetic-test-dem'
        is_measured = True
        supports_shadows = True

        def get_elevation(self, latitude, longitude):
            return 100 * (latitude + 89.54)

        def sample(self, latitude, longitude):
            from app.services.terrain import TerrainSample
            from app.services.geometry import spherical_surface_normal
            return TerrainSample(
                elevation_m=self.get_elevation(latitude, longitude), slope_deg=15 if longitude > 0 else 4,
                aspect_deg=90, surface_normal=spherical_surface_normal(latitude, longitude),
                source=self.source, is_measured=True,
            )

        def is_terrain_shadowed(self, latitude, longitude, solar_azimuth_deg, solar_elevation_deg):
            return longitude > 0

    result = compute_surface_screen(-89.54, 0, REFERENCE_TIME, 500, replace(providers, terrain=SyntheticTerrain()))
    assert result['available'] is True
    assert len(result['skyline']) == 36
    assert len(result['grid']) == 169
    assert result['collision_count'] > 0
    assert 0 < result['safe_fraction'] < 1
    assert result['coverage_fraction'] == 1
    assert any(cell['shadowed'] for cell in result['grid'])


def test_surface_offset_crosses_pole_without_clamping():
    latitude, longitude = _offset(-89.9, 0, 0, -15000)
    assert -90 < latitude < -89.5
    assert abs(longitude) == 180


def test_horizons_angle_parser_reads_quantity_four_csv():
    text = 'header\n$$SOE\n2026-Oct-12 00:00, , , 125.5, -3.25,\n$$EOE\n'
    assert _angle_row(text) == (125.5, -3.25)
    with pytest.raises(HorizonsUnavailable):
        _angle_row('No ephemeris available')
