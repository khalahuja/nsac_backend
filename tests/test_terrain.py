"""Terrain provider tests.

The critical property under test is honesty: the placeholder provider must
never claim to have measured anything, and shadow state must be "unknown"
rather than "false" when there is no DEM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.geometry import spherical_surface_normal
from app.services.terrain import (
    FlatTerrainProvider,
    RasterTerrainProvider,
    TerrainRasterError,
    create_terrain_provider,
)


@pytest.fixture
def flat() -> FlatTerrainProvider:
    return FlatTerrainProvider()


class TestFlatTerrainProvider:
    def test_labels_itself_as_unmeasured(self, flat: FlatTerrainProvider) -> None:
        assert flat.source == "flat-sphere"
        assert flat.is_measured is False
        assert flat.supports_shadows is False

    def test_elevation_and_slope_are_zero_by_definition(
        self, flat: FlatTerrainProvider
    ) -> None:
        assert flat.get_elevation(-89.4, 137.3) == 0.0
        assert flat.get_slope(-89.4, 137.3) == 0.0

    def test_aspect_is_undefined_on_a_flat_surface(self, flat: FlatTerrainProvider) -> None:
        assert flat.get_aspect(10.0, 20.0) is None

    def test_shadow_state_is_unknown_not_false(self, flat: FlatTerrainProvider) -> None:
        """Without a DEM the honest answer is None, not False."""
        assert flat.is_terrain_shadowed(0.0, 0.0, 180.0, 45.0) is None

    def test_normal_is_the_spherical_radial_direction(
        self, flat: FlatTerrainProvider
    ) -> None:
        for latitude, longitude in [(0.0, 0.0), (90.0, 0.0), (-90.0, 0.0), (45.0, -170.0)]:
            np.testing.assert_allclose(
                flat.get_surface_normal(latitude, longitude),
                spherical_surface_normal(latitude, longitude),
                atol=1e-12,
            )

    def test_sample_bundles_everything_with_provenance(
        self, flat: FlatTerrainProvider
    ) -> None:
        sample = flat.sample(-89.4, 137.3)
        assert sample.elevation_m == 0.0
        assert sample.slope_deg == 0.0
        assert sample.aspect_deg is None
        assert sample.source == "flat-sphere"
        assert sample.is_measured is False
        assert np.linalg.norm(sample.surface_normal) == pytest.approx(1.0)

    @pytest.mark.parametrize("latitude", [-90.0, 0.0, 90.0])
    @pytest.mark.parametrize("longitude", [-180.0, 0.0, 180.0])
    def test_handles_boundary_coordinates(
        self, flat: FlatTerrainProvider, latitude: float, longitude: float
    ) -> None:
        sample = flat.sample(latitude, longitude)
        assert np.linalg.norm(sample.surface_normal) == pytest.approx(1.0)


class TestTerrainProviderFactory:
    def test_defaults_to_the_flat_provider(self) -> None:
        provider = create_terrain_provider(use_dem=False, raster_path=Path("unused.tif"))
        assert isinstance(provider, FlatTerrainProvider)

    def test_falls_back_when_the_raster_is_missing(self, tmp_path: Path) -> None:
        """A missing DEM must degrade gracefully, not crash the app."""
        provider = create_terrain_provider(
            use_dem=True, raster_path=tmp_path / "no_such_dem.tif"
        )
        assert isinstance(provider, FlatTerrainProvider)
        assert provider.is_measured is False


class TestRasterTerrainProvider:
    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(TerrainRasterError):
            RasterTerrainProvider(tmp_path / "absent.tif")

    def test_declares_itself_measured(self) -> None:
        """Class-level contract: raster data is real measurement data."""
        assert RasterTerrainProvider.is_measured is True
