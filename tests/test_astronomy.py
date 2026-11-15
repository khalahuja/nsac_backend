"""Astronomy provider tests.

Validates the analytic ephemeris against independently known facts rather than
against its own output: orbital distances, the synodic period, frame
orthonormality, and the requirement that the app still works without SPICE.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from app.constants import SYNODIC_MONTH_DAYS
from app.services.astronomy import (
    J2000_JD,
    SimpleAstronomyProvider,
    create_astronomy_provider,
    datetime_to_julian_date,
    days_since_j2000,
)
from app.services.geometry import cartesian_to_lat_lon

REFERENCE_TIME = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def provider() -> SimpleAstronomyProvider:
    return SimpleAstronomyProvider()


class TestJulianDate:
    def test_j2000_epoch(self) -> None:
        """2000-01-01T12:00:00 UTC is JD 2451545.0 by definition."""
        epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert datetime_to_julian_date(epoch) == pytest.approx(J2000_JD, abs=1e-6)
        assert days_since_j2000(epoch) == pytest.approx(0.0, abs=1e-6)

    def test_unix_epoch(self) -> None:
        """1970-01-01T00:00:00 UTC is JD 2440587.5."""
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        assert datetime_to_julian_date(epoch) == pytest.approx(2440587.5, abs=1e-6)

    def test_one_day_advances_jd_by_one(self) -> None:
        first = datetime_to_julian_date(REFERENCE_TIME)
        second = datetime_to_julian_date(REFERENCE_TIME + timedelta(days=1))
        assert second - first == pytest.approx(1.0, abs=1e-9)

    def test_naive_datetime_assumed_utc(self) -> None:
        naive = datetime(2026, 9, 26, 12, 0, 0)
        assert datetime_to_julian_date(naive) == pytest.approx(
            datetime_to_julian_date(REFERENCE_TIME), abs=1e-9
        )


class TestSunGeometry:
    def test_returns_unit_vector(self, provider: SimpleAstronomyProvider) -> None:
        direction, _distance = provider.get_sun_vector_from_moon(REFERENCE_TIME)
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-12)

    def test_distance_is_about_one_au(self, provider: SimpleAstronomyProvider) -> None:
        """Sun-Moon distance stays within 1 AU +/- (eccentricity + lunar orbit)."""
        _direction, distance = provider.get_sun_vector_from_moon(REFERENCE_TIME)
        assert 1.46e8 < distance < 1.53e8

    def test_annual_distance_variation_matches_eccentricity(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """Earth's orbit is ~1.67% eccentric, so perihelion/aphelion differ ~3.4%."""
        distances = [
            provider.get_sun_vector_from_moon(
                datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
            )[1]
            for day in range(0, 365, 5)
        ]
        assert max(distances) / min(distances) == pytest.approx(1.034, abs=0.006)

    def test_subsolar_point_circles_the_moon_in_one_synodic_month(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """The sub-solar longitude must complete 360 deg per synodic month."""
        start = REFERENCE_TIME
        longitudes = []
        for hours in range(0, int(SYNODIC_MONTH_DAYS * 24), 6):
            direction, _ = provider.get_sun_vector_from_moon(start + timedelta(hours=hours))
            _latitude, longitude = cartesian_to_lat_lon(direction)
            longitudes.append(longitude)

        # Unwrap and measure the total swept angle.
        unwrapped = np.unwrap(np.radians(longitudes))
        swept_deg = abs(math.degrees(unwrapped[-1] - unwrapped[0]))
        assert swept_deg == pytest.approx(360.0, abs=15.0)

    def test_subsolar_latitude_stays_near_the_equator(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """The Moon's obliquity to the ecliptic is only 1.54 deg."""
        for days in range(0, 400, 7):
            direction, _ = provider.get_sun_vector_from_moon(
                REFERENCE_TIME + timedelta(days=days)
            )
            latitude, _longitude = cartesian_to_lat_lon(direction)
            assert abs(latitude) < 2.0


class TestEarthGeometry:
    def test_distance_matches_the_lunar_orbit(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """Perigee ~356,500 km, apogee ~406,700 km."""
        distances = [
            provider.get_earth_vector_from_moon(REFERENCE_TIME + timedelta(hours=hour))[1]
            for hour in range(0, 24 * 30, 6)
        ]
        assert 3.5e5 < min(distances) < 3.7e5
        assert 4.0e5 < max(distances) < 4.1e5

    def test_earth_stays_near_the_sub_earth_point(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """Tidal locking keeps Earth within ~10 deg of (0, 0) - optical libration."""
        for days in range(0, 60, 2):
            direction, _ = provider.get_earth_vector_from_moon(
                REFERENCE_TIME + timedelta(days=days)
            )
            latitude, longitude = cartesian_to_lat_lon(direction)
            assert abs(latitude) < 12.0
            assert abs(longitude) < 12.0


class TestMoonOrientation:
    def test_rotation_matrix_is_orthonormal(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        orientation = provider.get_moon_orientation(REFERENCE_TIME)
        matrix = orientation.j2000_to_body
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)

    def test_pole_is_near_the_iau_mean_values(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """alpha0 ~ 270 deg, delta0 ~ 66.5 deg, modulated by libration."""
        orientation = provider.get_moon_orientation(REFERENCE_TIME)
        assert 265.0 < orientation.right_ascension_deg < 275.0
        assert 64.0 < orientation.declination_deg < 69.0

    def test_prime_meridian_advances_one_turn_per_sidereal_month(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """W advances 13.176 deg/day, i.e. 360 deg in ~27.32 days."""
        first = provider.get_moon_orientation(REFERENCE_TIME)
        second = provider.get_moon_orientation(REFERENCE_TIME + timedelta(days=27.321661))
        difference = (second.prime_meridian_deg - first.prime_meridian_deg) % 360.0
        assert min(difference, 360.0 - difference) < 1.0


class TestGeometryBundle:
    def test_reports_provenance(self, provider: SimpleAstronomyProvider) -> None:
        geometry = provider.get_geometry(REFERENCE_TIME)
        assert geometry.source == "simple-analytic-ephemeris"
        assert "not a JPL-precision" in geometry.accuracy_note

    def test_subsolar_point_matches_the_sun_vector(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        geometry = provider.get_geometry(REFERENCE_TIME)
        latitude, longitude = cartesian_to_lat_lon(geometry.sun_direction_body)
        assert geometry.subsolar_latitude_deg == pytest.approx(latitude, abs=1e-9)
        assert geometry.subsolar_longitude_deg == pytest.approx(longitude, abs=1e-9)

    def test_j2000_sun_vector_is_a_unit_vector(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        geometry = provider.get_geometry(REFERENCE_TIME)
        assert np.linalg.norm(geometry.sun_direction_j2000) == pytest.approx(1.0, abs=1e-12)


class TestProviderFactory:
    def test_defaults_to_the_simple_provider(self) -> None:
        provider = create_astronomy_provider(
            use_spice=False, kernel_directory=Path("nonexistent")
        )
        assert isinstance(provider, SimpleAstronomyProvider)

    def test_falls_back_when_kernels_are_missing(self, tmp_path: Path) -> None:
        """The spec requires the app to work without SPICE installed."""
        provider = create_astronomy_provider(
            use_spice=True, kernel_directory=tmp_path / "does-not-exist"
        )
        assert isinstance(provider, SimpleAstronomyProvider)
        assert provider.source == "simple-analytic-ephemeris"

    def test_falls_back_when_kernel_directory_is_empty(self, tmp_path: Path) -> None:
        provider = create_astronomy_provider(use_spice=True, kernel_directory=tmp_path)
        assert isinstance(provider, SimpleAstronomyProvider)


class TestKnownAstronomicalEvents:
    def test_full_moon_has_subsolar_point_near_the_prime_meridian(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """At full Moon the near side is fully lit, so the sub-solar longitude
        is near 0 deg (the mean sub-Earth point).

        Full Moon occurred on 2026-09-26. Cross-checked against published
        lunar-phase tables, independent of this model.
        """
        geometry = provider.get_geometry(datetime(2026, 9, 26, 16, 49, tzinfo=timezone.utc))
        assert abs(geometry.subsolar_longitude_deg) < 8.0

    def test_new_moon_has_subsolar_point_near_the_far_side(
        self, provider: SimpleAstronomyProvider
    ) -> None:
        """At new Moon the far side is lit: sub-solar longitude near +/-180 deg.

        New Moon occurred on 2026-09-11.
        """
        geometry = provider.get_geometry(datetime(2026, 9, 11, 3, 27, tzinfo=timezone.utc))
        assert abs(abs(geometry.subsolar_longitude_deg) - 180.0) < 8.0
