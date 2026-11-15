"""Coordinate conversion and local-frame tests.

Covers the boundary cases called out in the project spec: the equator, both
poles, and both ends of the longitude range.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.geometry import (
    angle_between_deg,
    cartesian_to_lat_lon,
    lat_lon_to_cartesian,
    local_enu_basis,
    normalize,
    normalize_longitude,
    spherical_surface_normal,
)

# (latitude, longitude, expected unit vector)
KNOWN_POINTS = [
    (0.0, 0.0, [1.0, 0.0, 0.0]),  # prime meridian on the equator = sub-Earth point
    (0.0, 90.0, [0.0, 1.0, 0.0]),  # 90 deg East
    (0.0, -90.0, [0.0, -1.0, 0.0]),  # 90 deg West
    (0.0, 180.0, [-1.0, 0.0, 0.0]),  # far side centre
    (0.0, -180.0, [-1.0, 0.0, 0.0]),  # same point, opposite sign convention
    (90.0, 0.0, [0.0, 0.0, 1.0]),  # north pole
    (-90.0, 0.0, [0.0, 0.0, -1.0]),  # south pole
    (45.0, 0.0, [math.sqrt(0.5), 0.0, math.sqrt(0.5)]),
]


class TestLatLonToCartesian:
    @pytest.mark.parametrize(("latitude", "longitude", "expected"), KNOWN_POINTS)
    def test_known_points(
        self, latitude: float, longitude: float, expected: list[float]
    ) -> None:
        result = lat_lon_to_cartesian(latitude, longitude, radius=1.0)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_radius_scales_magnitude(self) -> None:
        vector = lat_lon_to_cartesian(30.0, 45.0, radius=1737.4)
        assert np.linalg.norm(vector) == pytest.approx(1737.4)

    def test_unit_radius_gives_unit_vector(self) -> None:
        for latitude in range(-90, 91, 15):
            for longitude in range(-180, 181, 30):
                vector = lat_lon_to_cartesian(float(latitude), float(longitude))
                assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)


class TestCartesianToLatLon:
    @pytest.mark.parametrize(("latitude", "longitude", "vector"), KNOWN_POINTS)
    def test_known_points(
        self, latitude: float, longitude: float, vector: list[float]
    ) -> None:
        result_latitude, result_longitude = cartesian_to_lat_lon(np.array(vector))
        assert result_latitude == pytest.approx(latitude, abs=1e-9)
        # Longitude is degenerate at the poles, and +/-180 denote the same point.
        if abs(latitude) < 90.0:
            assert math.cos(math.radians(result_longitude)) == pytest.approx(
                math.cos(math.radians(longitude)), abs=1e-9
            )
            assert math.sin(math.radians(result_longitude)) == pytest.approx(
                math.sin(math.radians(longitude)), abs=1e-9
            )

    def test_round_trip_is_identity(self) -> None:
        """lat/lon -> Cartesian -> lat/lon must recover the original values."""
        for latitude in (-89.4, -45.0, -0.001, 0.0, 12.3, 45.0, 89.9):
            for longitude in (-179.9, -137.3, -90.0, 0.0, 45.5, 137.3, 179.9):
                vector = lat_lon_to_cartesian(latitude, longitude, radius=3.7)
                back_latitude, back_longitude = cartesian_to_lat_lon(vector)
                assert back_latitude == pytest.approx(latitude, abs=1e-9)
                assert back_longitude == pytest.approx(longitude, abs=1e-9)

    def test_zero_vector_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero-length"):
            cartesian_to_lat_lon(np.zeros(3))


class TestNormalizeLongitude:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, 0.0),
            (180.0, 180.0),  # exact +180 is preserved, not folded to -180
            (-180.0, -180.0),
            (190.0, -170.0),
            (-190.0, 170.0),
            (540.0, -180.0),
            (360.0, 0.0),
        ],
    )
    def test_wrapping(self, value: float, expected: float) -> None:
        assert normalize_longitude(value) == pytest.approx(expected)


class TestSurfaceNormal:
    def test_normal_is_unit_length(self) -> None:
        normal = spherical_surface_normal(-89.4, 137.3)
        assert np.linalg.norm(normal) == pytest.approx(1.0)

    def test_normal_matches_spec_formula(self) -> None:
        """N = [cos(phi)cos(lam), cos(phi)sin(lam), sin(phi)]."""
        latitude, longitude = 23.5, -47.2
        phi, lam = math.radians(latitude), math.radians(longitude)
        expected = [
            math.cos(phi) * math.cos(lam),
            math.cos(phi) * math.sin(lam),
            math.sin(phi),
        ]
        np.testing.assert_allclose(
            spherical_surface_normal(latitude, longitude), expected, atol=1e-12
        )

    def test_poles(self) -> None:
        np.testing.assert_allclose(
            spherical_surface_normal(90.0, 0.0), [0.0, 0.0, 1.0], atol=1e-12
        )
        np.testing.assert_allclose(
            spherical_surface_normal(-90.0, 45.0), [0.0, 0.0, -1.0], atol=1e-12
        )

    def test_normal_equals_radial_direction(self) -> None:
        """On a sphere the normal is the normalised position vector."""
        position = lat_lon_to_cartesian(31.0, 118.0, radius=1737.4)
        np.testing.assert_allclose(
            spherical_surface_normal(31.0, 118.0), normalize(position), atol=1e-12
        )


class TestLocalEnuBasis:
    @pytest.mark.parametrize("latitude", [-90.0, -60.0, -0.0, 0.0, 37.5, 90.0])
    @pytest.mark.parametrize("longitude", [-180.0, -90.0, 0.0, 45.0, 180.0])
    def test_basis_is_orthonormal(self, latitude: float, longitude: float) -> None:
        east, north, up = local_enu_basis(latitude, longitude)
        for vector in (east, north, up):
            assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)
        assert np.dot(east, north) == pytest.approx(0.0, abs=1e-12)
        assert np.dot(east, up) == pytest.approx(0.0, abs=1e-12)
        assert np.dot(north, up) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("latitude", [-45.0, 0.0, 45.0])
    def test_basis_is_right_handed(self, latitude: float) -> None:
        """East x North must equal Up for a right-handed ENU triad."""
        east, north, up = local_enu_basis(latitude, 30.0)
        np.testing.assert_allclose(np.cross(east, north), up, atol=1e-12)

    def test_up_equals_surface_normal(self) -> None:
        _east, _north, up = local_enu_basis(-89.4, 137.3)
        np.testing.assert_allclose(up, spherical_surface_normal(-89.4, 137.3), atol=1e-12)

    def test_north_points_toward_pole(self) -> None:
        """At the equator, North should be the +Z direction."""
        _east, north, _up = local_enu_basis(0.0, 57.0)
        np.testing.assert_allclose(north, [0.0, 0.0, 1.0], atol=1e-12)


class TestAngleBetween:
    def test_parallel_vectors(self) -> None:
        assert angle_between_deg(np.array([1.0, 0, 0]), np.array([2.0, 0, 0])) == pytest.approx(0.0)

    def test_antiparallel_vectors(self) -> None:
        assert angle_between_deg(
            np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
        ) == pytest.approx(180.0)

    def test_perpendicular_vectors(self) -> None:
        assert angle_between_deg(
            np.array([1.0, 0, 0]), np.array([0.0, 1.0, 0])
        ) == pytest.approx(90.0)


class TestNormalize:
    def test_rejects_zero_vector(self) -> None:
        with pytest.raises(ValueError, match="zero-length"):
            normalize(np.zeros(3))

    def test_preserves_direction(self) -> None:
        result = normalize(np.array([3.0, 4.0, 0.0]))
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-12)
