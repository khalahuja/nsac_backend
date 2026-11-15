"""Solar incidence, elevation, azimuth and irradiance tests.

Uses synthetic Sun vectors so the geometry is checked independently of the
ephemeris. Covers the spec's required cases: Sun overhead, Sun exactly on the
horizon, and Sun below the horizon.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.constants import ASTRONOMICAL_UNIT_KM, SOLAR_CONSTANT_1AU_W_M2
from app.services.geometry import lat_lon_to_cartesian, spherical_surface_normal
from app.services.illumination import (
    compute_illumination,
    solar_azimuth_deg,
    solar_constant_at_distance,
    solar_elevation_deg,
    solar_incidence_angle_deg,
)

ONE_AU_KM = ASTRONOMICAL_UNIT_KM


def sun_at(latitude_deg: float, longitude_deg: float) -> np.ndarray:
    """Unit Sun vector whose sub-solar point is the given coordinate."""
    return lat_lon_to_cartesian(latitude_deg, longitude_deg, radius=1.0)


class TestSolarConstant:
    def test_exactly_one_au(self) -> None:
        assert solar_constant_at_distance(ONE_AU_KM) == pytest.approx(
            SOLAR_CONSTANT_1AU_W_M2
        )

    def test_inverse_square_law(self) -> None:
        """Doubling the distance must quarter the irradiance."""
        assert solar_constant_at_distance(2.0 * ONE_AU_KM) == pytest.approx(
            SOLAR_CONSTANT_1AU_W_M2 / 4.0
        )

    def test_perihelion_is_brighter_than_aphelion(self) -> None:
        perihelion = solar_constant_at_distance(0.9833 * ONE_AU_KM)
        aphelion = solar_constant_at_distance(1.0167 * ONE_AU_KM)
        assert perihelion > SOLAR_CONSTANT_1AU_W_M2 > aphelion
        # Annual variation is about +/-3.4%.
        assert perihelion / aphelion == pytest.approx(1.069, abs=0.005)

    def test_rejects_non_positive_distance(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            solar_constant_at_distance(0.0)


class TestSolarIncidence:
    def test_sun_directly_overhead(self) -> None:
        """Sub-solar point: incidence = 0, cos(theta) = 1."""
        latitude, longitude = 12.0, -34.0
        normal = spherical_surface_normal(latitude, longitude)
        assert solar_incidence_angle_deg(normal, sun_at(latitude, longitude)) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_sun_exactly_on_horizon(self) -> None:
        """A sub-solar point 90 deg away gives incidence = 90 deg."""
        normal = spherical_surface_normal(0.0, 0.0)
        assert solar_incidence_angle_deg(normal, sun_at(0.0, 90.0)) == pytest.approx(
            90.0, abs=1e-9
        )
        assert solar_incidence_angle_deg(normal, sun_at(90.0, 0.0)) == pytest.approx(
            90.0, abs=1e-9
        )

    def test_sun_below_horizon(self) -> None:
        """The antipode of the sub-solar point gives incidence = 180 deg."""
        normal = spherical_surface_normal(0.0, 0.0)
        assert solar_incidence_angle_deg(normal, sun_at(0.0, 180.0)) == pytest.approx(
            180.0, abs=1e-9
        )

    def test_matches_great_circle_distance(self) -> None:
        """Incidence equals the angular distance to the sub-solar point."""
        normal = spherical_surface_normal(0.0, 0.0)
        for separation in (0.0, 15.0, 45.0, 89.0, 120.0, 179.0):
            angle = solar_incidence_angle_deg(normal, sun_at(0.0, separation))
            assert angle == pytest.approx(separation, abs=1e-9)


class TestSolarElevation:
    def test_overhead_is_ninety_degrees(self) -> None:
        assert solar_elevation_deg(-89.4, 137.3, sun_at(-89.4, 137.3)) == pytest.approx(
            90.0, abs=1e-9
        )

    def test_horizon_is_zero(self) -> None:
        assert solar_elevation_deg(0.0, 0.0, sun_at(0.0, 90.0)) == pytest.approx(0.0, abs=1e-9)

    def test_antipode_is_minus_ninety(self) -> None:
        assert solar_elevation_deg(0.0, 0.0, sun_at(0.0, 180.0)) == pytest.approx(
            -90.0, abs=1e-9
        )

    def test_elevation_is_ninety_minus_incidence_on_sphere(self) -> None:
        """On a spherical Moon the spec's identity must hold exactly."""
        latitude, longitude = 33.0, 77.0
        normal = spherical_surface_normal(latitude, longitude)
        for sub_latitude, sub_longitude in [(0.0, 0.0), (45.0, 20.0), (-30.0, 150.0)]:
            sun = sun_at(sub_latitude, sub_longitude)
            incidence = solar_incidence_angle_deg(normal, sun)
            elevation = solar_elevation_deg(latitude, longitude, sun)
            assert elevation == pytest.approx(90.0 - incidence, abs=1e-9)


class TestSolarAzimuth:
    def test_sun_due_north(self) -> None:
        """From the equator, a Sun toward the north pole reads azimuth 0."""
        assert solar_azimuth_deg(0.0, 0.0, sun_at(90.0, 0.0)) == pytest.approx(0.0, abs=1e-9)

    def test_sun_due_east(self) -> None:
        assert solar_azimuth_deg(0.0, 0.0, sun_at(0.0, 90.0)) == pytest.approx(90.0, abs=1e-9)

    def test_sun_due_south(self) -> None:
        assert solar_azimuth_deg(0.0, 0.0, sun_at(-90.0, 0.0)) == pytest.approx(
            180.0, abs=1e-9
        )

    def test_sun_due_west(self) -> None:
        assert solar_azimuth_deg(0.0, 0.0, sun_at(0.0, -90.0)) == pytest.approx(
            270.0, abs=1e-9
        )

    def test_range_is_zero_to_360(self) -> None:
        for sub_latitude in (-80.0, -20.0, 0.0, 40.0, 85.0):
            for sub_longitude in range(-180, 181, 20):
                azimuth = solar_azimuth_deg(
                    25.0, -60.0, sun_at(sub_latitude, float(sub_longitude))
                )
                assert 0.0 <= azimuth < 360.0

    def test_northern_hemisphere_sees_equatorial_sun_to_the_south(self) -> None:
        assert solar_azimuth_deg(45.0, 10.0, sun_at(0.0, 10.0)) == pytest.approx(
            180.0, abs=1e-9
        )

    def test_southern_hemisphere_sees_equatorial_sun_to_the_north(self) -> None:
        assert solar_azimuth_deg(-45.0, 10.0, sun_at(0.0, 10.0)) == pytest.approx(
            0.0, abs=1e-9
        )


class TestComputeIllumination:
    def test_subsolar_point_receives_full_solar_constant(self) -> None:
        latitude, longitude = 0.0, 0.0
        result = compute_illumination(
            latitude_deg=latitude,
            longitude_deg=longitude,
            sun_direction_body=sun_at(latitude, longitude),
            sun_distance_km=ONE_AU_KM,
            surface_normal_body=spherical_surface_normal(latitude, longitude),
        )
        assert result.illuminated is True
        assert result.above_horizon is True
        assert result.incidence_angle_deg == pytest.approx(0.0, abs=1e-9)
        assert result.elevation_deg == pytest.approx(90.0, abs=1e-9)
        assert result.solar_irradiance_w_m2 == pytest.approx(SOLAR_CONSTANT_1AU_W_M2)

    def test_night_side_is_dark(self) -> None:
        """Sun below the horizon: illuminated False, irradiance exactly zero."""
        result = compute_illumination(
            latitude_deg=0.0,
            longitude_deg=0.0,
            sun_direction_body=sun_at(0.0, 180.0),
            sun_distance_km=ONE_AU_KM,
            surface_normal_body=spherical_surface_normal(0.0, 0.0),
        )
        assert result.illuminated is False
        assert result.above_horizon is False
        assert result.solar_irradiance_w_m2 == 0.0
        assert result.cos_incidence == 0.0
        assert result.elevation_deg < 0.0

    def test_sun_exactly_on_horizon_gives_zero_irradiance(self) -> None:
        """The terminator is the boundary case: cos(theta) = 0 exactly."""
        result = compute_illumination(
            latitude_deg=0.0,
            longitude_deg=0.0,
            sun_direction_body=sun_at(0.0, 90.0),
            sun_distance_km=ONE_AU_KM,
            surface_normal_body=spherical_surface_normal(0.0, 0.0),
        )
        assert result.elevation_deg == pytest.approx(0.0, abs=1e-9)
        assert result.incidence_angle_deg == pytest.approx(90.0, abs=1e-9)
        assert result.solar_irradiance_w_m2 == pytest.approx(0.0, abs=1e-6)
        assert result.illuminated is False

    def test_lambert_cosine_law(self) -> None:
        """Q = S0 cos(theta) for a range of incidence angles."""
        for separation_deg in (0.0, 30.0, 60.0, 75.0, 89.0):
            result = compute_illumination(
                latitude_deg=0.0,
                longitude_deg=0.0,
                sun_direction_body=sun_at(0.0, separation_deg),
                sun_distance_km=ONE_AU_KM,
                surface_normal_body=spherical_surface_normal(0.0, 0.0),
            )
            expected = SOLAR_CONSTANT_1AU_W_M2 * math.cos(math.radians(separation_deg))
            assert result.solar_irradiance_w_m2 == pytest.approx(expected, abs=1e-6)

    def test_terrain_shadow_overrides_illumination(self) -> None:
        """A shadowed point is dark even with the Sun high above the horizon."""
        result = compute_illumination(
            latitude_deg=0.0,
            longitude_deg=0.0,
            sun_direction_body=sun_at(0.0, 10.0),
            sun_distance_km=ONE_AU_KM,
            surface_normal_body=spherical_surface_normal(0.0, 0.0),
            terrain_shadowed=True,
        )
        assert result.above_horizon is True
        assert result.illuminated is False
        assert result.solar_irradiance_w_m2 == 0.0

    def test_unknown_terrain_shadow_does_not_block_light(self) -> None:
        """terrain_shadowed=None means 'unknown', not 'shadowed'."""
        result = compute_illumination(
            latitude_deg=0.0,
            longitude_deg=0.0,
            sun_direction_body=sun_at(0.0, 10.0),
            sun_distance_km=ONE_AU_KM,
            surface_normal_body=spherical_surface_normal(0.0, 0.0),
            terrain_shadowed=None,
        )
        assert result.illuminated is True
        assert result.terrain_shadowed is None

    def test_tilted_normal_changes_incidence_but_not_elevation(self) -> None:
        """A sloped surface sees a different incidence; the horizon is unchanged."""
        latitude, longitude = 0.0, 0.0
        sun = sun_at(0.0, 30.0)
        flat = compute_illumination(
            latitude_deg=latitude,
            longitude_deg=longitude,
            sun_direction_body=sun,
            sun_distance_km=ONE_AU_KM,
            surface_normal_body=spherical_surface_normal(latitude, longitude),
        )
        # Tilt the normal 20 deg toward the east (i.e. toward the Sun).
        tilted = compute_illumination(
            latitude_deg=latitude,
            longitude_deg=longitude,
            sun_direction_body=sun,
            sun_distance_km=ONE_AU_KM,
            surface_normal_body=spherical_surface_normal(0.0, 20.0),
        )
        assert tilted.elevation_deg == pytest.approx(flat.elevation_deg, abs=1e-9)
        assert tilted.incidence_angle_deg < flat.incidence_angle_deg
        assert tilted.solar_irradiance_w_m2 > flat.solar_irradiance_w_m2

    @pytest.mark.parametrize("latitude", [-90.0, 90.0])
    def test_poles_do_not_crash(self, latitude: float) -> None:
        result = compute_illumination(
            latitude_deg=latitude,
            longitude_deg=0.0,
            sun_direction_body=sun_at(0.0, 0.0),
            sun_distance_km=ONE_AU_KM,
            surface_normal_body=spherical_surface_normal(latitude, 0.0),
        )
        assert result.elevation_deg == pytest.approx(0.0, abs=1e-9)
        assert 0.0 <= result.azimuth_deg < 360.0
