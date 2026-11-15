"""Thermal model sanity checks.

These tests pin down the physics rather than the exact numbers: energy balance,
monotonicity, the darkness floor, and agreement with published lunar extremes.
"""

from __future__ import annotations

import pytest

from app.constants import (
    DEFAULT_ALBEDO,
    DEFAULT_EMISSIVITY,
    KELVIN_OFFSET,
    NIGHT_TEMPERATURE_FLOOR_K,
    SOLAR_CONSTANT_1AU_W_M2,
    STEFAN_BOLTZMANN,
)
from app.services.thermal import (
    EQUILIBRIUM_MODEL_NAME,
    RegolithThermalModel,
    background_flux_w_m2,
    equilibrium_temperature_k,
    estimate_temperature,
    is_physically_plausible,
    kelvin_to_celsius,
    subsolar_peak_temperature_k,
)


class TestEquilibriumTemperature:
    def test_darkness_returns_the_floor_exactly(self) -> None:
        """Zero irradiance must give the floor, never 0 K."""
        assert equilibrium_temperature_k(0.0) == pytest.approx(NIGHT_TEMPERATURE_FLOOR_K)

    def test_never_below_the_floor(self) -> None:
        for irradiance in (0.0, 0.001, 1.0, 10.0, 100.0):
            assert equilibrium_temperature_k(irradiance) >= NIGHT_TEMPERATURE_FLOOR_K

    def test_monotonic_in_irradiance(self) -> None:
        temperatures = [
            equilibrium_temperature_k(flux) for flux in (0.0, 50.0, 200.0, 700.0, 1361.0)
        ]
        assert temperatures == sorted(temperatures)
        assert len(set(temperatures)) == len(temperatures)

    def test_subsolar_peak_matches_observed_lunar_noon(self) -> None:
        """Diviner reports ~390 K at the equatorial subsolar point."""
        peak = subsolar_peak_temperature_k(SOLAR_CONSTANT_1AU_W_M2)
        assert 375.0 <= peak <= 400.0

    def test_energy_balance_is_exact(self) -> None:
        """epsilon sigma T^4 must equal absorbed flux plus background flux."""
        irradiance = 800.0
        temperature = equilibrium_temperature_k(irradiance)
        emitted = DEFAULT_EMISSIVITY * STEFAN_BOLTZMANN * temperature**4
        absorbed = (1.0 - DEFAULT_ALBEDO) * irradiance + background_flux_w_m2()
        assert emitted == pytest.approx(absorbed, rel=1e-12)

    def test_higher_albedo_gives_lower_temperature(self) -> None:
        dark = equilibrium_temperature_k(1000.0, albedo=0.05)
        bright = equilibrium_temperature_k(1000.0, albedo=0.30)
        assert dark > bright

    def test_results_are_physically_plausible(self) -> None:
        for irradiance in (0.0, 100.0, 680.0, 1361.0, 1420.0):
            assert is_physically_plausible(equilibrium_temperature_k(irradiance))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"albedo": 1.0}, "Albedo"),
            ({"albedo": -0.1}, "Albedo"),
            ({"emissivity": 0.0}, "Emissivity"),
            ({"emissivity": 1.5}, "Emissivity"),
        ],
    )
    def test_rejects_unphysical_parameters(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            equilibrium_temperature_k(500.0, **kwargs)

    def test_rejects_negative_irradiance(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            equilibrium_temperature_k(-1.0)


class TestBackgroundFlux:
    def test_defined_by_the_floor_temperature(self) -> None:
        """Q_background = epsilon sigma T_floor^4, by construction."""
        expected = DEFAULT_EMISSIVITY * STEFAN_BOLTZMANN * NIGHT_TEMPERATURE_FLOOR_K**4
        assert background_flux_w_m2() == pytest.approx(expected)

    def test_is_small_compared_to_daytime_flux(self) -> None:
        """The floor must not meaningfully brighten a sunlit surface."""
        assert background_flux_w_m2() < 0.01 * SOLAR_CONSTANT_1AU_W_M2


class TestEstimateTemperature:
    def test_reports_provenance_and_never_claims_measurement(self) -> None:
        result = estimate_temperature(500.0)
        assert result.model == EQUILIBRIUM_MODEL_NAME
        assert result.confidence == "experimental"
        assert result.is_measured is False
        assert "Diviner" in result.notes

    def test_celsius_matches_kelvin(self) -> None:
        result = estimate_temperature(300.0)
        assert result.temperature_c == pytest.approx(result.temperature_k - KELVIN_OFFSET)

    def test_absorbed_flux_accounts_for_albedo(self) -> None:
        result = estimate_temperature(1000.0, albedo=0.2)
        assert result.absorbed_flux_w_m2 == pytest.approx(800.0)

    def test_night_estimate_is_the_floor(self) -> None:
        result = estimate_temperature(0.0)
        assert result.temperature_k == pytest.approx(NIGHT_TEMPERATURE_FLOOR_K)
        assert result.temperature_c == pytest.approx(NIGHT_TEMPERATURE_FLOOR_K - KELVIN_OFFSET)


class TestKelvinToCelsius:
    @pytest.mark.parametrize(
        ("kelvin", "celsius"), [(273.15, 0.0), (0.0, -273.15), (373.15, 100.0)]
    )
    def test_conversion(self, kelvin: float, celsius: float) -> None:
        assert kelvin_to_celsius(kelvin) == pytest.approx(celsius)


class TestRegolithThermalModel:
    def test_is_an_honest_placeholder(self) -> None:
        """The unimplemented conduction model must refuse, not fabricate."""
        model = RegolithThermalModel()
        assert model.implemented is False
        with pytest.raises(NotImplementedError, match="placeholder"):
            model.simulate()
