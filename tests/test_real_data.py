"""Checks against the installed NASA/JPL data. Skipped when the data is absent."""
from pathlib import Path

import pytest

from app.config import Settings
from app.services.terrain import create_terrain_provider

DATA = Path(__file__).resolve().parents[2] / "data"
DEM = DATA / "terrain" / "ldem_80s_80m.lbl"


@pytest.mark.skipif(not DEM.exists() or not (DATA / "terrain" / "ldem_80s_80m.img").exists(), reason="LOLA DEM not installed")
def test_lola_polar_dem_matches_known_terrain():
    terrain = create_terrain_provider(use_dem=True, raster_path=DEM, enable_shadows=True)
    assert terrain.is_measured and "LDEM_80S_80M" in terrain.source
    # Heights above the 1737.4 km sphere (LOLA): pole on Shackleton's rim, floor ~4 km lower.
    assert 900 < terrain.get_elevation(-90, 0) < 1700
    assert -3300 < terrain.get_elevation(-89.67, 129.78) < -2300
    # Shackleton's floor sits under a ~20° rim wall; Connecting Ridge sees past every horizon.
    floor = terrain.horizon_profile(-89.67, 129.78, [0, 90, 180, 270])
    ridge = terrain.horizon_profile(-89.4398, -137.17, [0, 90, 180, 270])
    assert all(15 < h < 30 for h in floor)
    assert all(h < 1 for h in ridge)


@pytest.mark.skipif(not (DATA / "spice" / "de440s.bsp").exists(), reason="SPICE kernels not installed")
def test_spice_provider_loads_and_uses_moon_me():
    from app.services.astronomy import create_astronomy_provider

    provider = create_astronomy_provider(use_spice=True, kernel_directory=DATA / "spice")
    assert "spice" in provider.source.lower()
    assert provider._body_frame() == "MOON_ME"


@pytest.mark.skipif(not (DATA / "spice" / "de440s.bsp").exists() or not (DATA / "terrain" / "ldem_80s_80m.img").exists(), reason="real data not installed")
def test_full_swarm_selects_site_and_dates():
    from app.config import Settings
    from app.services.mission_swarm import MissionInputs, plan_mission
    from app.services.providers import build_providers

    providers = build_providers(Settings(use_spice=True, use_terrain_dem=True, use_terrain_shadows=True, terrain_raster_filename="ldem_80s_80m.lbl"))
    plan = plan_mission(MissionInputs(departure_utc="2027-01-01T00:00:00Z", target_latitude_deg=-90, target_longitude_deg=-65,
                                      site_mode="auto", landing_start_utc="2027-01-01T00:00:00Z", landing_end_utc="2027-03-31T00:00:00Z"), providers)
    names = [s.agent for s in plan.stages]
    assert names[:2] == ["Site selection agent", "Landing window agent"] and len(names) == 7
    site, window = plan.stages[0].outputs, plan.stages[1].outputs
    assert site["selected_site"] != "Requested target"          # the pole itself ranks low
    assert window["candidates_found"] >= 1 and window["terrain_verified"] is True
    # The chosen launch feeds the rest of the chain.
    assert plan.inputs.departure_utc.isoformat()[:16] == str(window["chosen_launch_utc"])[:16]


@pytest.mark.skipif(not (DATA / "spice" / "de440s.bsp").exists(), reason="SPICE kernels not installed")
def test_concurrent_spice_queries_match_serial_results():
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta, timezone

    import numpy as np

    from app.services.astronomy import SpiceAstronomyProvider

    # Separate providers still access the same native SPICE state.
    providers = [SpiceAstronomyProvider(DATA / "spice") for _ in range(2)]
    start = datetime(2026, 8, 24, tzinfo=timezone.utc)

    def query(index):
        provider = providers[index % 2]
        moment = start + timedelta(hours=index * 6)
        sun, distance = provider.get_sun_vector_from_moon(moment)
        earth, earth_distance = provider.get_earth_vector_from_moon(moment)
        orientation = provider.get_moon_orientation(moment)
        return np.concatenate((sun, [distance], earth, [earth_distance], orientation.j2000_to_body.ravel()))

    expected = [query(i) for i in range(120)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        actual = list(executor.map(query, range(120)))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
