from datetime import datetime, timezone
import json

import pytest
from pydantic import ValidationError
from pydantic import SecretStr
from fastapi.testclient import TestClient

from app.api import mission as mission_api

from app.services.mission_swarm import MissionInputs, plan_mission
from app.services.mission_windows import WindowSearchInputs, search_mission_windows
from app.config import Settings
from app.services.providers import build_providers


def inputs(**changes):
    return MissionInputs(departure_utc=datetime(2027, 1, 15, tzinfo=timezone.utc), **changes)


def test_graph_runs_all_agents_and_preserves_delta_v_budget():
    plan = plan_mission(inputs())
    assert len(plan.stages) == 7
    assert plan.stages[2].agent.startswith("Launch")
    assert plan.stages[-1].agent.startswith("Landing safety")
    descent = plan.stages[5].outputs
    # The upper stage performs TLI by default, so the spacecraft budget is LOI + descent.
    expected = descent["spacecraft_tli_delta_v_km_s"] + plan.stages[4].outputs["loi_delta_v_km_s"] + descent["descent_delta_v_km_s"]
    assert descent["spacecraft_tli_delta_v_km_s"] == 0
    assert plan.total_spacecraft_delta_v_km_s == pytest.approx(expected)
    assert plan.overall_status == "provisional_screening_pass"
    assert plan.missing_verifications


def test_propellant_shortfall_fails_review():
    plan = plan_mission(inputs(spacecraft_dry_mass_kg=6000))
    assert plan.overall_status == "screening_failed"
    assert plan.stages[5].checks


def test_launch_thrust_shortfall_fails_review():
    plan = plan_mission(inputs(launch_thrust_n=100000))
    assert plan.overall_status == "screening_failed"
    assert plan.stages[2].outputs["initial_twr"] < 1


def test_departure_requires_explicit_timezone():
    with pytest.raises(ValidationError):
        MissionInputs(departure_utc=datetime(2027, 1, 15))


def test_simulate_endpoint_returns_surface_forecast(client: TestClient):
    response = client.post("/api/mission/simulate", json={"inputs": inputs().model_dump(mode="json")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stages"][6]["outputs"]["astronomy_source"] == "simple-analytic-ephemeris"
    assert "longest_blackout_hours" in body["stages"][6]["outputs"]
    assert body["timeline"][0]["event"] == "Launch"


def test_stream_reports_actual_agent_completions(client: TestClient):
    with client.stream("POST", "/api/mission/simulate/stream", json={"inputs": inputs().model_dump(mode="json")}) as response:
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line]
    expected = ["site", "window", "launch", "transfer", "capture", "descent", "surface", "review"]
    assert [event["node"] for event in events if event["type"] == "running"] == expected
    assert [event["node"] for event in events if event["type"] == "completed"] == expected
    assert events[-1]["type"] == "result"
    assert len(events[-1]["plan"]["stages"]) == 7


def test_interpret_only_maps_validated_values(monkeypatch, client: TestClient):
    settings = mission_api.get_settings().model_copy(update={"groq_api_key": SecretStr("test-key")})
    monkeypatch.setattr(mission_api, "get_settings", lambda: settings)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"values":{"spacecraft_isp_s":420,"target_latitude_deg":-85},"unresolved":[]}'}}]}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            assert kwargs["json"]["messages"][1]["content"] == "Use 420 s Isp and land at 85 S"
            return Response()

    monkeypatch.setattr(mission_api.httpx, "AsyncClient", Client)
    response = client.post("/api/mission/interpret", headers={"Origin": "http://localhost:5173"}, json={"query": "Use 420 s Isp and land at 85 S", "base_inputs": inputs().model_dump(mode="json")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["extracted"] == {"spacecraft_isp_s": 420, "target_latitude_deg": -85}
    assert body["plan"]["inputs"]["spacecraft_isp_s"] == 420
    assert body["plan"]["inputs"]["landing_thrust_n"] == inputs().landing_thrust_n


def test_polar_window_search_keeps_unverified_dates_provisional():
    providers = build_providers(Settings(use_spice=False, use_terrain_dem=False))
    search = WindowSearchInputs(
        inputs=inputs(target_latitude_deg=-90, target_longitude_deg=-65),
        landing_start_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
        landing_end_utc=datetime(2027, 10, 12, 23, tzinfo=timezone.utc),
        min_continuous_sunlight_days=8,
    )
    result = search_mission_windows(search, providers)
    assert result.candidates
    assert result.best_plan is not None
    assert all(candidate.continuous_sunlight_days_screened >= 8 for candidate in result.candidates)
    assert all(not candidate.phase_targeted and not candidate.terrain_verified for candidate in result.candidates)
    assert all(candidate.estimated_launch_utc < candidate.estimated_landing_utc for candidate in result.candidates)
    assert all(search.landing_start_utc <= candidate.estimated_landing_utc <= search.landing_end_utc for candidate in result.candidates)
    assert any("all longitudes meet" in warning for warning in result.limitations)


def test_full_year_spice_search_returns_every_qualifying_interval():
    providers = build_providers(Settings(use_spice=True, use_terrain_dem=False))
    if providers.astronomy.source != "jpl-spice":
        pytest.skip("JPL SPICE kernels are not installed")
    search = WindowSearchInputs(
        inputs=inputs(target_latitude_deg=-8.33, target_longitude_deg=8.49),
        landing_start_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
        landing_end_utc=datetime(2027, 12, 31, 23, tzinfo=timezone.utc),
        min_continuous_sunlight_days=8,
    )
    result = search_mission_windows(search, providers, run_best_plan=False)
    assert len(result.candidates) == 11
    assert [c.estimated_landing_utc.month for c in result.candidates] == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert result.minimum_earth_elevation_deg == pytest.approx(73.6, abs=0.2)
    assert result.maximum_earth_elevation_deg == pytest.approx(80.7, abs=0.2)
    assert 176 <= result.total_screened_sunlight_hours / 24 <= 178
    assert all(c.sunlight_interval_start_utc <= c.estimated_landing_utc < c.sunlight_interval_end_utc for c in result.candidates)
    assert all(c.earth_contact_hours_during_required_stay == 192 for c in result.candidates)
    assert all(c.earth_contact_hours_in_sunlight_interval >= 192 for c in result.candidates)
    assert all(c.eclipse_screened and not c.phase_targeted for c in result.candidates)


def test_spice_forecast_marks_lunar_eclipse_samples():
    from app.services.forecast import compute_forecast
    providers = build_providers(Settings(use_spice=True, use_terrain_dem=False))
    if providers.astronomy.source != "jpl-spice":
        pytest.skip("JPL SPICE kernels are not installed")
    start = datetime(2027, 2, 20, 22, tzinfo=timezone.utc)
    result = compute_forecast(latitude_deg=-8.33, longitude_deg=8.49,
                              start=start, end=datetime(2027, 2, 21, 1, tzinfo=timezone.utc),
                              step_hours=1, providers=providers)
    assert [sample.solar_earth_overlap for sample in result.samples] == [False, True, True, False]


def test_window_range_requires_timezone_and_positive_duration():
    with pytest.raises(ValidationError):
        WindowSearchInputs(inputs=inputs(), landing_start_utc=datetime(2027, 1, 1), landing_end_utc=datetime(2027, 1, 10, tzinfo=timezone.utc))
    with pytest.raises(ValidationError):
        WindowSearchInputs(inputs=inputs(), landing_start_utc=datetime(2027, 1, 10, tzinfo=timezone.utc), landing_end_utc=datetime(2027, 1, 1, tzinfo=timezone.utc))


def test_window_endpoint_returns_a_reproducible_screen(client: TestClient):
    response = client.post("/api/mission/windows", json={
        "inputs": inputs(target_latitude_deg=-90, target_longitude_deg=-65).model_dump(mode="json"),
        "landing_start_utc": "2027-01-01T00:00:00Z",
        "landing_end_utc": "2027-02-20T23:00:00Z",
        "min_continuous_sunlight_days": 8,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["required_sunlight_days"] == 8
    assert "Moon phase" in " ".join(body["limitations"])


def test_window_search_enforces_independent_hour_constraints():
    providers = build_providers(Settings(use_spice=False, use_terrain_dem=False))
    result = search_mission_windows(WindowSearchInputs(
        inputs=inputs(),
        landing_start_utc=datetime(2027, 1, 1, tzinfo=timezone.utc),
        landing_end_utc=datetime(2027, 1, 31, 23, tzinfo=timezone.utc),
        min_continuous_sunlight_days=5 / 24,
        min_continuous_earth_hours=8,
    ), providers, run_best_plan=False)
    assert result.candidates
    assert result.required_sunlight_days * 24 == pytest.approx(5)
    assert result.required_earth_hours == 8
    assert all(candidate.continuous_earth_hours_screened >= 8 for candidate in result.candidates)


def test_swarm_without_range_keeps_fixed_site_and_departure():
    from app.services.mission_swarm import MissionInputs, plan_mission
    plan = plan_mission(MissionInputs(departure_utc="2027-01-15T12:00:00Z"))
    assert [s.agent for s in plan.stages][:3] == ["Site selection agent", "Landing window agent", "Launch and parking-orbit agent"]
    assert plan.stages[0].outputs["selected_site"] == "Connecting Ridge (C1)"


def test_removed_fuel_tank_field_is_rejected():
    with pytest.raises(ValidationError):
        inputs(spacecraft_propellant_capacity_kg=5000)


def test_more_loaded_fuel_increases_capacity_and_remaining_fuel():
    baseline = plan_mission(inputs())
    fueled = plan_mission(inputs(spacecraft_wet_mass_kg=7500))
    assert fueled.spacecraft_delta_v_capacity_km_s > baseline.spacecraft_delta_v_capacity_km_s
    assert fueled.stages[5].outputs["propellant_remaining_kg"] > baseline.stages[5].outputs["propellant_remaining_kg"]
    burn = fueled.stages[5].outputs
    assert burn["propellant_used_kg"] + burn["propellant_remaining_kg"] == pytest.approx(7500 - 2400)
    assert fueled.stages[3].outputs["tli_speed_km_s"] == baseline.stages[3].outputs["tli_speed_km_s"]


def test_orbit_altitude_changes_speed_and_period():
    low = plan_mission(inputs(lunar_orbit_altitude_km=100))
    high = plan_mission(inputs(lunar_orbit_altitude_km=400))
    assert high.stages[4].outputs["circular_speed_km_s"] < low.stages[4].outputs["circular_speed_km_s"]
    assert high.stages[4].outputs["orbit_period_hours"] > low.stages[4].outputs["orbit_period_hours"]


def test_orbit_count_and_thrust_change_mission_duration():
    baseline = plan_mission(inputs())
    waiting = plan_mission(inputs(lunar_wait_orbits=3))
    assert waiting.stages[5].outputs["mission_duration_hours"] - baseline.stages[5].outputs["mission_duration_hours"] == pytest.approx(2 * baseline.stages[4].outputs["orbit_period_hours"])
    faster = plan_mission(inputs(landing_thrust_n=90000))
    assert faster.stages[5].outputs["powered_descent_minutes"] == pytest.approx(baseline.stages[5].outputs["powered_descent_minutes"] * 22000 / 90000)
    assert faster.stages[5].outputs["mission_duration_hours"] < baseline.stages[5].outputs["mission_duration_hours"]


def test_window_offset_matches_configured_mission_duration():
    from app.services.mission_windows import _transfer_and_orbit_days
    configured = inputs(lunar_wait_orbits=4, landing_thrust_n=30000)
    plan = plan_mission(configured)
    assert _transfer_and_orbit_days(configured) * 24 == pytest.approx(plan.stages[5].outputs["mission_duration_hours"])


def test_realistic_descent_and_upper_stage_defaults():
    plan = plan_mission(inputs())
    launch, descent = plan.stages[2].outputs, plan.stages[5].outputs
    assert launch["upper_stage_delta_v_left_in_orbit_km_s"] >= plan.stages[3].outputs["tli_delta_v_km_s"]
    assert 8 < descent["powered_descent_minutes"] < 15          # Apollo 11: 12.6 min
    assert 1.5 < descent["twr_at_descent_start"] < 4
    assert plan.overall_status == "provisional_screening_pass"


def test_spacecraft_tli_option_costs_spacecraft_propellant():
    own = plan_mission(inputs(tli_by_upper_stage=False))
    assert own.stages[5].outputs["spacecraft_tli_delta_v_km_s"] > 3
    assert own.overall_status == "screening_failed"              # 6.5 t lander cannot also do TLI with 10 % reserve


def test_delta_v_reserve_is_enforced():
    plan = plan_mission(inputs(dv_margin_pct=100))
    assert any("reserve" in c for c in plan.stages[5].checks)


def test_oversized_engine_is_flagged():
    plan = plan_mission(inputs(landing_thrust_n=200_000))
    assert any("oversized" in c for c in plan.stages[5].checks)


def test_timeline_is_ordered():
    plan = plan_mission(inputs())
    times = [e.utc for e in plan.timeline]
    assert times == sorted(times)
    assert [e.event for e in plan.timeline][:6] == ["Launch", "Parking orbit", "Trans-lunar injection", "Lunar orbit insertion", "Powered descent starts", "Touchdown"]


def test_extra_earth_parking_orbits_delay_injection_without_extra_fuel():
    baseline = plan_mission(inputs())
    waiting = plan_mission(inputs(earth_parking_orbits=3))
    base = baseline.stages[3].outputs
    wait = waiting.stages[3].outputs
    dt = datetime.fromisoformat(wait["tli_utc"]) - datetime.fromisoformat(base["tli_utc"])
    assert dt.total_seconds() / 3600 == pytest.approx(2 * baseline.stages[2].outputs["orbit_period_hours"])
    assert waiting.total_spacecraft_delta_v_km_s == pytest.approx(baseline.total_spacecraft_delta_v_km_s)


def test_earth_orbit_raise_matches_two_impulse_hohmann_transfer():
    from math import sqrt, pi
    from app.services.mission_swarm import earth_orbit_raise, MU_E, R_E
    p = inputs(earth_staging_altitude_km=2000)
    burn = earth_orbit_raise(p)
    r1, r2 = R_E + 200, R_E + 2000
    a = (r1 + r2) / 2
    assert burn["departure_burn_km_s"] == pytest.approx(sqrt(MU_E/r1) * (sqrt(2*r2/(r1+r2)) - 1))
    assert burn["circularization_burn_km_s"] == pytest.approx(sqrt(MU_E/r2) * (1 - sqrt(2*r1/(r1+r2))))
    assert burn["coast_hours"] == pytest.approx(pi*sqrt(a**3/MU_E)/3600)


def test_higher_earth_orbit_is_charged_to_the_correct_vehicle():
    upper = plan_mission(inputs(earth_staging_altitude_km=2000, earth_staging_orbits=2))
    craft = plan_mission(inputs(earth_staging_altitude_km=2000, tli_by_upper_stage=False))
    assert upper.stages[5].outputs["spacecraft_earth_raise_delta_v_km_s"] == 0
    assert craft.stages[5].outputs["spacecraft_earth_raise_delta_v_km_s"] == pytest.approx(craft.stages[3].outputs["earth_orbit_raise_delta_v_km_s"])
    assert craft.total_spacecraft_delta_v_km_s == pytest.approx(sum(craft.stages[5].outputs[k] for k in ("spacecraft_earth_raise_delta_v_km_s", "spacecraft_tli_delta_v_km_s", "descent_delta_v_km_s")) + craft.stages[4].outputs["loi_delta_v_km_s"])
    assert any("orbit-raising" in event.event for event in upper.timeline)
    assert any("staging orbit" in event.event for event in upper.timeline)


def test_earth_staging_altitude_must_exceed_parking_orbit():
    with pytest.raises(ValidationError, match="staging altitude"):
        inputs(earth_staging_altitude_km=100)
