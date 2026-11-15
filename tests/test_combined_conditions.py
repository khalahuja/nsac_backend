from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json
import pytest
from app.api import chat as module
from app.services.science_windows import Criteria, matching_intervals

START = datetime(2027, 1, 1, tzinfo=timezone.utc)
CONTEXT = module.ForecastRequest(latitude=-89, longitude=0, start_datetime_utc=START,
    end_datetime_utc=START + timedelta(days=30), step_hours=1, display_timezone="UTC")


def sample(hour, **changes):
    values = dict(datetime_utc=START + timedelta(hours=hour), temperature_c=10,
        horizontal_irradiance_w_m2=100, earth_visible=hour >= 24, earth_elevation_deg=15,
        illuminated=True, solar_earth_overlap=False)
    return SimpleNamespace(**{**values, **changes})


def test_all_conditions_must_overlap_in_one_interval():
    criteria = Criteria(requireSun=True, minHours=72, minEarthHours=48,
        minTemperature=0, maxTemperature=20, minIrradiance=50, minEarthElevation=10)
    samples = [sample(h) for h in range(73)]
    assert matching_intervals(samples, criteria) == [(START, START + timedelta(hours=72))]
    for changes in [dict(temperature_c=25), dict(horizontal_irradiance_w_m2=20),
                    dict(illuminated=False), dict(solar_earth_overlap=True), dict(earth_elevation_deg=5)]:
        broken = [sample(h, **changes) if h == 40 else sample(h) for h in range(73)]
        assert matching_intervals(broken, criteria) == []
    assert matching_intervals([s for i, s in enumerate(samples) if i != 40], criteria) == []


@pytest.mark.parametrize("text", [
    "Find windows with three days sunlight and two days Earth in view",
    "Find windows with sunlight for 3 days and Earth contact for 2 days",
    "Find windows with temperature between -50 and 50 C and Earth above 10 degrees",
    "Find candidate sites with slope below 5 degrees and 3 days sunlight",
])
def test_audit_rejects_actions_that_drop_conditions(text):
    action = dict(page="windows", forecast=CONTEXT.model_dump(mode="json"), criteria={})
    reply, problem = module._parse_reply(json.dumps(dict(reply="Searching", action=action)), CONTEXT, text)
    assert reply is None and "omitted" in problem


def test_audit_accepts_full_combination():
    reply, problem = module._parse_reply(dict(reply="Searching", action=dict(page="windows",
        forecast=CONTEXT.model_dump(mode="json"), criteria=dict(requireSun=True, minHours=72,
            minEarthHours=48, minTemperature=-50, maxTemperature=50, minEarthElevation=10))), CONTEXT,
        "Find 3 days sunlight and 2 days Earth in view, temperature between -50 and 50 C, Earth above 10")
    assert reply is not None, problem


def test_unresolved_requirement_prevents_partial_search():
    reply, problem = module._parse_reply(dict(reply="Searching", unresolved_conditions=["Measured water probability is unavailable"],
        action=dict(page="sites", site_ranking={})), CONTEXT)
    assert reply.action is None
    assert "Measured water probability" in reply.reply


def test_candidate_followup_retains_all_other_conditions():
    old = module.Action(page="sites", site_ranking=dict(start_utc=START, end_utc=START + timedelta(days=30),
        min_sunlight_days=3, priority="water", max_slope_deg=5,
        criteria=dict(minEarthHours=48, minTemperature=-50, maxTemperature=50)))
    request = module.ChatRequest(context=CONTEXT, active_search=old,
        messages=[module.Message(role="user", content="Also require Earth above 10 degrees")])
    reply = module.Reply(reply="Searching", action=module.Action(page="sites",
        site_ranking=dict(criteria=dict(minEarthElevation=10))))
    merged = module._merge_candidate_followup(reply, request).action.site_ranking
    assert merged.min_sunlight_days == 3 and merged.priority == "water" and merged.max_slope_deg == 5
    assert merged.criteria.minEarthHours == 48 and merged.criteria.minTemperature == -50
    assert merged.criteria.minEarthElevation == 10 and merged.end_utc == old.site_ranking.end_utc


def test_candidate_combination_runs_shared_matcher_and_requires_measurements(monkeypatch):
    from app.services import site_ranking as ranking
    from app.services.site_ranking import SiteRankingRequest
    monkeypatch.setattr(ranking, "CANDIDATE_SITES", [("Pass", -89, 1), ("Hot", -88, 2), ("No terrain", -87, 3)])
    monkeypatch.setattr(ranking, "_site_metrics", lambda lat, lon, *args: {
        "slope_deg": None if lon == 3 else 2, "sunlit_pct": 100, "earth_visible_pct": 80})
    monkeypatch.setattr(ranking, "compute_surface_screen", lambda *args: {"safe_fraction": 0.95})
    monkeypatch.setattr(ranking, "nearest_ice_crater", lambda *args: ("Detected ice", 1))
    monkeypatch.setattr(ranking, "horizon_or_none", lambda *args: None)
    def forecast(**kwargs):
        assert kwargs["start"] == START and kwargs["end"] == START + timedelta(days=4)
        samples = []
        for h in range(97):
            row = sample(h, temperature_c=30 if kwargs["longitude_deg"] == 2 else 10,
                solar_azimuth_deg=45, solar_elevation_deg=10, earth_azimuth_deg=0)
            row.model_copy = lambda update, row=row: SimpleNamespace(**{**vars(row), **update})
            samples.append(row)
        return SimpleNamespace(samples=samples)
    monkeypatch.setattr(ranking, "compute_forecast", forecast)
    providers = SimpleNamespace(terrain=SimpleNamespace(source="test"), astronomy=SimpleNamespace(source="test"))
    result = ranking.rank_sites(SiteRankingRequest(start_utc=START, end_utc=START + timedelta(days=4),
        min_sunlight_days=3, priority="water", max_slope_deg=5, min_safe_fraction=0.9,
        max_ice_distance_km=2, criteria=Criteria(minEarthHours=48, minEarthElevation=10,
            minTemperature=0, maxTemperature=20, minIrradiance=50)), providers)
    assert [s.name for s in result.sites] == ["Pass"]
    assert result.sites[0].continuous_sunlight_days == 4


def test_correction_round_retries_semantically_incomplete_actions(monkeypatch):
    import asyncio
    from app.config import Settings
    calls = []
    partial = dict(reply="Searching", action=dict(page="windows", forecast=CONTEXT.model_dump(mode="json"),
        criteria=dict(requireSun=True, minHours=72)))
    complete = dict(reply="Searching", action=dict(page="windows", forecast=CONTEXT.model_dump(mode="json"),
        criteria=dict(requireSun=True, minHours=72, minEarthHours=48)))
    class Response:
        status_code = 200
        def __init__(self, body): self.body = body
        def json(self): return {"choices": [{"message": {"content": json.dumps(self.body)}}]}
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            calls.append(kwargs["json"])
            return Response(partial if len(calls) == 1 else complete)
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    reply, ok = asyncio.run(module._ask_groq(Settings(groq_api_key="test"), [dict(role="user",
        content="Find 3 days sunlight and 2 days Earth in view")], CONTEXT))
    assert ok and reply.action.criteria.minEarthHours == 48
    assert len(calls) == 2
    assert "Earth visibility duration" in calls[1]["messages"][-1]["content"]


def test_followup_ignores_unrelated_defaults_in_full_model_output():
    old = Criteria(requireSun=True, minHours=72, minEarthHours=48, minTemperature=-50, maxTemperature=50)
    new = Criteria.model_validate({**Criteria().model_dump(), "minEarthElevation": 10})
    merged = module._merge_criteria_fields(old, new, "also require Earth above 10")
    assert merged.requireSun and merged.minHours == 72
    assert merged.minTemperature == -50 and merged.maxTemperature == 50
    assert merged.minEarthElevation == 10 and merged.minEarthHours == 48
