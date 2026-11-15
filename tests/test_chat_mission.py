"""The chatbot must run the mission date search, never invent its results."""
import json
import pytest

from app.api import chat as module
from app.api.chat import ChatRequest, ForecastRequest, Message, Reply

CONTEXT = ForecastRequest(latitude=-89.54, longitude=0, start_datetime_utc="2027-01-03T00:00:00Z", end_datetime_utc="2027-02-02T00:00:00Z", step_hours=3, display_timezone="UTC")
ASK = [Message(role="user", content="reach -90 S 65 W between 1st January 2027 and 12th October 2027, launch and landing with 8 days of sunlight")]


def test_mission_search_action_is_accepted():
    reply, problem = module._parse_reply(json.dumps({
        "reply": "Running the date search; results appear below.",
        "action": {"page": "mission-plan", "mission_windows": {"latitude": -90, "longitude": -65, "landing_start_utc": "2027-01-01T00:00:00Z", "landing_end_utc": "2027-10-12T00:00:00Z", "min_sunlight_days": 8}},
    }), CONTEXT)
    assert reply is not None, problem
    assert reply.action.mission_windows.min_sunlight_days == 8


def test_invented_dates_are_suppressed():
    invented = Reply(reply="The earliest viable launch window starts on 2027-02-15, landing 2027-02-25.", action=None)
    guarded = module._guard_invented_results(invented, ASK)
    assert "2027-02-15" not in guarded.reply and "can't state dates" in guarded.reply


def test_dates_backed_by_a_computation_pass():
    ok = Reply.model_validate({"reply": "Searching 2027-01-01 to 2027-10-12.", "action": {"page": "swarm", "mission_windows": {"latitude": -90, "longitude": -65, "landing_start_utc": "2027-01-01T00:00:00Z", "landing_end_utc": "2027-10-12T00:00:00Z"}}})
    assert module._guard_invented_results(ok, ASK).reply == ok.reply


def test_earth_contact_followup_preserves_eight_day_sunlight_requirement():
    previous = module.Action(page="swarm", mission_windows={
        "latitude": -8.33, "longitude": 8.49,
        "landing_start_utc": "2027-01-01T00:00:00Z",
        "landing_end_utc": "2027-12-31T23:00:00Z", "min_sunlight_days": 8,
    })
    request = ChatRequest(messages=[Message(role="user", content="Yes, run windows search with Earth contact")],
                          context=CONTEXT, active_search=previous)
    proposed = Reply.model_validate({"reply": "Searching with Earth contact.",
        "action": {"page": "windows", "forecast": CONTEXT.model_dump(mode="json"),
                   "criteria": {"requireEarth": True}}})
    fixed = module._preserve_window_requirements(proposed, request)
    assert fixed.action.criteria.requireEarth and fixed.action.criteria.requireSun
    assert fixed.action.criteria.minHours == 192
    assert fixed.action.forecast.latitude == -8.33
    assert fixed.action.forecast.start_datetime_utc.year == 2027


def test_earth_threshold_followup_stays_a_window_search():
    previous = module.Action(page="windows", forecast=CONTEXT, criteria={"requireEarth": True})
    request = ChatRequest(messages=[Message(role="user", content="Show only times when Earth >10°")],
                          context=CONTEXT, active_search=previous)
    proposed = Reply.model_validate({"reply": "Generating a forecast", "action": {"page": "forecast", "forecast": CONTEXT.model_dump(mode="json")}})
    fixed = module._normalize_science_followup(proposed, request)
    assert fixed.action.page == "windows"
    assert fixed.action.criteria.requireEarth
    assert fixed.action.criteria.minEarthElevation == 10


def test_date_followup_preserves_previous_earth_threshold():
    previous = module.Action(page="windows", forecast=CONTEXT,
                             criteria={"requireEarth": True, "minEarthElevation": 10})
    shorter = CONTEXT.model_copy(update={"end_datetime_utc": module.datetime.fromisoformat("2027-01-10T00:00:00+00:00")})
    request = ChatRequest(messages=[Message(role="user", content="Show next week only")],
                          context=CONTEXT, active_search=previous)
    proposed = Reply.model_validate({"reply": "Generating a forecast", "action": {
        "page": "forecast", "forecast": shorter.model_dump(mode="json")}})
    fixed = module._normalize_science_followup(proposed, request)
    assert fixed.action.page == "windows"
    assert fixed.action.criteria.minEarthElevation == 10
    assert fixed.action.forecast.end_datetime_utc == shorter.end_datetime_utc


def test_candidate_finder_does_not_claim_water_probability_ranking():
    proposed = Reply.model_validate({"reply": "Here are the top sites with highest water probability.",
                                     "action": {"page": "sites"}})
    fixed = module._guard_site_ranking_claim(proposed)
    assert "does not calculate" in fixed.reply


def test_claimed_actions_without_computation_are_replaced():
    for text in ["South Pole A added to the comparison list.",
                 "Here are the top South-Pole sites filtered for high landing success and water probability.",
                 "Displaying comparison table for selected sites."]:
        guarded = module._guard_invented_results(Reply.model_validate({"reply": text, "action": {"page": "compare"}}), ASK)
        assert "can't add, select or show results" in guarded.reply, text
        assert guarded.action.page == "compare"


def test_site_ranking_action_passes_guard_and_routes_to_sites():
    reply, problem = module._parse_reply(json.dumps({"reply": "Here are the top sites; ranking below.", "action": {"page": "compare", "site_ranking": {}}}), CONTEXT)
    assert reply is not None, problem
    assert reply.action.page == "sites"
    assert module._guard_invented_results(reply, ASK).reply == reply.reply


def test_ice_crater_distance():
    from app.services.site_ranking import nearest_ice_crater
    assert nearest_ice_crater(-89.67, 129.78) == ("Shackleton", 0.0)
    name, km = nearest_ice_crater(-89.4398, -137.17)
    assert name == "Shackleton" and 5 < km < 15


def test_compare_sites_action_routes_to_compare_and_allows_claims():
    reply, problem = module._parse_reply(json.dumps({
        "reply": "Added Connecting Ridge (C1) and de Gerlache Rim (GR1) to the comparison list.",
        "action": {"page": "sites", "compare_sites": {"mode": "add", "sites": [
            {"name": "Connecting Ridge (C1)", "latitude": -89.4398, "longitude": -137.17},
            {"name": "de Gerlache Rim (GR1)", "latitude": -88.6952, "longitude": -68.2846}]}},
    }), CONTEXT)
    assert reply is not None, problem
    assert reply.action.page == "compare" and len(reply.action.compare_sites.sites) == 2
    assert module._guard_invented_results(reply, ASK).reply == reply.reply  # the claim is true now


def test_compare_sites_limited_to_five():
    reply, _ = module._parse_reply(json.dumps({"reply": "ok", "action": {"page": "compare", "compare_sites": {"sites": [
        {"name": f"P{i}", "latitude": -89, "longitude": i} for i in range(6)]}}}), CONTEXT)
    assert reply is None or reply.action is None or reply.action.compare_sites is None


def test_explicit_january_science_search_runs_without_provider(monkeypatch):
    import asyncio
    def unexpected_provider():
        raise AssertionError("Complete science request should not call the provider")
    monkeypatch.setattr(module, "get_settings", unexpected_provider)
    context = CONTEXT.model_copy(update={"display_timezone": "Asia/Calcutta"})
    previous = module.Action(page="swarm", mission_windows={
        "latitude": -90, "longitude": -65,
        "landing_start_utc": "2027-01-01T00:00:00Z",
        "landing_end_utc": "2027-02-01T00:00:00Z", "min_sunlight_days": 8,
    })
    request = ChatRequest(context=context, active_search=previous, messages=[Message(
        role="user", content="Find windows in January 2027 with Earth in view and temperature between -50 and 50 \u00b0C")])
    reply = asyncio.run(module.chat(request))
    action = reply.action
    assert action.page == "windows"
    assert action.forecast.latitude == context.latitude
    assert action.forecast.longitude == context.longitude
    assert action.forecast.start_datetime_utc.isoformat() == "2026-12-31T18:30:00+00:00"
    assert action.forecast.end_datetime_utc.isoformat() == "2027-01-31T18:29:59+00:00"
    assert action.forecast.step_hours == 1
    assert action.criteria.requireEarth
    assert not action.criteria.requireSun
    assert action.criteria.minHours == 0
    assert (action.criteria.minTemperature, action.criteria.maxTemperature) == (-50, 50)
    assert module._guard_invented_results(reply, request.messages) == reply


def test_month_science_parser_does_not_drop_extra_constraints():
    for suffix in [" and at least 8 days of sunlight", " at -90, -65"]:
        request = ChatRequest(context=CONTEXT, messages=[Message(role="user", content=
            "Find windows in January 2027 with Earth in view and temperature between -50 and 50 C" + suffix)])
        assert module._explicit_month_science_search(request) is None


def test_month_science_search_december_rollover_and_invalid_range():
    request = ChatRequest(context=CONTEXT, messages=[Message(role="user", content=
        "Find windows in December 2027 with Earth visible and temperature between -50 and 50 Celsius")])
    reply = module._explicit_month_science_search(request)
    assert reply.action.forecast.end_datetime_utc.isoformat() == "2027-12-31T23:59:59+00:00"
    request.messages[-1].content = request.messages[-1].content.replace("-50 and 50", "50 and -50")
    assert module._explicit_month_science_search(request).action is None


def test_separate_sunlight_and_earth_durations_run_without_provider(monkeypatch):
    import asyncio
    def unexpected_provider():
        raise AssertionError("Explicit duration search must not call the provider")
    monkeypatch.setattr(module, "get_settings", unexpected_provider)
    context = CONTEXT.model_copy(update={"display_timezone": "Asia/Calcutta"})
    request = ChatRequest(context=context, messages=[Message(role="user", content=
        "find me a time window with 3 days sunlight and 2 days earth in view in february 2027")])
    reply = asyncio.run(module.chat(request))
    assert reply.action.page == "windows"
    assert reply.action.criteria.minHours == 72
    assert reply.action.criteria.minEarthHours == 48
    assert reply.action.criteria.requireSun
    assert not reply.action.criteria.requireEarth
    assert reply.action.forecast.latitude == context.latitude
    assert reply.action.forecast.start_datetime_utc.isoformat() == "2027-01-31T18:30:00+00:00"
    assert reply.action.forecast.end_datetime_utc.isoformat() == "2027-02-28T18:29:59+00:00"
    assert module._guard_invented_results(reply, request.messages) == reply
    request.messages[-1].content += " and temperature above 20 C"
    assert module._explicit_duration_search(request) is None


def test_agent_swarm_hour_constraints_bypass_slow_provider(monkeypatch):
    import asyncio
    def unexpected_provider():
        raise AssertionError("Complete mission request should not call the language model")
    monkeypatch.setattr(module, "get_settings", unexpected_provider)
    context = CONTEXT.model_copy(update={"display_timezone": "Asia/Calcutta"})
    request = ChatRequest(context=context, messages=[Message(role="user", content=
        "run agent swarm and plan a mission for january 2027 where sunlight is there for atleast 5 hours "
        "and earth is visible for atleast 8 hours. calculate landing date etc and give all results")])
    reply = asyncio.run(module.chat(request))
    mission = reply.action.mission_windows
    assert reply.action.page == "swarm"
    assert mission.min_sunlight_days * 24 == pytest.approx(5)
    assert mission.min_earth_hours == 8
    assert mission.latitude == context.latitude and mission.longitude == context.longitude
    assert mission.landing_start_utc.isoformat() == "2026-12-31T18:30:00+00:00"
    assert mission.landing_end_utc.isoformat() == "2027-01-31T18:29:59+00:00"


def test_earth_threshold_followup_preserves_separate_contact_duration():
    previous = module.Action(page="windows", forecast=CONTEXT,
        criteria={"requireSun": True, "minHours": 72, "minEarthHours": 48})
    request = ChatRequest(context=CONTEXT, active_search=previous,
        messages=[Message(role="user", content="Earth above 10")])
    proposed = Reply.model_validate({"reply": "Searching", "action": {
        "page": "forecast", "forecast": CONTEXT.model_dump(mode="json")}})
    fixed = module._normalize_science_followup(proposed, request)
    assert fixed.action.criteria.minEarthHours == 48
    assert fixed.action.criteria.minHours == 72
    assert fixed.action.criteria.minEarthElevation == 10
    assert not fixed.action.criteria.requireEarth


def test_candidate_request_preserves_dates_and_sunlight_without_provider(monkeypatch):
    import asyncio
    def unexpected_provider():
        raise AssertionError("Explicit candidate search must not call the provider")
    monkeypatch.setattr(module, "get_settings", unexpected_provider)
    context = CONTEXT.model_copy(update={"display_timezone": "Asia/Calcutta"})
    request = ChatRequest(context=context, messages=[Message(role="user", content=
        "find me a candidate site on south pole with atleast 3 day sunlight , high water probability from 2 january,2027 to 4 februrary, 2027")])
    reply = asyncio.run(module.chat(request))
    ranking = reply.action.site_ranking
    assert ranking.min_sunlight_days == 3
    assert ranking.priority == "water"
    assert ranking.start_utc.isoformat() == "2027-01-01T18:30:00+00:00"
    assert ranking.end_utc.isoformat() == "2027-02-04T18:29:59+00:00"
    assert ranking.display_timezone == "Asia/Calcutta"
    assert "continuous" in reply.reply
    assert "not a measured water probability" in reply.reply
    request.messages[-1].content += " and Earth visible for 2 days"
    assert module._explicit_candidate_search(request) is None
