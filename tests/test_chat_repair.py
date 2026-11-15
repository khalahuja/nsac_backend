"""The assistant must cope with the many ways a model phrases the same search."""
import asyncio
import json

from app.api import chat as module
from app.api.chat import ChatRequest, ForecastRequest
from app.config import Settings

CONTEXT = ForecastRequest(latitude=-89.54, longitude=0, start_datetime_utc="2027-01-03T00:00:00Z", end_datetime_utc="2027-02-02T00:00:00Z", step_hours=3, display_timezone="Asia/Calcutta")


def parse(obj):
    reply, problem = module._parse_reply(json.dumps(obj) if not isinstance(obj, str) else obj, CONTEXT)
    assert reply is not None, problem
    return reply


def test_user_example_long_pole_search():
    """'-90 S, 35.09 W, Earth contact >= 3 h, 1 Jan 2027 - 3 Mar 2028' (427 days)."""
    reply = parse({
        "reply": "Searching for Earth-contact windows.",
        "action": {"page": "windows", "forecast": {"latitude": -90, "longitude": -35.09, "start_datetime_utc": "2027-01-01T00:00:00Z",
                   "end_datetime_utc": "2028-03-03T23:59:00Z", "step_hours": 1, "display_timezone": "UTC"},
                   "criteria": {"requireEarth": True, "minHours": 3}},
    })
    assert reply.action.forecast.latitude == -90 and reply.action.forecast.longitude == -35.09
    assert reply.action.forecast.step_hours == 1  # 10,272 samples fits the chunked limit
    assert reply.action.criteria.requireEarth and reply.action.criteria.minHours == 3


def test_repairs_common_slips():
    reply = parse("```json\n" + json.dumps({
        "reply": "ok",
        "action": {"page": "Science Windows", "forecast": {"longitude": 324.91, "start_datetime_utc": "2028-03-03T00:00:00Z",
                   "end_datetime_utc": "2027-01-01T00:00:00Z", "step_hours": None},
                   "criteria": {"requireEarth": True, "minHours": 3, "minTemperature": 50, "maxTemperature": -50, "maxIrradiance": None}},
    }) + "\n```")
    f = reply.action.forecast
    assert reply.action.page == "windows"
    assert round(f.longitude, 2) == -35.09                      # 0-360 east converted
    assert f.start_datetime_utc < f.end_datetime_utc            # reversed dates swapped
    assert f.latitude == CONTEXT.latitude                       # missing fields from context
    assert reply.action.criteria.minTemperature == -50          # swapped range


def test_criteria_without_forecast_uses_context_and_windows_page():
    reply = parse({"reply": "ok", "action": {"page": "forecast", "criteria": {"requireSun": True}}})
    assert reply.action.page == "windows"
    assert reply.action.forecast.latitude == CONTEXT.latitude


def test_too_fine_step_is_coarsened_to_fit():
    reply = parse({"reply": "ok", "action": {"page": "windows", "forecast": {"start_datetime_utc": "2027-01-01T00:00:00Z",
                   "end_datetime_utc": "2029-02-01T00:00:00Z", "step_hours": 0.5}, "criteria": {}}})
    assert reply.action.forecast.step_hours == 1  # 0.5 h would exceed 20,000 samples over 762 days


def test_correction_round_then_success(monkeypatch):
    answers = ['{"reply":"x","action":{"page":"nowhere"}}', '{"reply":"Opening","action":{"page":"windows"}}']
    calls = []

    class Response:
        status_code = 200
        headers = {}
        def __init__(self, text): self.text = text
        def json(self): return {"choices": [{"message": {"content": self.text}}]}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            calls.append(kwargs["json"])
            return Response(answers[len(calls) - 1])

    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key="k", groq_fallback_models=""))
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    result = asyncio.run(module.chat(ChatRequest(messages=[dict(role="user", content="open windows")], context=CONTEXT.model_dump())))
    assert result.action.page == "windows"
    assert "rejected" in calls[1]["messages"][-1]["content"]
