import asyncio
import pytest
from pydantic import ValidationError
from fastapi import HTTPException
from app.api import chat as module
from app.api.chat import Action, ChatRequest, Criteria, MissionWindowSearch
from app.config import Settings

CONTEXT = dict(latitude=-89.54, longitude=0, start_datetime_utc="2027-01-03T00:00:00Z", end_datetime_utc="2027-02-02T00:00:00Z", step_hours=3, display_timezone="Asia/Calcutta")

def test_rejects_reversed_or_excessive_ranges():
    for change in [dict(end_datetime_utc="2027-01-01T00:00:00Z"), dict(step_hours=0.01), dict(end_datetime_utc="2029-06-01T00:00:00Z")]:
        with pytest.raises(ValidationError):
            Action(page="windows", forecast={**CONTEXT, **change}, criteria={})

def test_rejects_invalid_filters_and_navigation():
    with pytest.raises(ValidationError):
        Criteria(minTemperature=35, maxTemperature=30)
    with pytest.raises(ValidationError):
        Action(page="windows", criteria={})
    with pytest.raises(ValidationError):
        Action(page="https://example.com")
    with pytest.raises(ValidationError):
        Action(page="forecast", tool="landing")
    assert Action(page="calculator", tool="landing").tool == "landing"


def test_swarm_page_is_a_valid_chat_context_and_action():
    request = ChatRequest(messages=[dict(role="user", content="Run the mission swarm")], page="swarm", context=CONTEXT)
    assert request.page == "swarm"
    assert Action(page="swarm").page == "swarm"


def test_mission_window_chat_action_requires_zoned_dates():
    with pytest.raises(ValidationError):
        MissionWindowSearch(latitude=-90, longitude=-65, landing_start_utc="2027-01-01", landing_end_utc="2027-10-12")
    action = Action(page="swarm", mission_windows=MissionWindowSearch(
        latitude=-90, longitude=-65,
        landing_start_utc="2027-01-01T00:00:00Z",
        landing_end_utc="2027-10-12T23:00:00Z",
        min_sunlight_days=8,
    ))
    assert action.mission_windows.min_sunlight_days == 8

def test_missing_key(monkeypatch):
    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key=""))
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.chat(ChatRequest(messages=[dict(role="user", content="Open windows")], context=CONTEXT)))
    assert error.value.status_code == 503

@pytest.mark.parametrize("content,expected", [(' {"reply":"Which year?","action":null}', None), ('{"reply":"Open windows","action":{"page":"windows"}}', "windows"), ('{"reply":"Bad","action":{"page":"execute"}}', "error")])
def test_model_output_validation(monkeypatch, content, expected):
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": content}}]}
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            assert kwargs["headers"]["Authorization"] == "Bearer test-key"
            assert kwargs["json"]["response_format"] == {"type": "json_object"}
            return Response()
    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key="test-key"))
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    request = ChatRequest(messages=[dict(role="user", content="Open windows")], context=CONTEXT)
    if expected == "error":
        # Unusable output becomes a helpful answer (after one correction round), not an error.
        result = asyncio.run(module.chat(request))
        assert result.action is None
        assert "couldn't turn it into a search" in result.reply
        assert result.options
    else:
        result = asyncio.run(module.chat(request))
        assert (result.action.page if result.action else None) == expected


@pytest.mark.parametrize("status,message", [(401,"API key"), (403,"permissions"), (404,"unavailable"), (429,"quota"), (500,"HTTP 500")])
def test_upstream_errors_are_actionable_and_do_not_expose_secrets(monkeypatch, status, message):
    class Response:
        status_code = status
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): return Response()
    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key="private-test-key"))
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.chat(ChatRequest(messages=[dict(role="user", content="Open windows")], context=CONTEXT)))
    assert message in error.value.detail
    assert "private-test-key" not in error.value.detail
