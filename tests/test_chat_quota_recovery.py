"""Provider quota recovery without real network calls or production credentials."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import chat as module
from app.api.chat import ChatRequest, Message
from app.config import Settings

CONTEXT = dict(latitude=-89.54, longitude=0, start_datetime_utc="2027-01-01T00:00:00Z",
               end_datetime_utc="2027-02-01T00:00:00Z", step_hours=1, display_timezone="UTC")


def fake_client(monkeypatch, replies):
    calls = []
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            calls.append(kwargs["json"])
            status, headers = replies.pop(0)
            return SimpleNamespace(status_code=status, headers=headers,
                json=lambda: {"choices": [{"message": {"content": '{"reply":"Ready","action":null}'}}]})
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    return calls


def ask(settings):
    return asyncio.run(module._ask_groq(settings, [{"role": "user", "content": "Explain thermal inertia"}],
                                        module.ForecastRequest(**CONTEXT)))


def test_provider_wait_is_propagated_and_repeated_requests_do_not_hit_cooled_models(monkeypatch):
    calls = fake_client(monkeypatch, [(429, {"retry-after": "120.5"}), (429, {"retry-after": "30"})])
    settings = Settings(groq_api_key="test", groq_model="primary", groq_fallback_models="backup")
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    for _ in range(2):
        with pytest.raises(HTTPException) as failure:
            ask(settings)
        assert failure.value.status_code == 503
        assert failure.value.headers["Retry-After"] == "30"
        assert failure.value.headers["X-Selene-Error"] == "upstream_rate_limit"
    assert len(calls) == 2


def test_after_reset_the_provider_is_tried_again(monkeypatch):
    calls = fake_client(monkeypatch, [(429, {"retry-after": "20"}), (200, {})])
    clock = [100]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    settings = Settings(groq_api_key="test", groq_model="primary", groq_fallback_models="")
    with pytest.raises(HTTPException):
        ask(settings)
    clock[0] = 121
    reply, success = ask(settings)
    assert success and reply.reply == "Ready"
    assert len(calls) == 2


def test_changing_credentials_does_not_reuse_an_old_quota_wait(monkeypatch):
    calls = fake_client(monkeypatch, [(429, {"retry-after": "90"}), (200, {})])
    with pytest.raises(HTTPException):
        ask(Settings(groq_api_key="first", groq_model="primary", groq_fallback_models=""))
    assert ask(Settings(groq_api_key="second", groq_model="primary", groq_fallback_models=""))[1]
    assert len(calls) == 2


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "invalid", None])
def test_invalid_retry_headers_do_not_create_negative_or_infinite_waits(value):
    assert module._retry_after(SimpleNamespace(headers={} if value is None else {"retry-after": value})) is None


def test_history_keeps_recent_complete_messages_and_never_truncates_the_latest_request():
    messages = [Message(role="user", content="old " * 1400), Message(role="assistant", content="reply"),
                Message(role="user", content="latest " * 700)]
    history = module._bounded_history(messages, 6, 6000)
    assert [m["content"] for m in history] == [messages[1].content, messages[2].content]
    assert sum(len(m["content"]) for m in history) <= 6000
    assert messages[0].content == "old " * 1400


def test_deterministic_search_still_runs_during_provider_cooldown(monkeypatch):
    settings = Settings(groq_api_key="test", groq_model="primary", groq_fallback_models="")
    fake_client(monkeypatch, [(429, {"retry-after": "300"})])
    with pytest.raises(HTTPException):
        ask(settings)
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    payload = ChatRequest(context=CONTEXT, messages=[dict(role="user",
        content="Find windows with 2 days sunlight and 1 days Earth in view in January 2027")])
    result = asyncio.run(module.chat(payload))
    assert result.action.criteria.requireSun
    assert result.action.criteria.minEarthHours == 24


def test_cors_exposes_upstream_recovery_headers(client):
    response = client.get("/api/chat/config", headers={"Origin": "http://localhost:5173", "User-Agent": "Mozilla/5.0"})
    assert response.status_code == 200
    exposed = response.headers["access-control-expose-headers"].lower()
    assert "retry-after" in exposed and "x-selene-error" in exposed
