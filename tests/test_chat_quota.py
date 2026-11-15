import asyncio

from app.api import chat as module
from app.api.chat import ChatRequest
from app.config import Settings

CONTEXT = dict(latitude=-89.54, longitude=0, start_datetime_utc="2027-01-03T00:00:00Z", end_datetime_utc="2027-02-02T00:00:00Z", step_hours=3, display_timezone="UTC")
GOOD = '{"reply":"Opening windows","action":{"page":"windows"}}'


def _client(statuses, calls):
    """Fake httpx client answering each post with the next status code."""
    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {}
        def json(self):
            return {"choices": [{"message": {"content": GOOD}}]}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            calls.append(kwargs["json"])
            return Response(statuses[len(calls) - 1])
    return Client


def _run(monkeypatch, statuses, **settings):
    calls = []
    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key="k", **settings))
    monkeypatch.setattr(module.httpx, "AsyncClient", _client(statuses, calls))
    request = ChatRequest(messages=[dict(role="user", content="Open windows")], context=CONTEXT)
    return asyncio.run(module.chat(request)), calls


def test_falls_back_to_next_model_on_quota(monkeypatch):
    reply, calls = _run(monkeypatch, [429, 200], groq_model="primary", groq_fallback_models="backup")
    assert reply.action.page == "windows"
    assert [c["model"] for c in calls] == ["primary", "backup"]


def test_repeated_question_is_served_from_cache(monkeypatch):
    _run(monkeypatch, [200], groq_model="primary", groq_fallback_models="")
    reply, calls = _run(monkeypatch, [200], groq_model="primary", groq_fallback_models="")
    assert reply.action.page == "windows" and calls == []


def test_request_is_slim(monkeypatch):
    _, calls = _run(monkeypatch, [200], groq_model="openai/gpt-oss-120b", groq_fallback_models="", chat_max_output_tokens=900)
    body = calls[0]
    assert body["max_tokens"] == 900
    assert body["reasoning_effort"] == "low"
    assert '"title"' not in body["messages"][0]["content"]


def test_only_recent_history_is_sent(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key="k", groq_fallback_models="", chat_history_messages=4))
    monkeypatch.setattr(module.httpx, "AsyncClient", _client([200], calls))
    turns = [dict(role="user" if i % 2 == 0 else "assistant", content=f"m{i}") for i in range(9)]
    asyncio.run(module.chat(ChatRequest(messages=turns, context=CONTEXT)))
    sent = calls[0]["messages"][1:]
    assert [m["content"] for m in sent] == ["m5", "m6", "m7", "m8"]


def test_bad_request_tries_next_configured_model(monkeypatch):
    reply, calls = _run(monkeypatch, [400, 200], groq_model="primary", groq_fallback_models="backup")
    assert reply.action.page == "windows"
    assert [call["model"] for call in calls] == ["primary", "backup"]


def test_json_generation_failure_retries_with_local_validation(monkeypatch):
    calls = []
    class Response:
        headers = {}
        def __init__(self, status): self.status_code = status
        def json(self):
            if self.status_code == 400:
                return {"error": {"code": "json_validate_failed", "message": "private-test-key and user content", "failed_generation": "malformed output"}}
            return {"choices": [{"message": {"content": GOOD}}]}
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs):
            calls.append(kwargs["json"])
            return Response(400 if len(calls) == 1 else 200)
    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key="k", groq_fallback_models=""))
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    reply = asyncio.run(module.chat(ChatRequest(messages=[dict(role="user", content="Open windows")], context=CONTEXT)))
    assert reply.action.page == "windows"
    assert len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    assert calls[0]["model"] == calls[1]["model"]


def test_all_bad_requests_are_bounded_and_do_not_expose_provider_output(monkeypatch):
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        _run(monkeypatch, [400, 400], groq_model="primary", groq_fallback_models="backup")
    assert error.value.status_code == 503
    assert "configured models" in error.value.detail
    assert "private-test-key" not in error.value.detail


def test_unavailable_model_tries_backup(monkeypatch):
    reply, calls = _run(monkeypatch, [404, 200], groq_model="primary", groq_fallback_models="backup")
    assert reply.action.page == "windows"
    assert len(calls) == 2


def test_server_error_tries_backup(monkeypatch):
    reply, calls = _run(monkeypatch, [503, 200], groq_model="primary", groq_fallback_models="backup")
    assert reply.action.page == "windows"
    assert len(calls) == 2


def test_missing_backup_does_not_hide_quota(monkeypatch):
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        _run(monkeypatch, [429, 404], groq_model="primary", groq_fallback_models="backup")
    assert "quota" in error.value.detail


def test_missing_backup_does_not_hide_bad_request(monkeypatch):
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        _run(monkeypatch, [400, 404], groq_model="primary", groq_fallback_models="backup")
    assert "valid response" in error.value.detail
