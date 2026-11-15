import asyncio

import httpx
import pytest
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

from app import chat_security
from app.api.chat import Action
from app.config import Settings
from pydantic import ValidationError


@pytest.mark.parametrize("declared", [None, b"8", b"2048"])
def test_actual_stream_limit_precedes_json_parsing(monkeypatch, declared):
    monkeypatch.setattr(chat_security, "get_settings", lambda: Settings(chat_max_body_bytes=1024))
    called = []
    async def downstream(scope, receive, send):
        called.append(True)
    messages = iter([{"type": "http.request", "body": b"x" * 700, "more_body": True}, {"type": "http.request", "body": b"x" * 700, "more_body": False}])
    async def receive(): return next(messages)
    sent = []
    async def send(message): sent.append(message)
    scope = {"type": "http", "path": "/api/chat", "method": "POST", "headers": [] if declared is None else [(b"content-length", declared)]}
    asyncio.run(chat_security.ChatBodyLimit(downstream)(scope, receive, send))
    assert not called
    assert sent[0]["status"] == 413


def test_stream_body_replayed_once(monkeypatch):
    monkeypatch.setattr(chat_security, "get_settings", lambda: Settings(chat_max_body_bytes=1024))
    messages = iter([{"type": "http.request", "body": b'{"hi":', "more_body": True}, {"type": "http.request", "body": b'1}', "more_body": False}, {"type": "http.disconnect"}])
    async def receive(): return next(messages)
    async def downstream(scope, replay, send):
        assert (await replay())["body"] == b'{"hi":1}'
        assert (await replay())["type"] == "http.disconnect"
        await JSONResponse({"ok": True})(scope, replay, send)
    sent = []
    async def send(message): sent.append(message)
    asyncio.run(chat_security.ChatBodyLimit(downstream)({"type": "http", "path": "/api/chat", "method": "POST", "headers": []}, receive, send))
    assert sent[0]["status"] == 200


@pytest.mark.parametrize("result,expected", [
    ({"success": True, "action": "chat", "hostname": "localhost"}, None),
    ({"success": False, "action": "chat", "hostname": "localhost"}, 403),
    ({"success": True, "action": "login", "hostname": "localhost"}, 403),
    ({"success": True, "action": "chat", "hostname": "external.example"}, 403),
    ({"success": True}, 403),
])
def test_challenge_checks_success_action_and_hostname(monkeypatch, result, expected):
    settings = Settings(turnstile_secret_key="test-private-key")
    monkeypatch.setattr(chat_security, "get_settings", lambda: settings)
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json):
            assert url.startswith("https://challenges.cloudflare.com/")
            assert json["secret"] == "test-private-key" and json["response"] == "test-token"
            return httpx.Response(200, json=result, request=httpx.Request("POST", url))
    monkeypatch.setattr(chat_security.httpx, "AsyncClient", Client)
    request = Request({"type": "http", "headers": [(b"origin", b"http://localhost:5173"), (b"x-turnstile-token", b"test-token")], "client": ("127.0.0.1", 1234)})
    if expected:
        with pytest.raises(HTTPException) as error:
            asyncio.run(chat_security.verify_chat_challenge(request))
        assert error.value.status_code == expected
        assert "test-private-key" not in error.value.detail
    else:
        asyncio.run(chat_security.verify_chat_challenge(request))


def test_challenge_missing_token_is_rejected(monkeypatch):
    monkeypatch.setattr(chat_security, "get_settings", lambda: Settings(turnstile_secret_key="test-private-key"))
    with pytest.raises(HTTPException) as error:
        asyncio.run(chat_security.verify_chat_challenge(Request({"type": "http", "headers": []})))
    assert error.value.status_code == 403


def test_mission_inputs_cannot_cross_tools_or_exceed_probability_bounds():
    assert Action(page="landing", inputs={"sigma_m": 40, "safe_terrain_pct": 95}).inputs.sigma_m == 40
    assert Action(page="orbit", inputs={"periapsis_km": 100, "apoapsis_km": 100}).page == "orbit"
    for action in [
        {"page": "landing", "inputs": {"vehicle_success_pct": 101}},
        {"page": "water", "inputs": {"sigma_m": 40}},
        {"page": "forecast", "inputs": {"region_query": "Nobile"}},
        {"page": "orbit", "inputs": {"periapsis_km": 100, "apoapsis_km": 50}},
        {"page": "orbit", "inputs": {"departure_local": "2027-02-30T12:00"}},
        {"page": "landing", "inputs": {"arbitrary_code": "execute"}},
    ]:
        with pytest.raises(ValidationError): Action(**action)


@pytest.mark.parametrize("headers,status", [
    ([(b"content-length", b"-1")], 400),
    ([(b"content-length", b"no")], 400),
    ([(b"content-length", b"1"), (b"content-length", b"2")], 400),
    ([(b"content-encoding", b"gzip")], 415),
    ([(b"content-length", b"3")], 400),
])
def test_malformed_body_headers_are_rejected(monkeypatch, headers, status):
    monkeypatch.setattr(chat_security, "get_settings", lambda: Settings(_env_file=None))
    async def downstream(scope, receive, send):
        raise AssertionError("Malformed request reached the endpoint")
    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}
    sent = []
    async def send(message): sent.append(message)
    asyncio.run(chat_security.RequestBodyLimit(downstream)(
        {"type": "http", "path": "/api/chat", "method": "POST", "headers": headers}, receive, send))
    assert sent[0]["status"] == status


def test_slow_body_is_rejected_without_calling_endpoint(monkeypatch):
    monkeypatch.setattr(chat_security, "get_settings", lambda: Settings(_env_file=None, api_body_timeout_seconds=1))
    async def downstream(scope, receive, send):
        raise AssertionError("Slow request reached the endpoint")
    async def receive():
        await asyncio.sleep(10)
        return {"type": "http.request", "body": b"{}", "more_body": False}
    sent = []
    async def send(message): sent.append(message)
    asyncio.run(chat_security.RequestBodyLimit(downstream)(
        {"type": "http", "path": "/api/chat", "method": "POST", "headers": []}, receive, send))
    assert sent[0]["status"] == 408
