from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import rate_limit
from app.api import chat as chat_module
from app.config import Settings
from app.main import create_app
from app.rate_limit import Rule, SlidingWindowLimiter

CONTEXT = dict(latitude=-89.54, longitude=0, start_datetime_utc="2027-01-03T00:00:00Z", end_datetime_utc="2027-02-02T00:00:00Z", step_hours=3, display_timezone="UTC")


def test_sliding_window_blocks_then_recovers():
    limiter = SlidingWindowLimiter([Rule(2, 60)])
    assert limiter.hit("a", now=0)[0]
    assert limiter.hit("a", now=1)[0]
    allowed, retry, limit, remaining = limiter.hit("a", now=2)
    assert not allowed and limit == 2 and remaining == 0
    assert 57 < retry <= 58
    assert limiter.hit("b", now=2)[0], "other IPs are unaffected"
    assert limiter.hit("a", now=60.5)[0], "window slides"


def test_tightest_rule_wins():
    limiter = SlidingWindowLimiter([Rule(10, 60), Rule(3, 3600)])
    for t in range(3):
        assert limiter.hit("a", now=t)[0]
    allowed, retry, limit, _ = limiter.hit("a", now=100)
    assert not allowed and limit == 3 and retry > 3000


def test_key_table_is_bounded():
    limiter = SlidingWindowLimiter([Rule(5, 60)], max_keys=50)
    for i in range(500):
        limiter.hit(f"ip{i}", now=i * 0.001)
    assert len(limiter._hits) <= 50


def test_forwarded_header_ignored_unless_trusted(monkeypatch):
    settings = Settings(_env_file=None, trusted_proxy_ips="127.0.0.1,10.0.0.1")
    monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
    app = FastAPI()

    @app.get("/ip")
    def ip(request: rate_limit.Request):
        return {"untrusted": rate_limit.client_ip(request, False), "trusted": rate_limit.client_ip(request, True)}

    body = TestClient(app, client=("127.0.0.1", 1234)).get("/ip", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"}).json()
    assert body["trusted"] == "9.9.9.9"
    assert body["untrusted"] != "9.9.9.9"


def test_chat_endpoint_returns_429_with_retry_after(monkeypatch):
    settings = Settings(groq_api_key="", chat_rate_limit_per_minute=2, chat_rate_limit_per_hour=100, chat_require_origin=False)
    monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
    monkeypatch.setattr(chat_module, "get_settings", lambda: settings)
    rate_limit.chat_limiter().reset()
    client = TestClient(create_app())
    body = {"messages": [{"role": "user", "content": "hi"}], "context": CONTEXT}
    first = client.post("/api/chat", json=body)
    assert first.status_code == 503  # no key configured, but the request was counted
    assert client.post("/api/chat", json=body).status_code == 503
    blocked = client.post("/api/chat", json=body)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert "wait" in blocked.json()["detail"]
    rate_limit.chat_limiter().reset()


def test_chat_rejects_external_and_missing_origins(monkeypatch):
    settings = Settings(groq_api_key="", chat_require_origin=True)
    monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
    monkeypatch.setattr(chat_module, "get_settings", lambda: settings)
    client = TestClient(create_app())
    body = {"messages": [{"role": "user", "content": "hi"}], "context": CONTEXT}
    assert client.post("/api/chat", json=body).status_code == 403
    assert client.post("/api/chat", json=body, headers={"Origin": "https://external.example"}).status_code == 403
    allowed = client.post("/api/chat", json=body, headers={"Origin": "http://localhost:5173"})
    assert allowed.status_code == 503  # passed origin protection; no Groq key configured


def test_chat_requires_json(monkeypatch):
    settings = Settings(groq_api_key="", chat_require_origin=False)
    monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
    monkeypatch.setattr(chat_module, "get_settings", lambda: settings)
    response = TestClient(create_app()).post("/api/chat", content="hello", headers={"Content-Type": "text/plain"})
    assert response.status_code == 415


def test_default_chat_limit_accepts_twelve_and_rejects_thirteenth(monkeypatch):
    settings = Settings(_env_file=None, groq_api_key="")
    monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
    monkeypatch.setattr(chat_module, "get_settings", lambda: settings)
    rate_limit.chat_limiter().reset()
    body = {"messages": [{"role": "user", "content": "hi"}], "context": CONTEXT}
    headers = {"Origin": "http://localhost:5173"}
    client = TestClient(create_app(), client=("203.0.113.10", 1000))
    for index in range(12):
        response = client.post("/api/chat", json=body, headers=headers)
        assert response.status_code == 503, (index, response.text)
    blocked = client.post("/api/chat", json=body, headers={**headers, "X-Forwarded-For": "198.51.100.99"})
    assert blocked.status_code == 429
    assert blocked.headers["X-RateLimit-Limit"] == "12"
    assert blocked.headers["X-RateLimit-Remaining"] == "0"
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60
    other = TestClient(create_app(), client=("203.0.113.11", 1000))
    assert other.post("/api/chat", json=body, headers=headers).status_code == 503
    # Mission-language interpretation spends the same upstream quota.
    assert client.post("/api/mission/interpret", json={}, headers=headers).status_code == 429
    rate_limit.chat_limiter().reset()


def test_active_buckets_cannot_be_evicted_to_reset_a_limit():
    limiter = SlidingWindowLimiter([Rule(1, 60)], max_keys=2)
    assert limiter.hit("original", now=0)[0]
    assert limiter.hit("other", now=1)[0]
    assert not limiter.hit("attacker-new-ip", now=2)[0]
    assert not limiter.hit("original", now=3)[0]
    assert limiter.hit("attacker-new-ip", now=61)[0]


def test_forwarding_requires_a_trusted_peer_and_uses_first_untrusted_hop(monkeypatch):
    settings = Settings(_env_file=None, trusted_proxy_ips="127.0.0.1,10.0.0.0/8")
    monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
    app = FastAPI()
    @app.get("/ip")
    def ip(request: rate_limit.Request):
        return {"ip": rate_limit.client_ip(request, True)}
    direct = TestClient(app, client=("203.0.113.2", 1234))
    assert direct.get("/ip", headers={"X-Forwarded-For": "8.8.8.8"}).json()["ip"] == "203.0.113.2"
    proxy = TestClient(app, client=("127.0.0.1", 1234))
    assert proxy.get("/ip", headers={"X-Forwarded-For": "8.8.8.8, 203.0.113.2, 10.0.0.1"}).json()["ip"] == "203.0.113.2"
    assert proxy.get("/ip", headers={"X-Forwarded-For": "garbage"}).json()["ip"] == "127.0.0.1"
    assert proxy.get("/ip", headers={"X-Forwarded-For": "::ffff:203.0.113.2"}).json()["ip"] == "203.0.113.2"


def test_chat_minute_limit_cannot_be_disabled_or_exceed_twelve():
    import pytest
    from pydantic import ValidationError
    for limit in (0, -1, 13):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, chat_rate_limit_per_minute=limit)
