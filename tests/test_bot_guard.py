from fastapi.testclient import TestClient

from app import bot_guard
from app.config import Settings
from app.main import create_app

BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}


def test_blocks_crawlers_and_missing_user_agent():
    patterns = Settings().bot_user_agent_patterns
    assert bot_guard.is_blocked_agent("Mozilla/5.0 (compatible; Googlebot/2.1)", patterns)
    assert bot_guard.is_blocked_agent("GPTBot/1.1", patterns)
    assert bot_guard.is_blocked_agent("python-requests/2.32", patterns)
    assert bot_guard.is_blocked_agent("", patterns)
    assert bot_guard.is_blocked_agent(None, patterns)
    assert not bot_guard.is_blocked_agent(BROWSER["User-Agent"], patterns)


def test_api_refuses_bots_but_serves_browsers():
    with TestClient(create_app()) as client:
        _check_bots(client)


def _check_bots(client):
    blocked = client.get("/api/status", headers={"User-Agent": "Scrapy/2.11"})
    assert blocked.status_code == 403
    assert blocked.headers["X-Robots-Tag"].startswith("noindex")
    ok = client.get("/api/status", headers=BROWSER)
    assert ok.status_code == 200
    assert ok.headers["X-Content-Type-Options"] == "nosniff"


def test_api_rate_limit(monkeypatch):
    settings = Settings(api_rate_limit_per_minute=3)
    monkeypatch.setattr(bot_guard, "get_settings", lambda: settings)
    bot_guard.api_limiter().reset()
    with TestClient(create_app()) as client:
        codes = [client.get("/api/status", headers=BROWSER).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    bot_guard.api_limiter().reset()


def test_api_rejects_foreign_origins_and_unsafe_methods():
    with TestClient(create_app()) as client:
        response = client.post("/api/environment", json={}, headers={"Origin": "https://attacker.example"})
        assert response.status_code == 403
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Cache-Control"] == "no-store"
        assert client.request("TRACE", "/api/health").status_code == 405
        assert client.get("/api/status", headers={"Origin": "null"}).status_code == 403


def test_security_headers_cover_body_rejections_hosts_and_preflights():
    with TestClient(create_app()) as client:
        rejected = client.post("/api/chat", content="hello", headers={"Content-Encoding": "gzip"})
        assert rejected.status_code == 415
        assert rejected.headers["X-Content-Type-Options"] == "nosniff"
        assert rejected.headers["Content-Security-Policy"].startswith("default-src 'none'")
        bad_host = client.get("/api/health", headers={"Host": "attacker.example"})
        assert bad_host.status_code == 400
        assert bad_host.headers["X-Frame-Options"] == "DENY"
        preflight = client.options("/api/chat", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })
        assert preflight.status_code == 200
        assert preflight.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        assert preflight.headers["X-Content-Type-Options"] == "nosniff"
        assert client.get("/api/status?" + "x" * 8193).status_code == 414
