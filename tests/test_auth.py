"""Account lifecycle and session security checks."""
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import COOKIE, database, limiter, router


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_DATABASE_PATH", str(tmp_path / "accounts.sqlite3"))
    limiter.reset()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app, headers={"X-Selene-Client": "web"}) as client:
        yield client


ACCOUNT = {"name": "Lunar Explorer", "email": "explorer@example.com", "password": "a long lunar password"}


def test_account_lifecycle(auth_client):
    client = auth_client
    assert client.get("/api/auth/me").status_code == 401
    response = client.post("/api/auth/signup", json=ACCOUNT)
    assert response.status_code == 201
    assert response.json()["name"] == ACCOUNT["name"]
    assert "password" not in response.json()
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    token = client.cookies.get(COOKIE)
    assert client.get("/api/auth/me").json()["email"] == ACCOUNT["email"]
    with database() as db:
        user = db.execute("SELECT * FROM users").fetchone()
        session = db.execute("SELECT * FROM sessions").fetchone()
        assert user["password"] != ACCOUNT["password"]
        assert session["token"] != token
    assert client.post("/api/auth/logout", json={}).status_code == 204
    client.cookies.set(COOKIE, token)
    assert client.get("/api/auth/me").status_code == 401
    response = client.post("/api/auth/login", json={"email": "EXPLORER@example.com", "password": ACCOUNT["password"]})
    assert response.status_code == 200
    assert client.get("/api/auth/me").status_code == 200
    with database() as db:
        db.execute("UPDATE sessions SET expires = ?", (time.time() - 1,))
    assert client.get("/api/auth/me").status_code == 401


def test_invalid_accounts_and_credentials(auth_client):
    client = auth_client
    assert client.post("/api/auth/signup", json={**ACCOUNT, "password": "short"}).status_code == 422
    assert client.post("/api/auth/signup", json={**ACCOUNT, "name": "   "}).status_code == 422
    assert client.post("/api/auth/signup", json={**ACCOUNT, "email": "bad-email"}).status_code == 422
    assert client.post("/api/auth/signup", json=ACCOUNT).status_code == 201
    assert client.post("/api/auth/signup", json=ACCOUNT).status_code == 409
    assert client.post("/api/auth/login", json={**ACCOUNT, "password": "incorrect long password"}).status_code == 401
    assert client.post("/api/auth/login", json={**ACCOUNT, "email": "unknown@example.com"}).status_code == 401


def test_signup_blocks_repeat_browser_and_gmail_alias(auth_client):
    client = auth_client
    assert client.post("/api/auth/signup", json={**ACCOUNT, "email": "lunar.explorer+first@gmail.com"}).status_code == 201
    # A canonical Gmail alias is the same account identity, even in a fresh browser.
    with other_browser(client) as fresh:
        duplicate = fresh.post("/api/auth/signup", json={**ACCOUNT, "email": "lunarexplorer@gmail.com"})
        assert duplicate.status_code == 409
    # A different address still cannot create another account from the browser
    # that already registered successfully.
    blocked = client.post("/api/auth/signup", json={**ACCOUNT, "email": "second@example.com"})
    assert blocked.status_code == 429
    assert "already been created" in blocked.json()["detail"]


def test_signup_network_quota_persists_across_browsers(auth_client):
    clients = [auth_client, *(other_browser(auth_client) for _ in range(3))]
    try:
        for index, client in enumerate(clients[:3]):
            assert client.post("/api/auth/signup", json={**ACCOUNT, "email": f"person{index}@example.com"}).status_code == 201
        response = clients[3].post("/api/auth/signup", json={**ACCOUNT, "email": "blocked@example.com"})
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) > 0
    finally:
        for client in clients[1:]:
            client.close()


def test_auth_config_and_name_control_characters(auth_client):
    assert auth_client.get("/api/auth/config").json() == {"challenge_required": False, "site_key": ""}
    assert auth_client.post("/api/auth/signup", json={**ACCOUNT, "name": "Lunar\u0000Explorer"}).status_code == 422


def test_request_protection_and_throttling(auth_client):
    client = auth_client
    assert client.post("/api/auth/signup", json=ACCOUNT, headers={"Origin": "https://untrusted.example"}).status_code == 403
    client.headers.pop("X-Selene-Client")
    assert client.post("/api/auth/signup", json=ACCOUNT).status_code == 403
    client.headers["X-Selene-Client"] = "web"
    for _ in range(10):
        assert client.post("/api/auth/login", json=ACCOUNT).status_code == 401
    assert client.post("/api/auth/login", json=ACCOUNT).status_code == 429


def other_browser(client):
    return TestClient(client.app, headers={"X-Selene-Client": "web"})


def test_session_persists_across_app_instances(auth_client):
    client = auth_client
    assert client.post("/api/auth/signup", json=ACCOUNT).status_code == 201
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as restarted:
        restarted.cookies.set(COOKIE, client.cookies.get(COOKIE))
        assert restarted.get("/api/auth/me").json()["email"] == ACCOUNT["email"]


def test_refresh_rotates_token_without_extending_absolute_expiry(auth_client):
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    old_token = client.cookies.get(COOKIE)
    original = client.get("/api/auth/sessions").json()["sessions"][0]
    response = client.post("/api/auth/refresh")
    assert response.status_code == 200
    assert client.cookies.get(COOKIE) != old_token
    current = client.get("/api/auth/sessions").json()["sessions"][0]
    assert current["session_id"] == original["session_id"]
    assert current["expires"] == original["expires"]
    assert "token" not in current
    assert "password" not in current
    with other_browser(client) as stolen:
        stolen.cookies.set(COOKIE, old_token)
        assert stolen.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me").status_code == 200


def test_idle_timeout_deletes_session_and_cookie(auth_client):
    from app.config import get_settings
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    with database() as db:
        db.execute("UPDATE sessions SET last_seen_at = ?", (time.time() - get_settings().auth_session_idle_seconds - 1,))
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert "Max-Age=0" in response.headers["set-cookie"]
    assert response.json()["detail"] == "Please log in."
    assert client.post("/api/auth/refresh").status_code == 401
    with database() as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_only_page_activity_updates_last_seen(auth_client):
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    previous = time.time() - 60
    with database() as db:
        db.execute("UPDATE sessions SET last_seen_at = ?", (previous,))
    assert client.get("/api/auth/me").status_code == 200
    with database() as db:
        assert db.execute("SELECT last_seen_at FROM sessions").fetchone()[0] == previous
    assert client.post("/api/auth/activity").status_code == 200
    with database() as db:
        assert db.execute("SELECT last_seen_at FROM sessions").fetchone()[0] > previous


def test_revoke_one_session_and_logout_all(auth_client):
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    with other_browser(client) as second:
        second.post("/api/auth/login", json=ACCOUNT)
        sessions = client.get("/api/auth/sessions").json()["sessions"]
        assert len(sessions) == 2
        assert sum(row["current"] for row in sessions) == 1
        target = next(row["session_id"] for row in sessions if not row["current"])
        assert client.delete(f"/api/auth/sessions/{target}").status_code == 204
        assert second.get("/api/auth/me").status_code == 401
        assert client.get("/api/auth/me").status_code == 200
        second.post("/api/auth/login", json=ACCOUNT)
        assert client.post("/api/auth/logout-all").status_code == 204
        assert client.get("/api/auth/me").status_code == 401
        assert second.get("/api/auth/me").status_code == 401


def test_session_ownership_and_self_revocation(auth_client):
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    own_id = client.get("/api/auth/sessions").json()["sessions"][0]["session_id"]
    with other_browser(client) as outsider:
        outsider.post("/api/auth/signup", json={**ACCOUNT, "email": "other@example.com"})
        assert outsider.delete(f"/api/auth/sessions/{own_id}").status_code == 404
    assert client.get("/api/auth/me").status_code == 200
    assert client.delete(f"/api/auth/sessions/{own_id}").status_code == 204
    assert client.cookies.get(COOKIE) is None
    assert client.get("/api/auth/me").status_code == 401


def test_legacy_session_schema_migrates(auth_client):
    import os
    from pathlib import Path
    from app.services.encrypted_store import open_encrypted
    from app.services.sessions import token_hash

    token = "legacy-cookie-token"
    with open_encrypted(Path(os.environ["AUTH_DATABASE_PATH"])) as db:
        db.executescript("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, password TEXT NOT NULL); CREATE TABLE sessions (token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires REAL NOT NULL);")
        db.execute("INSERT INTO users VALUES (1, ?, ?, ?)", (ACCOUNT["name"], ACCOUNT["email"], "legacy-password"))
        db.execute("INSERT INTO sessions VALUES (?, 1, ?)", (token_hash(token), time.time() + 3600))
    db.close()
    auth_client.cookies.set(COOKIE, token)
    assert auth_client.get("/api/auth/me").json()["email"] == ACCOUNT["email"]
    session = auth_client.get("/api/auth/sessions").json()["sessions"][0]
    assert session["session_id"]
    assert session["created_at"] > 0
    assert session["current"] is True


def test_new_login_rotates_existing_browser_session(auth_client):
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    previous = client.cookies.get(COOKIE)
    assert client.post("/api/auth/login", json=ACCOUNT).status_code == 200
    assert client.cookies.get(COOKIE) != previous
    assert len(client.get("/api/auth/sessions").json()["sessions"]) == 1
    with other_browser(client) as old:
        old.cookies.set(COOKIE, previous)
        assert old.get("/api/auth/me").status_code == 401


def test_session_mutations_require_request_guard(auth_client):
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    own_id = client.get("/api/auth/sessions").json()["sessions"][0]["session_id"]
    client.headers.pop("X-Selene-Client")
    assert client.post("/api/auth/refresh").status_code == 403
    assert client.post("/api/auth/logout-all").status_code == 403
    assert client.delete(f"/api/auth/sessions/{own_id}").status_code == 403
    assert client.get("/api/auth/me").status_code == 200


def test_three_minute_boundary_and_polling_cannot_keep_session_alive(auth_client, monkeypatch):
    from app.services import sessions
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    with database() as db:
        started = db.execute("SELECT last_seen_at FROM sessions").fetchone()[0]
    monkeypatch.setattr(sessions.time, "time", lambda: started + 179)
    assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/auth/refresh").status_code == 200
    monkeypatch.setattr(sessions.time, "time", lambda: started + 180)
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/activity").status_code == 401


def test_activity_keeps_active_page_signed_in(auth_client, monkeypatch):
    from app.services import sessions
    client = auth_client
    client.post("/api/auth/signup", json=ACCOUNT)
    with database() as db:
        started = db.execute("SELECT last_seen_at FROM sessions").fetchone()[0]
    monkeypatch.setattr(sessions.time, "time", lambda: started + 150)
    assert client.post("/api/auth/activity").status_code == 200
    monkeypatch.setattr(sessions.time, "time", lambda: started + 300)
    assert client.get("/api/auth/me").status_code == 200
    monkeypatch.setattr(sessions.time, "time", lambda: started + 330)
    assert client.get("/api/auth/me").status_code == 401
