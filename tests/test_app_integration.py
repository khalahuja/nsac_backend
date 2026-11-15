"""Public app must start and expose science without accounts or secrets."""
from app.main import create_app
from fastapi.testclient import TestClient


def test_unified_routes_are_public_and_auth_routes_are_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("SQLCIPHER_KEY", "")
    account_file = tmp_path / "must-not-exist.sqlite3"
    monkeypatch.setenv("AUTH_DATABASE_PATH", str(account_file))
    with TestClient(create_app()) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/status").json()["status"] == "ok"
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/api/chat", "/api/mission/simulate", "/api/environment", "/api/forecast"):
            assert path in paths
        assert not any(path.startswith("/api/auth") for path in paths)
        for path in ("/api/auth/signup", "/api/auth/login", "/api/auth/logout"):
            assert client.post(path, json={}).status_code == 404
        assert client.get("/api/auth/me").status_code == 404
        assert "set-cookie" not in client.get("/api/health").headers
    assert not account_file.exists()
