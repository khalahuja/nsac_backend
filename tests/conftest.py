"""Shared pytest fixtures."""

from __future__ import annotations

import os

# Unit tests run against the deterministic built-in models, whatever the local
# .env enables. Real-data checks live in test_real_data.py.
for _key in ("USE_SPICE", "USE_TERRAIN_DEM", "USE_TERRAIN_SHADOWS", "USE_DIVINER"):
    os.environ[_key] = "false"

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.environment import clear_environment_cache
from app.services.providers import ScienceProviders, build_providers

#: A fixed instant used across tests so results are reproducible.
REFERENCE_TIME = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings() -> Settings:
    """Settings with every optional dataset disabled (the MVP configuration)."""
    return Settings(
        use_spice=False,
        use_terrain_dem=False,
        use_diviner=False,
        use_terrain_shadows=False,
    )


@pytest.fixture
def providers(settings: Settings) -> ScienceProviders:
    """Provider bundle backed by the analytic ephemeris and spherical terrain."""
    clear_environment_cache()
    return build_providers(settings)


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client with the application lifespan executed."""
    clear_environment_cache()
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _fresh_api_limiter():
    """The site-wide limiter is process-global; give every test a clean slate."""
    from app import bot_guard
    bot_guard.api_limiter().reset()
    yield


@pytest.fixture(autouse=True)
def _fresh_chat_cache():
    """Chat replies are cached by message content; tests reuse messages."""
    from app.api import chat
    chat._CACHE.clear()
    chat._MODEL_COOLDOWNS.clear()
    yield


@pytest.fixture(autouse=True)
def _isolated_account_store(tmp_path, monkeypatch):
    """Never open developer accounts or use a production encryption key in tests."""
    import secrets
    monkeypatch.setenv("AUTH_DATABASE_PATH", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("SQLCIPHER_KEY", secrets.token_hex(32))
