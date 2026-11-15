"""Application configuration, loaded from environment variables / .env.

The provider pattern used throughout the science layer is driven entirely by
these flags. Every ``USE_*`` toggle defaults to ``False`` so that a fresh clone
with no NASA datasets installed still boots and produces clearly-labelled
approximate results.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root (…/backend/app/config.py -> …/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _default_data_dir(name: str) -> str:
    return str(PROJECT_ROOT / "data" / name)


class Settings(BaseSettings):
    """Runtime settings.

    Real environment variables override dotenv files. Dotenv files load from
    root, backend and backend/app, with later files overriding earlier ones.
    """

    # ``frozen=True`` makes Settings immutable *and hashable*, which lets the
    # provider bundle participate in the environment LRU cache key - swapping
    # configuration then naturally partitions the cache instead of serving
    # stale results.
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env", PROJECT_ROOT / "backend" / "app" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    groq_api_key: SecretStr = SecretStr("")
    #: Token-light primary: fewest prompt tokens and no hidden reasoning (tested: grounded answers, valid JSON).
    groq_model: str = "qwen/qwen3.8-27b"
    #: Tried in order when the main model hits its quota or is unavailable.
    #: Each Groq model has its own rate limits, so this multiplies capacity.
    groq_fallback_models: str = "openai/gpt-oss-20b,openai/gpt-oss-120b"
    #: Most recent chat messages sent upstream (older ones cost tokens every turn).
    chat_history_messages: int = Field(default=10, ge=2, le=16)
    #: Bound complete recent messages to keep long chats within token-rate budgets.
    chat_history_max_chars: int = Field(default=6000, ge=6000, le=30000)
    #: Upper bound on tokens the model may generate per reply.
    chat_max_output_tokens: int = Field(default=1200, ge=256, le=4096)

    # -- Chat rate limiting (per client IP) --------------------------------
    #: Rolling IP limits: 1-12 per minute; 0 disables only the hourly cap.
    chat_rate_limit_per_minute: int = Field(default=12, ge=1, le=12)
    chat_rate_limit_per_hour: int = Field(default=60, ge=0, le=10000)
    #: Reject cross-site browser requests and require an approved Origin for
    #: the paid chat endpoint. Disable only for trusted non-browser clients.
    chat_require_origin: bool = True
    chat_max_body_bytes: int = Field(default=65_536, ge=1024, le=1_048_576)
    #: Bound every mutating API request before JSON parsing. Authentication
    #: requests use a smaller hard-coded ceiling in the body-limit middleware.
    api_max_body_bytes: int = Field(default=1_048_576, ge=16_384, le=8_388_608)
    #: Optional production bot challenge. Public site key may be sent to browsers.
    turnstile_site_key: str = ""
    turnstile_secret_key: SecretStr = SecretStr("")
    #: Trust X-Forwarded-For / X-Real-IP. Enable only behind a reverse proxy
    #: that overwrites these headers, or clients can spoof their IP.
    trust_proxy_headers: bool = False
    #: Only these immediate peers may supply forwarding headers. Never use '*'.
    trusted_proxy_ips: str = "127.0.0.1,::1"
    #: Total time to receive an API request body, including chunked requests.
    api_body_timeout_seconds: float = Field(default=15, ge=1, le=60)

    # -- Site-wide API protection (see app/bot_guard.py) --------------------
    #: Requests per rolling minute per IP across all /api routes. 0 disables.
    api_rate_limit_per_minute: int = Field(default=300, ge=0)
    #: Refuse crawler/scraper user-agents (and requests with none) on /api.
    block_bot_user_agents: bool = True
    #: Comma-separated, case-insensitive substrings of blocked user-agents.
    bot_user_agent_patterns: str = (
        "bot,crawler,spider,crawl,slurp,scrapy,httrack,wget,python-requests,python-urllib,"
        "aiohttp,go-http-client,java/,libwww,okhttp,node-fetch,axios,scraper,"
        "gptbot,ccbot,claudebot,anthropic-ai,perplexitybot,bytespider,facebookexternalhit,"
        "semrush,ahrefs,mj12bot,dotbot,petalbot,yandex,baiduspider"
    )

    #: Host-header allow-list. Add the public hostname in production.
    trusted_hosts: str = "localhost,127.0.0.1,testserver,backend"

    # -- Server ------------------------------------------------------------
    api_title: str = "Lunar Environment Explorer API"
    api_version: str = "1.0.0"

    #: Comma-separated list of allowed CORS origins.
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173"
    )

    # -- Account sessions --------------------------------------------------
    sqlcipher_key: SecretStr = SecretStr("")
    auth_database_path: str = str(PROJECT_ROOT / "backend" / "data" / "accounts.sqlite3")
    auth_session_max_age_seconds: int = Field(default=604800, ge=60)
    auth_session_idle_seconds: int = Field(default=180, ge=60, le=180)
    #: Keep the login cookie across browser restarts. Off: users start logged out on every new visit.
    auth_persistent_cookie: bool = False
    # None enables Secure except for plain HTTP on loopback during development.
    auth_cookie_secure: bool | None = None
    #: Layered registration-abuse controls. The device limit is lifetime;
    #: the network limit is evaluated over the configured rolling window.
    auth_signup_max_per_device: int = Field(default=1, ge=1, le=20)
    auth_signup_max_per_ip: int = Field(default=3, ge=1, le=100)
    auth_signup_ip_window_seconds: int = Field(default=86400, ge=3600, le=2_592_000)

    # -- Dataset locations -------------------------------------------------
    spice_data_dir: str = Field(default_factory=lambda: _default_data_dir("spice"))
    terrain_data_dir: str = Field(default_factory=lambda: _default_data_dir("terrain"))
    diviner_data_dir: str = Field(default_factory=lambda: _default_data_dir("diviner"))

    nasa_data_dir: str = Field(default_factory=lambda: _default_data_dir("nasa"))

    # -- Provider toggles --------------------------------------------------
    #: Attempt to use JPL SPICE kernels for astronomy (Milestone 2).
    use_spice: bool = False
    #: Attempt to use a LOLA/SLDEM raster for terrain (Milestone 3).
    use_terrain_dem: bool = False
    #: Attempt to read LRO Diviner observations (Milestone 5).
    use_diviner: bool = False
    #: Enable DEM horizon ray-tracing for terrain shadows (Milestone 4).
    #: Requires ``use_terrain_dem`` as well.
    use_terrain_shadows: bool = False

    # -- Dataset filenames (relative to the matching *_data_dir) -----------
    #: Filename of the lunar DEM raster, e.g. "sldem2015_64ppd.tif".
    terrain_raster_filename: str = "lunar_dem.tif"

    # -- Science model parameters -----------------------------------------
    #: Override the default regolith albedo; ``None`` uses constants.DEFAULT_ALBEDO.
    albedo_override: float | None = None
    emissivity_override: float | None = None

    # -- Performance -------------------------------------------------------
    #: Number of cached environment computations held in memory.
    environment_cache_size: int = 2048
    #: Hard cap on samples a single /api/timeline request may produce.
    max_timeline_samples: int = 1200
    #: Excel export is one request, so it gets the same cap as the chunked website forecast.
    max_export_samples: int = 20_000

    # -- Derived helpers ---------------------------------------------------

    @property
    def spice_dir(self) -> Path:
        return Path(self.spice_data_dir).expanduser()

    @property
    def terrain_dir(self) -> Path:
        return Path(self.terrain_data_dir).expanduser()

    @property
    def diviner_dir(self) -> Path:
        return Path(self.diviner_data_dir).expanduser()

    @property
    def terrain_raster_path(self) -> Path:
        return self.terrain_dir / self.terrain_raster_filename

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def trusted_host_list(self) -> list[str]:
        return [host.strip() for host in self.trusted_hosts.split(",") if host.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests that patch the environment)."""
    get_settings.cache_clear()


__all__ = ["Settings", "get_settings", "reset_settings_cache", "PROJECT_ROOT"]
