"""Public chat, science and LangGraph API on port 8000."""
from contextlib import asynccontextmanager
import logging
import anyio.to_thread
from fastapi import FastAPI
from .api.chat import router as chat_router
from .api.mission import router as mission_router
from .api.routes import router as science_router
from .api.nasa import router as nasa_router
from .config import get_settings
from .http_app import create_http_app
from .services.environment import clear_environment_cache
from .services.providers import build_providers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bound simultaneous scientific work on small hosts.
    anyio.to_thread.current_default_thread_limiter().total_tokens = 2
    app.state.providers = build_providers(get_settings())
    clear_environment_cache()
    try:
        yield
    finally:
        providers = app.state.providers
        for provider in (providers.astronomy, providers.terrain):
            closer = getattr(provider, "unload", None) or getattr(provider, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.warning("Provider cleanup failed", exc_info=True)
        clear_environment_cache()


def create_app() -> FastAPI:
    app = create_http_app(get_settings().api_title, lifespan)
    app.include_router(chat_router)
    app.include_router(science_router)
    app.include_router(nasa_router)
    app.include_router(mission_router)

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": get_settings().api_title, "version": get_settings().api_version,
                "docs": "/docs", "status": "/api/status"}
    return app


app = create_app()
