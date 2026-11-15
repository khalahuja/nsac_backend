"""Loopback-only scientific preview for browser integration tests; no account store."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router as science_router
from app.api.nasa import router as nasa_router
from app.config import Settings
from app.services.providers import build_providers

@asynccontextmanager
async def lifespan(app):
    app.state.providers = build_providers(Settings(
        use_spice=True, use_terrain_dem=True, use_terrain_shadows=True,
        use_diviner=True, terrain_raster_filename="ldem_80s_80m.lbl",
    ))
    yield
    for provider in (app.state.providers.astronomy, app.state.providers.terrain):
        close = getattr(provider, "close", None) or getattr(provider, "unload", None)
        if close:
            close()

app = FastAPI(lifespan=lifespan)
app.include_router(science_router)
app.include_router(nasa_router)
