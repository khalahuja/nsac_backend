"""Shared HTTP policy, with no account, graph, or scientific imports."""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .bot_guard import bot_guard
from .chat_security import RequestBodyLimit
from .config import get_settings
from .security_headers import SecurityHeaders


def create_http_app(title: str, lifespan) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=title, version=settings.api_version, lifespan=lifespan)
    app.middleware("http")(bot_guard)
    app.add_middleware(RequestBodyLimit)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list,
                       allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["Content-Type", "X-Selene-Client", "X-Turnstile-Token"],
                       expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-Selene-Error"])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_host_list)
    app.add_middleware(SecurityHeaders)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        problems = "; ".join(f"{'.'.join(str(part) for part in error['loc'][1:]) or 'body'}: {error['msg']}" for error in exc.errors())
        return JSONResponse(status_code=422, content={"detail": problems or "Invalid request.", "error_type": "validation_error"})
    return app
