"""Site-wide protection for the JSON API against crawlers, scrapers and floods.

Three layers, applied to every ``/api`` request:

1. **Crawler/scraper user-agents are refused** (403). Search-engine and AI
   crawlers have no business calling a computational API, and a missing
   User-Agent is a strong signal of a crude script.
2. **Per-IP rate limit** across the whole API (429 + ``Retry-After``), generous
   enough for the app's own bursts (Compare Sites fetches several forecasts at
   once) but stopping floods. The chat endpoint keeps its own stricter limit.
3. **No-index / hardening headers** on every response, so nothing the API
   returns is indexed or framed.

Counters are per process; production should add a shared limiter or WAF.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

from fastapi import Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .rate_limit import Rule, SlidingWindowLimiter, client_ip

SECURITY_HEADERS = {
    "X-Robots-Tag": "noindex, nofollow, noarchive",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}


@lru_cache(maxsize=8)
def _bot_pattern(patterns: str) -> re.Pattern[str] | None:
    parts = [re.escape(p.strip()) for p in patterns.split(",") if p.strip()]
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


def is_blocked_agent(user_agent: str | None, patterns: str) -> bool:
    """True for a missing user-agent or one matching a crawler/scraper pattern."""
    if not user_agent or not user_agent.strip():
        return True
    pattern = _bot_pattern(patterns)
    return bool(pattern and pattern.search(user_agent))


_api_limiter: SlidingWindowLimiter | None = None
_api_limit_value: int | None = None


def api_limiter() -> SlidingWindowLimiter:
    global _api_limiter, _api_limit_value
    limit = get_settings().api_rate_limit_per_minute
    if _api_limiter is None or limit != _api_limit_value:
        _api_limiter = SlidingWindowLimiter([Rule(limit, 60)])
        _api_limit_value = limit
    return _api_limiter


def _with_headers(response, extra: dict[str, str] | None = None):
    for key, value in {**SECURITY_HEADERS, **(extra or {})}.items():
        response.headers.setdefault(key, value)
    return response


async def bot_guard(request: Request, call_next):
    """HTTP middleware: see module docstring."""
    path = request.url.path
    if not (path == "/api" or path.startswith("/api/")) or request.method == "OPTIONS":
        return await call_next(request)

    settings = get_settings()
    origin = request.headers.get("origin")
    if origin is not None and origin not in settings.cors_origin_list:
        return _with_headers(JSONResponse(
            status_code=403, content={"detail": "Request origin is not allowed."},
        ))
    if request.method not in {"GET", "HEAD", "POST"}:
        return _with_headers(JSONResponse(
            status_code=405, content={"detail": "Request method is not allowed."},
            headers={"Allow": "GET, HEAD, POST, OPTIONS"},
        ))
    if settings.block_bot_user_agents and is_blocked_agent(request.headers.get("user-agent"), settings.bot_user_agent_patterns):
        return _with_headers(JSONResponse(
            status_code=403,
            content={"detail": "Automated clients and crawlers are not allowed to use this API.", "error_type": "forbidden"},
        ))

    if settings.api_rate_limit_per_minute > 0:
        allowed, retry_after, _, _ = api_limiter().hit(client_ip(request, settings.trust_proxy_headers))
        if not allowed:
            wait = max(1, math.ceil(retry_after))
            return _with_headers(JSONResponse(
                status_code=429,
                content={"detail": f"Too many requests from your network. Please wait {wait} seconds.", "error_type": "rate_limited"},
            ), {"Retry-After": str(wait)})

    return _with_headers(await call_next(request))


__all__ = ["bot_guard", "is_blocked_agent", "api_limiter", "SECURITY_HEADERS"]
