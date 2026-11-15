"""Bound streamed request bodies before parsing; verify optional bot challenges."""
import asyncio
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import get_settings
from .rate_limit import client_ip


class RequestBodyLimit:
    """Reject oversized API bodies even when Content-Length is absent or false.

    Reading the ASGI stream here prevents chunked bodies from bypassing the
    declared-length check performed by endpoint dependencies.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        path = scope.get("path", "").rstrip("/")
        if scope["type"] != "http" or not path.startswith("/api/"):
            return await self.app(scope, receive, send)
        if len(scope.get("query_string", b"")) > 8192 or len(path) > 2048:
            return await JSONResponse({"detail": "Request URL is too long."}, 414)(scope, receive, send)
        if scope.get("method") not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        settings = get_settings()
        if path in ("/api/chat", "/api/mission/interpret"):
            limit = settings.chat_max_body_bytes
        elif path in ("/api/auth/login", "/api/auth/signup"):
            limit = min(settings.api_max_body_bytes, 16_384)
        else:
            limit = settings.api_max_body_bytes
        raw_headers = scope.get("headers", [])
        headers = dict(raw_headers)
        if sum(name == b"content-length" for name, _ in raw_headers) > 1:
            return await JSONResponse({"detail": "Duplicate Content-Length header."}, 400)(scope, receive, send)
        if headers.get(b"content-encoding", b"identity").strip().lower() not in (b"", b"identity"):
            return await JSONResponse({"detail": "Compressed request bodies are not supported."}, 415)(scope, receive, send)
        length = headers.get(b"content-length")
        try:
            declared = int(length) if length is not None else 0
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length header."}, 400)(scope, receive, send)
        if declared < 0:
            return await JSONResponse({"detail": "Invalid Content-Length header."}, 400)(scope, receive, send)
        body = bytearray()
        deadline = asyncio.get_running_loop().time() + settings.api_body_timeout_seconds
        if declared <= limit:
            while True:
                try:
                    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                    message = await asyncio.wait_for(receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    return await JSONResponse({"detail": "Request body timed out."}, 408)(scope, receive, send)
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > limit:
                    break
                body.extend(chunk)
                if not message.get("more_body", False):
                    if length is not None and len(body) != declared:
                        return await JSONResponse({"detail": "Content-Length does not match request body."}, 400)(scope, receive, send)
                    consumed = False
                    async def replay():
                        nonlocal consumed
                        if consumed:
                            return await receive()
                        consumed = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await self.app(scope, replay, send)
        response = JSONResponse({"detail": "Request body is too large."}, 413, headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"})
        await response(scope, receive, send)


async def verify_challenge(request: Request, action: str):
    settings = get_settings()
    secret = settings.turnstile_secret_key.get_secret_value()
    if not secret:
        return
    token = request.headers.get("x-turnstile-token", "")
    if not token or len(token) > 2048:
        raise HTTPException(403, "Please complete the assistant verification and try again.")
    expected_host = urlsplit(request.headers.get("origin", "")).hostname
    if not expected_host:
        raise HTTPException(403, "A website origin is required for assistant verification.")
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.post("https://challenges.cloudflare.com/turnstile/v0/siteverify", json={
                "secret": secret, "response": token,
                "remoteip": client_ip(request, settings.trust_proxy_headers),
            })
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(503, "Assistant verification is temporarily unavailable. Please retry.") from error
    if not isinstance(result, dict) or result.get("success") is not True or result.get("action") != action or result.get("hostname") != expected_host:
        raise HTTPException(403, "Human verification expired or failed. Please verify again.")


async def verify_chat_challenge(request: Request):
    await verify_challenge(request, "chat")


async def verify_signup_challenge(request: Request):
    await verify_challenge(request, "signup")


# Compatibility for imports in deployments that extend this middleware.
ChatBodyLimit = RequestBodyLimit
