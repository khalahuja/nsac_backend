"""In-memory, per-IP sliding-window rate limiting.

Used on endpoints that spend a paid upstream quota (the Groq-backed chat).
State lives in this process only: with several workers or replicas each keeps
its own counters, so put a shared limiter (e.g. at the reverse proxy or in
Redis) in front of a multi-instance deployment.
"""

from __future__ import annotations

import math
from functools import lru_cache
from ipaddress import ip_address, ip_network, IPv6Address
import threading
import time
from collections import deque
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response

from .config import get_settings


@dataclass(frozen=True)
class Rule:
    limit: int
    window_seconds: float


class SlidingWindowLimiter:
    """Counts hits per key inside one or more sliding time windows."""

    def __init__(self, rules: list[Rule], max_keys: int = 10_000) -> None:
        self.rules = [r for r in rules if r.limit > 0 and r.window_seconds > 0]
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._longest = max((r.window_seconds for r in self.rules), default=0.0)

    def hit(self, key: str, now: float | None = None) -> tuple[bool, float, int, int]:
        """Record a hit for ``key`` if allowed.

        Returns ``(allowed, retry_after_seconds, limit, remaining)`` for the
        tightest rule (the one with the fewest requests left).
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self.max_keys:
                    self._prune(now)
                if len(self._hits) >= self.max_keys:
                    # Never evict an active bucket: IP churn must not reset quotas.
                    limit = min((r.limit for r in self.rules), default=0)
                    return False, max(1.0, self._longest), limit, 0
                hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._longest:
                hits.popleft()

            retry_after = 0.0
            tightest = (math.inf, 0)  # (remaining, limit)
            for rule in self.rules:
                in_window = [t for t in hits if now - t < rule.window_seconds]
                if len(in_window) >= rule.limit:
                    retry_after = max(retry_after, rule.window_seconds - (now - in_window[-rule.limit]))
                remaining = rule.limit - len(in_window)
                if remaining < tightest[0]:
                    tightest = (remaining, rule.limit)

            if retry_after > 0:
                return False, retry_after, tightest[1], 0
            hits.append(now)
            return True, 0.0, tightest[1], max(0, int(tightest[0]) - 1)

    def _prune(self, now: float) -> None:
        stale = [k for k, h in self._hits.items() if not h or now - h[-1] >= self._longest]
        for key in stale:
            del self._hits[key]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def _address(value: str):
    address = ip_address(value.strip())
    return address.ipv4_mapped if isinstance(address, IPv6Address) and address.ipv4_mapped else address


@lru_cache(maxsize=16)
def _proxy_networks(config: str):
    try:
        return tuple(ip_network(item.strip(), strict=False) for item in config.split(",") if item.strip())
    except ValueError:
        # A malformed trust list must never enable trust in arbitrary callers.
        return ()


def client_ip(request: Request, trust_proxy: bool) -> str:
    """Canonical client IP; only configured immediate proxy peers are trusted."""
    raw_peer = request.client.host if request.client else "unknown"
    try:
        peer = _address(raw_peer)
    except ValueError:
        return raw_peer
    peer_key = str(peer)
    networks = _proxy_networks(get_settings().trusted_proxy_ips)

    def trusted(address):
        return any(address.version == network.version and address in network for network in networks)

    if not trust_proxy or not trusted(peer):
        return peer_key
    forwarded = request.headers.get("x-forwarded-for")
    raw = forwarded if forwarded is not None else request.headers.get("x-real-ip", "")
    if not raw or len(raw) > 2048:
        return peer_key
    parts = raw.split(",")
    if len(parts) > 32:
        return peer_key
    try:
        chain = [_address(part) for part in parts]
    except ValueError:
        return peer_key
    # Walk right to left, stopping at the first hop outside trusted networks.
    for address in reversed(chain):
        if not trusted(address):
            return str(address)
    return str(chain[0])


_chat_limiter: SlidingWindowLimiter | None = None
_chat_limiter_rules: tuple | None = None


def chat_limiter() -> SlidingWindowLimiter:
    """Process-wide limiter for /api/chat, rebuilt if the settings change."""
    global _chat_limiter, _chat_limiter_rules
    s = get_settings()
    rules = (s.chat_rate_limit_per_minute, s.chat_rate_limit_per_hour)
    if _chat_limiter is None or rules != _chat_limiter_rules:
        _chat_limiter = SlidingWindowLimiter([Rule(rules[0], 60), Rule(rules[1], 3600)])
        _chat_limiter_rules = rules
    return _chat_limiter


async def limit_chat(request: Request, response: Response) -> None:
    """FastAPI dependency: reject with 429 once an IP exceeds the chat limits."""
    settings = get_settings()
    allowed, retry_after, limit, remaining = chat_limiter().hit(client_ip(request, settings.trust_proxy_headers))
    if not allowed:
        wait = max(1, math.ceil(retry_after))
        raise HTTPException(
            status_code=429,
            detail=f"Too many messages from your network. Please wait {wait} seconds and try again.",
            headers={"Retry-After": str(wait), "X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"},
        )
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)


async def protect_chat(request: Request, response: Response) -> None:
    """Apply quota limits plus browser-origin and request-size checks.

    This is an application-layer guard for the paid upstream call. Production
    deployments should still put a shared rate limiter/WAF in front of every
    worker because the counters in this module are process-local.
    """
    settings = get_settings()
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Chat requests must use application/json.")

    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > settings.chat_max_body_bytes:
                raise HTTPException(status_code=413, detail="Chat request is too large.")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from error

    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        raise HTTPException(status_code=403, detail="Cross-site chat requests are not allowed.")

    origin = request.headers.get("origin")
    allowed_origins = set(settings.cors_origin_list)
    if origin and origin not in allowed_origins:
        raise HTTPException(status_code=403, detail="This website origin is not allowed to use the assistant.")
    if settings.chat_require_origin and not origin:
        raise HTTPException(status_code=403, detail="A trusted website origin is required to use the assistant.")

    await limit_chat(request, response)


__all__ = ["Rule", "SlidingWindowLimiter", "client_ip", "chat_limiter", "limit_chat", "protect_chat"]
