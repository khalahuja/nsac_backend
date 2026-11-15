"""Local account storage and revocable, opaque cookie sessions."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
import unicodedata

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from ..services.sessions import (
    COOKIE, IntegrityError, database, token_hash, start_session, get_current_session,
    get_current_user, cleanup_sessions, set_session_cookie, clear_session_cookie,
    cookie_is_secure,
)
from ..config import get_settings
from ..rate_limit import Rule, SlidingWindowLimiter, client_ip
from ..chat_security import verify_signup_challenge
from ..services.encrypted_store import encryption_key

router = APIRouter(prefix="/api/auth", tags=["accounts"])
login_limiter = SlidingWindowLimiter([Rule(10, 60), Rule(50, 3600)])
signup_limiter = SlidingWindowLimiter([Rule(5, 3600)])
SIGNUP_DEVICE_COOKIE = "selene_signup_device"


class _AuthLimiters:
    """Backward-compatible test hook that resets both independent buckets."""

    @staticmethod
    def reset():
        login_limiter.reset()
        signup_limiter.reset()


limiter = _AuthLimiters()


def guard(request: Request):
    if request.method != "GET":
        origin = request.headers.get("origin")
        allowed = set(get_settings().cors_origin_list) | {str(request.base_url).rstrip("/")}
        if origin and origin not in allowed:
            raise HTTPException(403, "Request origin is not allowed.")
        if request.headers.get("x-selene-client") != "web":
            raise HTTPException(403, "Missing request verification header.")
    if request.url.path.endswith(("/login", "/signup")):
        selected = signup_limiter if request.url.path.endswith("/signup") else login_limiter
        allowed, retry, _, _ = selected.hit(client_ip(request, get_settings().trust_proxy_headers))
        if not allowed:
            raise HTTPException(429, "Too many attempts. Please try again shortly.", headers={"Retry-After": str(int(retry) + 1)})


def email_identity(value: str) -> str:
    """Canonical account identity, including well-defined Gmail aliases."""
    local, domain = value.rsplit("@", 1)
    try:
        domain = domain.encode("idna").decode("ascii").casefold()
    except UnicodeError as error:
        raise ValueError("Enter a valid email address.") from error
    local = unicodedata.normalize("NFKC", local).casefold()
    if domain in {"gmail.com", "googlemail.com"}:
        domain = "gmail.com"
        local = local.split("+", 1)[0].replace(".", "")
    return f"{local}@{domain}"


def _abuse_hash(kind: str, value: str) -> str:
    key = bytes.fromhex(encryption_key())
    return hmac.new(key, f"{kind}\0{value}".encode(), hashlib.sha256).hexdigest()


def _device_token(request: Request) -> str:
    value = request.cookies.get(SIGNUP_DEVICE_COOKIE, "")
    return value if re.fullmatch(r"[A-Za-z0-9_-]{32,128}", value) else secrets.token_urlsafe(32)


def _set_device_cookie(response: Response, request: Request, token: str):
    response.set_cookie(SIGNUP_DEVICE_COOKIE, token, max_age=31536000, httponly=True,
                        secure=cookie_is_secure(request), samesite="strict", path="/api/auth")


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = unicodedata.normalize("NFKC", value.strip()).casefold()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid email address.")
        email_identity(value)
        return value


class Signup(Credentials):
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter your name.")
        if any(unicodedata.category(char).startswith("C") for char in value):
            raise ValueError("Name contains unsupported control characters.")
        return value


def password_hash(password: str, salt: str) -> str:
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


@router.get("/config")
def auth_config():
    settings = get_settings()
    required = bool(settings.turnstile_secret_key.get_secret_value())
    if required and not settings.turnstile_site_key:
        raise HTTPException(503, "Configure TURNSTILE_SITE_KEY to enable account verification.")
    return {"challenge_required": required, "site_key": settings.turnstile_site_key if required else ""}


@router.post("/signup", status_code=201, dependencies=[Depends(guard), Depends(verify_signup_challenge)])
def signup(body: Signup, request: Request, response: Response):
    settings = get_settings()
    now = time.time()
    device = _device_token(request)
    device_hash = hashlib.sha256(device.encode()).hexdigest()
    ip_hash = _abuse_hash("signup-ip", client_ip(request, settings.trust_proxy_headers))
    identity = email_identity(body.email)
    salt = secrets.token_hex(16)
    hashed = salt + ":" + password_hash(body.password, salt)
    with database() as db:
        # Populate identities for databases created before alias protection.
        for existing in db.execute("SELECT id, email FROM users WHERE email_key IS NULL").fetchall():
            db.execute("UPDATE users SET email_key = ? WHERE id = ?", (email_identity(existing["email"]), existing["id"]))
        if db.execute("SELECT 1 FROM users WHERE email_key = ?", (identity,)).fetchone():
            raise HTTPException(409, "Unable to create this account. Try logging in or use another email.")
        device_count = db.execute("SELECT COUNT(*) FROM signup_events WHERE device_hash = ?", (device_hash,)).fetchone()[0]
        if device_count >= settings.auth_signup_max_per_device:
            raise HTTPException(429, "An account has already been created from this browser. Please log in instead.")
        cutoff = now - settings.auth_signup_ip_window_seconds
        ip_count = db.execute("SELECT COUNT(*) FROM signup_events WHERE ip_hash = ? AND created_at > ?", (ip_hash, cutoff)).fetchone()[0]
        if ip_count >= settings.auth_signup_max_per_ip:
            oldest = db.execute("SELECT MIN(created_at) FROM signup_events WHERE ip_hash = ? AND created_at > ?", (ip_hash, cutoff)).fetchone()[0]
            retry = max(1, int(oldest + settings.auth_signup_ip_window_seconds - now) + 1)
            raise HTTPException(429, "Too many accounts were created from this network. Please try again later.", headers={"Retry-After": str(retry)})
        try:
            cursor = db.execute("INSERT INTO users(name, email, email_key, password) VALUES (?, ?, ?, ?)", (body.name, body.email, identity, hashed))
        except IntegrityError:
            raise HTTPException(409, "Unable to create this account. Try logging in or use another email.")
        db.execute("INSERT INTO signup_events(device_hash, ip_hash, user_id, created_at) VALUES (?, ?, ?, ?)",
                   (device_hash, ip_hash, cursor.lastrowid, now))
        user = {"id": cursor.lastrowid, "name": body.name, "email": body.email}
        result = start_session(db, response, request, user)
        _set_device_cookie(response, request, device)
        return result


@router.post("/login", dependencies=[Depends(guard)])
def login(body: Credentials, request: Request, response: Response):
    with database() as db:
        identity = email_identity(body.email)
        for existing in db.execute("SELECT id, email FROM users WHERE email_key IS NULL").fetchall():
            db.execute("UPDATE users SET email_key = ? WHERE id = ?", (email_identity(existing["email"]), existing["id"]))
        user = db.execute("SELECT * FROM users WHERE email = ? OR email_key = ? ORDER BY email = ? DESC LIMIT 1",
                          (body.email, identity, body.email)).fetchone()
        salt, expected = user["password"].split(":") if user else ("00" * 16, "00" * 64)
        actual = password_hash(body.password, salt)
        if not hmac.compare_digest(actual, expected) or not user:
            raise HTTPException(401, "Email or password is incorrect.")
        return start_session(db, response, request, user)


@router.get("/me")
def me(current: dict = Depends(get_current_session)):
    return {"id": current["user_id"], "name": current["name"], "email": current["email"],
            "idle_expires_at": current["last_seen_at"] + get_settings().auth_session_idle_seconds}


@router.post("/logout", status_code=204, dependencies=[Depends(guard)])
def logout(request: Request, response: Response):
    # Idempotent even when the cookie has expired or has already been revoked.
    with database() as db:
        db.execute("DELETE FROM sessions WHERE token = ?", (token_hash(request.cookies.get(COOKIE, "")),))
        cleanup_sessions(db)
    clear_session_cookie(response)


@router.get("/sessions")
def list_sessions(current: dict = Depends(get_current_session)):
    with database() as db:
        cleanup_sessions(db)
        rows = db.execute("SELECT session_id, created_at, last_seen_at, expires FROM sessions WHERE user_id = ? ORDER BY created_at DESC", (current["user_id"],)).fetchall()
    return {"sessions": [{**dict(row), "current": row["session_id"] == current["session_id"]} for row in rows]}


@router.post("/refresh", dependencies=[Depends(guard)])
def refresh(request: Request, response: Response, current: dict = Depends(get_current_session)):
    token = secrets.token_urlsafe(32)
    with database() as db:
        now = time.time()
        cleanup_sessions(db, now)
        # Compare-and-swap prevents a concurrently revoked token being renewed.
        cursor = db.execute("UPDATE sessions SET token = ? WHERE token = ? AND expires > ?", (token_hash(token), current["token"], now))
        changed = cursor.rowcount
    if not changed:
        raise HTTPException(401, "Session is no longer active.", headers={"Cache-Control": "no-store"})
    set_session_cookie(response, request, token, current["expires"])
    return {"id": current["user_id"], "name": current["name"], "email": current["email"]}


@router.delete("/sessions/{session_id}", status_code=204, dependencies=[Depends(guard)])
def revoke_session(session_id: str, response: Response, current: dict = Depends(get_current_session)):
    with database() as db:
        cursor = db.execute("DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, current["user_id"]))
        removed = cursor.rowcount
    if not removed:
        raise HTTPException(404, "Session not found.", headers={"Cache-Control": "no-store"})
    if session_id == current["session_id"]:
        clear_session_cookie(response)


@router.post("/logout-all", status_code=204, dependencies=[Depends(guard)])
def logout_all(response: Response, current: dict = Depends(get_current_session)):
    with database() as db:
        db.execute("DELETE FROM sessions WHERE user_id = ?", (current["user_id"],))
    clear_session_cookie(response)


@router.post("/activity", dependencies=[Depends(guard)])
def activity(response: Response, current: dict = Depends(get_current_session)):
    """Called only for genuine foreground page input, never idle polling."""
    now = time.time()
    with database() as db:
        cleanup_sessions(db, now)
        changed = db.execute("UPDATE sessions SET last_seen_at = ? WHERE token = ?", (now, current["token"])).rowcount
    if not changed:
        raise HTTPException(401, "Session is no longer active.", headers={"Cache-Control": "no-store"})
    response.headers["Cache-Control"] = "no-store"
    return {"idle_expires_at": now + get_settings().auth_session_idle_seconds}
