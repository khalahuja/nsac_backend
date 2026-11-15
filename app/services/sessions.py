"""SQLite-backed accounts and sessions shared across requests and workers."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import time

from fastapi import Depends, HTTPException, Request, Response

from ..config import get_settings
from .encrypted_store import open_encrypted, IntegrityError

COOKIE = "selene_session"


@contextmanager
def database():
    path = Path(os.environ.get("AUTH_DATABASE_PATH", get_settings().auth_database_path)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = open_encrypted(path)
    try:
        db.execute("PRAGMA foreign_keys = ON")
        # Serialize schema upgrades and session mutations across API workers.
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, password TEXT NOT NULL)")
        user_columns = {row["name"] for row in db.execute("PRAGMA table_info(users)")}
        if "email_key" not in user_columns:
            db.execute("ALTER TABLE users ADD COLUMN email_key TEXT")
        db.execute("CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires REAL NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS signup_events (
            id INTEGER PRIMARY KEY,
            device_hash TEXT NOT NULL,
            ip_hash TEXT NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id),
            created_at REAL NOT NULL
        )""")
        columns = {row["name"] for row in db.execute("PRAGMA table_info(sessions)")}
        for name, kind in (("session_id", "TEXT"), ("created_at", "REAL"), ("last_seen_at", "REAL")):
            if name not in columns:
                db.execute(f"ALTER TABLE sessions ADD COLUMN {name} {kind}")
        now = time.time()
        for row in db.execute("SELECT token FROM sessions WHERE session_id IS NULL").fetchall():
            db.execute("UPDATE sessions SET session_id = ?, created_at = ?, last_seen_at = ? WHERE token = ?", (secrets.token_urlsafe(18), now, now, row["token"]))
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS sessions_id ON sessions(session_id)")
        db.execute("CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id)")
        db.execute("CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires)")
        db.execute("CREATE INDEX IF NOT EXISTS sessions_idle ON sessions(last_seen_at)")
        db.execute("CREATE INDEX IF NOT EXISTS users_email_key ON users(email_key)")
        db.execute("CREATE INDEX IF NOT EXISTS signup_events_device ON signup_events(device_hash)")
        db.execute("CREATE INDEX IF NOT EXISTS signup_events_ip_time ON signup_events(ip_hash, created_at)")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def cleanup_sessions(db, now: float | None = None) -> int:
    now = time.time() if now is None else now
    cursor = db.execute("DELETE FROM sessions WHERE expires <= ? OR last_seen_at <= ?", (now, now - get_settings().auth_session_idle_seconds))
    return cursor.rowcount


def cookie_is_secure(request: Request) -> bool:
    configured = get_settings().auth_cookie_secure
    return configured if configured is not None else not (
        request.url.hostname in {"localhost", "127.0.0.1", "testserver"} and request.url.scheme == "http"
    )


def set_session_cookie(response: Response, request: Request, token: str, expires: float):
    secure = cookie_is_secure(request)
    # Browser-session cookie by default: closing the browser logs the user out.
    # The server-side expiry still bounds the session either way.
    max_age = max(0, int(expires - time.time())) if get_settings().auth_persistent_cookie else None
    response.set_cookie(COOKIE, token, max_age=max_age, httponly=True, secure=secure, samesite="lax", path="/")
    response.headers["Cache-Control"] = "no-store"


def clear_session_cookie(response: Response):
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="lax")
    response.headers["Cache-Control"] = "no-store"


def start_session(db, response: Response, request: Request, user):
    old_token = request.cookies.get(COOKIE)
    if old_token:
        db.execute("DELETE FROM sessions WHERE token = ?", (token_hash(old_token),))
    now = time.time()
    cleanup_sessions(db, now)
    token = secrets.token_urlsafe(32)
    expires = now + get_settings().auth_session_max_age_seconds
    db.execute("INSERT INTO sessions(token, user_id, expires, session_id, created_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)", (token_hash(token), user["id"], expires, secrets.token_urlsafe(18), now, now))
    set_session_cookie(response, request, token, expires)
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


def get_current_session(request: Request, response: Response) -> dict:
    """Validate a session without treating polling as user activity."""
    token = request.cookies.get(COOKIE, "")
    session = None
    with database() as db:
        now = time.time()
        cleanup_sessions(db, now)
        if 1 <= len(token) <= 128:
            row = db.execute("SELECT s.*, u.name, u.email FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ?", (token_hash(token),)).fetchone()
            if row:
                session = dict(row)
    response.headers["Cache-Control"] = "no-store"
    if session is None:
        rejected = Response()
        clear_session_cookie(rejected)
        raise HTTPException(401, "Please log in.", headers={"Cache-Control": "no-store", "Set-Cookie": rejected.headers["set-cookie"]})
    return session


def get_current_user(session: dict = Depends(get_current_session)) -> dict:
    return {"id": session["user_id"], "name": session["name"], "email": session["email"]}
