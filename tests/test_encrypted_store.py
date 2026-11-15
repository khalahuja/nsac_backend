"""Encryption verified with actual SQLCipher, never a mocked SQLite driver."""
import os
from pathlib import Path
import sqlite3
import pytest
from app.services.encrypted_store import open_encrypted
from app.services.sessions import database
from app.migrate_accounts import encrypt_database


def test_database_is_unreadable_to_plain_sqlite_and_wrong_key(monkeypatch):
    with database() as db:
        db.execute("INSERT INTO users(name,email,password) VALUES ('Secret Name','secret@example.com','hash')")
    path = Path(os.environ["AUTH_DATABASE_PATH"])
    contents = path.read_bytes()
    assert not contents.startswith(b"SQLite format 3")
    assert b"secret@example.com" not in contents
    with sqlite3.connect(path) as plain:
        with pytest.raises(sqlite3.DatabaseError):
            plain.execute("SELECT * FROM users").fetchall()
    monkeypatch.setenv("SQLCIPHER_KEY", "00" * 32)
    with pytest.raises(RuntimeError, match="Cannot open encrypted"):
        with database():
            pass


def test_missing_key_fails_without_creating_plaintext(monkeypatch):
    monkeypatch.setenv("SQLCIPHER_KEY", "")
    with pytest.raises(RuntimeError, match="SQLCIPHER_KEY must"):
        with database():
            pass
    assert not Path(os.environ["AUTH_DATABASE_PATH"]).exists()


def test_plaintext_requires_explicit_conversion(tmp_path):
    source = tmp_path / "plain.sqlite3"
    target = tmp_path / "encrypted.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO accounts VALUES (1, 'preserved')")
        db.execute("PRAGMA user_version=7")
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match="plaintext"):
        open_encrypted(source)
    encrypt_database(source, target)
    assert source.read_bytes() == original
    encrypted = open_encrypted(target)
    try:
        assert encrypted.execute("SELECT name FROM accounts").fetchone()[0] == "preserved"
        assert encrypted.execute("PRAGMA user_version").fetchone()[0] == 7
    finally:
        encrypted.close()
    with pytest.raises(ValueError, match="never overwritten"):
        encrypt_database(source, target)


def test_migration_cli_from_project_root(tmp_path, monkeypatch):
    import subprocess
    import sys
    from app.config import PROJECT_ROOT

    source = tmp_path / "cli.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE example (value TEXT)")
        db.execute("INSERT INTO example VALUES ('retained')")
    monkeypatch.setenv("AUTH_DATABASE_PATH", str(source))
    result = subprocess.run([sys.executable, "-m", "backend.app.migrate_accounts"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    target = source.with_name("cli.encrypted.sqlite3")
    assert target.exists()
    assert "AUTH_DATABASE_PATH=" in result.stdout
    db = open_encrypted(target)
    try:
        assert db.execute("SELECT value FROM example").fetchone()[0] == "retained"
    finally:
        db.close()
