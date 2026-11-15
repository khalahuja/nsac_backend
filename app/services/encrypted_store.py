"""SQLCipher connections. No plaintext fallback and no key stored on disk."""
from pathlib import Path
import os
import re
from sqlcipher3 import dbapi2 as sqlcipher
from ..config import get_settings

IntegrityError = sqlcipher.IntegrityError


def encryption_key() -> str:
    key = os.environ.get("SQLCIPHER_KEY", get_settings().sqlcipher_key.get_secret_value())
    if not re.fullmatch(r"[0-9a-fA-F]{64}", key):
        raise RuntimeError("SQLCIPHER_KEY must be a secret 64-character hex key (32 random bytes). Set it in the backend environment.")
    return key


def open_encrypted(path: Path, key: str | None = None):
    key = encryption_key() if key is None else key
    if not re.fullmatch(r"[0-9a-fA-F]{64}", key):
        raise ValueError("Invalid SQLCipher key format.")
    db = sqlcipher.connect(str(path), timeout=10)
    try:
        if not db.execute("PRAGMA cipher_version").fetchone():
            raise RuntimeError("The database driver does not support SQLCipher.")
        # PRAGMA cannot bind parameters: interpolate only validated hex.
        # Raw random 256-bit keys avoid password-KDF work on every connection.
        db.execute("PRAGMA key = \"x'" + key + "'\"")
        db.execute("PRAGMA cipher_compatibility = 4")
        db.execute("SELECT count(*) FROM sqlite_master").fetchone()
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA secure_delete = ON")
        db.execute("PRAGMA temp_store = MEMORY")
        db.execute("PRAGMA cache_size = -2048")
        db.row_factory = sqlcipher.Row
        return db
    except BaseException:
        db.close()
        raise RuntimeError("Cannot open encrypted account database. Check SQLCIPHER_KEY. Convert an existing plaintext file offline with python -m backend.app.migrate_accounts from the project root (or python -m app.migrate_accounts from backend) before starting this service.") from None
