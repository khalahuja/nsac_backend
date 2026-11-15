"""Offline, non-destructive plaintext -> SQLCipher conversion.

Stop the backend and configure SQLCIPHER_KEY first. From the project root run:
python -m backend.app.migrate_accounts
Or from backend: python -m app.migrate_accounts
Default paths use AUTH_DATABASE_PATH and an adjacent .encrypted.sqlite3 copy.
The plaintext source is retained for explicit backup/retirement by the owner.
"""
import argparse
from contextlib import closing
import os
from pathlib import Path
from .services.encrypted_store import encryption_key, open_encrypted, sqlcipher
from .config import get_settings


def encrypt_database(source: Path, destination: Path):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination or destination.exists():
        raise ValueError("Choose a new destination; existing files are never overwritten.")
    with source.open("rb") as f:
        if f.read(16) != b"SQLite format 3\x00":
            raise ValueError("Source must be a plaintext SQLite database.")
    if any(Path(str(source) + suffix).exists() for suffix in ("-wal", "-journal")):
        raise ValueError("Stop writers and checkpoint the source before conversion (WAL/journal exists).")
    key = encryption_key()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    db = None
    try:
        db = sqlcipher.connect(source.as_uri() + "?mode=ro", uri=True)
        db.execute("PRAGMA temp_store = MEMORY")
        db.execute("ATTACH DATABASE ? AS encrypted KEY \"x'" + key + "'\"", (str(destination),))
        db.execute("PRAGMA encrypted.cipher_compatibility = 4")
        db.execute("SELECT sqlcipher_export('encrypted')").fetchone()
        version = db.execute("PRAGMA user_version").fetchone()[0]
        db.execute(f"PRAGMA encrypted.user_version = {int(version)}")
        db.commit()
        db.execute("DETACH DATABASE encrypted")
        with closing(open_encrypted(destination, key)) as encrypted:
            result = encrypted.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError("Converted database failed integrity validation.")
            tables = db.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall()
            for (table,) in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                if db.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0] != encrypted.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]:
                    raise RuntimeError("Converted database row counts differ.")
    except BaseException:
        if db is not None:
            db.close()
            db = None
        destination.unlink(missing_ok=True)
        raise
    finally:
        if db is not None:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Plaintext file; defaults to AUTH_DATABASE_PATH.")
    parser.add_argument("--destination", type=Path, help="New encrypted file; defaults to <source>.encrypted.sqlite3.")
    args = parser.parse_args()
    source = (args.source or Path(get_settings().auth_database_path)).expanduser().resolve()
    destination = (args.destination or source.with_name(source.stem + ".encrypted" + source.suffix)).expanduser().resolve()
    try:
        encrypt_database(source, destination)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Migration failed: {error}\n")
    print("Encrypted copy verified. Original source retained.")
    print(f"Set AUTH_DATABASE_PATH={destination.as_posix()} before restarting the backend.")


if __name__ == "__main__":
    main()
