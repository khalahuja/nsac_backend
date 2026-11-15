"""Generate a random 256-bit SQLCipher key as exactly 64 hex characters.

Run: python backend/generate_key.py
Copy the output into SQLCIPHER_KEY in your environment or secret manager.
"""
import secrets


def main() -> None:
    print(secrets.token_hex(32))


if __name__ == "__main__":
    main()
