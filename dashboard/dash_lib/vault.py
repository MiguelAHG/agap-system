"""
vault.py

Password-based encryption of the restricted data files. The key is derived from the dashboard password (PBKDF2), so the
password is never stored anywhere: typing the correct password both unlocks the app and decrypts the data in memory.
No Streamlit imports, so the precompute notebook can use it too.

Files in data/dashboard_data/restricted_enc/:
  salt.bin     random salt (created once, then reused)
  check.enc    a short known text, used to test a password
  <name>.enc   one per restricted parquet file
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

ITERATIONS = 200_000
CHECK_TEXT = b"agap"


def derive_key(password: str, salt: bytes) -> bytes:
    raw = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS, dklen=32)
    return base64.urlsafe_b64encode(raw)


def encrypt(data: bytes, key: bytes) -> bytes:
    return Fernet(key).encrypt(data)


def decrypt(token: bytes, key: bytes) -> bytes:
    """Raises cryptography.fernet.InvalidToken when the key is wrong."""
    return Fernet(key).decrypt(token)


def check_key(check_token: bytes, key: bytes) -> bool:
    try:
        return decrypt(check_token, key) == CHECK_TEXT
    except InvalidToken:
        return False


def encrypt_folder(src_dir: Path, out_dir: Path, password: str) -> list:
    """Encrypt every .parquet in src_dir into out_dir. Returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    salt_path = out_dir / "salt.bin"
    if not salt_path.exists():
        salt_path.write_bytes(os.urandom(16))
    key = derive_key(password, salt_path.read_bytes())

    written = []
    (out_dir / "check.enc").write_bytes(encrypt(CHECK_TEXT, key))
    keep = {"salt.bin", "check.enc"}
    for path in sorted(Path(src_dir).glob("*.parquet")):
        target = out_dir / (path.name + ".enc")
        target.write_bytes(encrypt(path.read_bytes(), key))
        keep.add(target.name)
        written.append(target)
    for stale in out_dir.glob("*"):                           # files from an earlier run that no longer exist
        if stale.name not in keep:
            stale.unlink()
    return written
