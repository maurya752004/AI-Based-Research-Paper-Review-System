from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

TOKEN_SECRET = os.getenv("TOKEN_SECRET", "change-me-in-production")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "43200"))

DEFAULT_USERS = {
    "student1": {"password": "student123", "role": "student"},
    "reviewer1": {"password": "reviewer123", "role": "reviewer"},
    "admin": {"password": "admin123", "role": "admin"},
    "admin1": {"password": "admin123", "role": "admin"},
}


def _users() -> dict[str, dict[str, str]]:
    raw = os.getenv("APP_USERS_JSON", "").strip()
    if not raw:
        return DEFAULT_USERS
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return DEFAULT_USERS


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    user = _users().get(username)
    if not user:
        return None
    if user.get("password") != password:
        return None
    role = user.get("role", "student")
    return {"username": username, "role": role}


def _sign(payload_b64: str) -> str:
    signature = hmac.new(TOKEN_SECRET.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return signature


def issue_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    payload_json = json.dumps(payload, separators=(",", ":")).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode().rstrip("=")
    signature = _sign(payload_b64)
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> dict[str, Any] | None:
    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError:
        return None

    expected = _sign(payload_b64)
    if not hmac.compare_digest(signature, expected):
        return None

    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload_json = base64.urlsafe_b64decode((payload_b64 + padding).encode()).decode()
        payload = json.loads(payload_json)
    except Exception:
        return None

    exp = int(payload.get("exp", 0))
    if exp < int(time.time()):
        return None

    if "sub" not in payload or "role" not in payload:
        return None

    return payload
