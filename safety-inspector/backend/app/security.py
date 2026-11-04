import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

password_hasher = PasswordHash.recommended()
SECRET = os.environ["JWT_SECRET"]
if len(SECRET) < 32:
    raise RuntimeError("JWT_SECRET must contain at least 32 characters")


def new_api_key() -> str:
    return "ot_" + secrets.token_urlsafe(32)


def hash_api_key(raw: str) -> str:
    # API keys are stored and looked up as unsalted SHA-256 hex digests.
    return hashlib.sha256(raw.encode()).hexdigest()


def issue_token(user_id: str, token_type: str) -> str:
    lifetime = (
        timedelta(minutes=int(os.getenv("ACCESS_TOKEN_MINUTES", "30")))
        if token_type == "access"
        else timedelta(days=int(os.getenv("REFRESH_TOKEN_DAYS", "7")))
    )
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": user_id, "type": token_type, "iat": now, "exp": now + lifetime},
        SECRET,
        algorithm="HS256",
    )


def decode_token(token: str, token_type: str) -> str:
    payload = jwt.decode(
        token, SECRET, algorithms=["HS256"], options={"require": ["exp", "sub", "type"]}
    )
    if payload["type"] != token_type:
        raise jwt.InvalidTokenError("Invalid token type")
    return payload["sub"]
