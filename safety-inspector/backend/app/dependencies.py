from typing import Annotated

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_db
from .models import ApiKey, User
from .security import decode_token, hash_api_key

Db = Annotated[Session, Depends(get_db)]
bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def current_user(
    db: Db, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
):
    try:
        user_id = decode_token(credentials.credentials if credentials else "", "access")
    except jwt.InvalidTokenError:
        raise HTTPException(
            401,
            "Сессия истекла или токен недействителен",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(401, "Пользователь не найден")
    return user


def require_roles(*roles: str):
    def check(user: Annotated[User, Depends(current_user)]):
        if user.role not in roles:
            raise HTTPException(403, "Недостаточно прав")
        return user

    return check


def find_api_key(db: Session, key: str | None):
    if not key:
        return None
    return db.scalar(
        select(ApiKey).where(
            ApiKey.key_hash == hash_api_key(key), ApiKey.revoked_at.is_(None)
        )
    )


def require_api_key(db: Db, key: Annotated[str | None, Depends(api_key_header)]):
    row = find_api_key(db, key)
    if row is None:
        raise HTTPException(401, "API-ключ отсутствует, недействителен или отозван")
    return row
