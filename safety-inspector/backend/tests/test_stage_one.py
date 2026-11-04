import jwt
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, inspect, select


def login(client, role="admin"):
    response = client.post(
        "/api/v1/auth/login", json={"username": role, "password": role}
    )
    assert response.status_code == 200
    return response.json()


def test_schema_and_seed_are_idempotent(client):
    from app.database import SessionLocal, engine
    from app.models import Base, IncidentType, PpeType, User
    from app.seed import seed

    seed()
    assert len(Base.metadata.tables) == 19
    assert set(Base.metadata.tables).issubset(inspect(engine).get_table_names())
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(User)) == 4
        assert db.scalar(select(func.count()).select_from(PpeType)) == 9
        assert db.scalar(select(func.count()).select_from(IncidentType)) == 7
        master = db.scalar(select(User).where(User.role == "master"))
        assert master.area_id is not None
        assert master.password_hash != "master"


@pytest.mark.parametrize("role", ["admin", "engineer", "master", "viewer"])
def test_role_access(client, role):
    tokens = login(client, role)
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.json()["role"] == role
    assert "password_hash" not in response.json()
    assert client.get("/api/v1/users", headers=headers).status_code == (
        200 if role == "admin" else 403
    )
    assert len(client.get("/api/v1/settings/ppe-types", headers=headers).json()) == 9


def test_auth_failures_and_refresh(client):
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/settings").status_code == 401
    bad = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "wrong"}
    )
    assert bad.status_code == 401
    assert bad.json()["error"]["code"] == "unauthorized"
    tokens = login(client)
    assert (
        client.post(
            "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/auth/refresh", json={"refresh_token": tokens["access_token"]}
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['refresh_token']}"},
        ).status_code
        == 401
    )
    from app.security import SECRET

    expired = jwt.encode(
        {"sub": tokens["user"]["id"], "type": "access", "exp": 0},
        SECRET,
        algorithm="HS256",
    )
    assert (
        client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"}
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer malformed"}
        ).status_code
        == 401
    )
    assert client.post("/api/v1/auth/login", json={}).status_code == 422


def test_health_and_swagger(client):
    assert client.get("/health").json()["database"] == "ok"
    assert client.get("/api/docs").status_code == 200
    schema = client.get("/api/openapi.json").json()
    assert "/api/v1/auth/login" in schema["paths"]


def test_migration_roundtrip(client):
    from app.database import engine

    command.downgrade(Config("alembic.ini"), "base")
    assert "users" not in inspect(engine).get_table_names()
    command.upgrade(Config("alembic.ini"), "head")
    from app.seed import seed

    seed()
