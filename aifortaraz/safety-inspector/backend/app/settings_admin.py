from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException
from pydantic import Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .dependencies import Db
from .models import Area, Camera, IncidentType, PpeType, Setting, User, Zone
from .realtime import hub
from .schemas import StrictModel, Thresholds
from .security import password_hasher
from .stage_two import Admin, Editor, bump_config, validate_area

router = APIRouter(prefix="/api/v1", tags=["settings"])
Role = Literal["admin", "engineer", "master", "viewer"]
Severity = Literal["low", "medium", "high", "critical"]


class GeneralIn(Thresholds):
    enterprise_name: str = Field(min_length=1, max_length=200)
    workshop_name: str = Field(min_length=1, max_length=200)
    timezone: str = Field(min_length=1, max_length=100)
    theme: Literal["dark", "light"]
    alarm_sound: bool
    heartbeat_timeout_sec: int = Field(ge=1, le=3600)
    snapshot_retention_days: int = Field(ge=1, le=3650)

    @field_validator("timezone")
    @classmethod
    def timezone_exists(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Укажите часовой пояс IANA, например Asia/Almaty")
        return value


def local_date(db, stamp):
    from .stage_two import setting

    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(ZoneInfo(setting(db, "timezone", "UTC")))


def http_base_url(value):
    if not value:
        return ""
    try:
        url = urlsplit(value)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or any(c.isspace() or c == "\\" for c in value)
        ):
            raise ValueError
        _ = url.port
    except ValueError:
        raise ValueError(
            "Укажите базовый URL сайта, например https://inspector.example"
        )
    return value.rstrip("/")


async def commit_config(db):
    version = bump_config(db)
    db.commit()
    for camera_id in db.scalars(select(Camera.id)):
        await hub.publish(camera_id, version)


@router.put("/settings/general")
async def general(body: GeneralIn, db: Db, user: Editor):
    for key, value in body.model_dump().items():
        row = db.get(Setting, key)
        if row is None:
            db.add(Setting(key=key, value=value))
        else:
            row.value = value
    await commit_config(db)
    await hub.broadcast({"type": "settings.updated"})
    return body


class PpeIn(StrictModel):
    name_ru: str = Field(min_length=1, max_length=200)
    icon: str | None = Field(default=None, max_length=50)
    enabled: bool = True
    sort_order: int = Field(default=0, ge=0, le=10000)


class IncidentTypeIn(StrictModel):
    name_ru: str = Field(min_length=1, max_length=200)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    default_severity: Severity
    enabled: bool = True


class AreaIn(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


def save_code(db, model, code, body):
    import re

    if not re.fullmatch(r"[a-z][a-z0-9_]{0,49}", code):
        raise HTTPException(
            422, "Код должен содержать латинские строчные буквы, цифры и _"
        )
    row = db.get(model, code)
    if row is None:
        row = model(code=code)
        db.add(row)
    for key, value in body.model_dump().items():
        setattr(row, key, value)
    return row


@router.put("/settings/ppe-types/{code}")
async def save_ppe(code: str, body: PpeIn, db: Db, user: Editor):
    if not body.enabled and any(
        code in (rules or {}).get("required_ppe", [])
        for rules in db.scalars(select(Zone.rules))
    ):
        raise HTTPException(409, "Сначала удалите этот тип СИЗ из правил зон")
    save_code(db, PpeType, code, body)
    await commit_config(db)
    await hub.broadcast({"type": "settings.updated"})
    return {"code": code, **body.model_dump()}


@router.put("/settings/incident-types/{code}")
async def save_kind(code: str, body: IncidentTypeIn, db: Db, user: Editor):
    save_code(db, IncidentType, code, body)
    await commit_config(db)
    await hub.broadcast({"type": "settings.updated"})
    return {"code": code, **body.model_dump()}


@router.post("/settings/areas", status_code=201)
async def create_area(body: AreaIn, db: Db, user: Editor):
    row = Area(**body.model_dump())
    db.add(row)
    db.commit()
    await hub.broadcast({"type": "settings.updated"})
    return {"id": row.id, **body.model_dump()}


@router.put("/settings/areas/{area_id}")
async def save_area(area_id: str, body: AreaIn, db: Db, user: Editor):
    row = db.get(Area, area_id)
    if row is None:
        raise HTTPException(404, "Участок не найден")
    row.name, row.description = body.name, body.description
    await commit_config(db)
    await hub.broadcast({"type": "settings.updated"})
    return {"id": row.id, **body.model_dump()}


class AccountIn(StrictModel):
    username: str = Field(min_length=1, max_length=100, pattern=r"^\S+$")
    full_name: str = Field(min_length=1, max_length=200)
    role: Role
    area_id: str | None = None
    password: str | None = Field(default=None, min_length=8, max_length=256)


def save_account(db, row, body):
    if body.role == "master" and not body.area_id:
        raise HTTPException(422, "Укажите участок мастера")
    if body.area_id:
        validate_area(db, body.area_id)
    if row is None:
        if body.password is None:
            raise HTTPException(422, "Укажите пароль нового пользователя")
        row = User()
        db.add(row)
    for key, value in body.model_dump(exclude={"password"}).items():
        setattr(row, key, value)
    if body.password is not None:
        row.password_hash = password_hasher.hash(body.password)
    try:
        db.commit()
    except IntegrityError:
        raise HTTPException(409, "Логин уже занят")
    return {"id": row.id, **body.model_dump(exclude={"password"})}


@router.post("/users", status_code=201)
def create_user(body: AccountIn, db: Db, user: Admin):
    return save_account(db, None, body)


@router.put("/users/{user_id}")
def update_user(user_id: str, body: AccountIn, db: Db, user: Admin):
    row = db.get(User, user_id)
    if row is None:
        raise HTTPException(404, "Пользователь не найден")
    if row.id == user.id and body.role != "admin":
        raise HTTPException(409, "Нельзя снять с себя роль администратора")
    return save_account(db, row, body)


class ModelMetrics(StrictModel):
    map_percent: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    fps: float | None = Field(default=None, ge=0, le=10000, allow_inf_nan=False)
    latency_ms: float | None = Field(default=None, ge=0, le=60000, allow_inf_nan=False)


@router.get("/public/project", tags=["project"])
def project_information(db: Db):
    from .stage_two import setting

    return {
        "enterprise_name": setting(db, "enterprise_name", "Цифровой инспектор ОТ"),
        "model_metrics": ModelMetrics.model_validate(setting(db, "model_metrics", {})),
    }


@router.put("/settings/model-metrics")
async def save_model_metrics(body: ModelMetrics, db: Db, user: Editor):
    row = db.get(Setting, "model_metrics")
    if row is None:
        row = Setting(key="model_metrics")
        db.add(row)
    row.value = body.model_dump()
    db.commit()
    await hub.broadcast({"type": "settings.updated"})
    return body
