import hashlib
import secrets
from datetime import timedelta, timezone
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Integer, String, cast, delete, func, select, update

from .dependencies import Db, current_user, require_api_key, require_roles
from .models import (
    ApiKey,
    Area,
    Camera,
    CameraStatus,
    Incident,
    PpeType,
    Setting,
    User,
    Zone,
    utc_now,
)
from .realtime import hub
from .schemas import (
    CameraIn,
    CameraOut,
    ConfigRules,
    IngestConfig,
    KeyCreated,
    KeyIn,
    KeyOut,
    Thresholds,
    ZoneCreate,
    ZoneIn,
    ZoneOut,
)

router = APIRouter(prefix="/api/v1")
Reader = Annotated[User, Depends(current_user)]
Editor = Annotated[User, Depends(require_roles("admin", "engineer"))]
Admin = Annotated[User, Depends(require_roles("admin"))]
Ai = Annotated[ApiKey, Depends(require_api_key)]


def iso(value):
    return (
        value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        if value
        else None
    )


def camera_or_404(db, camera_id):
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, "Камера не найдена")
    return camera


def setting(db, key, fallback=None):
    row = db.get(Setting, key)
    return row.value if row else fallback


def bump_config(db):
    version = int(setting(db, "config_version", 1))
    result = db.execute(
        update(Setting)
        .where(
            Setting.key == "config_version",
            cast(cast(Setting.value, String), Integer) == version,
        )
        .values(value=version + 1)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        db.rollback()
        raise HTTPException(409, "Конфигурация изменилась. Загрузите данные заново")
    timestamp = db.get(Setting, "config_updated_at")
    if timestamp is None:
        db.add(Setting(key="config_updated_at", value=iso(utc_now())))
    else:
        timestamp.value = iso(utc_now())
    return version + 1


def status_payload(db, camera):
    import time
    from .live_runtime import runtime

    row = db.get(CameraStatus, camera.id)
    online = (
        row
        and row.last_heartbeat_at
        and (
            utc_now() - row.last_heartbeat_at.replace(tzinfo=timezone.utc)
        ).total_seconds()
        <= setting(db, "heartbeat_timeout_sec", 15)
    )
    return {
        "camera_id": camera.id,
        "relay_frame_age_sec": round(
            max(0, time.monotonic() - runtime.frames[camera.id].received_at), 3
        )
        if camera.id in runtime.frames
        else None,
        "source": (row.extra or {}).get("_source", "ai")
        if row and row.last_heartbeat_at
        else None,
        "status": row.status if online and camera.enabled else "offline",
        "fps": row.fps
        if online and camera.enabled and row.status != "offline"
        else None,
        "model_name": row.model_name if row else None,
        "model_version": row.model_version if row else None,
        "last_heartbeat_at": iso(row.last_heartbeat_at) if row else None,
    }


def camera_payload(db, camera):
    result = {
        column.name: getattr(camera, column.name) for column in Camera.__table__.columns
    }
    result["created_at"] = iso(camera.created_at)
    result.update(status_payload(db, camera))
    result.pop("camera_id")
    area = db.get(Area, camera.area_id)
    result["area_name"] = area.name if area else ""
    result["zones_count"] = db.scalar(
        select(func.count()).select_from(Zone).where(Zone.camera_id == camera.id)
    )
    result["incidents_7d"] = db.scalar(
        select(func.count())
        .select_from(Incident)
        .where(
            Incident.camera_id == camera.id,
            Incident.started_at >= utc_now() - timedelta(days=7),
        )
    )
    return result


def validate_area(db, area_id):
    if db.get(Area, area_id) is None:
        raise HTTPException(422, "Выбранный участок не существует")


@router.get("/cameras", response_model=list[CameraOut], tags=["cameras"])
def list_cameras(db: Db, user: Reader):
    return [
        camera_payload(db, c)
        for c in db.scalars(select(Camera).order_by(Camera.created_at))
    ]


@router.post("/cameras", response_model=CameraOut, status_code=201, tags=["cameras"])
async def create_camera(body: CameraIn, db: Db, user: Editor):
    validate_area(db, body.area_id)
    camera = Camera(**body.model_dump())
    db.add(camera)
    db.flush()
    db.add(CameraStatus(camera_id=camera.id, status="offline"))
    version = bump_config(db)
    db.commit()
    await hub.publish(camera.id, version)
    return camera_payload(db, camera)


@router.get("/cameras/{camera_id}", response_model=CameraOut, tags=["cameras"])
def get_camera(camera_id: str, db: Db, user: Reader):
    return camera_payload(db, camera_or_404(db, camera_id))


@router.put("/cameras/{camera_id}", response_model=CameraOut, tags=["cameras"])
async def update_camera(camera_id: str, body: CameraIn, db: Db, user: Editor):
    camera = camera_or_404(db, camera_id)
    validate_area(db, body.area_id)
    for key, value in body.model_dump().items():
        setattr(camera, key, value)
    version = bump_config(db)
    db.commit()
    await hub.publish(camera_id, version)
    return camera_payload(db, camera)


@router.delete("/cameras/{camera_id}", tags=["cameras"])
async def delete_camera(camera_id: str, db: Db, user: Editor):
    camera = camera_or_404(db, camera_id)
    if db.scalar(select(Incident.id).where(Incident.camera_id == camera_id).limit(1)):
        raise HTTPException(
            409, "Камера содержит инциденты. Отключите её вместо удаления"
        )
    from .models import ComplianceStat

    db.execute(delete(Zone).where(Zone.camera_id == camera_id))
    db.execute(delete(CameraStatus).where(CameraStatus.camera_id == camera_id))
    db.execute(delete(ComplianceStat).where(ComplianceStat.camera_id == camera_id))
    db.delete(camera)
    version = bump_config(db)
    db.commit()
    from .live_runtime import runtime

    runtime.clear_camera(camera_id)
    hub.latest_detections.pop(camera_id, None)
    await hub.publish(camera_id, version)
    return {"deleted": True}


@router.get("/cameras/{camera_id}/status", tags=["cameras"])
def camera_status(camera_id: str, db: Db, user: Reader):
    return status_payload(db, camera_or_404(db, camera_id))


@router.post("/cameras/{camera_id}/check-stream", tags=["cameras"])
async def check_stream(camera_id: str, db: Db, user: Editor):
    camera = camera_or_404(db, camera_id)
    url = camera.preview_url or (
        camera.stream_url if camera.stream_type == "mjpeg" else None
    )
    if not url or not url.startswith(("http://", "https://")):
        return {
            "ok": False,
            "message": "Для проверки укажите HTTP(S) URL превью. Декодирование RTSP и видеофайлов не выполняется",
        }
    try:
        async with httpx.AsyncClient(
            timeout=5, follow_redirects=False, trust_env=False
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not (
                    content_type.startswith("image/")
                    or "multipart/x-mixed-replace" in content_type
                ):
                    return {
                        "ok": False,
                        "message": "Источник не возвращает изображение или MJPEG",
                    }
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        return {
                            "ok": True,
                            "message": "Превью доступно",
                            "content_type": content_type,
                        }
        return {"ok": False, "message": "Источник вернул пустой ответ"}
    except httpx.HTTPError:
        return {
            "ok": False,
            "message": "Превью недоступно. Проверьте адрес и подключение к камере",
        }


def validate_rules(db, rules):
    known = set(db.scalars(select(PpeType.code).where(PpeType.enabled.is_(True))))
    if not set(rules.required_ppe).issubset(known) or len(
        set(rules.required_ppe)
    ) != len(rules.required_ppe):
        raise HTTPException(422, "Неизвестный, отключённый или повторяющийся тип СИЗ")
    if rules.responsible_user_id and db.get(User, rules.responsible_user_id) is None:
        raise HTTPException(422, "Ответственный пользователь не существует")


def apply_zone(db, body, camera_id, existing=None):
    validate_rules(db, body.rules)
    zone = existing or Zone(camera_id=camera_id)
    values = body.model_dump(exclude={"id", "camera_id"}, mode="json", by_alias=True)
    for key, value in values.items():
        setattr(zone, key, value)
    zone.updated_at = utc_now()
    if existing is None:
        if body.id:
            zone.id = str(body.id)
        db.add(zone)
    return zone


@router.get("/cameras/{camera_id}/zones", response_model=list[ZoneOut], tags=["zones"])
def list_zones(camera_id: str, db: Db, user: Reader):
    camera_or_404(db, camera_id)
    return db.scalars(
        select(Zone)
        .where(Zone.camera_id == camera_id)
        .order_by(Zone.sort_order, Zone.id)
    ).all()


@router.put("/cameras/{camera_id}/zones", response_model=list[ZoneOut], tags=["zones"])
async def replace_zones(camera_id: str, body: list[ZoneIn], db: Db, user: Editor):
    camera_or_404(db, camera_id)
    if len(body) > 100:
        raise HTTPException(422, "Максимум 100 зон на камеру")
    ids = [str(z.id) for z in body if z.id]
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "Идентификаторы зон должны быть уникальны")
    previous = {
        z.id: z for z in db.scalars(select(Zone).where(Zone.camera_id == camera_id))
    }
    for item in body:
        if (
            item.id
            and str(item.id) not in previous
            and db.get(Zone, str(item.id)) is not None
        ):
            raise HTTPException(409, "Зона принадлежит другой камере")
        validate_rules(db, item.rules)
    removed = set(previous) - set(ids)
    if removed and db.scalar(
        select(Incident.id).where(Incident.zone_id.in_(removed)).limit(1)
    ):
        raise HTTPException(409, "Нельзя удалить зону с инцидентами. Отключите её")
    for zone_id in removed:
        db.delete(previous[zone_id])
    result = [
        apply_zone(db, item, camera_id, previous.get(str(item.id))) for item in body
    ]
    version = bump_config(db)
    db.commit()
    await hub.publish(camera_id, version)
    return result


@router.post("/zones", response_model=ZoneOut, status_code=201, tags=["zones"])
async def create_zone(body: ZoneCreate, db: Db, user: Editor):
    camera_or_404(db, body.camera_id)
    if body.id and db.get(Zone, str(body.id)):
        raise HTTPException(409, "Зона с таким идентификатором уже существует")
    zone = apply_zone(db, body, body.camera_id)
    version = bump_config(db)
    db.commit()
    await hub.publish(body.camera_id, version)
    return zone


def zone_or_404(db, zone_id):
    zone = db.get(Zone, zone_id)
    if zone is None:
        raise HTTPException(404, "Зона не найдена")
    return zone


@router.put("/zones/{zone_id}", response_model=ZoneOut, tags=["zones"])
async def update_zone(zone_id: str, body: ZoneIn, db: Db, user: Editor):
    zone = zone_or_404(db, zone_id)
    if body.id and str(body.id) != zone_id:
        raise HTTPException(422, "Идентификатор зоны нельзя изменить")
    apply_zone(db, body, zone.camera_id, zone)
    version = bump_config(db)
    db.commit()
    await hub.publish(zone.camera_id, version)
    return zone


@router.delete("/zones/{zone_id}", tags=["zones"])
async def delete_zone(zone_id: str, db: Db, user: Editor):
    zone = zone_or_404(db, zone_id)
    if db.scalar(select(Incident.id).where(Incident.zone_id == zone_id).limit(1)):
        raise HTTPException(
            409, "Зона содержит инциденты. Отключите её вместо удаления"
        )
    camera_id = zone.camera_id
    db.delete(zone)
    version = bump_config(db)
    db.commit()
    await hub.publish(camera_id, version)
    return {"deleted": True}


def build_config(db, camera):
    thresholds = {
        key: setting(db, key, fallback)
        for key, fallback in [
            ("confirm_frames", 8),
            ("cooldown_sec", 60),
            ("min_confidence", 0.5),
        ]
    }
    zones = []
    for zone in db.scalars(
        select(Zone)
        .where(Zone.camera_id == camera.id, Zone.enabled.is_(True))
        .order_by(Zone.sort_order, Zone.id)
    ):
        rules = {
            key: value
            for key, value in zone.rules.items()
            if key in ConfigRules.model_fields
        }
        for key in ("confirm_frames", "cooldown_sec"):
            if rules.get(key) is None:
                rules[key] = thresholds[key]
        zones.append(
            {
                "id": zone.id,
                "name": zone.name,
                "type": zone.zone_type,
                "polygon": zone.polygon,
                "rules": ConfigRules(**rules),
            }
        )
    return IngestConfig(
        camera={
            key: getattr(camera, key)
            for key in ("id", "name", "stream_url", "stream_type")
        },
        frame_size_hint=(1280, 720),
        thresholds=thresholds,
        ppe_types=list(
            db.scalars(
                select(PpeType.code)
                .where(PpeType.enabled.is_(True))
                .order_by(PpeType.sort_order)
            )
        ),
        zones=zones,
        version=int(setting(db, "config_version", 1)),
        updated_at=setting(db, "config_updated_at", iso(camera.created_at)),
    )


@router.get("/ingest/config", response_model=list[IngestConfig], tags=["ingest"])
def all_config(db: Db, key: Ai):
    return [
        build_config(db, camera)
        for camera in db.scalars(
            select(Camera).where(Camera.ai_enabled.is_(True)).order_by(Camera.id)
        )
    ]


@router.get("/ingest/config/{camera_id}", response_model=IngestConfig, tags=["ingest"])
def ingest_config(camera_id: str, db: Db, key: Ai):
    return build_config(db, camera_or_404(db, camera_id))


@router.get(
    "/cameras/{camera_id}/config", response_model=IngestConfig, tags=["cameras"]
)
def browser_config(camera_id: str, db: Db, user: Reader):
    return build_config(db, camera_or_404(db, camera_id))


@router.get("/zone-responsibles", tags=["zones"])
def responsible_options(db: Db, user: Editor):
    return [
        {"id": u.id, "full_name": u.full_name}
        for u in db.scalars(select(User).order_by(User.full_name))
    ]


@router.put("/settings", response_model=Thresholds, tags=["settings"])
async def save_thresholds(body: Thresholds, db: Db, user: Editor):
    for key, value in body.model_dump().items():
        row = db.get(Setting, key)
        if row:
            row.value = value
        else:
            db.add(Setting(key=key, value=value))
    version = bump_config(db)
    db.commit()
    for camera_id in db.scalars(select(Camera.id)):
        await hub.publish(camera_id, version)
    return body


def key_payload(row):
    return {
        "id": row.id,
        "name": row.name,
        "created_at": iso(row.created_at),
        "revoked_at": iso(row.revoked_at),
    }


@router.get("/api-keys", response_model=list[KeyOut], tags=["api-keys"])
def list_keys(db: Db, user: Admin):
    return [
        key_payload(row)
        for row in db.scalars(select(ApiKey).order_by(ApiKey.created_at.desc()))
    ]


@router.post("/api-keys", response_model=KeyCreated, status_code=201, tags=["api-keys"])
def create_key(body: KeyIn, db: Db, user: Admin):
    raw = "ot_" + secrets.token_urlsafe(32)
    row = ApiKey(name=body.name, key_hash=hashlib.sha256(raw.encode()).hexdigest())
    db.add(row)
    db.commit()
    return {**key_payload(row), "key": raw}


@router.delete("/api-keys/{key_id}", response_model=KeyOut, tags=["api-keys"])
def revoke_key(key_id: str, db: Db, user: Admin):
    row = db.get(ApiKey, key_id)
    if row is None:
        raise HTTPException(404, "API-ключ не найден")
    if row.revoked_at is None:
        row.revoked_at = utc_now()
        db.commit()
    return key_payload(row)
