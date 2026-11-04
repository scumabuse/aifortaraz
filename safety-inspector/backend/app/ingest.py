import asyncio
import json
import logging
from datetime import timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import select, update
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .database import SessionLocal
from .dependencies import Db
from .incident_schemas import EndEvent, EventIn, HeartbeatIn, StatsIn
from .incidents import ingest_event, serialize_incident
from .models import (
    Camera,
    CameraStatus,
    ComplianceStat,
    Incident,
    IngestReceipt,
    utc_now,
)
from .realtime import hub
from .live_runtime import runtime
from .stage_two import (
    Ai,
    camera_or_404,
    iso,
    setting,
    status_payload,
    supported_ppe,
)
from .storage import check_snapshot, decode_snapshot

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])
logger = logging.getLogger("inspector")
MAX_REQUEST_BYTES = 8 * 1024 * 1024


async def parse_event(request: Request):
    content_type = request.headers.get("content-type", "").split(";")[0].lower()
    if content_type not in ("application/json", "multipart/form-data"):
        raise HTTPException(415, "Используйте application/json или multipart/form-data")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise HTTPException(
                413, "Размер запроса превышает 8 МБ; снимок — не более 5 МБ"
            )
        chunks.append(chunk)
    raw = b"".join(chunks)
    snapshot = None
    try:
        if content_type == "application/json":
            payload = json.loads(raw)
        else:

            async def receive():
                return {"type": "http.request", "body": raw, "more_body": False}

            parsed = Request(request.scope, receive=receive)
            async with parsed.form(
                max_files=1, max_fields=1, max_part_size=256 * 1024
            ) as form:
                if (
                    set(form.keys()) - {"payload", "snapshot"}
                    or len(form.getlist("payload")) != 1
                ):
                    raise HTTPException(
                        422,
                        "Ожидаются поле payload (JSON) и необязательный файл snapshot",
                    )
                text = form["payload"]
                if not isinstance(text, str):
                    raise HTTPException(422, "payload должен быть JSON-строкой")
                payload = json.loads(text)
                file = form.get("snapshot")
                if file is not None:
                    if not isinstance(file, UploadFile):
                        raise HTTPException(422, "snapshot должен быть файлом JPEG")
                    snapshot = await file.read(5 * 1024 * 1024 + 1)
                    if len(snapshot) > 5 * 1024 * 1024:
                        raise HTTPException(413, "Размер снимка превышает 5 МБ")
                if isinstance(payload, dict) and "snapshot_base64" in payload:
                    raise HTTPException(
                        422, "В multipart передайте снимок файлом snapshot"
                    )
        body = EventIn.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors())
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        raise HTTPException(422, "Не удалось прочитать JSON события")
    except StarletteHTTPException as exc:
        if exc.status_code == 400:
            raise HTTPException(
                422,
                "Некорректная multipart-форма; ожидаются payload и один JPEG-файл snapshot",
            )
        raise
    # Snapshot size and JPEG signature are checked before deduplication, so a malformed
    # snapshot is always a 422/413 (contract 4.1, 4.8).
    if snapshot is None:
        snapshot = decode_snapshot(body.snapshot_base64)
    return body, check_snapshot(snapshot)


@router.post(
    "/events",
    status_code=201,
    responses={
        200: {"description": "Same external_id"},
        202: {"description": "Cooldown deduplication"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": EventIn.model_json_schema()},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["payload"],
                        "properties": {
                            "payload": {
                                "type": "string",
                                "description": "JSON matching EventIn, without snapshot_base64",
                            },
                            "snapshot": {
                                "type": "string",
                                "format": "binary",
                                "description": "JPEG, max 5 MB",
                            },
                        },
                    }
                },
            },
        }
    },
)
async def create_event(request: Request, db: Db, key: Ai):
    body, snapshot = await parse_event(request)
    incident, code, notices, cooldown = await run_in_threadpool(
        ingest_event, db, body, snapshot
    )
    if code == 201:
        payload = serialize_incident(db, incident)
        await publish_incident(
            payload,
            [
                {"id": n.id, "recipient": n.recipient, "sent_at": iso(n.sent_at)}
                for n in notices
            ],
        )
    if code == 202:
        return JSONResponse(
            status_code=202, content={"deduplicated": True, "id": incident.id}
        )
    return JSONResponse(
        status_code=code,
        content={
            "id": incident.id,
            "status": incident.status,
            "severity": incident.severity,
            "deduplicated": code != 201,
        },
    )


async def publish_incident(payload, notices):
    await hub.broadcast({"type": "incident.created", "incident": payload})
    for notice in notices:
        await hub.broadcast(
            {
                "type": "notification",
                "id": notice["id"],
                "incident_id": payload["id"],
                "text": f"{payload['type_name']} · {payload['camera_name']}",
                "created_at": notice["sent_at"],
            },
            recipient=notice["recipient"],
        )


@router.patch("/events/{external_id:path}")
async def end_event(external_id: str, body: EndEvent, db: Db, key: Ai):
    db.execute(
        update(Incident)
        .where(Incident.external_id == external_id)
        .values(status=Incident.status)
    )
    incident = db.scalar(
        select(Incident)
        .where(Incident.external_id == external_id)
        .execution_options(populate_existing=True)
    )
    if incident is None:
        receipt = db.get(IngestReceipt, external_id)
        if receipt:
            db.execute(
                update(Incident)
                .where(Incident.id == receipt.incident_id)
                .values(status=Incident.status)
            )
            incident = db.get(Incident, receipt.incident_id, populate_existing=True)
    if incident is None:
        db.rollback()
        raise HTTPException(404, "Событие с таким external_id не найдено")
    if body.ended_at < incident.started_at.replace(tzinfo=timezone.utc):
        db.rollback()
        raise HTTPException(422, "Время окончания не может быть раньше начала")
    incident.ended_at = body.ended_at
    incident.duration_sec = (
        body.ended_at - incident.started_at.replace(tzinfo=timezone.utc)
    ).total_seconds()
    db.commit()
    result = serialize_incident(db, incident)
    await hub.broadcast({"type": "incident.updated", "incident": result})
    return result


def save_heartbeat(db, body, source="ai"):
    camera = camera_or_404(db, body.camera_id)
    db.execute(
        update(Camera).where(Camera.id == camera.id).values(enabled=Camera.enabled)
    )
    if source == "mock" and runtime.real_active(camera.id):
        db.rollback()
        return status_payload(db, camera)
    row = db.get(CameraStatus, camera.id)
    if row is None:
        row = CameraStatus(camera_id=camera.id)
        db.add(row)
    row.status = {
        "ok": "online",
        "online": "online",
        "offline": "offline",
        "degraded": "degraded",
        "error": "degraded",
    }[body.status]
    row.fps, row.model_name, row.model_version = (
        body.fps,
        body.model_name,
        body.model_version,
    )
    extra = {**body.extra, "_source": source}
    if source != "ai":
        # extra.supported_ppe describes the model of the latest AI heartbeat (contract 4.2);
        # heartbeats of the demo generator neither set nor clear it.
        extra.pop("supported_ppe", None)
        if "supported_ppe" in (row.extra or {}):
            extra["supported_ppe"] = row.extra["supported_ppe"]
    row.last_heartbeat_at, row.extra = utc_now(), extra
    db.commit()
    if source == "ai":
        runtime.mark_real(camera.id)
    return status_payload(db, camera)


@router.post("/heartbeat")
async def heartbeat(body: HeartbeatIn, db: Db, key: Ai):
    result = await run_in_threadpool(save_heartbeat, db, body)
    await hub.broadcast({"type": "camera.status", **result})
    return result


def save_stats(db, body, source="ai"):
    camera = camera_or_404(db, body.camera_id)
    db.execute(
        update(Camera).where(Camera.id == camera.id).values(enabled=Camera.enabled)
    )
    if source == "mock" and runtime.real_active(camera.id):
        db.rollback()
        return {"id": None, "compliance_percent": None}
    row = db.scalar(
        select(ComplianceStat).where(
            ComplianceStat.camera_id == body.camera_id,
            ComplianceStat.period_start == body.period_start,
            ComplianceStat.period_end == body.period_end,
        )
    )
    if row is None:
        row = ComplianceStat()
        db.add(row)
    elif source == "mock" and row.source == "ai":
        db.commit()
        return {
            "id": row.id,
            "compliance_percent": round(
                100 * row.persons_compliant / row.persons_total, 2
            )
            if row.persons_total
            else None,
        }
    for name, value in body.model_dump(mode="python").items():
        setattr(row, name, value)
    row.by_ppe = {code: counts.model_dump() for code, counts in body.by_ppe.items()}
    row.source = source
    db.commit()
    if source == "ai":
        runtime.mark_real(camera.id)
    return {
        "id": row.id,
        "compliance_percent": round(100 * row.persons_compliant / row.persons_total, 2)
        if row.persons_total
        else None,
    }


@router.post("/stats", status_code=201)
def stats(body: StatsIn, background_tasks: BackgroundTasks, db: Db, key: Ai):
    result = save_stats(db, body)
    background_tasks.add_task(
        hub.broadcast,
        {
            "type": "camera.status",
            **status_payload(db, camera_or_404(db, body.camera_id)),
        },
    )
    return result


async def expire_heartbeats():
    messages = []
    with SessionLocal() as db:
        timeout = setting(db, "heartbeat_timeout_sec", 15)
        for row in db.scalars(
            select(CameraStatus).where(CameraStatus.status != "offline")
        ):
            if (
                row.last_heartbeat_at
                and (
                    utc_now() - row.last_heartbeat_at.replace(tzinfo=timezone.utc)
                ).total_seconds()
                > timeout
            ):
                result = db.execute(
                    update(CameraStatus)
                    .where(
                        CameraStatus.camera_id == row.camera_id,
                        CameraStatus.last_heartbeat_at == row.last_heartbeat_at,
                        CameraStatus.status != "offline",
                    )
                    .values(status="offline")
                )
                if result.rowcount != 1:
                    continue
                messages.append(
                    {
                        "type": "camera.status",
                        "camera_id": row.camera_id,
                        "status": "offline",
                        "fps": None,
                        "model_name": row.model_name,
                        "model_version": row.model_version,
                        "last_heartbeat_at": iso(row.last_heartbeat_at),
                        "supported_ppe": supported_ppe(row),
                    }
                )
        db.commit()
    for message in messages:
        await hub.broadcast(message)


async def heartbeat_monitor():
    while True:
        await asyncio.sleep(1)
        try:
            await expire_heartbeats()
        except Exception:
            logger.exception("Heartbeat expiry check failed")
