import asyncio
import logging
import os
import random
import time
from datetime import timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import Field
from sqlalchemy import delete, select
from starlette.concurrency import run_in_threadpool

from .database import SessionLocal
from .demo_seed import demo_details, seed_demo_history, snapshot, synthetic_objects
from .dependencies import Db
from .incident_schemas import EventIn, HeartbeatIn, StatsIn, contract_external_id
from .incidents import ingest_event, serialize_incident
from .ingest import publish_incident, save_heartbeat, save_stats
from .live_runtime import runtime
from .models import (
    Camera,
    CameraStatus,
    Comment,
    ComplianceStat,
    Incident,
    IncidentEvent,
    IncidentType,
    Notification,
    Report,
    Setting,
    Zone,
    utc_now,
)
from .realtime import hub
from .schemas import StrictModel
from .stage_two import Admin, Editor, Reader, iso, setting, status_payload
from .storage import DATA_DIR, remove_snapshot

router = APIRouter(prefix="/api/v1/demo", tags=["demo"])
logger = logging.getLogger("inspector")


def load_config():
    with SessionLocal() as db:
        cameras = db.scalars(
            select(Camera)
            .where(Camera.enabled.is_(True), Camera.ai_enabled.is_(True))
            .order_by(Camera.created_at)
        ).all()
        result = []
        for camera in cameras:
            row = db.get(CameraStatus, camera.id)
            if (
                row
                and row.last_heartbeat_at
                and (row.extra or {}).get("_source") != "mock"
                and camera.id not in runtime.real_activity
            ):
                age = (
                    utc_now() - row.last_heartbeat_at.replace(tzinfo=timezone.utc)
                ).total_seconds()
                if age < 60:
                    runtime.mark_real(camera.id, age)
            zones = [
                {
                    "id": z.id,
                    "name": z.name,
                    "polygon": z.polygon,
                    "enabled": z.enabled,
                    "zone_type": z.zone_type,
                    "rules": z.rules,
                }
                for z in db.scalars(select(Zone).where(Zone.camera_id == camera.id))
            ]
            result.append({"id": camera.id, "name": camera.name, "zones": zones})
        return {
            "enabled": bool(setting(db, "demo_mode", False)),
            "rate": float(setting(db, "demo_rate", 1)),
            "cameras": result,
        }


def make_mock(camera_id, kind):
    with SessionLocal() as db:
        camera = db.get(Camera, camera_id)
        if camera is None:
            raise HTTPException(404, "Камера не найдена")
        if (
            not setting(db, "demo_mode", False)
            or not camera.enabled
            or not camera.ai_enabled
        ):
            raise HTTPException(409, "Включите демо-режим и анализ выбранной камеры")
        if runtime.real_active(camera_id):
            raise HTTPException(
                409,
                "Для камеры работает реальный ИИ; демо приостановлено на минуту тишины",
            )
        kind_row = db.get(IncidentType, kind)
        if kind_row is None or not kind_row.enabled:
            raise HTTPException(422, "Неизвестный тип события")
        zone = db.scalar(
            select(Zone)
            .where(Zone.camera_id == camera_id, Zone.enabled.is_(True))
            .order_by(Zone.sort_order)
        )
        now = utc_now()
        track = random.randint(10_000, 2_000_000_000)
        # Mocks go through the same strict ingest validation as the AI module (contract 4.1).
        body = EventIn(
            external_id=contract_external_id(camera_id, now, track, kind),
            camera_id=camera_id,
            zone_id=zone.id if zone else None,
            type=kind,
            confidence=0.94,
            track_id=track,
            started_at=now,
            ended_at=now + timedelta(seconds=12),
            details=demo_details(kind, track),
        )
        details = body.details
        raw = snapshot(camera.name, now, details["boxes"], kind_row.name_ru)
        incident, code, notices, _ = ingest_event(db, body, raw, source="mock")
        return (
            serialize_incident(db, incident),
            code,
            [
                {"id": n.id, "recipient": n.recipient, "sent_at": iso(n.sent_at)}
                for n in notices
            ],
        )


def reset_mock_data():
    with SessionLocal.begin() as db:
        ids = list(db.scalars(select(Incident.id).where(Incident.source == "mock")))
        query = select(Incident.id).where(Incident.source == "mock")
        for model in (Comment, IncidentEvent, Notification):
            db.execute(delete(model).where(model.incident_id.in_(query)))
        db.execute(delete(Incident).where(Incident.source == "mock"))
        db.execute(delete(ComplianceStat).where(ComplianceStat.source == "mock"))
        for report in db.scalars(select(Report)):
            if (report.params or {}).get("source") == "mock":
                db.delete(report)
        db.execute(delete(Setting).where(Setting.key == "demo_seed_state"))
    for incident_id in ids:
        remove_snapshot(incident_id)
    for name in ("demo-day.html", "demo-week.html"):
        (DATA_DIR / "reports" / name).unlink(missing_ok=True)
    return len(ids)


class DemoController:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.seeding = False
        self.error = None
        self.wake = asyncio.Event()

    async def simulate(self, camera_id, kind):
        async with self.lock:
            payload, code, notices = await run_in_threadpool(make_mock, camera_id, kind)
            if code == 201:
                await publish_incident(payload, notices)
            return payload

    async def seed(self):
        self.seeding = True
        self.error = None
        try:
            await run_in_threadpool(
                seed_demo_history,
                int(os.getenv("DEMO_SEED_INCIDENTS", "360")),
                int(os.getenv("DEMO_SEED_MINUTES", "43200")),
            )
        except Exception:
            self.error = "Не удалось подготовить демо-данные. Проверьте журнал сервера и свободное место"
            logger.exception("Demo seed failed")
        finally:
            self.seeding = False
        await hub.broadcast({"type": "demo.updated"})

    async def run(self):
        config = await run_in_threadpool(load_config)
        async with asyncio.TaskGroup() as group:
            if config["enabled"]:
                group.create_task(self.bootstrap())
            group.create_task(self.supervise(self.detections))
            group.create_task(self.supervise(self.events))

    async def bootstrap(self):
        async with self.lock:
            await self.seed()

    async def supervise(self, job):
        while True:
            try:
                await job()
            except Exception:
                logger.exception("Demo background job failed")
                await asyncio.sleep(2)

    async def detections(self):
        cached, loaded, last_heartbeat, last_stats = None, 0, 0, 0
        while True:
            now = time.monotonic()
            if cached is None or now - loaded >= 2:
                cached = await run_in_threadpool(load_config)
                loaded = now
            active = (
                [c for c in cached["cameras"] if not runtime.real_active(c["id"])]
                if cached["enabled"] and not runtime.demo_resetting
                else []
            )
            timestamp = time.time()
            for camera in active:
                message = {
                    "type": "detections",
                    "camera_id": camera["id"],
                    "ts": iso(utc_now()),
                    "frame_id": int(timestamp * 10),
                    "objects": synthetic_objects(
                        camera["id"], timestamp, camera["zones"]
                    ),
                    "source": "mock",
                }
                hub.latest_detections[camera["id"]] = message
                await hub.broadcast(message)
            if now - last_heartbeat >= 5:
                for camera in active:

                    def beat(camera=camera):
                        with SessionLocal() as db:
                            if runtime.demo_resetting or runtime.real_active(
                                camera["id"]
                            ):
                                return None
                            return save_heartbeat(
                                db,
                                HeartbeatIn(
                                    camera_id=camera["id"],
                                    fps=10,
                                    model_name="Демо-генератор",
                                    model_version="0.4",
                                    extra={"demo": True},
                                ),
                                source="mock",
                            )

                    status = await run_in_threadpool(beat)
                    if status:
                        await hub.broadcast({"type": "camera.status", **status})
                last_heartbeat = now
            if now - last_stats >= 60 and not self.seeding:
                for camera in active:

                    def record(camera=camera):
                        with SessionLocal() as db:
                            if (
                                runtime.demo_resetting
                                or self.seeding
                                or runtime.real_active(camera["id"])
                            ):
                                return
                            end = utc_now().replace(second=0, microsecond=0)
                            save_stats(
                                db,
                                StatsIn(
                                    camera_id=camera["id"],
                                    period_start=end - timedelta(minutes=1),
                                    period_end=end,
                                    persons_total=42,
                                    persons_compliant=40,
                                    by_ppe={
                                        "helmet": {"checked": 42, "compliant": 40},
                                        "vest": {"checked": 42, "compliant": 41},
                                    },
                                ),
                                source="mock",
                            )

                    await run_in_threadpool(record)
                last_stats = now
            await asyncio.sleep(0.1)

    async def events(self):
        while True:
            config = await run_in_threadpool(load_config)
            delay = random.uniform(20, 90) / max(0.1, config["rate"])
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=delay)
                continue
            except asyncio.TimeoutError:
                pass
            config = await run_in_threadpool(load_config)
            active = [c for c in config["cameras"] if not runtime.real_active(c["id"])]
            if not config["enabled"] or not active or runtime.demo_resetting:
                continue
            kind = random.choices(
                [
                    "ppe_violation",
                    "zone_intrusion",
                    "vehicle_proximity",
                    "overcrowding",
                    "fall_detected",
                    "person_immobile",
                ],
                weights=[70, 15, 8, 4, 1.5, 1.5],
            )[0]
            try:
                await self.simulate(random.choice(active)["id"], kind)
            except HTTPException:
                pass
            except Exception:
                logger.exception("Demo event failed")


controller = DemoController()


class DemoIn(StrictModel):
    enabled: bool
    rate: float = Field(ge=0.1, le=10, allow_inf_nan=False)


class SimulateIn(StrictModel):
    camera_id: str
    type: str = Field(min_length=1, max_length=50)


@router.get("")
def demo_status(db: Db, user: Reader):
    config = load_config()
    return {
        "enabled": config["enabled"],
        "rate": config["rate"],
        "seeding": controller.seeding,
        "error": controller.error,
        "active_camera_ids": [
            c["id"]
            for c in config["cameras"]
            if config["enabled"] and not runtime.real_active(c["id"])
        ],
        "real_camera_ids": [
            c["id"] for c in config["cameras"] if runtime.real_active(c["id"])
        ],
    }


@router.put("")
async def configure_demo(body: DemoIn, db: Db, user: Editor):
    async with controller.lock:
        for key, value in (("demo_mode", body.enabled), ("demo_rate", body.rate)):
            row = db.get(Setting, key)
            if row:
                row.value = value
            else:
                db.add(Setting(key=key, value=value))
        messages = []
        if not body.enabled:
            for row in db.scalars(select(CameraStatus)):
                if (row.extra or {}).get("_source") == "mock":
                    row.status = "offline"
                    camera = db.get(Camera, row.camera_id)
                    if camera:
                        messages.append(
                            {"type": "camera.status", **status_payload(db, camera)}
                        )
            for camera_id, frame in list(hub.latest_detections.items()):
                if frame.get("source") == "mock":
                    hub.latest_detections.pop(camera_id, None)
        db.commit()
        if body.enabled:
            state = db.get(Setting, "demo_seed_state")
            if state is None or not state.value.get("done"):
                runtime.demo_resetting = True
                try:
                    await controller.seed()
                finally:
                    runtime.demo_resetting = False
        for message in messages:
            await hub.broadcast(message)
        controller.wake.set()
    await hub.broadcast({"type": "demo.updated"})
    return {"enabled": body.enabled, "rate": body.rate}


@router.post("/simulate", status_code=201)
async def simulate(body: SimulateIn, db: Db, user: Editor):
    return await controller.simulate(body.camera_id, body.type)


@router.post("/reset")
async def reset_demo(db: Db, user: Admin):
    async with controller.lock:
        if not setting(db, "demo_mode", False):
            raise HTTPException(409, "Для пересоздания данных включите демо-режим")
        runtime.demo_resetting = True
        try:
            count = await run_in_threadpool(reset_mock_data)
            await controller.seed()
        finally:
            runtime.demo_resetting = False
        controller.wake.set()
    if controller.error:
        raise HTTPException(503, controller.error)
    await hub.broadcast({"type": "demo.updated"})
    return {"deleted": count, "seeded": True}
