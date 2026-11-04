"""Incident transactions and serializers; no detection or video processing."""

from datetime import timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from .models import (
    Area,
    Camera,
    Comment,
    Incident,
    IncidentEvent,
    IncidentType,
    IngestReceipt,
    Notification,
    User,
    Zone,
    new_id,
    utc_now,
)
from .stage_two import camera_or_404, iso, setting
from .storage import decode_snapshot, remove_snapshot, save_snapshot
from .telegram import enqueue_incident


def context_for(db, rows):
    camera_ids = {r.camera_id for r in rows}
    zone_ids = {r.zone_id for r in rows if r.zone_id}
    user_ids = {
        value
        for r in rows
        for value in (r.assignee_id, r.acknowledged_by, r.resolved_by)
        if value
    }
    cameras = {
        c.id: c for c in db.scalars(select(Camera).where(Camera.id.in_(camera_ids)))
    }
    return {
        "cameras": cameras,
        "zones": {
            z.id: z for z in db.scalars(select(Zone).where(Zone.id.in_(zone_ids)))
        },
        "users": {
            u.id: u for u in db.scalars(select(User).where(User.id.in_(user_ids)))
        },
        "types": {t.code: t for t in db.scalars(select(IncidentType))},
        "areas": {
            a.id: a
            for a in db.scalars(
                select(Area).where(Area.id.in_({c.area_id for c in cameras.values()}))
            )
        },
    }


def serialize_incident(db, row, context=None):
    context = context or context_for(db, [row])
    camera = context["cameras"].get(row.camera_id)
    zone = context["zones"].get(row.zone_id)
    kind = context["types"].get(row.type_code)
    area = context["areas"].get(camera.area_id) if camera else None
    result = {
        column.name: getattr(row, column.name)
        for column in Incident.__table__.columns
        if column.name not in ("snapshot_path", "thumbnail_path", "clip_path")
    }
    for key in (
        "started_at",
        "ended_at",
        "acknowledged_at",
        "resolved_at",
        "created_at",
    ):
        result[key] = iso(getattr(row, key))
    result.update(
        camera_name=camera.name if camera else "",
        area_id=camera.area_id if camera else None,
        area_name=area.name if area else "",
        zone_name=zone.name if zone else None,
        type_name=kind.name_ru if kind else row.type_code,
        snapshot_url=row.snapshot_path,
        thumbnail_url=row.thumbnail_path,
        clip_url=row.clip_path,
        reaction_sec=max(
            0,
            (
                row.acknowledged_at.replace(tzinfo=timezone.utc)
                - row.created_at.replace(tzinfo=timezone.utc)
            ).total_seconds(),
        )
        if row.acknowledged_at
        else None,
    )
    for field in ("assignee", "acknowledged", "resolved"):
        user_id = (
            row.assignee_id if field == "assignee" else getattr(row, f"{field}_by")
        )
        person = context["users"].get(user_id)
        result[f"{field}_name"] = person.full_name if person else None
    return result


def record_event(db, incident, event_type, user=None, payload=None):
    event = IncidentEvent(
        incident_id=incident.id,
        event_type=event_type,
        user_id=user.id if user else None,
        payload=payload or {},
        created_at=utc_now(),
    )
    db.add(event)
    return event


def get_incident(db, incident_id, lock=False):
    if lock:
        # A no-op write also obtains the SQLite write lock before reading a status.
        result = db.execute(
            update(Incident)
            .where(Incident.id == incident_id)
            .values(status=Incident.status)
        )
        if result.rowcount == 0:
            db.rollback()
            raise HTTPException(404, "Инцидент не найден")
    incident = db.get(Incident, incident_id, populate_existing=lock)
    if incident is None:
        raise HTTPException(404, "Инцидент не найден")
    return incident


def check_action(db, incident, user, action):
    allowed = user.role in ("admin", "engineer")
    if user.role == "master" and action in ("acknowledge", "resolve", "comment"):
        camera = db.get(Camera, incident.camera_id)
        allowed = camera is not None and camera.area_id == user.area_id
    if not allowed:
        db.rollback()
        raise HTTPException(403, "Недостаточно прав для этого действия на участке")
    valid_statuses = {
        "acknowledge": ("new", "acknowledged"),
        "resolve": ("acknowledged",),
        "false_positive": ("new", "acknowledged", "resolved", "false_positive"),
        "reopen": ("resolved", "false_positive"),
        "assign": ("new", "acknowledged", "resolved", "false_positive"),
        "comment": ("new", "acknowledged", "resolved", "false_positive"),
    }
    if incident.status not in valid_statuses[action]:
        db.rollback()
        raise HTTPException(409, "Действие недоступно для текущего статуса инцидента")


def apply_action(db, incident, user, action, value=None):
    check_action(db, incident, user, action)
    now = utc_now()
    if action == "acknowledge":
        if incident.status == "acknowledged":
            return False
        incident.status = "acknowledged"
        incident.acknowledged_at, incident.acknowledged_by = now, user.id
        record_event(db, incident, "acknowledged", user)
    elif action == "resolve":
        incident.status = "resolved"
        incident.resolved_at, incident.resolved_by, incident.resolution_note = (
            now,
            user.id,
            value,
        )
        record_event(db, incident, "resolved", user, {"note": value})
    elif action == "false_positive":
        if incident.status == "false_positive":
            return False
        incident.status = "false_positive"
        incident.false_positive_reason = value
        incident.resolved_at, incident.resolved_by = now, user.id
        record_event(db, incident, "false_positive", user, {"reason": value})
    elif action == "reopen":
        previous = incident.status
        incident.status = "acknowledged"
        if incident.acknowledged_at is None:
            incident.acknowledged_at, incident.acknowledged_by = now, user.id
        incident.resolved_at, incident.resolved_by = None, None
        incident.resolution_note, incident.false_positive_reason = None, None
        record_event(db, incident, "reopened", user, {"previous_status": previous})
    elif action == "assign":
        if value and db.get(User, value) is None:
            raise HTTPException(422, "Пользователь для назначения не найден")
        if value:
            assignee = db.get(User, value)
            camera = db.get(Camera, incident.camera_id)
            if assignee.role == "viewer" or (
                assignee.role == "master" and assignee.area_id != camera.area_id
            ):
                raise HTTPException(
                    422, "Ответственный должен иметь права на работу с инцидентом"
                )
        if incident.assignee_id == value:
            return False
        incident.assignee_id = value
        record_event(db, incident, "assigned", user, {"user_id": value})
    return True


def ingest_event(db, body, snapshot=None, source="ai"):
    camera = camera_or_404(db, body.camera_id)
    # Serialize same-camera writes across workers before querying the cooldown.
    db.execute(
        update(Camera).where(Camera.id == camera.id).values(enabled=Camera.enabled)
    )
    if source == "mock":
        from .live_runtime import runtime

        if runtime.real_active(camera.id):
            db.rollback()
            raise HTTPException(409, "Демо этой камеры приостановлено реальным ИИ")
    receipt = db.get(IngestReceipt, body.external_id)
    existing = (
        db.get(Incident, receipt.incident_id)
        if receipt
        else db.scalar(select(Incident).where(Incident.external_id == body.external_id))
    )
    if existing:
        db.commit()
        return existing, 200, [], False
    kind = db.get(IncidentType, body.type)
    if kind is None or not kind.enabled:
        raise HTTPException(422, "Неизвестный или отключённый тип инцидента")
    zone = db.get(Zone, body.zone_id) if body.zone_id else None
    if body.zone_id and (zone is None or zone.camera_id != camera.id):
        raise HTTPException(422, "Зона не найдена или принадлежит другой камере")
    rules = zone.rules if zone else {}
    cooldown = rules.get("cooldown_sec")
    if cooldown is None:
        cooldown = setting(db, "cooldown_sec", 60)
    if cooldown > 0:
        # Event time, not receive time: events spooled by the AI during an outage arrive in
        # a burst and must only be suppressed when they started within cooldown_sec of an
        # incident of the same camera, zone, type and track (in either order).
        window = timedelta(seconds=cooldown)
        existing = db.scalar(
            select(Incident)
            .where(
                Incident.camera_id == camera.id,
                Incident.zone_id == body.zone_id,
                Incident.type_code == body.type,
                Incident.track_id == body.track_id,
                Incident.started_at > body.started_at - window,
                Incident.started_at < body.started_at + window,
            )
            .order_by(Incident.started_at.desc())
            .limit(1)
        )
        if existing:
            db.add(IngestReceipt(external_id=body.external_id, incident_id=existing.id))
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                receipt = db.get(IngestReceipt, body.external_id)
                if receipt:
                    return db.get(Incident, receipt.incident_id), 200, [], False
                raise
            return existing, 202, [], True
    severity = body.severity or rules.get("severity_override")
    if not severity:
        severity = (
            "high"
            if (
                body.type == "ppe_violation"
                and zone
                and zone.zone_type == "danger"
                and "helmet" in body.details.get("missing_ppe", [])
            )
            else kind.default_severity
        )
    incident_id = new_id()
    if snapshot is None:
        snapshot = decode_snapshot(body.snapshot_base64)
    try:
        snapshot_path, thumbnail_path = save_snapshot(incident_id, snapshot)
        incident = Incident(
            id=incident_id,
            external_id=body.external_id,
            source=source,
            camera_id=camera.id,
            zone_id=body.zone_id,
            type_code=body.type,
            severity=severity,
            status="new",
            confidence=body.confidence,
            track_id=body.track_id,
            started_at=body.started_at,
            ended_at=body.ended_at,
            duration_sec=(body.ended_at - body.started_at).total_seconds()
            if body.ended_at
            else None,
            details=body.details,
            snapshot_path=snapshot_path,
            thumbnail_path=thumbnail_path,
        )
        db.add(incident)
        db.flush()
        db.add(IngestReceipt(external_id=body.external_id, incident_id=incident.id))
        created = record_event(db, incident, "created", payload={"source": source})
        recipients = db.scalars(select(User)).all()
        notices = []
        for person in recipients:
            if person.role == "master" and person.area_id != camera.area_id:
                continue
            notification = Notification(
                incident_id=incident.id,
                channel="inapp",
                recipient=person.id,
                status="sent",
                sent_at=utc_now(),
            )
            db.add(notification)
            notices.append(notification)
        notified = record_event(
            db,
            incident,
            "notified",
            payload={"channel": "inapp", "recipients": len(notices)},
        )
        # The timeline is ordered by time; a coarse clock (about 1 ms on Windows) can give
        # both events the same timestamp, so "notified" is kept after "created".
        notified.created_at = max(
            notified.created_at, created.created_at + timedelta(microseconds=1)
        )
        enqueue_incident(db, incident)
        db.commit()
        return incident, 201, notices, False
    except IntegrityError:
        db.rollback()
        remove_snapshot(incident_id)
        receipt = db.get(IngestReceipt, body.external_id)
        if receipt:
            return db.get(Incident, receipt.incident_id), 200, [], False
        existing = db.scalar(
            select(Incident).where(Incident.external_id == body.external_id)
        )
        if existing:
            return existing, 200, [], False
        raise
    except Exception:
        db.rollback()
        remove_snapshot(incident_id)
        raise


def comments_payload(db, incident_id):
    rows = db.execute(
        select(Comment, User.full_name)
        .join(User, User.id == Comment.user_id)
        .where(Comment.incident_id == incident_id)
        .order_by(Comment.created_at, Comment.id)
    ).all()
    return [
        {
            "id": c.id,
            "incident_id": c.incident_id,
            "user_id": c.user_id,
            "user_name": name,
            "text": c.text,
            "created_at": iso(c.created_at),
        }
        for c, name in rows
    ]


def timeline_payload(db, incident_id):
    rows = db.execute(
        select(IncidentEvent, User.full_name)
        .outerjoin(User, User.id == IncidentEvent.user_id)
        .where(IncidentEvent.incident_id == incident_id)
        .order_by(IncidentEvent.created_at, IncidentEvent.id)
    ).all()
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "user_id": e.user_id,
            "user_name": name,
            "payload": e.payload,
            "created_at": iso(e.created_at),
        }
        for e, name in rows
    ]
