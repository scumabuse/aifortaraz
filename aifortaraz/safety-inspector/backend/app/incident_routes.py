import csv
import io
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, case, func, or_, select, update

from .analytics import compliance_rows
from .dependencies import Db
from .incident_schemas import (
    AssignIn,
    BulkIn,
    CommentIn,
    FalsePositiveIn,
    ResolveIn,
    Severity,
    Status,
)
from .incidents import (
    apply_action,
    check_action,
    comments_payload,
    context_for,
    get_incident,
    record_event,
    serialize_incident,
    timeline_payload,
)
from .models import (
    Camera,
    Comment,
    ComplianceStat,
    Incident,
    IncidentType,
    Notification,
    User,
    Zone,
    utc_now,
)
from .realtime import hub
from .stage_two import Reader, iso, status_payload

router = APIRouter(prefix="/api/v1", tags=["incidents"])
SOURCES = ("ai", "mock", "manual")


class Filters:
    def __init__(
        self,
        from_: Annotated[datetime | None, Query(alias="from")] = None,
        to: datetime | None = None,
        camera_id: str | None = None,
        camera_ids: Annotated[str | None, Query(max_length=4096)] = None,
        area_id: str | None = None,
        zone_id: str | None = None,
        type: str | None = None,
        severity: Severity | None = None,
        status: Status | None = None,
        assignee_id: str | None = None,
        q: Annotated[str, Query(max_length=250)] = "",
        only_unacknowledged: bool = False,
        source: Annotated[
            str | None,
            Query(
                max_length=100,
                description="ai, mock or manual; several values separated by commas",
            ),
        ] = None,
    ):
        for value in (from_, to):
            if value is not None and value.tzinfo is None:
                raise HTTPException(422, "Укажите часовой пояс периода")
        if from_ and to and to < from_:
            raise HTTPException(422, "Конец периода должен быть не раньше начала")
        self.from_, self.to, self.camera_id, self.area_id = (
            from_,
            to,
            camera_id,
            area_id,
        )
        self.zone_id, self.type, self.severity, self.status = (
            zone_id,
            type,
            severity,
            status,
        )
        self.assignee_id, self.q, self.only_unacknowledged = (
            assignee_id,
            q.strip(),
            only_unacknowledged,
        )
        self.camera_ids = (
            list(dict.fromkeys(camera_ids.split(","))) if camera_ids else []
        )
        if len(self.camera_ids) > 100:
            raise HTTPException(422, "Допускается не более 100 камер в фильтре")
        self.sources = list(
            dict.fromkeys(
                value.strip() for value in (source or "").split(",") if value.strip()
            )
        )
        if not set(self.sources).issubset(SOURCES):
            raise HTTPException(422, "Источник инцидента: ai, mock или manual")

    def conditions(self):
        result = []
        if self.camera_ids:
            result.append(Incident.camera_id.in_(self.camera_ids))
        if self.sources:
            result.append(Incident.source.in_(self.sources))
        for column, value in [
            (Incident.camera_id, self.camera_id),
            (Camera.area_id, self.area_id),
            (Incident.zone_id, self.zone_id),
            (Incident.type_code, self.type),
            (Incident.severity, self.severity),
            (Incident.status, self.status),
        ]:
            if value is not None:
                result.append(column == value)
        if self.from_:
            result.append(Incident.started_at >= self.from_.astimezone(timezone.utc))
        if self.to:
            result.append(Incident.started_at <= self.to.astimezone(timezone.utc))
        if self.assignee_id:
            result.append(
                Incident.assignee_id.is_(None)
                if self.assignee_id == "unassigned"
                else Incident.assignee_id == self.assignee_id
            )
        if self.only_unacknowledged:
            result.append(Incident.status == "new")
        if self.q:
            result.append(
                or_(
                    *[
                        column.icontains(self.q, autoescape=True)
                        for column in (
                            Incident.id,
                            Incident.external_id,
                            Camera.name,
                            Zone.name,
                        )
                    ]
                )
            )
        return result

    def query(self):
        return (
            select(Incident)
            .join(Camera, Camera.id == Incident.camera_id)
            .outerjoin(Zone, Zone.id == Incident.zone_id)
            .where(*self.conditions())
        )


FilterDependency = Annotated[Filters, Depends()]
Sort = Literal[
    "started_at",
    "-started_at",
    "created_at",
    "-created_at",
    "severity",
    "-severity",
    "status",
    "-status",
]


def order_columns(sort):
    field = sort.removeprefix("-")
    expression = (
        case({"low": 0, "medium": 1, "high": 2, "critical": 3}, value=Incident.severity)
        if field == "severity"
        else getattr(Incident, field)
    )
    return (
        expression.desc() if sort.startswith("-") else expression.asc(),
        Incident.id.desc(),
    )


@router.get("/incidents")
def list_incidents(
    db: Db,
    user: Reader,
    filters: FilterDependency,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    sort: Sort = "-started_at",
):
    query = filters.query()
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = db.scalars(
        query.order_by(*order_columns(sort))
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    context = context_for(db, rows)
    return {
        "items": [serialize_incident(db, row, context) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/incidents/export.csv")
def export_csv(
    db: Db, user: Reader, filters: FilterDependency, sort: Sort = "-started_at"
):
    query = filters.query().order_by(*order_columns(sort))

    def safe(value):
        value = "" if value is None else str(value)
        return (
            "'" + value
            if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
            else value
        )

    def generate():
        output = io.StringIO()
        writer = csv.writer(output, delimiter=";")
        writer.writerow(
            [
                "ID",
                "Время UTC",
                "Тип",
                "Камера",
                "Зона",
                "Критичность",
                "Статус",
                "Ответственный",
                "Реакция, с",
            ]
        )
        yield "\ufeff" + output.getvalue()
        for row in db.scalars(query.execution_options(yield_per=100)):
            item = serialize_incident(db, row)
            item["severity"] = {
                "low": "Низкая",
                "medium": "Средняя",
                "high": "Высокая",
                "critical": "Критическая",
            }[item["severity"]]
            item["status"] = {
                "new": "Новый",
                "acknowledged": "Принят",
                "resolved": "Закрыт",
                "false_positive": "Ложное срабатывание",
            }[item["status"]]
            output.seek(0)
            output.truncate(0)
            writer.writerow(
                [
                    safe(item[key])
                    for key in [
                        "id",
                        "started_at",
                        "type_name",
                        "camera_name",
                        "zone_name",
                        "severity",
                        "status",
                        "assignee_name",
                        "reaction_sec",
                    ]
                ]
            )
            yield output.getvalue()

    return StreamingResponse(
        generate(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="incidents.csv"'},
    )


@router.get("/incident-assignees")
def assignees(db: Db, user: Reader):
    return [
        {
            "id": row.id,
            "full_name": row.full_name,
            "role": row.role,
            "area_id": row.area_id,
        }
        for row in db.scalars(
            select(User).where(User.role != "viewer").order_by(User.full_name)
        )
    ]


@router.post("/incidents/bulk")
async def bulk(body: BulkIn, db: Db, user: Reader):
    rows = [
        get_incident(db, incident_id, lock=True) for incident_id in sorted(body.ids)
    ]
    for row in rows:
        check_action(db, row, user, body.action)
    changed = [
        row for row in rows if apply_action(db, row, user, body.action, body.reason)
    ]
    db.commit()
    result = [serialize_incident(db, row) for row in rows]
    for row in changed:
        await hub.broadcast(
            {"type": "incident.updated", "incident": serialize_incident(db, row)}
        )
    return {"items": result, "updated": len(changed)}


@router.get("/incidents/{incident_id}")
def incident_detail(incident_id: str, db: Db, user: Reader):
    incident = get_incident(db, incident_id)
    newer = or_(
        Incident.started_at > incident.started_at,
        and_(Incident.started_at == incident.started_at, Incident.id > incident.id),
    )
    older = or_(
        Incident.started_at < incident.started_at,
        and_(Incident.started_at == incident.started_at, Incident.id < incident.id),
    )
    return {
        **serialize_incident(db, incident),
        "timeline": timeline_payload(db, incident_id),
        "comments": comments_payload(db, incident_id),
        "previous_id": db.scalar(
            select(Incident.id)
            .where(newer)
            .order_by(Incident.started_at, Incident.id)
            .limit(1)
        ),
        "next_id": db.scalar(
            select(Incident.id)
            .where(older)
            .order_by(Incident.started_at.desc(), Incident.id.desc())
            .limit(1)
        ),
    }


async def action_result(db, incident, user, action, value=None):
    try:
        changed = apply_action(db, incident, user, action, value)
        db.commit()
    except Exception:
        db.rollback()
        raise
    result = serialize_incident(db, incident)
    if changed:
        await hub.broadcast({"type": "incident.updated", "incident": result})
    return result


@router.post("/incidents/{incident_id}/acknowledge")
async def acknowledge(incident_id: str, db: Db, user: Reader):
    return await action_result(
        db, get_incident(db, incident_id, lock=True), user, "acknowledge"
    )


@router.post("/incidents/{incident_id}/assign")
async def assign(incident_id: str, body: AssignIn, db: Db, user: Reader):
    return await action_result(
        db, get_incident(db, incident_id, lock=True), user, "assign", body.user_id
    )


@router.post("/incidents/{incident_id}/resolve")
async def resolve(incident_id: str, body: ResolveIn, db: Db, user: Reader):
    return await action_result(
        db, get_incident(db, incident_id, lock=True), user, "resolve", body.note
    )


@router.post("/incidents/{incident_id}/false-positive")
async def false_positive(incident_id: str, body: FalsePositiveIn, db: Db, user: Reader):
    return await action_result(
        db,
        get_incident(db, incident_id, lock=True),
        user,
        "false_positive",
        body.reason,
    )


@router.post("/incidents/{incident_id}/reopen")
async def reopen(incident_id: str, db: Db, user: Reader):
    return await action_result(
        db, get_incident(db, incident_id, lock=True), user, "reopen"
    )


@router.get("/incidents/{incident_id}/comments")
def get_comments(incident_id: str, db: Db, user: Reader):
    get_incident(db, incident_id)
    return comments_payload(db, incident_id)


@router.post("/incidents/{incident_id}/comments", status_code=201)
async def add_comment(incident_id: str, body: CommentIn, db: Db, user: Reader):
    incident = get_incident(db, incident_id, lock=True)
    check_action(db, incident, user, "comment")
    comment = Comment(incident_id=incident_id, user_id=user.id, text=body.text)
    db.add(comment)
    record_event(db, incident, "commented", user, {"text": body.text})
    db.commit()
    await hub.broadcast(
        {"type": "incident.updated", "incident": serialize_incident(db, incident)}
    )
    return next(c for c in comments_payload(db, incident_id) if c["id"] == comment.id)


@router.get("/notifications/inbox", tags=["notifications"])
def inbox(
    db: Db,
    user: Reader,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 30,
):
    condition = (Notification.channel == "inapp", Notification.recipient == user.id)
    unread = db.scalar(
        select(func.count())
        .select_from(Notification)
        .where(*condition, Notification.read_at.is_(None))
    )
    total = db.scalar(select(func.count()).select_from(Notification).where(*condition))
    rows = db.execute(
        select(Notification, Incident, Camera.name)
        .join(Incident, Incident.id == Notification.incident_id)
        .join(Camera, Camera.id == Incident.camera_id)
        .where(*condition)
        .order_by(Notification.sent_at.desc(), Notification.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return {
        "items": [
            {
                "id": n.id,
                "incident_id": i.id,
                "type_code": i.type_code,
                "severity": i.severity,
                "text": f"{i.details.get('title', '') or db.get(IncidentType, i.type_code).name_ru} · {camera_name}",
                "camera_name": camera_name,
                "created_at": iso(n.sent_at),
                "read_at": iso(n.read_at),
            }
            for n, i, camera_name in rows
        ],
        "unread_count": unread,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post("/notifications/read-all", tags=["notifications"])
def read_all(db: Db, user: Reader):
    result = db.execute(
        update(Notification)
        .where(
            Notification.channel == "inapp",
            Notification.recipient == user.id,
            Notification.read_at.is_(None),
        )
        .values(read_at=utc_now())
    )
    db.commit()
    return {"updated": result.rowcount}


@router.get("/notifications/log", tags=["notifications"])
def notification_log(
    db: Db,
    user: Reader,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 30,
):
    query = select(Notification)
    if user.role not in ("admin", "engineer"):
        query = query.where(Notification.recipient == user.id)
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = db.scalars(
        query.order_by(Notification.sent_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "total": total,
        "items": [
            {
                "id": n.id,
                "incident_id": n.incident_id,
                "channel": n.channel,
                "recipient": n.recipient,
                "status": n.status,
                "error": n.error,
                "sent_at": iso(n.sent_at),
            }
            for n in rows
        ],
    }


@router.get("/analytics/summary", tags=["analytics"])
def summary(db: Db, user: Reader, filters: FilterDependency):
    query = filters.query().subquery()
    reaction = (
        (func.julianday(query.c.acknowledged_at) - func.julianday(query.c.created_at))
        * 86400
        if db.bind.dialect.name == "sqlite"
        else func.extract("epoch", query.c.acknowledged_at - query.c.created_at)
    )
    total, opened, critical, mean_reaction = db.execute(
        select(
            func.count(),
            func.sum(case((query.c.status.in_(["new", "acknowledged"]), 1), else_=0)),
            func.sum(case((query.c.severity == "critical", 1), else_=0)),
            func.avg(reaction),
        ).select_from(query)
    ).one()
    conditions = []
    if filters.camera_id:
        conditions.append(ComplianceStat.camera_id == filters.camera_id)
    if filters.area_id:
        conditions.append(
            ComplianceStat.camera_id.in_(
                select(Camera.id).where(Camera.area_id == filters.area_id)
            )
        )
    if filters.from_:
        conditions.append(
            ComplianceStat.period_start >= filters.from_.astimezone(timezone.utc)
        )
    if filters.to:
        conditions.append(
            ComplianceStat.period_end <= filters.to.astimezone(timezone.utc)
        )
    stats = compliance_rows(*conditions).subquery()
    persons, compliant = db.execute(
        select(func.sum(stats.c.persons_total), func.sum(stats.c.persons_compliant))
    ).one()
    camera_query = select(Camera)
    if filters.camera_id:
        camera_query = camera_query.where(Camera.id == filters.camera_id)
    if filters.area_id:
        camera_query = camera_query.where(Camera.area_id == filters.area_id)
    cameras = db.scalars(camera_query).all()
    return {
        "incidents_total": total,
        "open_incidents": opened or 0,
        "critical_incidents": critical or 0,
        "mean_reaction_sec": round(max(0, mean_reaction), 2)
        if mean_reaction is not None
        else None,
        "compliance_percent": round(compliant / persons * 100, 2) if persons else None,
        "online_cameras": sum(
            status_payload(db, c)["status"] == "online" for c in cameras
        ),
        "total_cameras": len(cameras),
    }
