import base64
import csv
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import Field
from sqlalchemy import func, select, update

from .analytics import Period, counts, date_key, incident_rows, overview
from .dependencies import Db
from .incidents import context_for, get_incident, serialize_incident
from .models import Incident, PpeType, Report, Setting, User, Zone, new_id
from .schemas import StrictModel
from .settings_admin import local_date
from .stage_two import Editor, Reader, iso, setting
from .storage import DATA_DIR, SNAPSHOT_DIR
from .telegram import SEVERITY

router = APIRouter(prefix="/api/v1", tags=["reports"])
templates = Environment(
    loader=FileSystemLoader(Path(__file__).with_name("templates")),
    autoescape=select_autoescape(("html",)),
)
REPORT_DIR = DATA_DIR / "reports"
CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
STATUS = {
    "new": "Новый",
    "acknowledged": "Принят",
    "resolved": "Закрыт",
    "false_positive": "Ложное срабатывание",
}
TITLES = {
    "summary": "Сводка за период",
    "prescription": "Предписание",
    "act": "Акт о нарушении требований охраны труда",
}
ROW_LIMIT = 500


def local_text(db, stamp):
    return local_date(db, stamp).strftime("%d.%m.%Y %H:%M")


def image_data(path):
    if not path or not re.fullmatch(
        r"/static/snapshots/[0-9a-f-]+(?:-thumb)?\.jpg", path
    ):
        return None
    try:
        raw = (SNAPSHOT_DIR / path.rsplit("/", 1)[1]).read_bytes()
    except OSError:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def info(row, author):
    return dict(
        id=row.id,
        type=row.report_type,
        title=row.params.get("title", row.report_type),
        from_=iso(row.period_from),
        to=iso(row.period_to),
        author=author,
        created_at=iso(row.created_at),
        has_csv=bool(row.params.get("csv")),
        incident_id=row.params.get("incident_id"),
        number=row.params.get("number"),
    )


def stored_report(db, report_id):
    row = db.get(Report, report_id)
    if row is None:
        raise HTTPException(404, "Документ не найден")
    return row


@router.get("/reports")
def list_reports(db: Db, user: Reader, page: int = Query(1, ge=1, le=100000)):
    items = db.execute(
        select(Report, User.full_name)
        .join(User, User.id == Report.created_by)
        .order_by(Report.created_at.desc(), Report.id)
        .offset((page - 1) * 20)
        .limit(20)
    )
    return {
        "items": [info(row, author) for row, author in items],
        "total": db.scalar(select(func.count()).select_from(Report)),
    }


@router.get("/reports/{report_id}/document")
def document(report_id: str, db: Db, user: Reader, download: bool = False):
    row = stored_report(db, report_id)
    name = row.file_path.rsplit("/", 1)[-1] if row.file_path else ""
    if not re.fullmatch(r"(?:[0-9a-f-]+|demo-day|demo-week)\.html", name):
        raise HTTPException(404, "Файл документа не найден")
    path = REPORT_DIR / name
    if not path.is_file():
        raise HTTPException(404, "Файл документа не найден")
    return FileResponse(
        path,
        media_type="text/html; charset=utf-8",
        headers={"Content-Security-Policy": CSP, "Cache-Control": "no-store"},
        filename=name if download else None,
    )


@router.get("/reports/{report_id}/data.csv")
def report_csv(report_id: str, db: Db, user: Reader):
    row = stored_report(db, report_id)
    if not row.params.get("csv"):
        raise HTTPException(404, "У этого документа нет CSV")
    path = REPORT_DIR / f"{row.id}.csv"
    if not path.is_file():
        raise HTTPException(404, "Файл CSV не найден")
    return FileResponse(
        path, media_type="text/csv; charset=utf-8", filename=f"{row.id}.csv"
    )


def save_document(db, user, kind, start, end, context, params, items=None):
    report_id = new_id()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORT_DIR / f"{report_id}.html"
    csv_path = REPORT_DIR / f"{report_id}.csv"
    params = {"title": TITLES[kind], **params, "csv": items is not None}
    content = templates.get_template(
        "act.html" if kind == "act" else "summary.html"
    ).render(**context)
    row = Report(
        id=report_id,
        report_type=kind,
        period_from=start,
        period_to=end,
        params=params,
        file_path=html_path.name,
        created_by=user.id,
    )
    try:
        html_path.write_text(content, encoding="utf-8")
        if items is not None:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(
                    [
                        "ID",
                        "Дата и время",
                        "Тип",
                        "Камера",
                        "Зона",
                        "Критичность",
                        "Статус",
                        "Ответственный",
                    ]
                )
                for item in items:
                    fields = (
                        "id",
                        "time",
                        "type_name",
                        "camera_name",
                        "zone_name",
                        "severity_label",
                        "status_label",
                        "assignee_name",
                    )
                    writer.writerow([csv_cell(item.get(key)) for key in fields])
        db.add(row)
        db.commit()
    except Exception:
        html_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise
    return info(row, user.full_name)


def csv_cell(value):
    value = str(value) if value is not None else ""
    return (
        "'" + value
        if value.startswith(("\t", "\r", "\n"))
        or value.lstrip().startswith(("=", "+", "-", "@"))
        else value
    )


class ReportIn(StrictModel):
    kind: Literal["summary", "prescription"] = "summary"
    period: Period
    date_from: date | None = None
    date_to: date | None = None
    area_id: str | None = None
    camera_ids: list[str] = Field(default_factory=list, max_length=100)
    measures: str = Field(default="", max_length=3000)
    deadline: date | None = None


@router.post("/reports", status_code=201)
def create_report(body: ReportIn, db: Db, user: Editor):
    from .models import Area

    if body.kind == "prescription" and (
        not body.area_id or not body.measures.strip() or not body.deadline
    ):
        raise HTTPException(422, "Для предписания укажите участок, меры и срок")
    area = db.get(Area, body.area_id) if body.area_id else None
    if body.area_id and area is None:
        raise HTTPException(404, "Участок не найден")
    data = overview(
        db,
        user,
        period=body.period,
        date_from=body.date_from,
        date_to=body.date_to,
        area_id=body.area_id,
        dashboard=False,
        camera_ids=body.camera_ids,
    )
    ids = [camera["id"] for camera in data["cameras"]]
    if set(body.camera_ids) - set(ids):
        raise HTTPException(
            422, "Выбранная камера не найдена или находится на другом участке"
        )
    start, end = (
        datetime.fromisoformat(data["start"]),
        datetime.fromisoformat(data["end"]),
    )
    rows = incident_rows(start, end, ids)
    records = list(
        db.scalars(
            select(Incident)
            .join(rows, rows.c.id == Incident.id)
            .order_by(Incident.started_at.desc(), Incident.id)
            .limit(ROW_LIMIT + 1)
        )
    )
    if len(records) > ROW_LIMIT:
        raise HTTPException(
            422, "В отчёте больше 500 инцидентов. Сократите период или выберите камеры"
        )
    cache = context_for(db, records)
    items = []
    for record in records:
        item = serialize_incident(db, record, cache)
        item.update(
            time=local_text(db, record.started_at),
            severity_label=SEVERITY[record.severity],
            status_label=STATUS[record.status],
            image=image_data(record.thumbnail_path),
        )
        items.append(item)
    max_type = max((item["count"] for item in data["by_type"]), default=1)
    daily = [(d["date"], sum(d["counts"].values())) for d in data["daily"]]
    daily_max = max((count for _, count in daily), default=0) or 1
    top_camera = max(data["cameras"], key=lambda item: item["count"], default=None)
    hour = date_key(db, rows.c.started_at, ZoneInfo(data["timezone"]), "hour")
    peak_hour = max(
        counts(db, rows, hour), key=lambda item: (item[1], -int(item[0])), default=None
    )
    context = dict(
        title=TITLES[body.kind],
        kind=body.kind,
        start=local_text(db, start),
        end=local_text(db, end),
        enterprise=setting(db, "enterprise_name"),
        workshop=setting(db, "workshop_name"),
        area=area.name if area else "Все участки",
        author=user.full_name,
        items=items,
        data=data,
        daily=daily,
        daily_max=daily_max,
        max_type=max_type,
        top_camera=top_camera,
        peak=max(daily, key=lambda item: item[1], default=None),
        peak_hour=peak_hour,
        measures=body.measures,
        deadline=body.deadline,
    )
    return save_document(db, user, body.kind, start, end, context, {}, items)


class ActIn(StrictModel):
    enterprise: str = Field(max_length=200)
    workshop: str = Field(max_length=200)
    area: str = Field(max_length=200)
    time: str = Field(max_length=100)
    camera: str = Field(max_length=200)
    zone: str = Field(max_length=200)
    violation: str = Field(max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    regulation: str = Field(max_length=500)
    responsible: str = Field(max_length=200)
    measures: str = Field(max_length=3000)
    deadline: str = Field(max_length=100)


def draft_fields(db, incident):
    item = serialize_incident(db, incident)
    zone = db.get(Zone, incident.zone_id) if incident.zone_id else None
    names = {p.code: p.name_ru for p in db.scalars(select(PpeType))}
    missing = ", ".join(
        names.get(code, code) for code in item["details"].get("missing_ppe", [])
    )
    description = f"{item['type_name']}. Камера «{item['camera_name']}»"
    if item["zone_name"]:
        description += f", зона «{item['zone_name']}»"
    if incident.track_id is not None:
        description += f", трек #{incident.track_id}"
    if missing:
        description += f". Отсутствующие СИЗ: {missing}"
    if incident.duration_sec is not None:
        description += f". Длительность {incident.duration_sec:g} с"
    if incident.confidence is not None:
        description += f". Уверенность системы {incident.confidence:.0%}"
    return ActIn(
        enterprise=setting(db, "enterprise_name", ""),
        workshop=setting(db, "workshop_name", ""),
        area=item["area_name"],
        time=local_text(db, incident.started_at),
        camera=item["camera_name"],
        zone=item["zone_name"] or "—",
        violation=item["type_name"],
        description=description[:3999] + ".",
        regulation=((zone.rules or {}).get("regulation_ref") or "") if zone else "",
        responsible=item["assignee_name"] or "",
        measures=incident.resolution_note or "",
        deadline="",
    )


@router.post("/incidents/{incident_id}/act/draft", response_model=ActIn)
def act_draft(incident_id: str, db: Db, user: Editor):
    return draft_fields(db, get_incident(db, incident_id))


def act_context(body, incident, number, author):
    boxes = (incident.details or {}).get("boxes", [])
    box = next(
        (obj.get("bbox") for obj in boxes if obj.get("track_id") == incident.track_id),
        None,
    )
    return dict(
        title=TITLES["act"],
        act=body,
        number=number,
        author=author,
        image=image_data(incident.snapshot_path),
        box=box,
    )


@router.post("/incidents/{incident_id}/act/preview", response_class=HTMLResponse)
def preview_act(incident_id: str, body: ActIn, db: Db, user: Editor):
    incident = get_incident(db, incident_id)
    html = templates.get_template("act.html").render(
        **act_context(body, incident, "Черновик", user.full_name)
    )
    return HTMLResponse(
        html, headers={"Content-Security-Policy": CSP, "Cache-Control": "no-store"}
    )


@router.post("/incidents/{incident_id}/act", status_code=201)
def create_act(incident_id: str, body: ActIn, db: Db, user: Editor):
    incident = get_incident(db, incident_id)
    db.execute(
        update(Setting).where(Setting.key == "act_counter").values(value=Setting.value)
    )
    counter = db.get(Setting, "act_counter", populate_existing=True)
    counter.value += 1
    number = f"АКТ-{local_date(db, incident.started_at).year}-{counter.value:06d}"
    context = act_context(body, incident, number, user.full_name)
    return save_document(
        db,
        user,
        "act",
        incident.started_at,
        incident.ended_at or incident.started_at,
        context,
        {"incident_id": incident.id, "number": number},
    )
