from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import case, func, or_, select

from .dependencies import Db
from .models import Camera, ComplianceStat, Incident, IncidentType, Zone, utc_now
from .stage_two import Reader, iso, setting, status_payload

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
Period = Literal["today", "shift", "7d", "30d", "custom"]


def period_bounds(db, period, date_from=None, date_to=None):
    zone = ZoneInfo(setting(db, "timezone", "UTC"))
    now = utc_now().astimezone(zone)
    midnight = datetime.combine(now.date(), time(), zone)
    if period == "custom":
        if not date_from or not date_to or not 0 <= (date_to - date_from).days < 366:
            raise HTTPException(422, "Укажите период до 366 дней")
        start = datetime.combine(date_from, time(), zone)
        end = datetime.combine(date_to + timedelta(days=1), time(), zone)
        span = end - start
    else:
        end = now
        if period == "shift":
            start = midnight.replace(hour=20 if now.hour >= 20 else 8)
            if now.hour < 8:
                start = midnight.replace(hour=20) - timedelta(days=1)
            span = timedelta(hours=12)
        else:
            days = {"today": 1, "7d": 7, "30d": 30}[period]
            start = midnight - timedelta(days=days - 1)
            span = timedelta(days=days)
    return (
        start.astimezone(timezone.utc),
        end.astimezone(timezone.utc),
        (start - span).astimezone(timezone.utc),
        (end - span).astimezone(timezone.utc),
        zone,
    )


def sql_local(db, column, zone):
    if db.bind.dialect.name == "sqlite":
        return func.local_datetime(column, zone.key)
    return func.timezone(zone.key, column)


def date_key(db, column, zone, part):
    local = sql_local(db, column, zone)
    formats = {
        "day": ("%Y-%m-%d", "YYYY-MM-DD"),
        "hour": ("%H", "HH24"),
        "weekday": ("%w", "D"),
    }
    sqlite, postgres = formats[part]
    return (
        func.strftime(sqlite, local)
        if db.bind.dialect.name == "sqlite"
        else func.to_char(local, postgres)
    )


def incident_rows(start, end, camera_ids):
    return (
        select(Incident)
        .where(
            Incident.camera_id.in_(camera_ids),
            Incident.started_at >= start,
            Incident.started_at < end,
        )
        .subquery()
    )


def reaction_seconds(db, rows):
    if db.bind.dialect.name == "sqlite":
        seconds = (
            func.julianday(rows.c.acknowledged_at) - func.julianday(rows.c.created_at)
        ) * 86400
    else:
        seconds = func.extract("epoch", rows.c.acknowledged_at - rows.c.created_at)
    return case((seconds >= 0, seconds))


def percent(numerator, denominator):
    return round(100 * (numerator or 0) / denominator, 2) if denominator else None


def rounded(value):
    return round(float(value), 2) if value is not None else None


def kpis(db, rows, reaction, stats):
    total, opened, critical, false, mean = db.execute(
        select(
            func.count(),
            func.count().filter(rows.c.status.in_(["new", "acknowledged"])),
            func.count().filter(rows.c.severity == "critical"),
            func.count().filter(rows.c.status == "false_positive"),
            func.avg(reaction),
        ).select_from(rows)
    ).one()
    persons, compliant = db.execute(
        select(func.sum(stats.c.persons_total), func.sum(stats.c.persons_compliant))
    ).one()
    return dict(
        incidents_total=total,
        open_incidents=opened,
        critical_incidents=critical,
        false_positive_percent=percent(false, total),
        mean_reaction_sec=rounded(mean),
        compliance_percent=percent(compliant, persons),
    )


def compliance_rows(*conditions):
    """Statistics behind the compliance KPI (contract 4.3).

    A camera with real statistics (source ai) under ``conditions`` is measured by those only;
    mock statistics count only for cameras without real ones, so one camera never blends both.
    """
    real = select(ComplianceStat.camera_id).where(
        *conditions, ComplianceStat.source == "ai"
    )
    return select(ComplianceStat).where(
        *conditions,
        or_(ComplianceStat.source == "ai", ComplianceStat.camera_id.not_in(real)),
    )


def stats_rows(start, end, camera_ids):
    return compliance_rows(
        ComplianceStat.camera_id.in_(camera_ids),
        ComplianceStat.period_start >= start,
        ComplianceStat.period_end <= end,
    ).subquery()


def counts(db, rows, *columns):
    return db.execute(
        select(*columns, func.count()).select_from(rows).group_by(*columns)
    ).all()


@router.get("/overview")
def overview(
    db: Db,
    user: Reader,
    period: Period = "7d",
    date_from: date | None = None,
    date_to: date | None = None,
    area_id: str | None = None,
    camera_id: str | None = None,
    dashboard: bool = Query(False),
    camera_ids: list[str] | None = Query(None, max_length=100),
):
    start, end, previous_start, previous_end, zone = period_bounds(
        db, period, date_from, date_to
    )
    query = select(Camera).order_by(Camera.name)
    if area_id:
        query = query.where(Camera.area_id == area_id)
    if camera_id:
        query = query.where(Camera.id == camera_id)
    if camera_ids:
        query = query.where(Camera.id.in_(camera_ids))
    cameras = db.scalars(query).all()
    camera_ids = [camera.id for camera in cameras]
    rows = incident_rows(start, end, camera_ids)
    previous = incident_rows(previous_start, previous_end, camera_ids)
    reaction = reaction_seconds(db, rows)
    stats = stats_rows(start, end, camera_ids)
    current = kpis(db, rows, reaction, stats)
    comparison = kpis(
        db,
        previous,
        reaction_seconds(db, previous),
        stats_rows(previous_start, previous_end, camera_ids),
    )
    catalog = {kind.code: kind.name_ru for kind in db.scalars(select(IncidentType))}
    by_type = sorted(
        counts(db, rows, rows.c.type_code), key=lambda row: (-row[1], row[0])
    )
    result = dict(
        start=iso(start),
        end=iso(end),
        previous_start=iso(previous_start),
        previous_end=iso(previous_end),
        timezone=zone.key,
        current=current,
        previous=comparison,
        types=catalog,
        by_type=[dict(name=catalog.get(code, code), count=n) for code, n in by_type],
    )
    if dashboard:
        today_start, today_end, *_ = period_bounds(db, "today")
        today = incident_rows(today_start, today_end, camera_ids)
        hours = dict(counts(db, today, date_key(db, today.c.started_at, zone, "hour")))
        latest_thumbnails = (
            select(
                Incident.camera_id,
                Incident.thumbnail_path,
                func.row_number()
                .over(
                    partition_by=Incident.camera_id,
                    order_by=(Incident.started_at.desc(), Incident.id),
                )
                .label("rank"),
            )
            .where(
                Incident.camera_id.in_(camera_ids), Incident.thumbnail_path.is_not(None)
            )
            .subquery()
        )
        thumbnails = dict(
            db.execute(
                select(
                    latest_thumbnails.c.camera_id, latest_thumbnails.c.thumbnail_path
                ).where(latest_thumbnails.c.rank == 1)
            ).all()
        )
        status = [
            dict(
                id=c.id,
                name=c.name,
                preview_url=c.preview_url or thumbnails.get(c.id),
                **status_payload(db, c),
            )
            for c in cameras
        ]
        return dict(
            **result,
            cameras=status,
            hourly=[
                dict(name=f"{h:02}:00", count=hours.get(f"{h:02}", 0))
                for h in range(24)
            ],
        )

    days = {}
    local_day = start.astimezone(zone).date()
    while local_day <= end.astimezone(zone).date():
        if datetime.combine(local_day, time(), zone) >= end:
            break
        days[local_day.isoformat()] = dict.fromkeys(catalog, 0)
        local_day += timedelta(days=1)
    day = date_key(db, rows.c.started_at, zone, "day")
    for key, code, count in counts(db, rows, day, rows.c.type_code):
        days[key][code] = count
    stats_day = date_key(db, stats.c.period_start, zone, "day")
    daily_ppe = {
        key: percent(compliant, persons)
        for key, persons, compliant in db.execute(
            select(
                stats_day,
                func.sum(stats.c.persons_total),
                func.sum(stats.c.persons_compliant),
            ).group_by(stats_day)
        )
    }
    current_camera = {
        camera: (total, critical, rounded(mean))
        for camera, total, critical, mean in db.execute(
            select(
                rows.c.camera_id,
                func.count(),
                func.count().filter(rows.c.severity == "critical"),
                func.avg(reaction),
            ).group_by(rows.c.camera_id)
        )
    }
    previous_camera = dict(counts(db, previous, previous.c.camera_id))
    hour = date_key(db, rows.c.started_at, zone, "hour")
    weekday = date_key(db, rows.c.started_at, zone, "weekday")
    histogram = case(
        (reaction < 60, "0–59 с"),
        (reaction < 300, "1–5 мин"),
        (reaction < 900, "5–15 мин"),
        (reaction < 3600, "15–60 мин"),
        else_="≥ 60 мин",
    )
    buckets = dict(
        db.execute(
            select(histogram, func.count())
            .select_from(rows)
            .where(reaction.is_not(None))
            .group_by(histogram)
        ).all()
    )
    zone_counts = db.execute(
        select(Zone.name, Camera.name, func.count())
        .select_from(rows)
        .join(Camera, Camera.id == rows.c.camera_id)
        .outerjoin(Zone, Zone.id == rows.c.zone_id)
        .group_by(rows.c.camera_id, rows.c.zone_id, Zone.name, Camera.name)
        .order_by(func.count().desc(), Camera.name, Zone.name)
    )
    return dict(
        **result,
        daily=[
            dict(date=key, counts=value, compliance=daily_ppe.get(key))
            for key, value in days.items()
        ],
        heat=[
            dict(
                day=(int(d) - (1 if db.bind.dialect.name == "sqlite" else 2)) % 7,
                hour=int(h),
                count=n,
            )
            for d, h, n in counts(db, rows, weekday, hour)
        ],
        zones=[
            dict(name=f"{camera} · {name or 'Без зоны'}", count=n)
            for name, camera, n in zone_counts
        ],
        reaction=[
            dict(name=name, count=buckets.get(name, 0))
            for name in ("0–59 с", "1–5 мин", "5–15 мин", "15–60 мин", "≥ 60 мин")
        ],
        cameras=[
            dict(
                id=c.id,
                name=c.name,
                count=total,
                critical=critical,
                reaction=mean,
                previous=previous_camera.get(c.id, 0),
            )
            for c in cameras
            for total, critical, mean in [current_camera.get(c.id, (0, 0, None))]
        ],
    )
