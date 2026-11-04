"""Offline demo fixtures only; images and trajectories are synthetic, never analysed."""

import io
import math
import random
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont, ImageOps
from sqlalchemy import insert, select

from .database import SessionLocal
from .incident_schemas import contract_external_id
from .models import (
    Camera,
    Comment,
    ComplianceStat,
    FloorPlan,
    Incident,
    IncidentEvent,
    IncidentType,
    IngestReceipt,
    Notification,
    Report,
    Setting,
    User,
    Zone,
    utc_now,
)
from .stage_two import iso
from .storage import DATA_DIR, save_snapshot


def font(size):
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def point_in_polygon(x, y, polygon):
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            inside = not inside
    return inside


def synthetic_objects(camera_id, timestamp, zones):
    offset = sum(map(ord, camera_id)) % 17
    objects = []
    for index, track in enumerate((17, 24, 31)):
        x = 0.12 + (math.sin(timestamp * 0.14 + index * 1.7 + offset) + 1) * 0.32
        y = 0.36 + 0.07 * math.cos(timestamp * 0.12 + index)
        ppe = {
            "helmet": int(timestamp / 17 + index) % 5 != 0,
            "vest": int(timestamp / 23 + index) % 7 != 0,
            "glasses": int(timestamp / 19 + index) % 6 != 0,
            "gloves": int(timestamp / 29 + index) % 8 != 0,
            "mask": True,
            "boots": True,
            "earmuffs": True,
            "harness": True,
            "face_shield": True,
        }
        memberships = [
            z
            for z in zones
            if z["enabled"] and point_in_polygon(x + 0.042, y + 0.32, z["polygon"])
        ]
        danger = any(z["rules"].get("entry_forbidden") for z in memberships)
        violation = any(
            not ppe.get(code, False)
            for z in memberships
            for code in z["rules"].get("required_ppe", [])
        )
        objects.append(
            {
                "track_id": track,
                "label": "person",
                "bbox": [round(x, 5), round(y, 5), 0.085, 0.32],
                "conf": 0.94,
                "ppe": ppe,
                "zones": [z["id"] for z in memberships],
                "state": "danger" if danger else "violation" if violation else "ok",
            }
        )
    objects.append(
        {
            "track_id": 3,
            "label": "forklift",
            "bbox": [
                round(0.5 + 0.2 * math.sin(timestamp * 0.08 + offset), 5),
                0.67,
                0.18,
                0.17,
            ],
            "conf": 0.96,
            "ppe": {},
            "zones": [],
            "state": "ok",
        }
    )
    return objects


def snapshot(camera_name, when, boxes, title):
    image = None
    assets = Path(__file__).resolve().parents[2] / "scripts/demo_assets/snapshots"
    candidates = (
        sorted(
            p for p in assets.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if assets.exists()
        else []
    )
    if candidates:
        try:
            with Image.open(
                candidates[int(when.timestamp()) % len(candidates)]
            ) as supplied:
                image = ImageOps.fit(supplied.convert("RGB"), (1280, 720))
        except (OSError, ValueError):
            image = None
    if image is None:
        image = Image.new("RGB", (1280, 720), "#17232d")
        draw = ImageDraw.Draw(image)
        for x in range(0, 1280, 80):
            draw.line((x, 0, x, 720), fill="#263845")
        for y in range(0, 720, 80):
            draw.line((0, y, 1280, y), fill="#263845")
        for x, y, w, h in (
            (80, 110, 250, 180),
            (870, 95, 270, 200),
            (900, 440, 200, 170),
        ):
            draw.rounded_rectangle(
                (x, y, x + w, y + h),
                radius=12,
                fill="#2a3c48",
                outline="#466170",
                width=3,
            )
        for box in boxes:
            x, y, w, h = box["bbox"]
            left, top, width, height = (
                int(x * 1280),
                int(y * 720),
                int(w * 1280),
                int(h * 720),
            )
            draw.ellipse(
                (left + width * 0.25, top, left + width * 0.75, top + height * 0.26),
                fill="#93a5af",
            )
            draw.rounded_rectangle(
                (
                    left + width * 0.15,
                    top + height * 0.3,
                    left + width * 0.85,
                    top + height,
                ),
                radius=15,
                fill="#667f8e",
            )
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 76), fill="#111c25")
    draw.text((26, 14), camera_name, fill="#d7e9e4", font=font(23))
    draw.text((26, 45), f"{iso(when)}  |  {title}", fill="#8db4a5", font=font(15))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=82)
    return output.getvalue()


def demo_details(kind, track=17):
    """details of a synthetic event with exactly the keys of contract 4.1 for its type."""
    result = {
        "boxes": [
            {
                "track_id": track,
                "label": "person",
                "bbox": [0.41, 0.22, 0.12, 0.46],
                "conf": 0.94,
            }
        ],
        "frame_size": [1280, 720],
    }
    result.update(
        {
            "ppe_violation": {"missing_ppe": ["helmet"], "present_ppe": ["vest"]},
            "vehicle_proximity": {
                "distance_m": 1.8,
                "person_track_id": track,
                "vehicle_track_id": 3,
            },
            "zone_intrusion": {"duration_sec": 14},
            "fall_detected": {"pose_conf": 0.93},
            "person_immobile": {"immobile_sec": 65},
            "overcrowding": {"count": 4, "max_persons": 2},
            "custom": {"title": "Учебное событие"},
        }.get(kind, {})
    )
    return result


def set_state(db, value):
    row = db.get(Setting, "demo_seed_state")
    if row is None:
        db.add(Setting(key="demo_seed_state", value=value))
    else:
        row.value = value


def seed_demo_history(count=360, minutes=30 * 24 * 60):
    with SessionLocal() as db:
        marker = db.get(Setting, "demo_seed_state")
        state = (
            dict(marker.value)
            if marker
            else {
                "generation": str(uuid4()),
                "anchor": iso(utc_now().replace(second=0, microsecond=0)),
                "incidents_done": False,
                "stats_cameras": {},
                "done": False,
            }
        )
        if state["done"]:
            return {"seeded": False}
        set_state(db, state)
        db.commit()
        cameras = db.scalars(
            select(Camera)
            .where(Camera.enabled.is_(True), Camera.ai_enabled.is_(True))
            .order_by(Camera.id)
        ).all()
        users = db.scalars(select(User)).all()
        types = {t.code: t for t in db.scalars(select(IncidentType))}
        zones = {
            c.id: db.scalars(
                select(Zone).where(Zone.camera_id == c.id).order_by(Zone.sort_order)
            ).all()
            for c in cameras
        }
        timezone_name = db.get(Setting, "timezone").value
    if not cameras:
        return {"seeded": False, "reason": "no_cameras"}
    anchor = datetime.fromisoformat(state["anchor"].replace("Z", "+00:00"))
    rng = random.Random(state["generation"])
    actor = next(u for u in users if u.role == "engineer")
    if not state["incidents_done"]:
        kinds = rng.choices(
            [
                "ppe_violation",
                "zone_intrusion",
                "vehicle_proximity",
                "overcrowding",
                "fall_detected",
                "person_immobile",
            ],
            weights=[70, 15, 8, 4, 1.5, 1.5],
            k=count,
        )
        statuses = (
            ["resolved"] * int(count * 0.6)
            + ["acknowledged"] * int(count * 0.2)
            + ["new"] * int(count * 0.1)
        )
        statuses += ["false_positive"] * (count - len(statuses))
        rng.shuffle(statuses)
        for index, kind in enumerate(kinds):
            if (
                kind in ("fall_detected", "person_immobile")
                and statuses[index] == "new"
            ):
                swap = next(
                    (
                        i
                        for i, k in enumerate(kinds)
                        if k not in ("fall_detected", "person_immobile")
                        and statuses[i] == "resolved"
                    ),
                    None,
                )
                if swap is not None:
                    statuses[index], statuses[swap] = statuses[swap], statuses[index]
        incidents, receipts, events, comments, notices = [], [], [], [], []
        local = anchor.astimezone(ZoneInfo(timezone_name))
        days = [local.date() - timedelta(days=i) for i in range(30)]
        day_weights = [1.7 if d.weekday() in (0, 4) else 1 for d in days]
        for index, (kind, status) in enumerate(zip(kinds, statuses)):
            camera = cameras[index % len(cameras)]
            zone = (
                zones[camera.id][index % len(zones[camera.id])]
                if zones[camera.id]
                else None
            )
            day = (
                days[index]
                if index < len(days)
                else rng.choices(days, weights=day_weights)[0]
            )
            hour = rng.choices(
                range(24),
                weights=[
                    9
                    if h in (8, 20)
                    else 4
                    if h in (7, 9, 19, 21)
                    else 2
                    if 10 <= h < 19
                    else 1
                    for h in range(24)
                ],
            )[0]
            start = datetime(
                day.year,
                day.month,
                day.day,
                hour,
                rng.randrange(60),
                rng.randrange(60),
                tzinfo=ZoneInfo(timezone_name),
            ).astimezone(timezone.utc)
            start = min(start, anchor - timedelta(minutes=45))
            created = start + timedelta(seconds=rng.randint(0, 3))
            acknowledged = (
                created + timedelta(minutes=rng.randint(1, 40))
                if status in ("resolved", "acknowledged")
                else None
            )
            resolved = (
                min(
                    anchor,
                    (acknowledged or created) + timedelta(minutes=rng.randint(5, 90)),
                )
                if status in ("resolved", "false_positive")
                else None
            )
            external = contract_external_id(camera.id, start, index + 100, kind)
            incident_id = str(
                uuid5(NAMESPACE_URL, f"demo-history-{state['generation']}-{index}")
            )
            details = demo_details(kind, index + 100)
            snapshot_path, thumbnail_path = save_snapshot(
                incident_id,
                snapshot(camera.name, start, details["boxes"], types[kind].name_ru),
            )
            severity = (
                "high"
                if kind == "ppe_violation" and zone and zone.zone_type == "danger"
                else types[kind].default_severity
            )
            row = dict(
                id=incident_id,
                external_id=external,
                source="mock",
                camera_id=camera.id,
                zone_id=zone.id if zone else None,
                type_code=kind,
                severity=severity,
                status=status,
                confidence=round(rng.uniform(0.86, 0.98), 2),
                track_id=index + 100,
                started_at=start,
                ended_at=start + timedelta(seconds=18),
                duration_sec=18,
                snapshot_path=snapshot_path,
                thumbnail_path=thumbnail_path,
                details=details,
                assignee_id=actor.id,
                created_at=created,
                acknowledged_at=acknowledged,
                acknowledged_by=actor.id if acknowledged else None,
                resolved_at=resolved,
                resolved_by=actor.id if resolved else None,
                resolution_note="Проведён инструктаж, требования безопасности восстановлены"
                if status == "resolved"
                else None,
                false_positive_reason="Проверка снимка: учебное ложное срабатывание"
                if status == "false_positive"
                else None,
            )
            incidents.append(row)
            receipts.append(
                dict(external_id=external, incident_id=incident_id, created_at=created)
            )

            def event(event_type, at, payload=None, user_id=None):
                events.append(
                    dict(
                        id=str(uuid4()),
                        incident_id=incident_id,
                        event_type=event_type,
                        created_at=at,
                        user_id=user_id,
                        payload=payload or {},
                    )
                )

            event("created", created, {"source": "mock"})
            event(
                "notified", created + timedelta(milliseconds=100), {"channel": "inapp"}
            )
            if acknowledged:
                event("acknowledged", acknowledged, user_id=actor.id)
            if resolved:
                event(
                    "resolved" if status == "resolved" else "false_positive",
                    resolved,
                    {"note": row["resolution_note"]}
                    if status == "resolved"
                    else {"reason": row["false_positive_reason"]},
                    actor.id,
                )
            for number in range(1 + index % 2):
                comment_at = (acknowledged or created) + timedelta(
                    seconds=20 + number * 20
                )
                text = (
                    "Учебные данные: информация передана ответственному"
                    if number == 0
                    else "Выполнена проверка участка по снимку"
                )
                comments.append(
                    dict(
                        id=str(uuid4()),
                        incident_id=incident_id,
                        user_id=actor.id,
                        text=text,
                        created_at=comment_at,
                    )
                )
                event("commented", comment_at, {"text": text}, actor.id)
            for user in users:
                if user.role == "master" and user.area_id != camera.area_id:
                    continue
                sent = created + timedelta(milliseconds=100)
                notices.append(
                    dict(
                        id=str(uuid4()),
                        incident_id=incident_id,
                        channel="inapp",
                        recipient=user.id,
                        status="sent",
                        sent_at=sent,
                        read_at=None
                        if status == "new" and anchor - created < timedelta(days=1)
                        else sent + timedelta(minutes=2),
                    )
                )
        with SessionLocal.begin() as db:
            for model, rows in (
                (Incident, incidents),
                (IngestReceipt, receipts),
                (IncidentEvent, events),
                (Comment, comments),
                (Notification, notices),
            ):
                db.execute(insert(model), rows)
            state = {**state, "incidents_done": True}
            set_state(db, state)
    for camera in cameras:
        offset = state["stats_cameras"].get(camera.id, 0)
        while offset < minutes:
            end = min(minutes, offset + 2000)
            rows = []
            for index in range(offset, end):
                start = anchor - timedelta(minutes=minutes - index)
                peak = start.astimezone(ZoneInfo(timezone_name)).hour in (8, 20)
                total = 28 + (index + sum(map(ord, camera.id))) % 29
                rate = (0.89 if peak else 0.93) + 0.035 * (
                    1 + math.sin(index * 0.07)
                ) / 2
                compliant = round(total * rate)
                rows.append(
                    dict(
                        id=str(uuid4()),
                        camera_id=camera.id,
                        period_start=start,
                        period_end=start + timedelta(minutes=1),
                        persons_total=total,
                        persons_compliant=compliant,
                        source="mock",
                        by_ppe={
                            "helmet": {
                                "checked": total,
                                "compliant": min(total, compliant + 1),
                            },
                            "vest": {"checked": total, "compliant": compliant},
                        },
                    )
                )
            with SessionLocal.begin() as db:
                db.execute(insert(ComplianceStat), rows)
                state = {
                    **state,
                    "stats_cameras": {**state["stats_cameras"], camera.id: end},
                }
                set_state(db, state)
            offset = end
    with SessionLocal.begin() as db:
        plans_dir = DATA_DIR / "plans"
        plans_dir.mkdir(exist_ok=True)
        for area_id in {c.area_id for c in cameras}:
            if (
                db.scalar(select(FloorPlan).where(FloorPlan.area_id == area_id))
                is not None
            ):
                continue
            image = Image.new("RGB", (1200, 700), "#17232d")
            draw = ImageDraw.Draw(image)
            draw.rectangle((80, 100, 1120, 610), outline="#5c8a7a", width=5)
            draw.line((600, 100, 600, 610), fill="#405b68", width=4)
            draw.text(
                (90, 30), "Демонстрационный план участка", font=font(27), fill="#c4ded3"
            )
            for camera in cameras:
                if camera.area_id != area_id:
                    continue
                x, y = (
                    int((camera.plan_x or 0.5) * 1000) + 100,
                    int((camera.plan_y or 0.5) * 450) + 130,
                )
                draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="#65d9b2")
                draw.text((x + 20, y - 12), camera.name, font=font(18), fill="#d7e6df")
            name = f"demo-area-{area_id}.png"
            image.save(plans_dir / name)
            db.add(
                FloorPlan(
                    area_id=area_id,
                    image_path=f"/static/plans/{name}",
                    width=1200,
                    height=700,
                )
            )
        report_dir = DATA_DIR / "reports"
        report_dir.mkdir(exist_ok=True)
        for kind, days_count in (("day", 1), ("week", 7)):
            title = (
                "Учебная сводка за день"
                if kind == "day"
                else "Учебная сводка за неделю"
            )
            name = f"demo-{kind}.html"
            content = (
                f'<!doctype html><html lang="ru"><meta charset="utf-8"><title>{title}</title><style>body{{font:16px Arial;padding:40px;color:#17232d}}table{{border-collapse:collapse}}td,th{{border:1px solid #aaa;padding:12px}}@media print{{body{{padding:0}}}}</style><h1>{title}</h1><p>Механический цех №3 · ДЕМО-ДАННЫЕ</p><p>{iso(anchor - timedelta(days=days_count))} — {iso(anchor)}</p><table><tr><th>Камера</th><th>Назначение</th></tr>'
                + "".join(
                    f"<tr><td>{escape(c.name)}</td><td>Контроль безопасности</td></tr>"
                    for c in cameras
                )
                + "</table><p>Подготовленный учебный отчёт. Полный генератор отчётов будет добавлен отдельно.</p></html>"
            )
            (report_dir / name).write_text(content, encoding="utf-8")
            db.add(
                Report(
                    report_type=kind,
                    period_from=anchor - timedelta(days=days_count),
                    period_to=anchor,
                    params={"source": "mock", "title": title},
                    file_path=f"/static/reports/{name}",
                    created_by=actor.id,
                )
            )
        set_state(db, {**state, "done": True})
    return {"seeded": True, "incidents": count, "stats": minutes * len(cameras)}
