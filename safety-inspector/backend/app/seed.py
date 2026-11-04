import logging
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy import select

from .database import SessionLocal
from .models import (
    ApiKey,
    Area,
    Camera,
    CameraStatus,
    Zone,
    IncidentType,
    NotificationSettings,
    PpeType,
    Setting,
    User,
    utc_now,
)
from .security import hash_api_key, new_api_key, password_hasher

logger = logging.getLogger("inspector")
AI_KEY_NAME = "ai-module"
AI_KEY_FILE = Path(__file__).resolve().parents[2] / ".secrets" / "ai_api_key"
DEFAULT_MJPEG_BASE_URL = "http://localhost:8001"
CAMERA_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,35}")
CENTRAL = [[0.15, 0.12], [0.85, 0.12], [0.85, 0.96], [0.15, 0.96]]
LEFT = [[0.12, 0.2], [0.6, 0.2], [0.6, 0.8], [0.12, 0.8]]
ENTRANCE = [[0.7, 0.1], [0.95, 0.1], [0.95, 0.7], [0.7, 0.7]]


def zone_rules(required, max_persons=None, distance=None):
    return {
        "required_ppe": required,
        "max_persons": max_persons,
        "min_vehicle_distance_m": distance,
        "entry_forbidden": False,
        "schedule": None,
        "confirm_frames": None,
        "cooldown_sec": None,
        "severity_override": None,
        "regulation_ref": None,
        "responsible_user_id": None,
    }


# Demo and integration cameras. The ids are shared with the AI module's configuration
# (contract 4.4); each camera gets its main zone and the entrance zone.
DEMO_CAMERAS = [
    (
        "cam-1",
        "Пресс №2",
        0,
        [
            ("danger", CENTRAL, zone_rules(["helmet", "vest"], 2)),
            ("ppe", ENTRANCE, zone_rules(["helmet"], 2)),
        ],
    ),
    (
        "cam-2",
        "Сварочный участок",
        1,
        [
            ("ppe", CENTRAL, zone_rules(["glasses", "gloves"])),
            ("ppe", ENTRANCE, zone_rules(["glasses", "gloves"])),
        ],
    ),
    (
        "cam-3",
        "Склад заготовок",
        2,
        [
            ("ppe", LEFT, zone_rules(["vest"])),
            ("ppe", ENTRANCE, zone_rules(["vest"])),
        ],
    ),
    (
        "cam-4",
        "Проезд погрузчиков",
        2,
        [
            ("vehicle", LEFT, zone_rules(["vest", "helmet"], distance=3)),
            ("ppe", ENTRANCE, zone_rules(["vest", "helmet"], distance=3)),
        ],
    ),
]


def ai_camera_ids():
    """AI_CAMERAS: comma-separated ids of cameras whose video comes from the AI module."""
    result = []
    for value in os.getenv("AI_CAMERAS", "").split(","):
        value = value.strip()
        if not value or value in result:
            continue
        if not CAMERA_ID.fullmatch(value):
            logger.warning("AI_CAMERAS: ignoring the invalid camera id %r", value)
            continue
        result.append(value)
    return result


def mjpeg_base_url():
    value = os.getenv("AI_MJPEG_BASE_URL", "").strip().rstrip("/")
    if not value:
        return DEFAULT_MJPEG_BASE_URL
    parts = urlsplit(value)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or any(char.isspace() for char in value)
    ):
        logger.warning(
            "AI_MJPEG_BASE_URL must be an http(s) URL; using %s", DEFAULT_MJPEG_BASE_URL
        )
        return DEFAULT_MJPEG_BASE_URL
    return value


def seed_cameras(db, areas):
    """Insert missing demo cameras (DEMO_MODE, once per database) and AI_CAMERAS cameras.

    Existing cameras and their zones are never modified, so edits made in the interface
    survive restarts.
    """
    ai_ids = ai_camera_ids()
    base_url = mjpeg_base_url()
    demo = (
        os.getenv("DEMO_MODE", "true").lower() == "true"
        and db.get(Setting, "stage_two_seeded") is None
    )
    specs = {spec[0]: (index, spec) for index, spec in enumerate(DEMO_CAMERAS)}
    wanted = [spec[0] for spec in DEMO_CAMERAS if demo or spec[0] in ai_ids]
    wanted += [camera_id for camera_id in ai_ids if camera_id not in specs]
    for camera_id in wanted:
        if db.get(Camera, camera_id) is not None:
            continue
        index, (_, name, area_index, zones) = specs.get(
            camera_id, (None, (camera_id, f"Камера {camera_id}", 0, []))
        )
        source = (
            {
                "stream_type": "mjpeg",
                "stream_url": f"{base_url}/stream/{camera_id}.mjpeg",
                "description": "Аннотированный MJPEG-поток модуля ИИ",
            }
            if camera_id in ai_ids
            else {
                "stream_type": "file",
                "stream_url": f"/demo/cam{index + 1}.mp4",
                "description": "Демонстрационный источник; видео добавляет команда",
            }
        )
        db.add(
            Camera(
                id=camera_id,
                name=name,
                area_id=areas[area_index].id,
                plan_x=0.2 + index * 0.2 if index is not None else None,
                plan_y=0.5 if index is not None else None,
                ai_enabled=True,
                **source,
            )
        )
        db.flush()
        db.add(CameraStatus(camera_id=camera_id, status="offline"))
        for order, (zone_type, polygon, rules) in enumerate(zones):
            db.add(
                Zone(
                    camera_id=camera_id,
                    name=name if order == 0 else "Вход на участок",
                    color="#65d9b2" if order == 0 else "#eab54d",
                    zone_type=zone_type,
                    polygon=polygon,
                    rules=rules,
                    sort_order=order,
                )
            )
    if demo:
        db.add(Setting(key="stage_two_seeded", value=True))


def write_secret(path, value):
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        return True
    except OSError:
        logger.warning("Could not write the AI API key to %s", path, exc_info=True)
        return False


def seed_ai_api_key(db):
    """Ensure the AI module's API key; return a newly generated plaintext key or None.

    With SEED_AI_API_KEY the only non-revoked "ai-module" key has exactly that value (other
    "ai-module" keys are revoked). Without it a random key is created once, when no
    non-revoked "ai-module" key exists, and written to AI_KEY_FILE. Keys are stored hashed.
    """
    active = db.scalars(
        select(ApiKey).where(ApiKey.name == AI_KEY_NAME, ApiKey.revoked_at.is_(None))
    ).all()
    raw = os.getenv("SEED_AI_API_KEY", "").strip()
    if raw:
        digest = hash_api_key(raw)
        stale = [row for row in active if row.key_hash != digest]
        if active and not stale:
            return None
        if len(stale) == len(active):
            db.add(ApiKey(name=AI_KEY_NAME, key_hash=digest))
        for row in stale:
            row.revoked_at = utc_now()
        logger.info("AI API key seeded from environment")
        return None
    if active:
        return None
    raw = new_api_key()
    db.add(ApiKey(name=AI_KEY_NAME, key_hash=hash_api_key(raw)))
    return raw


def seed():
    with SessionLocal.begin() as db:
        areas = []
        for name in ["Прессовый участок", "Сварочный участок", "Склад"]:
            area = db.scalar(select(Area).where(Area.name == name))
            if area is None:
                area = Area(name=name)
                db.add(area)
                db.flush()
            areas.append(area)
        for role, full_name in [
            ("admin", "Администратор"),
            ("engineer", "Инженер ОТ"),
            ("master", "Мастер участка"),
            ("viewer", "Наблюдатель"),
        ]:
            if db.scalar(select(User).where(User.username == role)) is None:
                db.add(
                    User(
                        username=role,
                        full_name=full_name,
                        role=role,
                        password_hash=password_hasher.hash(role),
                        area_id=areas[0].id if role == "master" else None,
                    )
                )
        ppe = [
            ("helmet", "Каска"),
            ("vest", "Жилет"),
            ("glasses", "Очки"),
            ("gloves", "Перчатки"),
            ("mask", "Респиратор"),
            ("boots", "Защитная обувь"),
            ("earmuffs", "Наушники"),
            ("harness", "Страховочная привязь"),
            ("face_shield", "Защитный щиток"),
        ]
        for index, (code, name) in enumerate(ppe):
            if db.get(PpeType, code) is None:
                db.add(
                    PpeType(code=code, name_ru=name, icon="shield", sort_order=index)
                )
        types = [
            ("ppe_violation", "Нарушение СИЗ", "medium"),
            ("zone_intrusion", "Вход в опасную зону", "high"),
            ("vehicle_proximity", "Опасное сближение с техникой", "high"),
            ("fall_detected", "Падение человека", "critical"),
            ("person_immobile", "Человек неподвижен", "critical"),
            ("overcrowding", "Превышение числа людей в зоне", "low"),
            ("custom", "Прочее", "medium"),
        ]
        colors = {
            "low": "#8194ac",
            "medium": "#eab54d",
            "high": "#ef873e",
            "critical": "#f35c64",
        }
        for code, name, severity in types:
            if db.get(IncidentType, code) is None:
                db.add(
                    IncidentType(
                        code=code,
                        name_ru=name,
                        default_severity=severity,
                        color=colors[severity],
                    )
                )
        defaults = {
            "enterprise_name": "Механический цех №3",
            "workshop_name": "Механический цех №3",
            "act_counter": 0,
            "timezone": "Asia/Yekaterinburg",
            "demo_mode": os.getenv("DEMO_MODE", "true").lower() == "true",
            "confirm_frames": 8,
            "cooldown_sec": 60,
            "min_confidence": 0.5,
            "heartbeat_timeout_sec": 15,
            "snapshot_retention_days": 30,
            "theme": "dark",
            "alarm_sound": False,
            "config_version": 1,
        }
        for key, value in defaults.items():
            if db.get(Setting, key) is None:
                db.add(Setting(key=key, value=value))
        if db.get(NotificationSettings, 1) is None:
            db.add(NotificationSettings(id=1, chat_ids=[], rules={}))
        if db.get(Setting, "config_updated_at") is None:
            db.add(
                Setting(
                    key="config_updated_at",
                    value=utc_now().isoformat().replace("+00:00", "Z"),
                )
            )
        seed_cameras(db, areas)
        generated = seed_ai_api_key(db)
        # Written before the commit: if the commit fails, the next start creates a new key.
        saved = write_secret(AI_KEY_FILE, generated) if generated else False
    if generated:
        place = f"saved to {AI_KEY_FILE}" if saved else "not saved to a file"
        print(f"AI module API key (shown once, {place}): {generated}", flush=True)


if __name__ == "__main__":
    seed()
