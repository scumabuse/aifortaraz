import os

from sqlalchemy import select

from .database import SessionLocal
from .models import (
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
from .security import password_hasher


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
        if (
            os.getenv("DEMO_MODE", "true").lower() == "true"
            and db.get(Setting, "stage_two_seeded") is None
        ):
            camera_specs = [
                ("Пресс №2", 0, "danger", ["helmet"], 2, None),
                ("Сварочный участок", 1, "ppe", ["glasses", "gloves"], None, None),
                ("Склад заготовок", 2, "ppe", ["vest"], None, None),
                ("Проезд погрузчиков", 2, "vehicle", ["vest", "helmet"], None, 3),
            ]
            for index, (
                name,
                area_index,
                zone_type,
                required,
                maximum,
                distance,
            ) in enumerate(camera_specs):
                camera = Camera(
                    name=name,
                    area_id=areas[area_index].id,
                    stream_type="file",
                    stream_url=f"/demo/cam{index + 1}.mp4",
                    plan_x=0.2 + index * 0.2,
                    plan_y=0.5,
                    description="Демонстрационный источник; видео добавляет команда",
                )
                db.add(camera)
                db.flush()
                db.add(CameraStatus(camera_id=camera.id, status="offline"))
                for order in range(2):
                    db.add(
                        Zone(
                            camera_id=camera.id,
                            name=name if order == 0 else "Вход на участок",
                            color="#65d9b2" if order == 0 else "#eab54d",
                            zone_type=zone_type if order == 0 else "ppe",
                            polygon=[[0.12, 0.2], [0.6, 0.2], [0.6, 0.8], [0.12, 0.8]]
                            if order == 0
                            else [[0.7, 0.1], [0.95, 0.1], [0.95, 0.7], [0.7, 0.7]],
                            rules={
                                "required_ppe": required,
                                "max_persons": maximum,
                                "min_vehicle_distance_m": distance,
                                "entry_forbidden": False,
                                "schedule": None,
                                "confirm_frames": None,
                                "cooldown_sec": None,
                                "severity_override": None,
                                "regulation_ref": None,
                                "responsible_user_id": None,
                            },
                            sort_order=order,
                        )
                    )
            db.add(Setting(key="stage_two_seeded", value=True))


if __name__ == "__main__":
    seed()
