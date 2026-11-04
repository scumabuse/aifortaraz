"""Seed of the integration: fixed camera ids, AI cameras and the AI module's API key.

Rows 4, 57, 88 and 104 of the conformance matrix (INTEGRATION_INVENTORY.md). Every test
seeds its own empty database; the last test checks the seeded application of the suite.
"""

import hashlib
import logging
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker

CENTRAL = [[0.15, 0.12], [0.85, 0.12], [0.85, 0.96], [0.15, 0.96]]
LEFT = [[0.12, 0.2], [0.6, 0.2], [0.6, 0.8], [0.12, 0.8]]
ENTRANCE = [[0.7, 0.1], [0.95, 0.1], [0.95, 0.7], [0.7, 0.7]]
RULE_KEYS = {
    "required_ppe",
    "entry_forbidden",
    "max_persons",
    "min_vehicle_distance_m",
    "schedule",
    "confirm_frames",
    "cooldown_sec",
    "severity_override",
}


@pytest.fixture
def fresh(client, tmp_path, monkeypatch):
    """An empty database for app.seed.seed() and a key file inside tmp_path."""
    from app import seed
    from app.models import Base

    engine = create_engine(f"sqlite:///{(tmp_path / 'seed.db').as_posix()}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(seed, "SessionLocal", sessions)
    monkeypatch.setattr(seed, "AI_KEY_FILE", tmp_path / ".secrets" / "ai_api_key")
    for name in ("DEMO_MODE", "AI_CAMERAS", "AI_MJPEG_BASE_URL", "SEED_AI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield sessions
    engine.dispose()


def cameras_of(sessions):
    from app.models import Area, Camera, Zone

    with sessions() as db:
        result = {}
        for camera in db.scalars(select(Camera).order_by(Camera.id)):
            zones = db.scalars(
                select(Zone)
                .where(Zone.camera_id == camera.id)
                .order_by(Zone.sort_order)
            ).all()
            result[camera.id] = {
                "name": camera.name,
                "area": db.get(Area, camera.area_id).name,
                "stream_type": camera.stream_type,
                "stream_url": camera.stream_url,
                "ai_enabled": camera.ai_enabled,
                "plan": (camera.plan_x, camera.plan_y),
                "zones": [
                    (
                        zone.name,
                        zone.zone_type,
                        zone.polygon,
                        zone.rules["required_ppe"],
                        zone.rules["max_persons"],
                        zone.rules["min_vehicle_distance_m"],
                    )
                    for zone in zones
                ],
            }
        return result


def test_demo_mode_seeds_the_four_cameras_with_fixed_ids(fresh, monkeypatch):
    """Rows 57, 104: cam-1..cam-4 keep the demo names, areas, positions and zones."""
    from app.seed import seed

    monkeypatch.setenv("DEMO_MODE", "true")
    seed()
    cameras = cameras_of(fresh)
    assert list(cameras) == ["cam-1", "cam-2", "cam-3", "cam-4"]
    expected = {
        "cam-1": ("Пресс №2", "Прессовый участок", 0.2),
        "cam-2": ("Сварочный участок", "Сварочный участок", 0.4),
        "cam-3": ("Склад заготовок", "Склад", 0.6),
        "cam-4": ("Проезд погрузчиков", "Склад", 0.8),
    }
    for index, (camera_id, (name, area, plan_x)) in enumerate(expected.items(), 1):
        camera = cameras[camera_id]
        assert (camera["name"], camera["area"]) == (name, area)
        assert camera["plan"] == pytest.approx((plan_x, 0.5))
        assert camera["stream_type"] == "file" and camera["ai_enabled"] is True
        assert camera["stream_url"] == f"/demo/cam{index}.mp4"
    assert cameras["cam-1"]["zones"] == [
        ("Пресс №2", "danger", CENTRAL, ["helmet", "vest"], 2, None),
        ("Вход на участок", "ppe", ENTRANCE, ["helmet"], 2, None),
    ]
    assert cameras["cam-2"]["zones"] == [
        ("Сварочный участок", "ppe", CENTRAL, ["glasses", "gloves"], None, None),
        ("Вход на участок", "ppe", ENTRANCE, ["glasses", "gloves"], None, None),
    ]
    assert cameras["cam-3"]["zones"] == [
        ("Склад заготовок", "ppe", LEFT, ["vest"], None, None),
        ("Вход на участок", "ppe", ENTRANCE, ["vest"], None, None),
    ]
    assert cameras["cam-4"]["zones"] == [
        ("Проезд погрузчиков", "vehicle", LEFT, ["vest", "helmet"], None, 3),
        ("Вход на участок", "ppe", ENTRANCE, ["vest", "helmet"], None, 3),
    ]
    seed()
    assert cameras_of(fresh) == cameras


def test_ai_cameras_are_seeded_without_demo_mode_with_mjpeg_urls(fresh, monkeypatch):
    """Rows 88, 104: AI_CAMERAS cameras point to the AI module's MJPEG streams."""
    from app.schemas import CameraIn
    from app.models import Camera
    from app.seed import seed
    from app.stage_two import build_config

    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("AI_CAMERAS", " cam-2, cam-1,,cam-2")
    monkeypatch.setenv("AI_MJPEG_BASE_URL", "http://ai-host:8001/")
    seed()
    cameras = cameras_of(fresh)
    assert list(cameras) == ["cam-1", "cam-2"]
    for camera_id in cameras:
        assert cameras[camera_id]["stream_type"] == "mjpeg"
        assert cameras[camera_id]["stream_url"] == (
            f"http://ai-host:8001/stream/{camera_id}.mjpeg"
        )
        assert cameras[camera_id]["ai_enabled"] is True
    assert cameras["cam-1"]["zones"][0][1:4] == ("danger", CENTRAL, ["helmet", "vest"])
    assert cameras["cam-2"]["zones"][0][1:4] == ("ppe", CENTRAL, ["glasses", "gloves"])
    with fresh() as db:
        camera = db.get(Camera, "cam-1")
        CameraIn.model_validate(
            {
                key: getattr(camera, key)
                for key in ("name", "area_id", "stream_type", "stream_url", "plan_x")
            }
        )
        config = build_config(db, camera).model_dump(mode="json", by_alias=True)
    # GET /api/v1/ingest/config/cam-1, contract 4.4.
    assert config["camera"] == {
        "id": "cam-1",
        "name": "Пресс №2",
        "stream_url": "http://ai-host:8001/stream/cam-1.mjpeg",
        "stream_type": "mjpeg",
    }
    assert config["frame_size_hint"] == [1280, 720]
    assert config["thresholds"] == {
        "confirm_frames": 8,
        "cooldown_sec": 60,
        "min_confidence": 0.5,
    }
    assert config["ppe_types"] == [
        "helmet",
        "vest",
        "glasses",
        "gloves",
        "mask",
        "boots",
        "earmuffs",
        "harness",
        "face_shield",
    ]
    main = config["zones"][0]
    assert set(main) == {"id", "name", "type", "polygon", "rules"}
    assert (main["name"], main["type"], main["polygon"]) == (
        "Пресс №2",
        "danger",
        CENTRAL,
    )
    assert main["rules"] == {
        "required_ppe": ["helmet", "vest"],
        "entry_forbidden": False,
        "max_persons": 2,
        "min_vehicle_distance_m": None,
        "schedule": None,
        "confirm_frames": 8,
        "cooldown_sec": 60,
        "severity_override": None,
    }
    assert config["version"] == 1 and config["updated_at"].endswith("Z")
    # Turning DEMO_MODE on later adds only the missing demo cameras.
    monkeypatch.setenv("DEMO_MODE", "true")
    seed()
    later = cameras_of(fresh)
    assert list(later) == ["cam-1", "cam-2", "cam-3", "cam-4"]
    assert (
        later["cam-1"] == cameras["cam-1"] and later["cam-3"]["stream_type"] == "file"
    )


def test_demo_mode_with_ai_cameras_keeps_mocks_for_the_others(fresh, monkeypatch):
    """Row 88: cam-1/cam-2 from the AI module, cam-3/cam-4 demo files, all AI-enabled."""
    from app.seed import seed

    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("AI_CAMERAS", "cam-1,cam-2")
    seed()
    cameras = cameras_of(fresh)
    assert [cameras[c]["stream_url"] for c in cameras] == [
        "http://localhost:8001/stream/cam-1.mjpeg",
        "http://localhost:8001/stream/cam-2.mjpeg",
        "/demo/cam3.mp4",
        "/demo/cam4.mp4",
    ]
    assert [cameras[c]["stream_type"] for c in cameras] == [
        "mjpeg",
        "mjpeg",
        "file",
        "file",
    ]
    assert all(camera["ai_enabled"] for camera in cameras.values())


def test_seed_never_overwrites_cameras_and_ignores_invalid_values(
    fresh, monkeypatch, caplog
):
    """Rows 57, 104: idempotent (insert only if missing); bad env values are warnings."""
    from app.models import Camera, Zone
    from app.seed import seed

    monkeypatch.setenv("DEMO_MODE", "true")
    seed()
    with fresh.begin() as db:
        camera = db.get(Camera, "cam-1")
        camera.name, camera.stream_url = "Пресс №2 (изменено)", "/demo/other.mp4"
        db.delete(
            db.scalar(
                select(Zone).where(Zone.camera_id == "cam-1", Zone.sort_order == 1)
            )
        )
    monkeypatch.setenv("AI_CAMERAS", "cam-1,bad id!,cam-9")
    monkeypatch.setenv("AI_MJPEG_BASE_URL", "ftp://ai-host")
    with caplog.at_level(logging.WARNING, logger="inspector"):
        seed()
    messages = [record.getMessage() for record in caplog.records]
    assert "AI_CAMERAS: ignoring the invalid camera id 'bad id!'" in messages
    assert (
        "AI_MJPEG_BASE_URL must be an http(s) URL; using http://localhost:8001"
        in messages
    )
    cameras = cameras_of(fresh)
    assert cameras["cam-1"]["name"] == "Пресс №2 (изменено)"
    assert cameras["cam-1"]["stream_url"] == "/demo/other.mp4"
    assert len(cameras["cam-1"]["zones"]) == 1
    assert cameras["cam-9"] == {
        "name": "Камера cam-9",
        "area": "Прессовый участок",
        "stream_type": "mjpeg",
        "stream_url": "http://localhost:8001/stream/cam-9.mjpeg",
        "ai_enabled": True,
        "plan": (None, None),
        "zones": [],
    }
    with fresh() as db:
        assert db.scalar(select(func.count()).select_from(Camera)) == 5


def active_ai_keys(sessions):
    from app.models import ApiKey

    with sessions() as db:
        return [
            row.key_hash
            for row in db.scalars(
                select(ApiKey).where(
                    ApiKey.name == "ai-module", ApiKey.revoked_at.is_(None)
                )
            )
        ]


def test_random_ai_key_is_created_once_printed_and_saved(fresh, capsys):
    """Row 4: without SEED_AI_API_KEY a random key is stored hashed, shown once, saved."""
    from app import seed as module
    from app.dependencies import find_api_key

    module.seed()
    printed = capsys.readouterr().out
    key = Path(module.AI_KEY_FILE).read_text(encoding="utf-8")
    assert key.startswith("ot_") and len(key) > 40
    assert printed.count(key) == 1 and str(module.AI_KEY_FILE) in printed
    assert active_ai_keys(fresh) == [hashlib.sha256(key.encode()).hexdigest()]
    with fresh() as db:
        assert find_api_key(db, key).name == "ai-module"
    if os.name != "nt":
        assert Path(module.AI_KEY_FILE).stat().st_mode & 0o777 == 0o600
    module.seed()
    assert capsys.readouterr().out == ""
    assert Path(module.AI_KEY_FILE).read_text(encoding="utf-8") == key
    assert active_ai_keys(fresh) == [hashlib.sha256(key.encode()).hexdigest()]
    website = Path(__file__).resolve().parents[2]
    assert ".secrets/" in (website / ".gitignore").read_text().splitlines()
    assert module.AI_KEY_FILE.name == "ai_api_key"


def test_env_ai_key_is_seeded_rotated_and_never_logged(
    fresh, monkeypatch, capsys, caplog
):
    """Row 4: SEED_AI_API_KEY creates or rotates the ai-module key; only the fact is logged."""
    from app.models import ApiKey
    from app.seed import seed

    seed()
    capsys.readouterr()
    for value in ("  env-key-one  ", "env-key-one", "env-key-two"):
        monkeypatch.setenv("SEED_AI_API_KEY", value)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="inspector"):
            seed()
        notices = [
            record
            for record in caplog.records
            if record.getMessage() == "AI API key seeded from environment"
        ]
        # The same value again is a no-op (idempotent across restarts).
        assert len(notices) == (0 if value == "env-key-one" else 1)
        assert "env-key" not in caplog.text and "env-key" not in capsys.readouterr().out
        assert active_ai_keys(fresh) == [
            hashlib.sha256(value.strip().encode()).hexdigest()
        ]
    with fresh() as db:
        rows = db.scalars(select(ApiKey).where(ApiKey.name == "ai-module")).all()
        # Random key, env-key-one and env-key-two; the two older ones are revoked.
        assert len(rows) == 3
        assert sum(row.revoked_at is None for row in rows) == 1


def test_seeded_application_serves_the_cam_1_config_to_the_env_key(client):
    """Rows 4, 57, 88: GET /api/v1/ingest/config/cam-1 with the SEED_AI_API_KEY key."""
    key = {"X-API-Key": os.environ["SEED_AI_API_KEY"]}
    response = client.get("/api/v1/ingest/config/cam-1", headers=key)
    assert response.status_code == 200, response.text
    config = response.json()
    assert set(config) == {
        "camera",
        "frame_size_hint",
        "thresholds",
        "ppe_types",
        "zones",
        "version",
        "updated_at",
    }
    # AI_CAMERAS is empty in the suite, so cam-1 is still a demo file camera.
    assert config["camera"] == {
        "id": "cam-1",
        "name": "Пресс №2",
        "stream_url": "/demo/cam1.mp4",
        "stream_type": "file",
    }
    assert [zone["name"] for zone in config["zones"]] == ["Пресс №2", "Вход на участок"]
    main = config["zones"][0]
    assert (main["type"], main["polygon"]) == ("danger", CENTRAL)
    assert main["rules"]["required_ppe"] == ["helmet", "vest"]
    assert set(main["rules"]) == RULE_KEYS
    listed = client.get("/api/v1/ingest/config", headers=key).json()
    assert {"cam-1", "cam-2", "cam-3", "cam-4"} <= {c["camera"]["id"] for c in listed}
    assert (
        client.get("/api/v1/ingest/config/cam-1", headers={"X-API-Key": "x"}).json()[
            "error"
        ]["code"]
        == "unauthorized"
    )
