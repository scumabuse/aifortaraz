import asyncio
import base64

import httpx
from sqlalchemy import delete

from test_stage_four import image_bytes, login


def ingest_context(client):
    admin, _ = login(client)
    area = client.get("/api/v1/settings/areas", headers=admin).json()[0]
    camera = client.post(
        "/api/v1/cameras",
        headers=admin,
        json={
            "name": "Stage six camera",
            "area_id": area["id"],
            "stream_type": "file",
            "stream_url": "/demo/qa.mp4",
        },
    ).json()
    key = client.post(
        "/api/v1/api-keys", headers=admin, json={"name": "Stage six key"}
    ).json()
    return admin, area, camera, {"X-API-Key": key["key"]}


def create_event(client, camera, headers, external_id, severity="high", track=15):
    from app.models import utc_now

    response = client.post(
        "/api/v1/ingest/events",
        headers=headers,
        json={
            "external_id": external_id,
            "camera_id": camera["id"],
            "type": "ppe_violation",
            "severity": severity,
            "started_at": utc_now().isoformat(),
            "confidence": 0.91,
            "track_id": track,
            "details": {
                "missing_ppe": ["helmet"],
                "boxes": [
                    {
                        "bbox": [0.1, 0.1, 0.3, 0.3],
                        "label": "person",
                        "track_id": track,
                        "conf": 0.91,
                    }
                ],
            },
            "snapshot_base64": base64.b64encode(image_bytes()).decode(),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_report_act_and_csv_lifecycle(client):
    admin, area, camera, key = ingest_context(client)
    viewer, _ = login(client, "viewer")
    incident = create_event(client, camera, key, "stage-six-report")
    root = f"/api/v1/incidents/{incident}/act"
    assert client.post(root + "/draft", headers=viewer).status_code == 403
    draft = client.post(root + "/draft", headers=admin)
    assert draft.status_code == 200, draft.text
    assert "#15" in draft.json()["description"] and "91%" in draft.json()["description"]
    act = {**draft.json(), "measures": "<script>alert(1)</script>"}
    preview = client.post(root + "/preview", headers=admin, json=act)
    assert preview.status_code == 200
    assert (
        "&lt;script&gt;" in preview.text and "data:image/jpeg;base64," in preview.text
    )
    assert 'class="box"' in preview.text
    saved = client.post(root, headers=admin, json=act)
    assert saved.status_code == 201, saved.text
    assert saved.json()["number"].startswith("АКТ-")
    path = f"/api/v1/reports/{saved.json()['id']}/document"
    assert client.get(path).status_code == 401
    assert saved.json()["number"] in client.get(path, headers=viewer).text

    report = client.post(
        "/api/v1/reports",
        headers=admin,
        json={
            "period": "7d",
            "camera_ids": [camera["id"]],
            "area_id": area["id"],
        },
    )
    assert report.status_code == 201, report.text
    path = "/api/v1/reports/" + report.json()["id"]
    html = client.get(path + "/document", headers=viewer).text
    assert camera["name"] in html and "Сводка по камерам" in html
    csv_response = client.get(path + "/data.csv", headers=viewer)
    assert csv_response.status_code == 200 and incident in csv_response.text
    assert (
        client.post(
            "/api/v1/reports",
            headers=admin,
            json={
                "kind": "prescription",
                "period": "7d",
                "area_id": area["id"],
                "measures": "Инструктаж",
                "deadline": "2027-01-01",
            },
        ).status_code
        == 201
    )


def test_telegram_rules_dedupe_and_delivery_journal(client, monkeypatch):
    from app import telegram
    from app.database import SessionLocal
    from app.models import NotificationSettings, TelegramDelivery

    admin, _, camera, key = ingest_context(client)
    viewer, _ = login(client, "viewer")
    route = "/api/v1/settings/telegram"
    assert client.get(route, headers=viewer).status_code == 403
    claim = telegram.claim
    monkeypatch.setattr(telegram, "claim", lambda: None)
    token = "123456:" + "a" * 35
    local = telegram.local_date
    monkeypatch.setattr(
        telegram, "local_date", lambda db, stamp: local(db, stamp).replace(hour=23)
    )
    config = {
        "token": token,
        "recipients": [{"chat_id": "-1001234", "label": "Мастер"}],
        "rules": {
            "enabled": True,
            "base_url": "https://inspector.example",
            "quiet_start": "22:00",
            "quiet_end": "06:00",
            "cooldown_sec": 300,
        },
    }
    saved = client.put(route, headers=admin, json=config)
    assert (
        saved.status_code == 200
        and token not in saved.text
        and saved.json()["has_token"]
    )
    try:
        skipped = create_event(client, camera, key, "telegram-quiet")
        sent = create_event(client, camera, key, "telegram-critical", "critical", 16)
        cooldown = create_event(
            client, camera, key, "telegram-cooldown", "critical", 17
        )
        test = client.post(route + "/test", headers=admin, json={"chat_id": "-1001234"})
        assert test.status_code == 202
        jobs = {
            item["incident_id"]: item
            for item in client.get(route + "/journal", headers=admin).json()["items"]
        }
        assert jobs[skipped]["error"] == "Тихие часы"
        assert jobs[cooldown]["error"] == "Кулдаун камеры и типа"
        assert jobs[sent]["status"] == "pending"
        assert jobs[None]["status"] == "pending"
        delivery_id, attempt, credential, recipient, caption, photo = claim()
        assert (delivery_id, attempt, credential, recipient) == (
            jobs[sent]["id"],
            1,
            token,
            "-1001234",
        )
        assert f"https://inspector.example/incidents/{sent}" in caption
        assert photo.startswith(b"\xff\xd8")
        telegram.finish(jobs[sent]["id"], 1, "Сбой сети", 0)
        with SessionLocal.begin() as db:
            row = db.get(TelegramDelivery, jobs[sent]["id"])
            assert row.status == "pending"
            row.attempts = telegram.MAX_ATTEMPTS
        telegram.finish(jobs[sent]["id"], telegram.MAX_ATTEMPTS, "Сбой сети", 0)
        with SessionLocal() as db:
            assert db.get(TelegramDelivery, jobs[sent]["id"]).status == "failed"
    finally:
        with SessionLocal.begin() as db:
            db.execute(delete(TelegramDelivery))
            row = db.get(NotificationSettings, 1)
            row.telegram_bot_token, row.chat_ids, row.rules = None, [], {}


def test_telegram_transport_uses_photo_and_handles_rate_limit_and_offline():
    from app.telegram import send

    async def exercise():
        def handler(request):
            assert request.url.path.endswith("/sendPhoto")
            assert "multipart/form-data" in request.headers["content-type"]
            assert b"incident.jpg" in request.content
            return httpx.Response(
                429,
                json={
                    "ok": False,
                    "error_code": 429,
                    "parameters": {"retry_after": 12},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            error, wait = await send(client, "token", "123", "caption", b"jpeg")
            assert error and wait == 12

        def offline(request):
            raise httpx.ConnectError("offline test", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(offline)) as client:
            error, wait = await send(client, "token", "123", "caption", b"jpeg")
            assert error == "Сеть Telegram недоступна или истекло время ожидания"
            assert wait == 0

    asyncio.run(exercise())


def test_settings_permissions_and_account_creation(client):
    admin, _ = login(client)
    editor, _ = login(client, "engineer")
    viewer, _ = login(client, "viewer")
    config = client.get("/api/v1/settings", headers=admin).json()
    keys = (
        "enterprise_name",
        "workshop_name",
        "timezone",
        "theme",
        "alarm_sound",
        "confirm_frames",
        "cooldown_sec",
        "min_confidence",
        "heartbeat_timeout_sec",
        "snapshot_retention_days",
    )
    original = {key: config[key] for key in keys}
    edited = {**original, "theme": "light", "workshop_name": "Цех для проверки"}
    assert (
        client.put("/api/v1/settings/general", headers=viewer, json=edited).status_code
        == 403
    )
    assert (
        client.put(
            "/api/v1/settings/general",
            headers=editor,
            json={**edited, "timezone": "Nowhere"},
        ).status_code
        == 422
    )
    assert (
        client.put("/api/v1/settings/general", headers=editor, json=edited).status_code
        == 200
    )
    assert client.get("/api/v1/settings", headers=viewer).json()["theme"] == "light"
    assert (
        client.put(
            "/api/v1/settings/general", headers=editor, json=original
        ).status_code
        == 200
    )
    account = {
        "username": "qa_stage_six",
        "full_name": "Новый наблюдатель",
        "role": "viewer",
        "password": "example-secret",
    }
    assert client.post("/api/v1/users", headers=editor, json=account).status_code == 403
    result = client.post("/api/v1/users", headers=admin, json=account)
    assert result.status_code == 201 and account["password"] not in result.text
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": account["username"], "password": account["password"]},
        ).status_code
        == 200
    )
    update = client.put(
        "/api/v1/users/" + result.json()["id"],
        headers=admin,
        json={**account, "role": "engineer", "password": None},
    )
    assert update.status_code == 200 and update.json()["role"] == "engineer"
    from app.database import SessionLocal
    from app.models import User

    with SessionLocal.begin() as db:
        db.execute(delete(User).where(User.id == result.json()["id"]))
