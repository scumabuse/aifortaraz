from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from test_stage_four import image_bytes, login


def test_overview_weights_ppe_and_compares_local_period(client, monkeypatch):
    from app import analytics
    from app.database import SessionLocal
    from app.models import Area, Camera, ComplianceStat, Incident

    headers, _ = login(client, "viewer")
    start = datetime(2026, 1, 4, 19, tzinfo=timezone.utc)
    monkeypatch.setattr(analytics, "utc_now", lambda: start + timedelta(hours=8))
    with SessionLocal() as db:
        area = Area(name="Analytics regression")
        db.add(area)
        db.flush()
        area_id = area.id
        camera = Camera(
            name="Analytics camera",
            area_id=area.id,
            stream_type="file",
            stream_url="/demo/qa.mp4",
        )
        db.add(camera)
        db.flush()
        camera_id = camera.id
        for offset, people, compliant in [(0, 2, 2), (1, 8, 2), (-1440, 10, 9)]:
            stamp = start + timedelta(minutes=offset)
            db.add(
                ComplianceStat(
                    camera_id=camera_id,
                    period_start=stamp,
                    period_end=stamp + timedelta(minutes=1),
                    persons_total=people,
                    persons_compliant=compliant,
                )
            )
        for offset, status in [
            (0, "acknowledged"),
            (1, "false_positive"),
            (-1440, "new"),
        ]:
            stamp = start + timedelta(minutes=offset)
            db.add(
                Incident(
                    camera_id=camera_id,
                    source="ai",
                    type_code="ppe_violation",
                    severity="critical",
                    status=status,
                    started_at=stamp,
                    created_at=stamp,
                    acknowledged_at=stamp + timedelta(seconds=35)
                    if status == "acknowledged"
                    else None,
                )
            )
        db.commit()

    response = client.get(
        "/api/v1/analytics/overview",
        params={"camera_id": camera_id, "period": "today"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["current"] == {
        "incidents_total": 2,
        "open_incidents": 1,
        "critical_incidents": 2,
        "false_positive_percent": 50,
        "mean_reaction_sec": 35,
        "compliance_percent": 40,
    }
    assert result["previous"]["incidents_total"] == 1
    assert result["previous"]["compliance_percent"] == 90
    assert result["daily"][0]["date"] == "2026-01-05"
    assert result["daily"][0]["compliance"] == 40
    assert result["heat"] == [{"day": 0, "hour": 0, "count": 2}]
    assert result["reaction"][0]["count"] == 1
    assert result["cameras"][0]["previous"] == 1
    with SessionLocal() as db:
        db.execute(delete(Incident).where(Incident.camera_id == camera_id))
        db.execute(delete(ComplianceStat).where(ComplianceStat.camera_id == camera_id))
        db.execute(delete(Camera).where(Camera.id == camera_id))
        db.execute(delete(Area).where(Area.id == area_id))
        db.commit()


def test_plan_image_and_position_permissions_and_persistence(client):
    headers, _ = login(client, "engineer")
    viewer, _ = login(client, "viewer")
    area = client.get("/api/v1/settings/areas", headers=headers).json()[0]
    camera = client.post(
        "/api/v1/cameras",
        headers=headers,
        json={
            "name": "Plan test",
            "area_id": area["id"],
            "stream_type": "file",
            "stream_url": "/demo/qa.mp4",
        },
    ).json()
    image_path = f"/api/v1/plan/{area['id']}/image"
    position_path = f"/api/v1/plan/cameras/{camera['id']}"
    files = {"file": ("plan.jpg", image_bytes(), "image/jpeg")}
    assert client.put(image_path, headers=viewer, files=files).status_code == 403
    assert (
        client.put(
            image_path, headers=headers, files={"file": ("plan.png", b"invalid")}
        ).status_code
        == 422
    )
    uploaded = client.put(image_path, headers=headers, files=files)
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()["width"] == 320
    assert client.get(uploaded.json()["url"]).status_code == 200
    assert (
        client.patch(
            position_path, headers=viewer, json={"x": 0.25, "y": 0.75}
        ).status_code
        == 403
    )
    assert (
        client.patch(
            position_path, headers=headers, json={"x": -1, "y": 0.75}
        ).status_code
        == 422
    )
    assert (
        client.patch(
            position_path, headers=headers, json={"x": 0.25, "y": 0.75}
        ).status_code
        == 200
    )
    plan = client.get(
        "/api/v1/plan", headers=viewer, params={"area_id": area["id"]}
    ).json()["areas"][0]
    assert plan["image"] == uploaded.json()
    marker = next(marker for marker in plan["cameras"] if marker["id"] == camera["id"])
    assert (marker["x"], marker["y"], marker["count"]) == (0.25, 0.75, 0)
