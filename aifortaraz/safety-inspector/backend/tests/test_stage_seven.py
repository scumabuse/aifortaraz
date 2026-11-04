import importlib.util
import io
from pathlib import Path

from PIL import Image

from test_stage_four import login


def test_public_metrics_are_empty_until_editor_publishes_them(client):
    from app.database import SessionLocal
    from app.models import Setting

    path = "/api/v1/public/project"
    settings = "/api/v1/settings/model-metrics"
    initial = client.get(path)
    assert initial.status_code == 200
    assert set(initial.json()) == {"enterprise_name", "model_metrics"}
    assert all(value is None for value in initial.json()["model_metrics"].values())

    editor, _ = login(client, "engineer")
    viewer, _ = login(client, "viewer")
    metrics = {"map_percent": 91.5, "fps": 22, "latency_ms": 35.4}
    assert client.put(settings, headers=viewer, json=metrics).status_code == 403
    assert (
        client.put(
            settings, headers=editor, json={**metrics, "map_percent": 101}
        ).status_code
        == 422
    )
    try:
        result = client.put(settings, headers=editor, json=metrics)
        assert result.status_code == 200, result.text
        assert client.get(path).json()["model_metrics"] == metrics
    finally:
        with SessionLocal.begin() as db:
            row = db.get(Setting, "model_metrics")
            if row is not None:
                db.delete(row)


def test_fake_client_generates_valid_contract_metadata_and_jpeg():
    import base64

    from app.incident_schemas import DetectionsIn

    path = Path(__file__).resolve().parents[2] / "scripts" / "fake_ai_client.py"
    spec = importlib.util.spec_from_file_location("fake_ai_client", path)
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    frame = fake.synthetic_frame()
    objects = fake.detect(frame)
    message = DetectionsIn.model_validate(
        {
            "type": "detections",
            "camera_id": "test-camera",
            "frame_id": 1,
            "ts": "2026-11-04T08:00:00Z",
            "objects": objects,
        }
    )
    assert message.objects[0].track_id == 88
    raw = base64.b64decode(fake.encode_jpeg(frame), validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == "JPEG"
        assert image.size == (960, 540)
