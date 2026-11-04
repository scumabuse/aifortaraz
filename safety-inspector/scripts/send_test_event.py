"""Send contract metadata and a generated JPEG; no camera access or ML code."""

import argparse
import base64
from datetime import datetime, timedelta, timezone
import getpass
import io
import json
import os
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(
        description="Send a local stage-3 integration test event"
    )
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--zone")
    parser.add_argument(
        "--type",
        default="ppe_violation",
        choices=[
            "ppe_violation",
            "zone_intrusion",
            "vehicle_proximity",
            "fall_detected",
            "person_immobile",
            "overcrowding",
            "custom",
        ],
    )
    parser.add_argument("--format", default="json", choices=["json", "multipart"])
    parser.add_argument("--external-id", default=None)
    parser.add_argument("--track", type=int, default=17)
    parser.add_argument("--severity", choices=["low", "medium", "high", "critical"])
    parser.add_argument("--end-after", type=float, default=12)
    args = parser.parse_args()
    key = os.getenv("AI_API_KEY") or getpass.getpass("API key (hidden): ")
    now = datetime.now(timezone.utc)
    image = Image.new("RGB", (1280, 720), (22, 33, 44))
    drawing = ImageDraw.Draw(image)
    for x in range(0, 1280, 80):
        drawing.line((x, 0, x, 720), fill=(39, 55, 68))
    for y in range(0, 720, 80):
        drawing.line((0, y, 1280, y), fill=(39, 55, 68))
    drawing.ellipse((550, 170, 640, 260), fill=(133, 150, 161))
    drawing.rounded_rectangle((530, 265, 660, 550), radius=25, fill=(98, 119, 133))
    drawing.text(
        (30, 30),
        f"TEST METADATA / {args.type} / TRACK #{args.track}",
        fill=(220, 234, 238),
    )
    output = io.BytesIO()
    image.save(output, "JPEG", quality=88)
    body = {
        "external_id": args.external_id or str(uuid4()),
        "camera_id": args.camera,
        "zone_id": args.zone,
        "type": args.type,
        "severity": args.severity,
        "confidence": 0.93,
        "track_id": args.track,
        "started_at": now.isoformat(),
        "ended_at": None,
        "details": {
            "missing_ppe": ["helmet"],
            "present_ppe": ["vest"],
            "boxes": [
                {
                    "track_id": args.track,
                    "label": "person",
                    "bbox": [0.4, 0.2, 0.15, 0.6],
                    "conf": 0.93,
                }
            ],
            "frame_size": [1280, 720],
        },
    }
    if args.type != "ppe_violation":
        body["details"].pop("missing_ppe")
        body["details"].pop("present_ppe")
    body["details"].update(
        {
            "fall_detected": {"pose_conf": 0.94},
            "person_immobile": {"immobile_sec": 65},
            "vehicle_proximity": {
                "distance_m": 1.7,
                "person_track_id": args.track,
                "vehicle_track_id": 3,
            },
            "zone_intrusion": {"zone_id": args.zone, "duration_sec": 12},
            "overcrowding": {"count": 4, "max_persons": 2},
            "custom": {"title": "Тестовое событие"},
        }.get(args.type, {})
    )
    with httpx.Client(
        base_url=args.host, headers={"X-API-Key": key}, timeout=15
    ) as client:
        heartbeat = client.post(
            "/api/v1/ingest/heartbeat",
            json={
                "camera_id": args.camera,
                "fps": 24,
                "model_name": "contract-test",
                "model_version": "0.3",
                "status": "ok",
                "extra": {},
            },
        )
        heartbeat.raise_for_status()
        stats = client.post(
            "/api/v1/ingest/stats",
            json={
                "camera_id": args.camera,
                "period_start": (now - timedelta(minutes=1)).isoformat(),
                "period_end": now.isoformat(),
                "persons_total": 42,
                "persons_compliant": 39,
                "by_ppe": {"helmet": {"checked": 42, "compliant": 40}},
            },
        )
        stats.raise_for_status()
        if args.format == "multipart":
            response = client.post(
                "/api/v1/ingest/events",
                data={"payload": json.dumps(body)},
                files={"snapshot": ("frame.jpg", output.getvalue(), "image/jpeg")},
            )
        else:
            response = client.post(
                "/api/v1/ingest/events",
                json={
                    **body,
                    "snapshot_base64": base64.b64encode(output.getvalue()).decode(),
                },
            )
        print(response.status_code, response.text)
        response.raise_for_status()
        if response.status_code == 201 and args.end_after >= 0:
            ended = client.patch(
                f"/api/v1/ingest/events/{body['external_id']}",
                json={
                    "ended_at": (now + timedelta(seconds=args.end_after)).isoformat()
                },
            )
            ended.raise_for_status()
        print("external_id:", body["external_id"])


if __name__ == "__main__":
    main()
