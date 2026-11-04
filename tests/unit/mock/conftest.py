"""Fixtures of the strict mock backend tests: app, TestClient, fake clock and payload factories.

Payload factories build the contract's own example messages (docs/CONTRACT.md 4.1-4.5) with
real timestamps where the contract shows ``"..."``; tests mutate copies of them.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import warnings
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from safety_ai.clock import FakeClock
from tools.mock_backend import DEFAULT_CONFIG_PATH, create_app, mock_state
from tools.mock_state import MockState

with warnings.catch_warnings():
    # starlette 1.x prefers httpx2 for its TestClient; httpx works the same way here.
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

API_KEY = "test-key"
AUTH = {"X-API-Key": API_KEY}
START_UTC = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)

CONTRACT_EVENT: dict[str, Any] = {
    "external_id": "cam-1-1762179667-17-ppe_violation",
    "camera_id": "cam-1",
    "zone_id": "zone-press-2",
    "type": "ppe_violation",
    "severity": None,
    "confidence": 0.91,
    "track_id": 17,
    "started_at": "2026-11-03T14:21:07Z",
    "ended_at": None,
    "details": {
        "missing_ppe": ["helmet"],
        "present_ppe": ["vest"],
        "boxes": [
            {"track_id": 17, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93}
        ],
        "frame_size": [1280, 720],
    },
}

CONTRACT_HEARTBEAT: dict[str, Any] = {
    "camera_id": "cam-1",
    "fps": 11.8,
    "model_name": "ppe_s_sh17",
    "model_version": "2026-11-04",
    "status": "ok",
    "extra": {
        "device": "cuda:0",
        "backend": "tensorrt",
        "queue_depth": 0,
        "frames_dropped": 12,
        "stream": "connected",
    },
}

CONTRACT_STATS: dict[str, Any] = {
    "camera_id": "cam-1",
    "period_start": "2026-11-03T14:21:00Z",
    "period_end": "2026-11-03T14:22:00Z",
    "persons_total": 42,
    "persons_compliant": 39,
    "by_ppe": {
        "helmet": {"checked": 42, "compliant": 40},
        "vest": {"checked": 42, "compliant": 41},
    },
}

CONTRACT_CONFIG: dict[str, Any] = {
    "camera": {
        "id": "cam-1",
        "name": "Press #2",
        "stream_url": "rtsp://192.168.1.10/live",
        "stream_type": "rtsp",
    },
    "frame_size_hint": [1280, 720],
    "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
    "ppe_types": [
        "helmet",
        "vest",
        "glasses",
        "gloves",
        "mask",
        "boots",
        "earmuffs",
        "harness",
        "face_shield",
    ],
    "zones": [
        {
            "id": "zone-press-2",
            "name": "Press #2",
            "type": "danger",
            "polygon": [[0.10, 0.20], [0.55, 0.18], [0.60, 0.80], [0.12, 0.85]],
            "rules": {
                "required_ppe": ["helmet", "glasses"],
                "entry_forbidden": False,
                "max_persons": 2,
                "min_vehicle_distance_m": None,
                "schedule": None,
                "confirm_frames": 8,
                "cooldown_sec": 60,
                "severity_override": None,
            },
        }
    ],
    "version": 42,
    "updated_at": "2026-11-03T12:00:00Z",
}

CONTRACT_DETECTIONS: dict[str, Any] = {
    "type": "detections",
    "camera_id": "cam-1",
    "ts": "2026-11-03T14:21:07.250Z",
    "frame_id": 12345,
    "objects": [
        {
            "track_id": 17,
            "label": "person",
            "bbox": [0.41, 0.22, 0.12, 0.46],
            "conf": 0.93,
            "ppe": {"helmet": False, "vest": True, "glasses": None},
            "zones": ["zone-press-2"],
            "state": "violation",
        },
        {"track_id": 3, "label": "forklift", "bbox": [0.70, 0.50, 0.20, 0.30], "conf": 0.88},
    ],
}

_PERSON_BOX = {"track_id": 17, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93}
_FORKLIFT_BOX = {"track_id": 3, "label": "forklift", "bbox": [0.70, 0.50, 0.20, 0.30], "conf": 0.88}
_DETAILS_BY_TYPE: dict[str, dict[str, Any]] = {
    "ppe_violation": {"missing_ppe": ["helmet"], "present_ppe": ["vest"]},
    "zone_intrusion": {"duration_sec": 1.5},
    "vehicle_proximity": {"distance_m": 1.8, "person_track_id": 17, "vehicle_track_id": 3},
    "fall_detected": {"pose_conf": 0.74},
    "person_immobile": {"immobile_sec": 31.0},
    "overcrowding": {"count": 3, "max_persons": 2},
    "custom": {"title": "Spilled oil near press #2"},
}


def make_jpeg(width: int, height: int, *, quality: int = 85, seed: int = 0) -> bytes:
    """A real JPEG of the given size with a deterministic colour."""
    color = ((seed * 53) % 256, (seed * 97 + 40) % 256, (seed * 31 + 80) % 256)
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


class Payloads:
    """Builders of valid contract messages; every call returns a fresh deep copy."""

    def contract_event(self) -> dict[str, Any]:
        """The 4.1 example verbatim."""
        return copy.deepcopy(CONTRACT_EVENT)

    def event(
        self,
        event_type: str = "ppe_violation",
        *,
        camera_id: str = "cam-1",
        zone_id: str | None = "cam1-shop-floor",
        track_id: int = 17,
        seconds: int = 1762179667,
        started_at: str = "2026-11-03T14:21:07Z",
        **overrides: Any,
    ) -> dict[str, Any]:
        """A valid event of ``event_type`` (details per contract 4.1)."""
        boxes = [dict(_PERSON_BOX, track_id=track_id)]
        if event_type == "vehicle_proximity":
            boxes.append(dict(_FORKLIFT_BOX))
        details = copy.deepcopy(_DETAILS_BY_TYPE[event_type])
        if event_type == "vehicle_proximity":
            details["person_track_id"] = track_id
        details.update(boxes=boxes, frame_size=[1280, 720])
        payload: dict[str, Any] = {
            "external_id": f"{camera_id}-{seconds}-{track_id}-{event_type}",
            "camera_id": camera_id,
            "zone_id": zone_id,
            "type": event_type,
            "severity": None,
            "confidence": 0.91,
            "track_id": track_id,
            "started_at": started_at,
            "ended_at": None,
            "details": details,
        }
        payload.update(overrides)
        return payload

    def heartbeat(self, **overrides: Any) -> dict[str, Any]:
        """The 4.2 example (plus overrides)."""
        payload = copy.deepcopy(CONTRACT_HEARTBEAT)
        payload.update(overrides)
        return payload

    def stats(self, **overrides: Any) -> dict[str, Any]:
        """The 4.3 example with minute-aligned periods (plus overrides)."""
        payload = copy.deepcopy(CONTRACT_STATS)
        payload.update(overrides)
        return payload

    def config(self) -> dict[str, Any]:
        """The 4.4 example."""
        return copy.deepcopy(CONTRACT_CONFIG)

    def detections(self, **overrides: Any) -> dict[str, Any]:
        """The 4.5 detections example (plus overrides)."""
        payload = copy.deepcopy(CONTRACT_DETECTIONS)
        payload.update(overrides)
        return payload

    def frame(
        self,
        *,
        camera_id: str = "cam-1",
        ts: str = "2026-11-03T14:21:07.250Z",
        width: int = 960,
        height: int = 540,
        seed: int = 0,
    ) -> dict[str, Any]:
        """A 4.5 frame message holding a real JPEG (quality 70)."""
        jpeg = make_jpeg(width, height, quality=70, seed=seed)
        return {
            "type": "frame",
            "camera_id": camera_id,
            "ts": ts,
            "jpeg_base64": base64.b64encode(jpeg).decode("ascii"),
        }

    def ws_heartbeat(self, **overrides: Any) -> dict[str, Any]:
        """The 4.2 heartbeat payload plus ``"type": "heartbeat"`` (4.5)."""
        payload = {"type": "heartbeat", **copy.deepcopy(CONTRACT_HEARTBEAT)}
        payload.update(overrides)
        return payload


PostEvent = Callable[..., Any]


def contract_error(response: Any) -> dict[str, Any]:
    """The ``error`` object of an error response; asserts the contract's exact envelope."""
    body = response.json()
    assert set(body) == {"error"}, body
    error: dict[str, Any] = body["error"]
    assert set(error) == {"code", "message", "details"}, error
    assert isinstance(error["code"], str) and isinstance(error["message"], str), error
    assert isinstance(error["details"], dict), error
    return error


def contract_error_lines(response: Any) -> list[str]:
    """``details.fields`` of an error response as readable ``location: message`` lines."""
    lines = []
    for field in contract_error(response)["details"].get("fields", []):
        location = ".".join(str(part) for part in field["location"])
        lines.append(f"{location}: {field['message']}" if location else field["message"])
    return lines


@pytest.fixture
def error_of() -> Callable[[Any], dict[str, Any]]:
    """``error_of(response)`` -> the contract ``error`` object (envelope checked)."""
    return contract_error


@pytest.fixture
def error_lines() -> Callable[[Any], list[str]]:
    """``error_lines(response)`` -> the field errors as ``location: message`` lines."""
    return contract_error_lines


@pytest.fixture
def payloads() -> Payloads:
    """Factories of valid contract messages."""
    return Payloads()


@pytest.fixture
def jpeg() -> Callable[..., bytes]:
    """``jpeg(width, height, quality=85, seed=0) -> bytes``."""
    return make_jpeg


@pytest.fixture
def clock() -> FakeClock:
    """Fake clock of the mock (arrival times), starting at 2026-11-03T14:00:00Z."""
    return FakeClock(START_UTC, start_mono=500.0)


@pytest.fixture
def config_path() -> Path:
    """The shipped ``tools/mock_config.yaml``."""
    return DEFAULT_CONFIG_PATH


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Data directory of the mock under test."""
    return tmp_path / "mock_data"


@pytest.fixture
def app(config_path: Path, data_dir: Path, clock: FakeClock) -> Any:
    """The mock app with API key ``test-key`` and the fake clock."""
    return create_app(config_path, data_dir, API_KEY, clock=clock)


@pytest.fixture
def state(app: Any) -> MockState:
    """The app's state."""
    return mock_state(app)


@pytest.fixture
def client(app: Any) -> Iterator[TestClient]:
    """TestClient sending ``X-API-Key: test-key`` by default."""
    with TestClient(app, headers=AUTH) as test_client:
        yield test_client


@pytest.fixture
def anon_client(app: Any) -> Iterator[TestClient]:
    """TestClient without an API key."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_client_factory() -> Callable[..., TestClient]:
    """``TestClient`` itself, for tests that build their own app."""
    return TestClient


@pytest.fixture
def post_event(client: TestClient) -> PostEvent:
    """POST an event: multipart by default, the JSON alternative with ``as_json=True``."""

    def _post(
        payload: Any,
        snapshot: bytes | None = None,
        *,
        as_json: bool = False,
        test_client: TestClient | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        target = test_client if test_client is not None else client
        image = snapshot if snapshot is not None else make_jpeg(1280, 720)
        if as_json:
            body = dict(payload, snapshot_base64=base64.b64encode(image).decode("ascii"))
            return target.post("/api/v1/ingest/events", json=body, headers=headers)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return target.post(
            "/api/v1/ingest/events",
            data={"payload": text},
            files={"snapshot": ("snapshot.jpg", image, "image/jpeg")},
            headers=headers,
        )

    return _post
