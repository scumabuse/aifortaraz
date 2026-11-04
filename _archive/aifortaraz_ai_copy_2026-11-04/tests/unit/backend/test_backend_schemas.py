"""Contract conformance of safety_ai/backend/schemas.py (spec section 4).

Outgoing messages (AI -> site) must reproduce every contract example byte-for-byte as compact
JSON; incoming messages (site -> AI) must parse every example and keep its values. Invalid
variants of every message type are rejected.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from safety_ai.backend import schemas as s

# ---------------------------------------------------------------------------- contract examples

EVENT_EXAMPLE = """{
  "external_id": "cam-1-1762179667-17-ppe_violation",
  "camera_id": "cam-1",
  "zone_id": "zone-press-2",
  "type": "ppe_violation",
  "severity": null,
  "confidence": 0.91,
  "track_id": 17,
  "started_at": "2026-11-03T14:21:07Z",
  "ended_at": null,
  "details": {
    "missing_ppe": ["helmet"],
    "present_ppe": ["vest"],
    "boxes": [{"track_id": 17, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93}],
    "frame_size": [1280, 720]
  }
}"""

PATCH_EXAMPLE = '{"ended_at": "2026-11-03T14:21:19Z"}'

RESPONSE_201 = '{"id": "evt-42", "status": "new", "severity": "medium", "deduplicated": false}'
RESPONSE_202 = '{"deduplicated": true, "id": "evt-41"}'

# Contract 4.2 verbatim plus ``model_family`` (DECISIONS.md D-020: the AI always sends it).
HEARTBEAT_EXAMPLE = (
    '{"camera_id": "cam-1", "fps": 11.8, "model_name": "ppe_s_sh17", "model_version": '
    '"2026-11-04", "status": "ok", "extra": {"device": "cuda:0", "backend": "tensorrt", '
    '"queue_depth": 0, "frames_dropped": 12, "stream": "connected", "model_family": "yolo26"}}'
)

# Contract 4.3 with the elided "..." timestamps filled in (minute-aligned, spec 5.11).
STATS_EXAMPLE = (
    '{"camera_id": "cam-1", "period_start": "2026-11-03T14:21:00Z", "period_end": '
    '"2026-11-03T14:22:00Z", "persons_total": 42, "persons_compliant": 39, "by_ppe": '
    '{"helmet": {"checked": 42, "compliant": 40}, "vest": {"checked": 42, "compliant": 41}}}'
)

CONFIG_EXAMPLE = """{
  "camera": {"id": "cam-1", "name": "Press #2", "stream_url": "rtsp://192.168.1.10/live", "stream_type": "rtsp"},
  "frame_size_hint": [1280, 720],
  "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
  "ppe_types": ["helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "harness", "face_shield"],
  "zones": [
    {
      "id": "zone-press-2",
      "name": "Press #2",
      "type": "danger",
      "polygon": [[0.10, 0.20], [0.55, 0.18], [0.60, 0.80], [0.12, 0.85]],
      "rules": {
        "required_ppe": ["helmet", "glasses"],
        "entry_forbidden": false,
        "max_persons": 2,
        "min_vehicle_distance_m": null,
        "schedule": null,
        "confirm_frames": 8,
        "cooldown_sec": 60,
        "severity_override": null
      }
    }
  ],
  "version": 42,
  "updated_at": "2026-11-03T12:00:00Z"
}"""  # noqa: E501 (contract 4.4 example, verbatim)

SCHEDULE_EXAMPLE = '{"days": [1, 2, 3, 4, 5], "from": "08:00", "to": "20:00"}'

# Contract 4.5 with the elided "ts" filled in (milliseconds, DECISIONS.md D-011).
DETECTIONS_EXAMPLE = (
    '{"type": "detections", "camera_id": "cam-1", "ts": "2026-11-03T14:21:07.250Z", '
    '"frame_id": 12345, "objects": [{"track_id": 17, "label": "person", "bbox": '
    '[0.41, 0.22, 0.12, 0.46], "conf": 0.93, "ppe": {"helmet": false, "vest": true, '
    '"glasses": null}, "zones": ["zone-press-2"], "state": "violation"}, {"track_id": 3, '
    '"label": "forklift", "bbox": [0.70, 0.50, 0.20, 0.30], "conf": 0.88}]}'
)
FRAME_EXAMPLE = (
    '{"type": "frame", "camera_id": "cam-1", "ts": "2026-11-03T14:21:07.250Z", '
    '"jpeg_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"}'
)
CONFIG_UPDATED_EXAMPLE = '{"type": "config.updated", "camera_id": "cam-1", "version": 43}'
SERVER_ERROR_EXAMPLE = '{"type": "error", "message": "invalid detections message"}'


def compact(text: str) -> str:
    """The canonical compact JSON form of an example (what model_dump_json produces)."""
    return json.dumps(json.loads(text), separators=(",", ":"))


def event_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(EVENT_EXAMPLE)
    data.update(overrides)
    return data


def heartbeat_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(HEARTBEAT_EXAMPLE)
    data.update(overrides)
    return data


def stats_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(STATS_EXAMPLE)
    data.update(overrides)
    return data


def config_dict() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(CONFIG_EXAMPLE)
    return data


# ---------------------------------------------------------------------------- outgoing: exact


@pytest.mark.parametrize(
    ("model", "example"),
    [
        (s.EventPayload, EVENT_EXAMPLE),
        (s.EventEndPatch, PATCH_EXAMPLE),
        (s.HeartbeatPayload, HEARTBEAT_EXAMPLE),
        (s.StatsPayload, STATS_EXAMPLE),
        (s.DetectionsMessage, DETECTIONS_EXAMPLE),
        (s.FrameMessage, FRAME_EXAMPLE),
    ],
    ids=["event", "patch", "heartbeat", "stats", "detections", "frame"],
)
def test_outgoing_examples_round_trip_byte_for_byte(
    model: type[s.StrictModel], example: str
) -> None:
    parsed = model.model_validate_json(example)
    assert parsed.model_dump_json() == compact(example)
    again = model.model_validate_json(parsed.model_dump_json())
    assert again == parsed


def test_heartbeat_ws_message_carries_type_and_the_http_payload() -> None:
    message = s.HeartbeatMessage.model_validate({**heartbeat_dict(), "type": "heartbeat"})
    dumped = json.loads(message.model_dump_json())
    assert dumped.pop("type") == "heartbeat"
    assert dumped == json.loads(
        s.HeartbeatPayload.model_validate(heartbeat_dict()).model_dump_json()
    )
    assert s.HeartbeatMessage.model_validate_json(message.model_dump_json()) == message


@pytest.mark.parametrize("event_type", s.EVENT_TYPES)
def test_every_event_type_round_trips(event_type: str, make_payload: Any) -> None:
    zone = None if event_type in ("fall_detected", "person_immobile") else "zone-1"
    payload = make_payload(event_type=event_type, zone_id=zone)
    text = payload.model_dump_json()
    assert s.EventPayload.model_validate_json(text).model_dump_json() == text
    assert isinstance(payload.details, s.DETAILS_BY_TYPE[event_type])


def test_event_timestamps_are_utc_whole_seconds_with_z() -> None:
    plus5 = timezone(timedelta(hours=5))
    payload = s.EventPayload.model_validate(
        event_dict(
            started_at=datetime(2026, 11, 3, 19, 21, 7, 900_000, tzinfo=plus5),
            ended_at="2026-11-03T14:21:19.5Z",
        )
    )
    dumped = json.loads(payload.model_dump_json())
    assert dumped["started_at"] == "2026-11-03T14:21:07Z"
    assert dumped["ended_at"] == "2026-11-03T14:21:19Z"
    assert payload.started_at.tzinfo is not None


def test_ws_timestamps_keep_milliseconds() -> None:
    message = s.DetectionsMessage(
        camera_id="cam-1",
        ts=datetime(2026, 11, 3, 14, 21, 7, 250_999, tzinfo=timezone.utc),
        frame_id=1,
        objects=[],
    )
    assert json.loads(message.model_dump_json())["ts"] == "2026-11-03T14:21:07.250Z"


def test_bbox_and_confidence_are_rounded_in_json() -> None:
    box = s.Box(track_id=1, label="helmet", bbox=(0.123456, 0.2, 0.3, 0.4), conf=0.987654)
    assert json.loads(box.model_dump_json()) == {
        "track_id": 1,
        "label": "helmet",
        "bbox": [0.1235, 0.2, 0.3, 0.4],
        "conf": 0.988,
    }


def test_external_id_helpers() -> None:
    started = datetime(2026, 11, 3, 14, 21, 7, 999_000, tzinfo=timezone.utc)
    external_id = s.make_external_id("cam-1", started, 17, "ppe_violation")
    assert (
        external_id == f"cam-1-{int(started.replace(microsecond=0).timestamp())}-17-ppe_violation"
    )
    assert s.external_id_matches(external_id, "cam-1", 17, "ppe_violation")
    assert not s.external_id_matches(external_id, "cam-2", 17, "ppe_violation")
    assert not s.external_id_matches("cam-1-12x4-17-ppe_violation", "cam-1", 17, "ppe_violation")
    assert not s.external_id_matches(external_id, "cam-1", 18, "ppe_violation")


def test_zone_id_may_be_null_for_fall_and_immobile_only(make_payload: Any) -> None:
    for event_type in ("fall_detected", "person_immobile"):
        assert make_payload(event_type=event_type, zone_id=None).zone_id is None
    with pytest.raises(ValidationError, match="zone_id may be null only"):
        make_payload(event_type="zone_intrusion", zone_id=None)


# ---------------------------------------------------------------------------- outgoing: invalid


def _box(**overrides: Any) -> dict[str, Any]:
    box: dict[str, Any] = {"track_id": 17, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46]}
    box["conf"] = 0.93
    box.update(overrides)
    return box


def _details(**overrides: Any) -> dict[str, Any]:
    details: dict[str, Any] = json.loads(EVENT_EXAMPLE)["details"]
    details.update(overrides)
    return details


INVALID_EVENTS: dict[str, dict[str, Any]] = {
    "unknown top-level field": event_dict(extra_field=1),
    "unknown details field": event_dict(details=_details(extra=1)),
    "unknown box field": event_dict(details=_details(boxes=[_box(extra=1)])),
    "unknown event type": event_dict(type="explosion"),
    "unknown severity": event_dict(severity="urgent"),
    "confidence above 1": event_dict(confidence=1.2),
    "confidence below 0": event_dict(confidence=-0.1),
    "confidence NaN": event_dict(confidence=float("nan")),
    "negative track id": event_dict(track_id=-1, external_id="cam-1-1762179667--1-ppe_violation"),
    "naive started_at": event_dict(started_at="2026-11-03T14:21:07"),
    "unparseable started_at": event_dict(started_at="yesterday"),
    "ended_at before started_at": event_dict(ended_at="2026-11-03T14:21:06Z"),
    "external_id of another camera": event_dict(external_id="cam-2-1762179667-17-ppe_violation"),
    "external_id of another track": event_dict(external_id="cam-1-1762179667-18-ppe_violation"),
    "external_id of another type": event_dict(external_id="cam-1-1762179667-17-fall_detected"),
    "external_id without seconds": event_dict(external_id="cam-1-x-17-ppe_violation"),
    "empty external_id": event_dict(external_id=""),
    "invalid camera id": event_dict(
        camera_id="cam 1", external_id="cam 1-1762179667-17-ppe_violation"
    ),
    "null zone for ppe_violation": event_dict(zone_id=None),
    "details of another type": event_dict(
        details={"duration_sec": 1.0, "boxes": [_box()], "frame_size": [1280, 720]}
    ),
    "empty missing_ppe": event_dict(details=_details(missing_ppe=[])),
    "unknown PPE code": event_dict(details=_details(missing_ppe=["cape"])),
    "no boxes": event_dict(details=_details(boxes=[])),
    "unknown box label": event_dict(details=_details(boxes=[_box(label="dog")])),
    "bbox outside 0..1": event_dict(details=_details(boxes=[_box(bbox=[1.2, 0.1, 0.1, 0.1])])),
    "bbox past the frame": event_dict(details=_details(boxes=[_box(bbox=[0.9, 0.1, 0.5, 0.1])])),
    "bbox with three values": event_dict(details=_details(boxes=[_box(bbox=[0.1, 0.1, 0.1])])),
    "negative bbox size": event_dict(details=_details(boxes=[_box(bbox=[0.1, 0.1, -0.1, 0.1])])),
    "zero frame width": event_dict(details=_details(frame_size=[0, 720])),
    "missing details": {k: v for k, v in event_dict().items() if k != "details"},
}


@pytest.mark.parametrize("case", sorted(INVALID_EVENTS))
def test_invalid_event_payloads_are_rejected(case: str) -> None:
    with pytest.raises(ValidationError):
        s.EventPayload.model_validate(INVALID_EVENTS[case])


def test_proximity_needs_two_boxes(make_payload: Any) -> None:
    payload = make_payload(event_type="vehicle_proximity")
    data = payload.model_dump()
    data["details"]["boxes"] = data["details"]["boxes"][:1]
    with pytest.raises(ValidationError):
        s.EventPayload.model_validate(data)


@pytest.mark.parametrize(
    ("model", "valid", "invalid"),
    [
        (s.ZoneIntrusionDetails, {"duration_sec": 2.0}, {"duration_sec": -1.0}),
        (s.FallDetectedDetails, {"pose_conf": 0.8}, {"pose_conf": 1.5}),
        (s.PersonImmobileDetails, {"immobile_sec": 30.0}, {"immobile_sec": -2.0}),
        (s.OvercrowdingDetails, {"count": 3, "max_persons": 2}, {"count": -1, "max_persons": 2}),
        (s.CustomDetails, {"title": "check"}, {"title": ""}),
        (
            s.VehicleProximityDetails,
            {"distance_m": 1.0, "person_track_id": 1, "vehicle_track_id": 2},
            {"distance_m": -1.0, "person_track_id": 1, "vehicle_track_id": 2},
        ),
    ],
    ids=["intrusion", "fall", "immobile", "overcrowding", "custom", "proximity"],
)
def test_details_variants_validate_their_fields(
    model: type[s.StrictModel], valid: dict[str, Any], invalid: dict[str, Any]
) -> None:
    common = {"boxes": [_box(), _box(track_id=3, label="forklift")], "frame_size": [1, 1]}
    assert model.model_validate({**valid, **common})
    with pytest.raises(ValidationError):
        model.model_validate({**invalid, **common})
    with pytest.raises(ValidationError):
        model.model_validate({**valid, **common, "unexpected": 1})


def test_outgoing_models_are_frozen(make_payload: Any) -> None:
    payload = make_payload()
    with pytest.raises(ValidationError):
        payload.confidence = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "data",
    [
        {"ended_at": "2026-11-03T14:21:19"},
        {"ended_at": "2026-11-03T14:21:19Z", "extra": 1},
        {},
    ],
    ids=["naive", "extra field", "missing"],
)
def test_invalid_patch_bodies_are_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        s.EventEndPatch.model_validate(data)


INVALID_HEARTBEATS: dict[str, dict[str, Any]] = {
    "unknown status": heartbeat_dict(status="offline"),
    "negative fps": heartbeat_dict(fps=-1),
    "empty model name": heartbeat_dict(model_name=""),
    "unknown field": heartbeat_dict(gpu=1),
    "unknown stream state": heartbeat_dict(extra={**heartbeat_dict()["extra"], "stream": "up"}),
    "unknown extra field": heartbeat_dict(extra={**heartbeat_dict()["extra"], "temp": 70}),
    "negative queue depth": heartbeat_dict(extra={**heartbeat_dict()["extra"], "queue_depth": -1}),
    "missing extra": {k: v for k, v in heartbeat_dict().items() if k != "extra"},
}


@pytest.mark.parametrize("case", sorted(INVALID_HEARTBEATS))
def test_invalid_heartbeats_are_rejected(case: str) -> None:
    with pytest.raises(ValidationError):
        s.HeartbeatPayload.model_validate(INVALID_HEARTBEATS[case])


def test_heartbeat_fps_is_rounded() -> None:
    payload = s.HeartbeatPayload.model_validate(heartbeat_dict(fps=11.8765))
    assert json.loads(payload.model_dump_json())["fps"] == 11.88


INVALID_STATS: dict[str, dict[str, Any]] = {
    "window not minute aligned": stats_dict(period_start="2026-11-03T14:21:05Z"),
    "end before start": stats_dict(period_end="2026-11-03T14:20:00Z"),
    "empty window": stats_dict(period_end="2026-11-03T14:21:00Z"),
    "more compliant than total": stats_dict(persons_compliant=43),
    "more compliant than checked": stats_dict(by_ppe={"helmet": {"checked": 1, "compliant": 2}}),
    "unknown PPE code": stats_dict(by_ppe={"cape": {"checked": 1, "compliant": 1}}),
    "negative total": stats_dict(persons_total=-1),
    "unknown field": stats_dict(persons_seen=1),
}


@pytest.mark.parametrize("case", sorted(INVALID_STATS))
def test_invalid_stats_are_rejected(case: str) -> None:
    with pytest.raises(ValidationError):
        s.StatsPayload.model_validate(INVALID_STATS[case])


def _detections(**object_overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(DETECTIONS_EXAMPLE)
    data["objects"][0].update(object_overrides)
    return data


INVALID_DETECTIONS: dict[str, dict[str, Any]] = {
    "unknown person state": _detections(state="panic"),
    "person without ppe": {
        **_detections(),
        "objects": [{k: v for k, v in _detections()["objects"][0].items() if k != "ppe"}],
    },
    "unknown PPE key": _detections(ppe={"cape": True}),
    "non-person, non-vehicle label": _detections(label="helmet"),
    "vehicle with person fields": {
        **_detections(),
        "objects": [{**json.loads(DETECTIONS_EXAMPLE)["objects"][1], "state": "ok"}],
    },
    "bbox outside 0..1": _detections(bbox=[0.5, 0.5, 0.6, 0.1]),
    "naive ts": {**_detections(), "ts": "2026-11-03T14:21:07"},
    "wrong type": {**_detections(), "type": "frame"},
    "negative frame id": {**_detections(), "frame_id": -1},
}


@pytest.mark.parametrize("case", sorted(INVALID_DETECTIONS))
def test_invalid_detections_are_rejected(case: str) -> None:
    with pytest.raises(ValidationError):
        s.DetectionsMessage.model_validate(INVALID_DETECTIONS[case])


@pytest.mark.parametrize("jpeg_base64", ["iVBORw0KGgoAAAANSUhEUg==", "/9j/", ""])
def test_frame_messages_must_carry_a_jpeg(jpeg_base64: str) -> None:
    data = json.loads(FRAME_EXAMPLE)
    data["jpeg_base64"] = jpeg_base64
    with pytest.raises(ValidationError):
        s.FrameMessage.model_validate(data)


# ---------------------------------------------------------------------------- incoming


def test_post_responses_parse() -> None:
    created = s.EventAccepted.model_validate_json(RESPONSE_201)
    assert (created.id, created.status, created.severity, created.deduplicated) == (
        "evt-42",
        "new",
        "medium",
        False,
    )
    assert created.model_dump_json() == compact(RESPONSE_201)
    deduplicated = s.EventAccepted.model_validate_json(RESPONSE_202)
    assert (deduplicated.id, deduplicated.deduplicated) == ("evt-41", True)
    assert json.loads(deduplicated.model_dump_json(exclude_unset=True)) == json.loads(RESPONSE_202)


def test_post_response_numeric_id_becomes_text_and_unknown_fields_are_ignored() -> None:
    accepted = s.EventAccepted.model_validate({"id": 7, "status": "new", "extra": "x"})
    assert accepted.id == "7"
    with pytest.raises(ValidationError):
        s.EventAccepted.model_validate({"id": True})
    with pytest.raises(ValidationError):
        s.EventAccepted.model_validate({"status": "new"})


def test_config_example_parses_and_keeps_every_value() -> None:
    config = s.CameraConfig.model_validate_json(CONFIG_EXAMPLE)
    assert config.camera.id == "cam-1"
    assert config.version == 42
    assert config.zones[0].rules.required_ppe == ["helmet", "glasses"]
    assert config.zones[0].enabled is True
    assert config.updated_at == datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    # Lenient incoming model: values are kept, floats normalized (60 -> 60.0).
    dumped = json.loads(config.model_dump_json(by_alias=True, exclude_unset=True))
    assert dumped == json.loads(CONFIG_EXAMPLE)
    assert s.CameraConfig.model_validate_json(config.model_dump_json(by_alias=True)) == config


def test_schedule_example_round_trips_byte_for_byte() -> None:
    window = s.ScheduleWindow.model_validate_json(SCHEDULE_EXAMPLE)
    assert (window.start_minute, window.end_minute) == (8 * 60, 20 * 60)
    assert window.model_dump_json(by_alias=True) == compact(SCHEDULE_EXAMPLE)


@pytest.mark.parametrize(
    ("example", "model"),
    [
        (CONFIG_UPDATED_EXAMPLE, s.ConfigUpdatedMessage),
        (SERVER_ERROR_EXAMPLE, s.ServerErrorMessage),
    ],
    ids=["config.updated", "error"],
)
def test_server_messages_round_trip_byte_for_byte(example: str, model: type[Any]) -> None:
    message = s.parse_server_message(json.loads(example))
    assert isinstance(message, model)
    assert message.model_dump_json() == compact(example)


def test_server_error_offending_id() -> None:
    plain = s.parse_server_message(json.loads(SERVER_ERROR_EXAMPLE))
    assert isinstance(plain, s.ServerErrorMessage)
    assert plain.offending_id() is None
    for key in ("id", "message_id", "msg_id", "ref", "frame_id", "external_id"):
        message = s.parse_server_message({"type": "error", "message": "bad", key: 12})
        assert isinstance(message, s.ServerErrorMessage)
        assert message.offending_id() == "12"
    nulled = s.parse_server_message({"type": "error", "message": "bad", "id": None})
    assert isinstance(nulled, s.ServerErrorMessage) and nulled.offending_id() is None


@pytest.mark.parametrize("data", [None, [], "config.updated", {"type": "hello"}, {}])
def test_unknown_server_messages_return_none(data: Any) -> None:
    assert s.parse_server_message(data) is None


def test_invalid_config_updated_is_rejected() -> None:
    with pytest.raises(ValidationError):
        s.parse_server_message({"type": "config.updated", "version": "forty-three"})


def test_config_updated_without_camera_means_all() -> None:
    message = s.parse_server_message({"type": "config.updated"})
    assert isinstance(message, s.ConfigUpdatedMessage)
    assert (message.camera_id, message.version) == (None, None)


def test_unknown_ppe_codes_are_dropped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    data = config_dict()
    data["ppe_types"] = ["helmet", "jetpack"]
    data["zones"][0]["rules"]["required_ppe"] = ["helmet", "cape"]
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.schemas"):
        config = s.CameraConfig.model_validate(data)
    assert config.ppe_types == ["helmet"]
    assert config.zones[0].rules.required_ppe == ["helmet"]
    assert "jetpack" in caplog.text and "cape" in caplog.text


def test_config_defaults_and_unknown_fields() -> None:
    config = s.CameraConfig.model_validate(
        {
            "camera": {"id": "cam-9"},
            "brand_new_field": {"x": 1},
            "updated_at": "2026-11-03T12:00:00",
        }
    )
    assert config.zones == [] and config.version == 0
    assert config.ppe_types == list(s.PPE_CODE_VALUES)
    assert config.updated_at is not None and config.updated_at.tzinfo == timezone.utc
    assert config.thresholds.confirm_frames is None


def _config_with_zone(**zone_overrides: Any) -> dict[str, Any]:
    data = config_dict()
    data["zones"][0].update(zone_overrides)
    return data


def _config_with_rules(**rule_overrides: Any) -> dict[str, Any]:
    data = config_dict()
    data["zones"][0]["rules"].update(rule_overrides)
    return data


INVALID_CONFIGS: dict[str, dict[str, Any]] = {
    "polygon point outside 0..1": _config_with_zone(polygon=[[0.1, 0.1], [1.5, 0.1], [0.5, 0.5]]),
    "polygon with two points": _config_with_zone(polygon=[[0.1, 0.1], [0.5, 0.5]]),
    "polygon not a list": _config_with_zone(polygon="square"),
    "polygon point with three values": _config_with_zone(polygon=[[0.1, 0.1, 0.1]] * 3),
    "duplicate zone ids": {**config_dict(), "zones": config_dict()["zones"] * 2},
    "schedule hour 25": _config_with_rules(
        schedule=[{"days": [1], "from": "25:00", "to": "26:00"}]
    ),
    "schedule 24:30": _config_with_rules(schedule=[{"days": [1], "from": "08:00", "to": "24:30"}]),
    "schedule not HH:MM": _config_with_rules(schedule=[{"days": [1], "from": "8am", "to": "9"}]),
    "schedule day 0": _config_with_rules(schedule=[{"days": [0], "from": "08:00", "to": "09:00"}]),
    "schedule without days": _config_with_rules(
        schedule=[{"days": [], "from": "08:00", "to": "09:00"}]
    ),
    "zero vehicle distance": _config_with_rules(min_vehicle_distance_m=0),
    "negative max persons": _config_with_rules(max_persons=-1),
    "zero confirm frames": _config_with_rules(confirm_frames=0),
    "unknown severity override": _config_with_rules(severity_override="extreme"),
    "min_confidence above 1": {**config_dict(), "thresholds": {"min_confidence": 2}},
    "invalid camera id": {**config_dict(), "camera": {"id": ""}},
    "missing camera": {k: v for k, v in config_dict().items() if k != "camera"},
}


@pytest.mark.parametrize("case", sorted(INVALID_CONFIGS))
def test_invalid_configs_are_rejected(case: str) -> None:
    with pytest.raises(ValidationError):
        s.CameraConfig.model_validate(INVALID_CONFIGS[case])


def test_polygon_points_within_tolerance_are_clamped() -> None:
    config = s.CameraConfig.model_validate(
        _config_with_zone(polygon=[[-0.0000001, 0.0], [1.0000001, 0.0], [0.5, 1.0]])
    )
    assert config.zones[0].polygon == [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]


def test_schedule_window_past_midnight_and_24_00() -> None:
    late = s.ScheduleWindow.model_validate({"days": [5], "from": "22:00", "to": "06:00"})
    assert (late.start_minute, late.end_minute) == (22 * 60, 6 * 60)
    day = s.ScheduleWindow.model_validate({"days": [7], "from": "00:00", "to": "24:00"})
    assert day.end_minute == 24 * 60


def test_parse_config_list_accepts_arrays_wrappers_and_single_objects() -> None:
    one = config_dict()
    assert [c.camera.id for c in s.parse_config_list([one])] == ["cam-1"]
    for key in ("cameras", "configs", "items", "data"):
        assert [c.camera.id for c in s.parse_config_list({key: [one]})] == ["cam-1"]
    assert [c.camera.id for c in s.parse_config_list(one)] == ["cam-1"]
    assert s.parse_config_list([]) == []


@pytest.mark.parametrize("data", ["cameras", 7, {"unrelated": []}, None])
def test_parse_config_list_rejects_other_shapes(data: Any) -> None:
    with pytest.raises(ValueError, match="JSON array"):
        s.parse_config_list(data)


def test_parse_config_list_rejects_invalid_entries() -> None:
    with pytest.raises(ValidationError):
        s.parse_config_list([{"camera": {"id": ""}}])
