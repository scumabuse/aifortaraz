"""Strict validation models of the mock: contract examples pass, every rule has a failing case."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pytest

from tools import mock_models as m


def _expect_error(func: Callable[[], object], fragment: str) -> m.ContractError:
    with pytest.raises(m.ContractError) as info:
        func()
    joined = " | ".join(info.value.errors)
    assert fragment in joined, joined
    return info.value


# --------------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-11-03T14:21:07Z", datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc)),
        ("2026-11-03T14:21:07.250Z", datetime(2026, 11, 3, 14, 21, 7, 250000, tzinfo=timezone.utc)),
        ("2026-11-03T14:21:07.5Z", datetime(2026, 11, 3, 14, 21, 7, 500000, tzinfo=timezone.utc)),
        (
            "2026-11-03T14:21:07.123456789Z",
            datetime(2026, 11, 3, 14, 21, 7, 123456, tzinfo=timezone.utc),
        ),
    ],
)
def test_parse_z_timestamp_accepts_utc_with_z(raw: str, expected: datetime) -> None:
    assert m.parse_z_timestamp(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "2026-11-03T14:21:07",
        "2026-11-03T14:21:07+00:00",
        "2026-11-03 14:21:07Z",
        "2026-11-03T14:21:07z",
        "2026-11-03T14:21Z",
        "2026-11-03T14:21:07.Z",
        "2026-11-03T14:21:07Z\n",
        "2026-02-30T14:21:07Z",
        "2026-11-03T25:00:00Z",
        "٢026-11-03T14:21:07Z",
        1762179667,
        None,
    ],
)
def test_parse_z_timestamp_rejects_everything_else(raw: object) -> None:
    with pytest.raises(ValueError):
        m.parse_z_timestamp(raw)


# --------------------------------------------------------------------------------------------
# 4.1 events: the contract example and one failing case per rule
# --------------------------------------------------------------------------------------------


def test_contract_event_example_is_accepted(payloads: Any, jpeg: Any) -> None:
    validated = m.validate_event_submission(payloads.contract_event(), jpeg(1280, 720))
    assert isinstance(validated.event, m.PpeViolationEvent)
    assert validated.event.external_id == "cam-1-1762179667-17-ppe_violation"
    assert validated.snapshot_size == (1280, 720)
    assert validated.payload == payloads.contract_event()


@pytest.mark.parametrize("event_type", list(m.EVENT_TYPES))
def test_every_event_type_is_accepted(payloads: Any, event_type: str) -> None:
    zone = None if event_type in ("fall_detected", "person_immobile") else "zone-a"
    event = m.validate_event_payload(payloads.event(event_type, zone_id=zone))
    assert event.type == event_type


def test_custom_event_may_have_no_boxes(payloads: Any) -> None:
    payload = payloads.event("custom")
    payload["details"]["boxes"] = []
    assert m.validate_event_payload(payload).type == "custom"


def test_ended_at_equal_to_started_at_is_accepted(payloads: Any) -> None:
    payload = payloads.event(ended_at="2026-11-03T14:21:07Z")
    assert m.validate_event_payload(payload).ended_at == m.parse_z_timestamp("2026-11-03T14:21:07Z")


def test_bbox_within_tolerance_is_accepted(payloads: Any) -> None:
    payload = payloads.event()
    payload["details"]["boxes"][0]["bbox"] = [0.5, 0.5, 0.5005, 0.5009]
    m.validate_event_payload(payload)


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    keys = path.split(".")

    def mutate(payload: dict[str, Any]) -> None:
        target: Any = payload
        for key in keys[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        last = keys[-1]
        if isinstance(target, list):
            target[int(last)] = value
        else:
            target[last] = value

    return mutate


def _delete(path: str) -> Callable[[dict[str, Any]], None]:
    keys = path.split(".")

    def mutate(payload: dict[str, Any]) -> None:
        target: Any = payload
        for key in keys[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        del target[keys[-1]]

    return mutate


EVENT_RULES: list[tuple[str, str, Callable[[dict[str, Any]], None], str]] = [
    ("type enum", "ppe_violation", _set("type", "fire"), "does not match any of the expected"),
    ("severity enum", "ppe_violation", _set("severity", "urgent"), "severity"),
    ("zone_id null", "ppe_violation", _set("zone_id", None), "zone_id may be null only"),
    ("zone_id empty", "zone_intrusion", _set("zone_id", ""), "zone_id"),
    ("track_id negative", "ppe_violation", _set("track_id", -1), "track_id"),
    ("track_id string", "ppe_violation", _set("track_id", "17"), "track_id"),
    ("track_id bool", "ppe_violation", _set("track_id", True), "track_id"),
    ("confidence > 1", "ppe_violation", _set("confidence", 1.2), "confidence"),
    ("confidence < 0", "ppe_violation", _set("confidence", -0.1), "confidence"),
    ("confidence string", "ppe_violation", _set("confidence", "0.9"), "confidence"),
    (
        "external_id camera",
        "ppe_violation",
        _set("external_id", "cam-2-1762179667-17-ppe_violation"),
        "must match",
    ),
    (
        "external_id track",
        "ppe_violation",
        _set("external_id", "cam-1-1762179667-18-ppe_violation"),
        "must match",
    ),
    (
        "external_id type",
        "ppe_violation",
        _set("external_id", "cam-1-1762179667-17-zone_intrusion"),
        "must match",
    ),
    (
        "external_id seconds",
        "ppe_violation",
        _set("external_id", "cam-1-2026-11-03T14:21:07-17-ppe_violation"),
        "must match",
    ),
    (
        "external_id suffix",
        "ppe_violation",
        _set("external_id", "cam-1-1762179667-17-ppe_violation-2"),
        "must match",
    ),
    (
        "started_at without Z",
        "ppe_violation",
        _set("started_at", "2026-11-03T14:21:07"),
        "trailing Z",
    ),
    (
        "started_at offset",
        "ppe_violation",
        _set("started_at", "2026-11-03T14:21:07+00:00"),
        "trailing Z",
    ),
    ("started_at null", "ppe_violation", _set("started_at", None), "started_at"),
    ("ended_at format", "ppe_violation", _set("ended_at", "2026-11-03 14:21:19Z"), "ended_at"),
    (
        "ended_at before started_at",
        "ppe_violation",
        _set("ended_at", "2026-11-03T14:21:06Z"),
        "ended_at must not be earlier",
    ),
    ("missing key", "ppe_violation", _delete("severity"), "severity: Field required"),
    ("missing ended_at key", "ppe_violation", _delete("ended_at"), "ended_at: Field required"),
    ("unknown key", "ppe_violation", _set("priority", 1), "priority: Extra inputs"),
    ("details extra key", "ppe_violation", _set("details.note", "x"), "note: Extra inputs"),
    ("missing_ppe missing", "ppe_violation", _delete("details.missing_ppe"), "missing_ppe"),
    ("missing_ppe empty", "ppe_violation", _set("details.missing_ppe", []), "missing_ppe"),
    ("missing_ppe code", "ppe_violation", _set("details.missing_ppe", ["hat"]), "missing_ppe"),
    ("present_ppe code", "ppe_violation", _set("details.present_ppe", ["Vest"]), "present_ppe"),
    (
        "details of another type",
        "zone_intrusion",
        _set("details.missing_ppe", ["helmet"]),
        "missing_ppe: Extra inputs",
    ),
    ("duration_sec missing", "zone_intrusion", _delete("details.duration_sec"), "duration_sec"),
    ("duration_sec negative", "zone_intrusion", _set("details.duration_sec", -1), "duration_sec"),
    ("distance_m negative", "vehicle_proximity", _set("details.distance_m", -0.5), "distance_m"),
    (
        "vehicle_track_id missing",
        "vehicle_proximity",
        _delete("details.vehicle_track_id"),
        "vehicle_track_id",
    ),
    (
        "proximity track_id is the person's",
        "vehicle_proximity",
        _set("details.person_track_id", 99),
        "must equal details.person_track_id",
    ),
    (
        "proximity needs a vehicle box",
        "vehicle_proximity",
        _set("details.boxes.1.label", "person"),
        "box of the vehicle involved",
    ),
    ("pose_conf > 1", "fall_detected", _set("details.pose_conf", 1.5), "pose_conf"),
    ("immobile_sec negative", "person_immobile", _set("details.immobile_sec", -3), "immobile_sec"),
    ("count float", "overcrowding", _set("details.count", 3.0), "count"),
    ("max_persons missing", "overcrowding", _delete("details.max_persons"), "max_persons"),
    ("title empty", "custom", _set("details.title", ""), "title"),
    ("boxes missing", "ppe_violation", _delete("details.boxes"), "boxes: Field required"),
    (
        "person box required",
        "ppe_violation",
        _set("details.boxes.0.label", "helmet"),
        "box of the person involved",
    ),
    ("box label", "ppe_violation", _set("details.boxes.0.label", "worker"), "label"),
    ("box track_id null", "ppe_violation", _set("details.boxes.0.track_id", None), "track_id"),
    ("box conf > 1", "ppe_violation", _set("details.boxes.0.conf", 1.01), "conf"),
    ("bbox 3 values", "ppe_violation", _set("details.boxes.0.bbox", [0.1, 0.1, 0.2]), "bbox"),
    ("bbox value > 1", "ppe_violation", _set("details.boxes.0.bbox", [1.1, 0, 0.1, 0.1]), "bbox"),
    ("bbox negative", "ppe_violation", _set("details.boxes.0.bbox", [-0.1, 0, 0.1, 0.1]), "bbox"),
    (
        "bbox x + w",
        "ppe_violation",
        _set("details.boxes.0.bbox", [0.9, 0.1, 0.2, 0.1]),
        "x + w = 1.1000 exceeds 1",
    ),
    (
        "bbox y + h",
        "ppe_violation",
        _set("details.boxes.0.bbox", [0.1, 0.5, 0.1, 0.502]),
        "y + h = 1.0020 exceeds 1",
    ),
    ("bbox pixels", "ppe_violation", _set("details.boxes.0.bbox", [410, 220, 120, 460]), "bbox"),
    ("frame_size 3", "ppe_violation", _set("details.frame_size", [1280, 720, 3]), "frame_size"),
    ("frame_size zero", "ppe_violation", _set("details.frame_size", [0, 720]), "frame_size"),
    ("frame_size float", "ppe_violation", _set("details.frame_size", [1280.0, 720]), "frame_size"),
    ("frame_size missing", "ppe_violation", _delete("details.frame_size"), "frame_size"),
]


@pytest.mark.parametrize(
    ("event_type", "mutate", "fragment"),
    [(event_type, mutate, fragment) for _, event_type, mutate, fragment in EVENT_RULES],
    ids=[rule_id for rule_id, *_ in EVENT_RULES],
)
def test_event_rule_violations_are_rejected(
    payloads: Any, event_type: str, mutate: Callable[[dict[str, Any]], None], fragment: str
) -> None:
    payload = payloads.event(event_type)
    mutate(payload)
    _expect_error(lambda: m.validate_event_payload(payload), fragment)


@pytest.mark.parametrize("event_type", ["fall_detected", "person_immobile"])
def test_zone_id_may_be_null_for_fall_and_immobile(payloads: Any, event_type: str) -> None:
    assert m.validate_event_payload(payloads.event(event_type, zone_id=None)).zone_id is None


def test_nan_confidence_is_rejected(payloads: Any) -> None:
    payload = json.loads(json.dumps(payloads.event()).replace("0.91", "NaN"))
    _expect_error(lambda: m.validate_event_payload(payload), "confidence")


def test_event_payload_must_be_an_object() -> None:
    _expect_error(lambda: m.validate_event_payload(["not", "an", "object"]), "JSON object")


def test_external_id_matches_digits_only() -> None:
    assert m.external_id_matches("cam-1-1762179667-17-ppe_violation", "cam-1", 17, "ppe_violation")
    assert m.external_id_matches("cam.1:a-0-0-custom", "cam.1:a", 0, "custom")
    assert not m.external_id_matches("cam-1--17-ppe_violation", "cam-1", 17, "ppe_violation")
    assert not m.external_id_matches("cam-1-12a-17-ppe_violation", "cam-1", 17, "ppe_violation")
    assert not m.external_id_matches("xcam-1-1-17-ppe_violation", "cam-1", 17, "ppe_violation")


# --------------------------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------------------------


def test_snapshot_must_be_present(payloads: Any) -> None:
    _expect_error(lambda: m.validate_event_submission(payloads.event(), None), "snapshot: required")


def test_snapshot_magic_bytes(payloads: Any) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    _expect_error(
        lambda: m.validate_event_submission(payloads.event(), png), "magic bytes FF D8 FF missing"
    )


def test_snapshot_truncated_jpeg_is_rejected(payloads: Any, jpeg: Any) -> None:
    data = jpeg(640, 360)
    _expect_error(
        lambda: m.validate_event_submission(payloads.event(), data[: len(data) // 2]), "snapshot:"
    )


def test_snapshot_longest_side_at_most_1280(payloads: Any, jpeg: Any) -> None:
    payload = payloads.event()
    payload["details"]["frame_size"] = [1920, 1080]
    _expect_error(
        lambda: m.validate_event_submission(payload, jpeg(1920, 1080)),
        "1920x1080, the longest side must be <= 1280 px",
    )


def test_snapshot_must_keep_the_frame_aspect_ratio(payloads: Any, jpeg: Any) -> None:
    _expect_error(
        lambda: m.validate_event_submission(payloads.event(), jpeg(640, 640)), "aspect ratio"
    )


def test_snapshot_cannot_be_larger_than_the_original_frame(payloads: Any, jpeg: Any) -> None:
    payload = payloads.event()
    payload["details"]["frame_size"] = [640, 360]
    _expect_error(
        lambda: m.validate_event_submission(payload, jpeg(1280, 720)), "larger than details"
    )


def test_downscaled_snapshot_of_a_larger_frame_is_accepted(payloads: Any, jpeg: Any) -> None:
    payload = payloads.event()
    payload["details"]["frame_size"] = [1920, 1080]
    assert m.validate_event_submission(payload, jpeg(1280, 720)).snapshot_size == (1280, 720)


def test_payload_and_snapshot_errors_are_reported_together(payloads: Any) -> None:
    payload = payloads.event(severity="urgent")
    error = _expect_error(lambda: m.validate_event_submission(payload, b"GIF89a"), "severity")
    assert any("magic bytes" in line for line in error.errors)


def test_inspect_jpeg_rejects_non_jpeg_images() -> None:
    disguised = bytes((0xFF, 0xD8, 0xFF)) + b"definitely not a jpeg body" * 4
    _expect_error(lambda: m.inspect_jpeg(disguised, max_side=1280, what="snapshot"), "snapshot:")


def test_decode_base64_is_strict() -> None:
    assert m.decode_base64(base64.b64encode(b"abc").decode(), what="x") == b"abc"
    _expect_error(lambda: m.decode_base64("data:image/jpeg;base64,AAAA", what="x"), "x:")
    _expect_error(lambda: m.decode_base64("YWJj\nZA==", what="x"), "x:")


# --------------------------------------------------------------------------------------------
# PATCH, 4.2 heartbeat, 4.3 stats
# --------------------------------------------------------------------------------------------


def test_patch_body_is_exactly_ended_at() -> None:
    assert m.validate_patch({"ended_at": "2026-11-03T14:21:19Z"}).ended_at == datetime(
        2026, 11, 3, 14, 21, 19, tzinfo=timezone.utc
    )
    _expect_error(lambda: m.validate_patch({}), "ended_at: Field required")
    _expect_error(
        lambda: m.validate_patch({"ended_at": "2026-11-03T14:21:19Z", "reason": "x"}),
        "reason: Extra inputs",
    )
    _expect_error(lambda: m.validate_patch({"ended_at": "2026-11-03T14:21:19"}), "trailing Z")
    _expect_error(lambda: m.validate_patch({"ended_at": None}), "ended_at")


def test_contract_heartbeat_example_is_accepted(payloads: Any) -> None:
    heartbeat = m.validate_heartbeat(payloads.heartbeat())
    assert heartbeat.extra.model_family is None


def test_heartbeat_model_family_is_optional(payloads: Any) -> None:
    payload = payloads.heartbeat()
    payload["extra"]["model_family"] = "yolo26"
    assert m.validate_heartbeat(payload).extra.model_family == "yolo26"


HEARTBEAT_RULES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("status enum", _set("status", "offline"), "status"),
    ("fps negative", _set("fps", -1.0), "fps"),
    ("fps string", _set("fps", "11.8"), "fps"),
    ("model_name empty", _set("model_name", ""), "model_name"),
    ("model_version missing", _delete("model_version"), "model_version"),
    ("camera_id missing", _delete("camera_id"), "camera_id"),
    ("top-level extra key", _set("uptime", 5), "uptime: Extra inputs"),
    ("extra unknown key", _set("extra.gpu_temp", 61), "gpu_temp: Extra inputs"),
    ("extra device missing", _delete("extra.device"), "device"),
    ("extra backend missing", _delete("extra.backend"), "backend"),
    ("queue_depth negative", _set("extra.queue_depth", -1), "queue_depth"),
    ("frames_dropped float", _set("extra.frames_dropped", 1.5), "frames_dropped"),
    ("stream missing", _delete("extra.stream"), "stream"),
    ("model_family null", _set("extra.model_family", None), "model_family"),
]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [(mutate, fragment) for _, mutate, fragment in HEARTBEAT_RULES],
    ids=[rule_id for rule_id, *_ in HEARTBEAT_RULES],
)
def test_heartbeat_rule_violations_are_rejected(
    payloads: Any, mutate: Callable[[dict[str, Any]], None], fragment: str
) -> None:
    payload = payloads.heartbeat()
    mutate(payload)
    _expect_error(lambda: m.validate_heartbeat(payload), fragment)


def test_contract_stats_example_is_accepted(payloads: Any) -> None:
    stats = m.validate_stats(payloads.stats())
    assert stats.by_ppe["vest"].compliant == 41


STATS_RULES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("period_start Z", _set("period_start", "2026-11-03T14:21:00"), "trailing Z"),
    ("period_end Z", _set("period_end", "2026-11-03T14:22:00+00:00"), "trailing Z"),
    ("start aligned", _set("period_start", "2026-11-03T14:21:30Z"), "aligned to a whole minute"),
    ("end aligned", _set("period_end", "2026-11-03T14:22:00.500Z"), "aligned to a whole minute"),
    ("end after start", _set("period_end", "2026-11-03T14:21:00Z"), "period_end must be after"),
    ("compliant <= total", _set("persons_compliant", 43), "exceeds persons_total"),
    ("total negative", _set("persons_total", -1), "persons_total"),
    ("by_ppe code", _set("by_ppe.hardhat", {"checked": 1, "compliant": 1}), "by_ppe.hardhat"),
    ("checked int", _set("by_ppe.helmet.checked", 4.5), "checked"),
    ("compliant <= checked", _set("by_ppe.helmet.compliant", 43), "exceeds checked"),
    ("checked <= total", _set("by_ppe.vest", {"checked": 50, "compliant": 1}), "exceeds persons"),
    ("by_ppe extra key", _set("by_ppe.helmet.missing", 2), "missing: Extra inputs"),
    ("missing by_ppe", _delete("by_ppe"), "by_ppe: Field required"),
]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [(mutate, fragment) for _, mutate, fragment in STATS_RULES],
    ids=[rule_id for rule_id, *_ in STATS_RULES],
)
def test_stats_rule_violations_are_rejected(
    payloads: Any, mutate: Callable[[dict[str, Any]], None], fragment: str
) -> None:
    payload = payloads.stats()
    mutate(payload)
    _expect_error(lambda: m.validate_stats(payload), fragment)


# --------------------------------------------------------------------------------------------
# 4.5 WebSocket messages
# --------------------------------------------------------------------------------------------


def test_contract_detections_example_is_accepted(payloads: Any) -> None:
    analysis = m.analyse_ws_message(json.dumps(payloads.detections()), None)
    assert analysis.valid, analysis.errors
    assert analysis.kind == "detections"
    assert analysis.camera_id == "cam-1"
    assert isinstance(analysis.message, m.DetectionsMessageModel)


def test_contract_frame_example_is_accepted(payloads: Any) -> None:
    analysis = m.analyse_ws_message(json.dumps(payloads.frame()), None)
    assert analysis.valid, analysis.errors
    assert analysis.jpeg_size == (960, 540)
    assert analysis.jpeg is not None and analysis.jpeg.startswith(m.JPEG_MAGIC)
    assert isinstance(analysis.record, dict)
    assert analysis.record["jpeg_base64"].endswith("base64 chars>")


def test_contract_ws_heartbeat_is_accepted(payloads: Any) -> None:
    analysis = m.analyse_ws_message(json.dumps(payloads.ws_heartbeat()), None)
    assert analysis.valid, analysis.errors
    assert isinstance(analysis.message, m.HeartbeatMessageModel)


WS_RULES: list[tuple[str, str, Callable[[dict[str, Any]], None], str]] = [
    ("person without ppe", "detections", _delete("objects.0.ppe"), "ppe: Field required"),
    ("person without zones", "detections", _delete("objects.0.zones"), "zones: Field required"),
    ("person without state", "detections", _delete("objects.0.state"), "state: Field required"),
    ("person state enum", "detections", _set("objects.0.state", "warning"), "state"),
    ("ppe value", "detections", _set("objects.0.ppe", {"helmet": "no"}), "helmet"),
    ("ppe key", "detections", _set("objects.0.ppe", {"hardhat": True}), "hardhat"),
    ("zones items", "detections", _set("objects.0.zones", [7]), "zones.0"),
    (
        "ppe outside zones",
        "detections",
        _set("objects.0.zones", []),
        "ppe must be empty when the person is in no zone",
    ),
    ("vehicle with ppe", "detections", _set("objects.1.ppe", {}), "ppe: Extra inputs"),
    ("vehicle with zones", "detections", _set("objects.1.zones", ["z"]), "zones: Extra inputs"),
    ("vehicle with state", "detections", _set("objects.1.state", "ok"), "state: Extra inputs"),
    ("object label", "detections", _set("objects.1.label", "helmet"), "does not match"),
    ("object bbox", "detections", _set("objects.0.bbox", [0.9, 0.9, 0.2, 0.2]), "exceeds 1"),
    ("object conf", "detections", _set("objects.1.conf", 2), "conf"),
    ("ts Z", "detections", _set("ts", "2026-11-03T14:21:07.250"), "trailing Z"),
    ("frame_id negative", "detections", _set("frame_id", -1), "frame_id"),
    ("objects missing", "detections", _delete("objects"), "objects: Field required"),
    ("camera_id slash", "detections", _set("camera_id", "cam/1"), "camera_id"),
    ("detections extra", "detections", _set("fps", 12), "fps: Extra inputs"),
    ("frame ts", "frame", _set("ts", "yesterday"), "trailing Z"),
    ("frame base64", "frame", _set("jpeg_base64", "not base64!"), "jpeg_base64"),
    ("frame not jpeg", "frame", _set("jpeg_base64", "R0lGODlhAQABAAAAACw="), "magic bytes"),
    ("frame extra", "frame", _set("width", 960), "width: Extra inputs"),
    ("heartbeat extra key", "heartbeat", _set("extra.gpu", "x"), "gpu: Extra inputs"),
    ("heartbeat status", "heartbeat", _set("status", "down"), "status"),
]


@pytest.mark.parametrize(
    ("kind", "mutate", "fragment"),
    [(kind, mutate, fragment) for _, kind, mutate, fragment in WS_RULES],
    ids=[rule_id for rule_id, *_ in WS_RULES],
)
def test_ws_rule_violations_are_rejected(
    payloads: Any, kind: str, mutate: Callable[[dict[str, Any]], None], fragment: str
) -> None:
    builders = {
        "detections": payloads.detections,
        "frame": payloads.frame,
        "heartbeat": payloads.ws_heartbeat,
    }
    message = builders[kind]()
    mutate(message)
    analysis = m.analyse_ws_message(json.dumps(message), None)
    assert not analysis.valid
    assert analysis.kind == kind
    assert analysis.message is None
    assert any(fragment in error for error in analysis.errors), analysis.errors


def test_ws_frame_longest_side_at_most_960(payloads: Any) -> None:
    analysis = m.analyse_ws_message(json.dumps(payloads.frame(width=1280, height=720)), None)
    assert not analysis.valid
    assert any("longest side must be <= 960 px" in error for error in analysis.errors)


@pytest.mark.parametrize(
    ("text", "data", "kind", "fragment"),
    [
        (None, b"\x00\x01", "malformed", "binary frame of 2 bytes"),
        ("{not json", None, "malformed", "<message>:"),
        ("[1, 2]", None, "malformed", "got list"),
        ('{"camera_id": "cam-1"}', None, "unknown", "type: None is not one of"),
        ('{"type": "events", "camera_id": "cam-1"}', None, "unknown", "'events' is not one of"),
    ],
)
def test_ws_malformed_and_unknown_messages(
    text: str | None, data: bytes | None, kind: str, fragment: str
) -> None:
    analysis = m.analyse_ws_message(text, data)
    assert analysis.kind == kind
    assert not analysis.valid
    assert any(fragment in error for error in analysis.errors), analysis.errors


def test_person_outside_zones_with_empty_ppe_is_valid(payloads: Any) -> None:
    message = payloads.detections()
    message["objects"][0].update(zones=[], ppe={}, state="ok")
    assert m.analyse_ws_message(json.dumps(message), None).valid


# --------------------------------------------------------------------------------------------
# 4.4 configuration models
# --------------------------------------------------------------------------------------------


def _schedule(days: list[int], start: object, end: object) -> list[dict[str, object]]:
    return [{"days": days, "from": start, "to": end}]


def test_contract_config_example_is_valid(payloads: Any) -> None:
    config = m.validate_camera_config(payloads.config())
    assert config.version == 42
    assert config.zones[0].rules.required_ppe == ["helmet", "glasses"]


def test_config_serializes_to_the_contract_structure(payloads: Any) -> None:
    raw = payloads.config()
    raw["zones"][0]["rules"]["schedule"] = _schedule([1, 2, 3, 4, 5], "08:00", "20:00")
    config = m.validate_camera_config(raw)
    data = m.camera_config_to_contract(config, 43, "2026-11-03T12:30:00Z")
    assert data["version"] == 43
    assert data["updated_at"] == "2026-11-03T12:30:00Z"
    assert set(data) == {
        "camera",
        "frame_size_hint",
        "thresholds",
        "ppe_types",
        "zones",
        "version",
        "updated_at",
    }
    zone = data["zones"][0]
    assert "enabled" not in zone
    assert zone["rules"]["schedule"] == [{"days": [1, 2, 3, 4, 5], "from": "08:00", "to": "20:00"}]
    assert set(zone["rules"]) == {
        "required_ppe",
        "entry_forbidden",
        "max_persons",
        "min_vehicle_distance_m",
        "schedule",
        "confirm_frames",
        "cooldown_sec",
        "severity_override",
    }


def test_zone_enabled_is_served_only_when_configured(payloads: Any) -> None:
    raw = payloads.config()
    raw["zones"][0]["enabled"] = False
    data = m.camera_config_to_contract(m.validate_camera_config(raw), 1, "2026-11-03T12:00:00Z")
    assert data["zones"][0]["enabled"] is False


CONFIG_RULES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("zone type", _set("zones.0.type", "parking"), "zones.0.type"),
    ("polygon points", _set("zones.0.polygon", [[0.1, 0.1], [0.2, 0.2]]), "polygon"),
    ("polygon range", _set("zones.0.polygon.0", [1.2, 0.1]), "polygon"),
    ("schedule time number", _set("zones.0.rules.schedule", _schedule([1], 480, "20:00")), "from"),
    (
        "schedule time range",
        _set("zones.0.rules.schedule", _schedule([1], "08:00", "25:00")),
        "outside 00:00..24:00",
    ),
    ("schedule day", _set("zones.0.rules.schedule", _schedule([0], "08:00", "20:00")), "days"),
    ("rule ppe code", _set("zones.0.rules.required_ppe", ["hardhat"]), "required_ppe"),
    ("rule extra", _set("zones.0.rules.max_speed", 3), "max_speed: Extra inputs"),
    ("severity override", _set("zones.0.rules.severity_override", "extreme"), "severity_override"),
    ("distance positive", _set("zones.0.rules.min_vehicle_distance_m", 0), "min_vehicle_dist"),
    ("confirm positive", _set("thresholds.confirm_frames", 0), "confirm_frames"),
    ("camera id", _delete("camera.id"), "camera.id"),
    ("duplicate zones", lambda c: c["zones"].append(dict(c["zones"][0])), "duplicate zone ids"),
    ("updated_at Z", _set("updated_at", "2026-11-03 12:00"), "trailing Z"),
]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [(mutate, fragment) for _, mutate, fragment in CONFIG_RULES],
    ids=[rule_id for rule_id, *_ in CONFIG_RULES],
)
def test_config_rule_violations_are_rejected(
    payloads: Any, mutate: Callable[[dict[str, Any]], None], fragment: str
) -> None:
    config = payloads.config()
    mutate(config)
    _expect_error(lambda: m.validate_camera_config(config), fragment)


def test_config_file_model_requires_unique_cameras(payloads: Any) -> None:
    with pytest.raises(ValueError, match="duplicate camera ids"):
        m.MockConfigFileModel.model_validate({"cameras": [payloads.config(), payloads.config()]})


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def test_excerpt_elides_base64_and_truncates() -> None:
    text = m.excerpt_of({"snapshot_base64": "A" * 5000, "camera_id": "cam-1"})
    assert text == '{"snapshot_base64":"<5000 base64 chars>","camera_id":"cam-1"}'
    assert m.excerpt_of("x" * 2000).endswith("...")
    assert len(m.excerpt_of("x" * 2000)) == 1000
    assert m.excerpt_of(b"\x00\x01") == "<binary 2 bytes>"
    assert m.excerpt_of([{"jpeg_base64": "QUJD"}]) == '[{"jpeg_base64":"<4 base64 chars>"}]'
    assert m.excerpt_of({1, 2}).startswith("{")


def test_parse_json_object_checks_content_type_and_shape() -> None:
    assert m.parse_json_object("application/json; charset=utf-8", b'{"a": 1}', what="x") == {"a": 1}
    assert m.parse_json_object("application/vnd.api+json", b"{}", what="x") == {}
    _expect_error(lambda: m.parse_json_object(None, b"{}", what="x"), "<missing>")
    _expect_error(lambda: m.parse_json_object("text/plain", b"{}", what="x"), "text/plain")
    _expect_error(lambda: m.parse_json_object("application/json", b"{", what="x"), "<body>")
    _expect_error(lambda: m.parse_json_object("application/json", b"[]", what="x"), "JSON object")


def test_media_type_of() -> None:
    assert m.media_type_of("Multipart/Form-Data; boundary=abc") == "multipart/form-data"
    assert m.media_type_of(None) == ""


def test_format_validation_errors_lists_locations(payloads: Any) -> None:
    payload = payloads.heartbeat(status="bad")
    error = _expect_error(lambda: m.validate_heartbeat(payload), "status:")
    assert error.reason == "invalid heartbeat"
    assert str(error).startswith("invalid heartbeat: status:")
