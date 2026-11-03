"""Pydantic models of the integration contract with the website (docs/CONTRACT.md, spec 4).

Outgoing messages (AI -> site) are strict: unknown fields are forbidden (``extra="forbid"``),
enums and ranges are validated and timestamps are UTC with a trailing ``Z``. Event, PATCH and
stats timestamps use whole seconds (matching the contract examples and the external_id);
WebSocket ``ts`` values carry milliseconds.

Incoming messages (site -> AI: config, WebSocket server messages, POST responses) validate the
fields we rely on and ignore unknown ones for forward compatibility.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Annotated, Any, Final, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from safety_ai.clock import format_utc_millis, format_utc_seconds, to_utc

__all__ = [
    "DETAILS_BY_TYPE",
    "EVENT_TYPES",
    "PPE_CODE_VALUES",
    "Box",
    "BoxLabel",
    "CameraConfig",
    "CameraInfo",
    "ConfigUpdatedMessage",
    "CustomDetails",
    "DetectedObject",
    "DetectionsMessage",
    "EventAccepted",
    "EventDetails",
    "EventEndPatch",
    "EventPayload",
    "EventType",
    "FallDetectedDetails",
    "FrameMessage",
    "HeartbeatExtra",
    "HeartbeatMessage",
    "HeartbeatPayload",
    "OvercrowdingDetails",
    "PersonImmobileDetails",
    "PersonObject",
    "PersonStateName",
    "PpeCode",
    "PpeStat",
    "PpeViolationDetails",
    "ScheduleWindow",
    "ServerErrorMessage",
    "ServerMessage",
    "Severity",
    "StatsPayload",
    "Thresholds",
    "VehicleObject",
    "VehicleProximityDetails",
    "Zone",
    "ZoneIntrusionDetails",
    "ZoneRules",
    "external_id_matches",
    "make_external_id",
    "parse_config_list",
    "parse_server_message",
]

_log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# Enumerations (contract 4.1, 4.5)
# --------------------------------------------------------------------------------------------

PpeCode = Literal[
    "helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "harness", "face_shield"
]
PPE_CODE_VALUES: Final[tuple[str, ...]] = (
    "helmet",
    "vest",
    "glasses",
    "gloves",
    "mask",
    "boots",
    "earmuffs",
    "harness",
    "face_shield",
)
BareLabel = Literal["head", "face", "hands", "foot", "ear"]
VehicleLabel = Literal["vehicle", "forklift", "truck", "car"]
BoxLabel = Literal[
    "person",
    "vehicle",
    "forklift",
    "truck",
    "car",
    "helmet",
    "vest",
    "glasses",
    "gloves",
    "mask",
    "boots",
    "earmuffs",
    "harness",
    "face_shield",
    "head",
    "face",
    "hands",
    "foot",
    "ear",
]
EventType = Literal[
    "ppe_violation",
    "zone_intrusion",
    "vehicle_proximity",
    "fall_detected",
    "person_immobile",
    "overcrowding",
    "custom",
]
EVENT_TYPES: Final[tuple[str, ...]] = (
    "ppe_violation",
    "zone_intrusion",
    "vehicle_proximity",
    "fall_detected",
    "person_immobile",
    "overcrowding",
    "custom",
)
Severity = Literal["low", "medium", "high", "critical"]
PersonStateName = Literal["ok", "violation", "danger"]
HeartbeatStatus = Literal["ok", "degraded"]
StreamStateName = Literal["connecting", "connected", "reconnecting", "ended", "failed"]

_ZONE_OPTIONAL_TYPES: Final[frozenset[str]] = frozenset({"fall_detected", "person_immobile"})

# --------------------------------------------------------------------------------------------
# Field types
# --------------------------------------------------------------------------------------------


def _utc_whole_seconds(value: datetime) -> datetime:
    return to_utc(value).replace(microsecond=0)


def _utc_exact(value: datetime) -> datetime:
    return to_utc(value)


UtcSeconds = Annotated[
    datetime,
    AfterValidator(_utc_whole_seconds),
    PlainSerializer(format_utc_seconds, return_type=str, when_used="json"),
]
"""Aware UTC timestamp truncated to seconds; JSON form ``2026-11-03T14:21:07Z``."""

UtcMillis = Annotated[
    datetime,
    AfterValidator(_utc_exact),
    PlainSerializer(format_utc_millis, return_type=str, when_used="json"),
]
"""Aware UTC timestamp; JSON form ``2026-11-03T14:21:07.250Z``."""

Unit = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PosFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegInt = Annotated[int, Field(ge=0)]
PosInt = Annotated[int, Field(gt=0)]
CameraId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
ZoneId = Annotated[str, Field(min_length=1, max_length=128)]

_BBOX_TOLERANCE: Final[float] = 1e-3


def _check_bbox_extent(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    if x + w > 1.0 + _BBOX_TOLERANCE or y + h > 1.0 + _BBOX_TOLERANCE:
        raise ValueError(f"bbox {list(box)} extends past the frame (x+w and y+h must be <= 1)")
    return box


def _round_list(values: tuple[float, ...]) -> list[float]:
    return [round(v, 4) for v in values]


NormBBox = Annotated[
    tuple[Unit, Unit, Unit, Unit],
    AfterValidator(_check_bbox_extent),
    PlainSerializer(_round_list, return_type=list[float], when_used="json"),
]
"""``[x, y, w, h]`` normalized to the original frame; serialized rounded to 4 decimals."""

Confidence = Annotated[
    Unit, PlainSerializer(lambda v: round(v, 3), return_type=float, when_used="json")
]
FrameSize = tuple[PosInt, PosInt]


class StrictModel(BaseModel):
    """Base for outgoing messages: unknown fields are rejected, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class LenientModel(BaseModel):
    """Base for incoming messages: unknown fields are ignored for forward compatibility."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


# --------------------------------------------------------------------------------------------
# 4.1 Events
# --------------------------------------------------------------------------------------------


class Box(StrictModel):
    """One box in ``details.boxes``; PPE item boxes carry the owning person's track id."""

    track_id: NonNegInt
    label: BoxLabel
    bbox: NormBBox
    conf: Confidence


class PpeViolationDetails(StrictModel):
    """``details`` of a ``ppe_violation`` event."""

    missing_ppe: list[PpeCode] = Field(min_length=1)
    present_ppe: list[PpeCode]
    boxes: list[Box] = Field(min_length=1)
    frame_size: FrameSize


class ZoneIntrusionDetails(StrictModel):
    """``details`` of a ``zone_intrusion`` event."""

    duration_sec: NonNegFloat
    boxes: list[Box] = Field(min_length=1)
    frame_size: FrameSize


class VehicleProximityDetails(StrictModel):
    """``details`` of a ``vehicle_proximity`` event."""

    distance_m: NonNegFloat
    person_track_id: NonNegInt
    vehicle_track_id: NonNegInt
    boxes: list[Box] = Field(min_length=2)
    frame_size: FrameSize


class FallDetectedDetails(StrictModel):
    """``details`` of a ``fall_detected`` event."""

    pose_conf: Unit
    boxes: list[Box] = Field(min_length=1)
    frame_size: FrameSize


class PersonImmobileDetails(StrictModel):
    """``details`` of a ``person_immobile`` event."""

    immobile_sec: NonNegFloat
    boxes: list[Box] = Field(min_length=1)
    frame_size: FrameSize


class OvercrowdingDetails(StrictModel):
    """``details`` of an ``overcrowding`` event."""

    count: NonNegInt
    max_persons: NonNegInt
    boxes: list[Box] = Field(min_length=1)
    frame_size: FrameSize


class CustomDetails(StrictModel):
    """``details`` of a ``custom`` event."""

    title: str = Field(min_length=1, max_length=200)
    boxes: list[Box] = Field(min_length=1)
    frame_size: FrameSize


EventDetails = Union[
    PpeViolationDetails,
    ZoneIntrusionDetails,
    VehicleProximityDetails,
    FallDetectedDetails,
    PersonImmobileDetails,
    OvercrowdingDetails,
    CustomDetails,
]

DETAILS_BY_TYPE: Final[dict[str, type[StrictModel]]] = {
    "ppe_violation": PpeViolationDetails,
    "zone_intrusion": ZoneIntrusionDetails,
    "vehicle_proximity": VehicleProximityDetails,
    "fall_detected": FallDetectedDetails,
    "person_immobile": PersonImmobileDetails,
    "overcrowding": OvercrowdingDetails,
    "custom": CustomDetails,
}


def make_external_id(camera_id: str, started_at: datetime, track_id: int, event_type: str) -> str:
    """Build ``{camera_id}-{unix_seconds_at_start}-{track_id}-{type}`` (contract 4.1)."""
    seconds = math.floor(to_utc(started_at).timestamp())
    return f"{camera_id}-{seconds}-{track_id}-{event_type}"


def external_id_matches(external_id: str, camera_id: str, track_id: int, event_type: str) -> bool:
    """Check the ``{camera_id}-<digits>-{track_id}-{type}`` pattern of an external id.

    The seconds part is checked for shape only: the contract's own example pairs
    ``1762179667`` (2025-11-03T14:21:07Z) with ``started_at`` 2026-11-03T14:21:07Z, so an exact
    seconds check would reject the contract example (DECISIONS.md D-010). Our allocator always
    derives the seconds from ``started_at``; unit tests enforce that.
    """
    prefix = f"{camera_id}-"
    suffix = f"-{track_id}-{event_type}"
    if not (external_id.startswith(prefix) and external_id.endswith(suffix)):
        return False
    middle = external_id[len(prefix) : len(external_id) - len(suffix)]
    return middle.isdigit()


class EventPayload(StrictModel):
    """Body of ``POST /api/v1/ingest/events`` (the ``payload`` form field, or the JSON body)."""

    external_id: str = Field(min_length=1, max_length=256)
    camera_id: CameraId
    zone_id: ZoneId | None
    type: EventType
    severity: Severity | None
    confidence: Confidence
    track_id: NonNegInt
    started_at: UtcSeconds
    ended_at: UtcSeconds | None
    details: EventDetails

    @model_validator(mode="before")
    @classmethod
    def _pick_details_model(cls, data: Any) -> Any:
        """Validate ``details`` against the model selected by ``type`` (clear error messages)."""
        if isinstance(data, dict):
            event_type = data.get("type")
            details = data.get("details")
            model = DETAILS_BY_TYPE.get(event_type) if isinstance(event_type, str) else None
            if model is not None and isinstance(details, dict):
                data = {**data, "details": model.model_validate(details)}
        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Enforce the cross-field rules of contract 4.1."""
        expected_model = DETAILS_BY_TYPE[self.type]
        if not isinstance(self.details, expected_model):
            raise ValueError(f"details for type {self.type!r} must be {expected_model.__name__}")
        if not external_id_matches(self.external_id, self.camera_id, self.track_id, self.type):
            raise ValueError(
                f"external_id {self.external_id!r} must match "
                f"'{self.camera_id}-<unix_seconds>-{self.track_id}-{self.type}'"
            )
        if self.zone_id is None and self.type not in _ZONE_OPTIONAL_TYPES:
            raise ValueError("zone_id may be null only for fall_detected and person_immobile")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at must not be earlier than started_at")
        return self


class EventEndPatch(StrictModel):
    """Body of ``PATCH /api/v1/ingest/events/{external_id}``."""

    ended_at: UtcSeconds


class EventAccepted(LenientModel):
    """Response of ``POST /api/v1/ingest/events`` (201, 200 idempotent, 202 deduplicated)."""

    id: str
    status: str | None = None
    severity: str | None = None
    deduplicated: bool = False

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, value: Any) -> Any:
        """Backends may return numeric ids; store them as strings."""
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value


# --------------------------------------------------------------------------------------------
# 4.2 Heartbeat, 4.3 Stats
# --------------------------------------------------------------------------------------------


class HeartbeatExtra(StrictModel):
    """``extra`` object of a heartbeat."""

    device: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    queue_depth: NonNegInt
    frames_dropped: NonNegInt
    stream: StreamStateName
    model_family: str = Field(min_length=1)
    """Detector family chosen at startup (``yolo26`` or ``yolo11``, spec 5.3)."""


class HeartbeatPayload(StrictModel):
    """Body of ``POST /api/v1/ingest/heartbeat`` (every 5 s per camera)."""

    camera_id: CameraId
    fps: Annotated[NonNegFloat, PlainSerializer(lambda v: round(v, 2), return_type=float)]
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    status: HeartbeatStatus
    extra: HeartbeatExtra


class PpeStat(StrictModel):
    """Per-item compliance counters of a stats window."""

    checked: NonNegInt
    compliant: NonNegInt

    @model_validator(mode="after")
    def _compliant_le_checked(self) -> Self:
        """``compliant`` can never exceed ``checked``."""
        if self.compliant > self.checked:
            raise ValueError("compliant must be <= checked")
        return self


class StatsPayload(StrictModel):
    """Body of ``POST /api/v1/ingest/stats`` (every 60 s per camera, minute-aligned windows)."""

    camera_id: CameraId
    period_start: UtcSeconds
    period_end: UtcSeconds
    persons_total: NonNegInt
    persons_compliant: NonNegInt
    by_ppe: dict[PpeCode, PpeStat]

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        """Windows are aligned to whole minutes and counts are consistent (spec 5.11)."""
        if self.period_end <= self.period_start:
            raise ValueError("period_end must be after period_start")
        if self.period_start.second != 0 or self.period_end.second != 0:
            raise ValueError("stats windows must be aligned to whole minutes")
        if self.persons_compliant > self.persons_total:
            raise ValueError("persons_compliant must be <= persons_total")
        return self


# --------------------------------------------------------------------------------------------
# 4.5 WebSocket messages
# --------------------------------------------------------------------------------------------


class PersonObject(StrictModel):
    """A person in a ``detections`` message; ``ppe`` holds only the required items."""

    track_id: NonNegInt
    label: Literal["person"]
    bbox: NormBBox
    conf: Confidence
    ppe: dict[PpeCode, bool | None]
    zones: list[ZoneId]
    state: PersonStateName


class VehicleObject(StrictModel):
    """A vehicle in a ``detections`` message."""

    track_id: NonNegInt
    label: VehicleLabel
    bbox: NormBBox
    conf: Confidence


DetectedObject = Annotated[Union[PersonObject, VehicleObject], Field(discriminator="label")]


class DetectionsMessage(StrictModel):
    """``{"type": "detections", ...}`` — at most 15 per second per camera."""

    type: Literal["detections"] = "detections"
    camera_id: CameraId
    ts: UtcMillis
    frame_id: NonNegInt
    objects: list[DetectedObject]


class FrameMessage(StrictModel):
    """``{"type": "frame", ...}`` — optional annotated JPEG (<= 10/s, <= 960 px, quality 70)."""

    type: Literal["frame"] = "frame"
    camera_id: CameraId
    ts: UtcMillis
    jpeg_base64: str = Field(min_length=8)

    @field_validator("jpeg_base64")
    @classmethod
    def _looks_like_jpeg(cls, value: str) -> str:
        """Base64 of a JPEG always starts with ``/9j/`` (bytes FF D8 FF)."""
        if not value.startswith("/9j/"):
            raise ValueError("jpeg_base64 must be a base64-encoded JPEG")
        return value


class HeartbeatMessage(HeartbeatPayload):
    """``{"type": "heartbeat", ...}`` — the heartbeat payload of 4.2 sent over the WebSocket."""

    type: Literal["heartbeat"] = "heartbeat"


class ConfigUpdatedMessage(LenientModel):
    """Server -> AI: the configuration of ``camera_id`` changed (re-read it immediately)."""

    type: Literal["config.updated"]
    camera_id: str | None = None
    version: int | None = None


class ServerErrorMessage(LenientModel):
    """Server -> AI: an error report; logged at WARNING with any message id it carries."""

    model_config = ConfigDict(extra="allow", frozen=True)

    type: Literal["error"]
    message: str = ""

    def offending_id(self) -> str | None:
        """Return the id of the offending message when the server included one."""
        extra = self.model_extra or {}
        for key in ("id", "message_id", "msg_id", "ref", "frame_id", "external_id"):
            if key in extra and extra[key] is not None:
                return str(extra[key])
        return None


ServerMessage = Annotated[
    Union[ConfigUpdatedMessage, ServerErrorMessage], Field(discriminator="type")
]


def parse_server_message(data: Any) -> ConfigUpdatedMessage | ServerErrorMessage | None:
    """Parse a decoded server WebSocket message; unknown types return None."""
    if not isinstance(data, dict):
        return None
    kind = data.get("type")
    if kind == "config.updated":
        return ConfigUpdatedMessage.model_validate(data)
    if kind == "error":
        return ServerErrorMessage.model_validate(data)
    return None


# --------------------------------------------------------------------------------------------
# 4.4 Configuration (site -> AI)
# --------------------------------------------------------------------------------------------


def _drop_unknown_ppe(value: Any) -> Any:
    """Ignore PPE codes this build does not know instead of rejecting the whole config."""
    if isinstance(value, list):
        known = [v for v in value if v in PPE_CODE_VALUES]
        unknown = [v for v in value if v not in PPE_CODE_VALUES]
        if unknown:
            _log.warning("ignoring unknown PPE codes in config: %s", unknown)
        return known
    return value


PpeList = Annotated[list[PpeCode], BeforeValidator(_drop_unknown_ppe)]


def _parse_hhmm(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"time {value!r} must be HH:MM")
    hours, minutes = int(parts[0]), int(parts[1])
    if not (0 <= hours <= 24 and 0 <= minutes <= 59) or (hours == 24 and minutes != 0):
        raise ValueError(f"time {value!r} is out of range")
    return hours * 60 + minutes


class ScheduleWindow(LenientModel):
    """A schedule entry: active on ``days`` (1 = Monday ... 7 = Sunday) from ``from`` to ``to``.

    Times are the camera's local time. A window whose ``to`` is earlier than ``from`` runs past
    midnight and belongs to the day it starts on.
    """

    days: list[Annotated[int, Field(ge=1, le=7)]] = Field(min_length=1)
    from_: str = Field(alias="from")
    to: str

    @field_validator("from_", "to")
    @classmethod
    def _valid_time(cls, value: str) -> str:
        """Times must be HH:MM in 00:00..24:00."""
        _parse_hhmm(value)
        return value

    @property
    def start_minute(self) -> int:
        """Minutes after local midnight when the window opens."""
        return _parse_hhmm(self.from_)

    @property
    def end_minute(self) -> int:
        """Minutes after local midnight when the window closes."""
        return _parse_hhmm(self.to)


class ZoneRules(LenientModel):
    """Rule fields of a zone; every non-null field is applied (contract 4.4)."""

    required_ppe: PpeList | None = None
    entry_forbidden: bool | None = None
    max_persons: NonNegInt | None = None
    min_vehicle_distance_m: PosFloat | None = None
    schedule: list[ScheduleWindow] | None = None
    confirm_frames: PosInt | None = None
    cooldown_sec: NonNegFloat | None = None
    severity_override: Severity | None = None


def _clamp_polygon(value: Any) -> Any:
    if isinstance(value, list):
        out = []
        for point in value:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                px, py = point
                if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                    if not (-1e-6 <= px <= 1 + 1e-6 and -1e-6 <= py <= 1 + 1e-6):
                        raise ValueError(f"polygon point {list(point)} is outside 0..1")
                    out.append((min(1.0, max(0.0, float(px))), min(1.0, max(0.0, float(py)))))
                    continue
            out.append(point)
        return out
    return value


class Zone(LenientModel):
    """A zone polygon (normalized to the frame) with its rules; ``type`` is only a UI hint."""

    id: ZoneId
    name: str = ""
    type: str = "ppe"
    polygon: Annotated[list[tuple[float, float]], BeforeValidator(_clamp_polygon)] = Field(
        min_length=3
    )
    rules: ZoneRules = Field(default_factory=ZoneRules)
    enabled: bool = True


class CameraInfo(LenientModel):
    """The ``camera`` object of a config response."""

    id: CameraId
    name: str | None = None
    stream_url: str | None = None
    stream_type: str | None = None


class Thresholds(LenientModel):
    """Camera-level thresholds; null values fall back to the local configuration."""

    confirm_frames: PosInt | None = None
    cooldown_sec: NonNegFloat | None = None
    min_confidence: Unit | None = None


class CameraConfig(LenientModel):
    """Response of ``GET /api/v1/ingest/config/{camera_id}`` (one element of the list endpoint)."""

    camera: CameraInfo
    frame_size_hint: FrameSize | None = None
    thresholds: Thresholds = Field(default_factory=Thresholds)
    ppe_types: PpeList = Field(default_factory=lambda: list(PPE_CODE_VALUES))
    zones: list[Zone] = Field(default_factory=list)
    version: int = 0
    updated_at: datetime | None = None

    @field_validator("updated_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        """Naive timestamps are interpreted as UTC (the contract mandates UTC)."""
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def _unique_zone_ids(self) -> Self:
        """Zone ids identify rule state; duplicates would merge unrelated zones."""
        ids = [z.id for z in self.zones]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate zone ids: {duplicates}")
        return self


def parse_config_list(data: Any) -> list[CameraConfig]:
    """Parse ``GET /api/v1/ingest/config``.

    The contract does not show this response; the most literal reading is a JSON array of
    per-camera config objects. An object wrapping the array under ``cameras``, ``configs``,
    ``items`` or ``data`` is accepted too (DECISIONS.md D-012).
    """
    if isinstance(data, dict):
        for key in ("cameras", "configs", "items", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            if "camera" in data:
                data = [data]
    if not isinstance(data, list):
        raise ValueError("config list response must be a JSON array of camera configs")
    return [CameraConfig.model_validate(item) for item in data]
