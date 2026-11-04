"""Strict, independent validation models of the integration contract (docs/CONTRACT.md, spec 4).

The strict mock backend (:mod:`tools.mock_backend`) validates every message the AI module sends
with the models of this module. They are written from the contract text, never imported from
``safety_ai.backend.schemas``, so a bug in the client's own schemas cannot hide itself
(DECISIONS.md D-024). Nothing in ``tools/mock_*.py`` imports the ``safety_ai`` package.

Rules enforced on AI -> site messages:

* every object rejects unknown keys (``extra="forbid"``) and every key shown in the contract is
  required; values are validated strictly (no string-to-number coercion, booleans are not
  numbers, no NaN or infinity);
* timestamps are strings ``YYYY-MM-DDTHH:MM:SS[.fff]Z``: the raw text is checked first, then
  parsed into an aware UTC datetime;
* events (4.1): type enum, severity enum or null, ``zone_id`` null only for ``fall_detected`` and
  ``person_immobile``, ``track_id`` int >= 0, confidence 0..1, ``external_id`` of the form
  ``{camera_id}-<digits>-{track_id}-{type}`` (D-010), ``ended_at >= started_at``, details with
  exactly the fields of the event type plus ``boxes`` and ``frame_size``; every non-custom event
  carries at least one ``person`` box, ``vehicle_proximity`` also a vehicle box and its
  ``track_id`` equals ``details.person_track_id``; boxes are ``[x, y, w, h]`` normalized to the
  original frame (each value 0..1, ``x + w`` and ``y + h`` at most ``1 + 1e-3``);
* snapshots: JPEG (magic bytes ``FF D8 FF``, decodable), longest side <= 1280 px, the full raw
  frame (aspect ratio of ``details.frame_size``, never larger than it);
* heartbeat (4.2), stats (4.3, with the 5.11 consistency rules: minute-aligned windows,
  ``persons_compliant <= persons_total``, ``compliant <= checked <= persons_total``) and the
  WebSocket messages of 4.5 (``detections``, ``frame`` with a JPEG of longest side <= 960 px,
  ``heartbeat``).

The camera configuration models (4.4) validate the mock's own YAML file and the bodies of
``POST /debug/set-config/{camera_id}`` with the same strictness.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Final, Literal, TypeVar

from PIL import Image
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from typing_extensions import Self

__all__ = [
    "BOX_LABELS",
    "DEFAULT_SEVERITY",
    "EVENT_TYPES",
    "MAX_FRAME_SIDE",
    "MAX_SNAPSHOT_SIDE",
    "PPE_CODES",
    "VEHICLE_LABELS",
    "WS_MESSAGE_TYPES",
    "AnyEvent",
    "AnyWsMessage",
    "BoxModel",
    "CameraConfigModel",
    "CameraInfoModel",
    "ContractError",
    "CustomDetails",
    "CustomEvent",
    "DetectionsMessageModel",
    "EventEndPatchModel",
    "FallDetectedDetails",
    "FallDetectedEvent",
    "FrameMessageModel",
    "HeartbeatExtraModel",
    "HeartbeatMessageModel",
    "HeartbeatModel",
    "MockConfigFileModel",
    "OvercrowdingDetails",
    "OvercrowdingEvent",
    "PersonImmobileDetails",
    "PersonImmobileEvent",
    "PersonObjectModel",
    "PpeStatModel",
    "PpeViolationDetails",
    "PpeViolationEvent",
    "ScheduleWindowModel",
    "StatsModel",
    "ThresholdsModel",
    "ValidatedEvent",
    "VehicleObjectModel",
    "VehicleProximityDetails",
    "VehicleProximityEvent",
    "WsAnalysis",
    "ZoneIntrusionDetails",
    "ZoneIntrusionEvent",
    "ZoneModel",
    "ZoneRulesModel",
    "analyse_ws_message",
    "camera_config_to_contract",
    "decode_base64",
    "excerpt_of",
    "external_id_matches",
    "format_validation_errors",
    "inspect_jpeg",
    "media_type_of",
    "parse_json_object",
    "parse_z_timestamp",
    "snapshot_geometry_problems",
    "validate_camera_config",
    "validate_event_payload",
    "validate_event_submission",
    "validate_heartbeat",
    "validate_patch",
    "validate_stats",
    "validate_ws_message",
]

# --------------------------------------------------------------------------------------------
# Contract enumerations (4.1, 4.5)
# --------------------------------------------------------------------------------------------

PpeCode = Literal[
    "helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "harness", "face_shield"
]
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
VehicleLabel = Literal["vehicle", "forklift", "truck", "car"]
Severity = Literal["low", "medium", "high", "critical"]

PPE_CODES: Final[tuple[PpeCode, ...]] = (
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
VEHICLE_LABELS: Final[tuple[str, ...]] = ("vehicle", "forklift", "truck", "car")
BOX_LABELS: Final[tuple[str, ...]] = (
    "person",
    *VEHICLE_LABELS,
    *PPE_CODES,
    "head",
    "face",
    "hands",
    "foot",
    "ear",
)
EVENT_TYPES: Final[tuple[str, ...]] = (
    "ppe_violation",
    "zone_intrusion",
    "vehicle_proximity",
    "fall_detected",
    "person_immobile",
    "overcrowding",
    "custom",
)
WS_MESSAGE_TYPES: Final[tuple[str, ...]] = ("detections", "frame", "heartbeat")

DEFAULT_SEVERITY: Final[Mapping[str, str]] = {
    "fall_detected": "critical",
    "person_immobile": "critical",
    "zone_intrusion": "high",
    "vehicle_proximity": "high",
    "ppe_violation": "medium",
    "overcrowding": "medium",
    "custom": "low",
}
"""Severity the backend assigns when the payload sends ``null`` and the zone has no override."""

MAX_SNAPSHOT_SIDE: Final[int] = 1280
MAX_FRAME_SIDE: Final[int] = 960
BBOX_TOLERANCE: Final[float] = 1e-3
ASPECT_TOLERANCE: Final[float] = 0.02
"""Relative tolerance between the snapshot's and ``details.frame_size``'s aspect ratios."""

JPEG_MAGIC: Final[bytes] = bytes((0xFF, 0xD8, 0xFF))
_JPEG_FORMATS: Final[frozenset[str]] = frozenset({"JPEG", "MPO"})
_ZONE_OPTIONAL_TYPES: Final[frozenset[str]] = frozenset({"fall_detected", "person_immobile"})
_EXCERPT_CHARS: Final[int] = 1000
_Z_TIMESTAMP: Final[re.Pattern[str]] = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]+))?Z"
)
_HHMM: Final[re.Pattern[str]] = re.compile(r"([0-9]{2}):([0-9]{2})")

_T = TypeVar("_T")


class ContractError(Exception):
    """A message broke the integration contract; ``errors`` lists every problem found."""

    def __init__(self, reason: str, errors: Sequence[str] = ()) -> None:
        """Create the error from a short ``reason`` and the detailed ``errors`` lines."""
        self.reason = reason
        self.errors: tuple[str, ...] = tuple(errors) if errors else (reason,)
        super().__init__(f"{reason}: {'; '.join(self.errors)}")


# --------------------------------------------------------------------------------------------
# Field types
# --------------------------------------------------------------------------------------------


def parse_z_timestamp(value: object) -> datetime:
    """Parse ``YYYY-MM-DDTHH:MM:SS[.fff]Z`` into an aware UTC datetime.

    Args:
        value: The raw JSON value; anything but a string matching the pattern is rejected.

    Returns:
        The timestamp as a timezone-aware UTC datetime (fractions beyond microseconds are cut).

    Raises:
        ValueError: The value is not a string, does not match the pattern or is not a real date.
    """
    if not isinstance(value, str):
        raise ValueError(f"timestamp must be a string like 2026-11-03T14:21:07Z, got {value!r}")
    match = _Z_TIMESTAMP.fullmatch(value)
    if match is None:
        raise ValueError(
            f"timestamp {value!r} must be ISO 8601 UTC with a trailing Z "
            "(YYYY-MM-DDTHH:MM:SS[.fff]Z)"
        )
    year, month, day, hour, minute, second = (int(match.group(i)) for i in range(1, 7))
    fraction = match.group(7) or ""
    microsecond = int(fraction[:6].ljust(6, "0")) if fraction else 0
    try:
        return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"timestamp {value!r} is not a valid date and time: {exc}") from exc


def _parse_optional_z_timestamp(value: object) -> datetime | None:
    return None if value is None else parse_z_timestamp(value)


def _check_bbox_extent(box: list[float]) -> list[float]:
    x, y, w, h = box
    if x + w > 1.0 + BBOX_TOLERANCE:
        raise ValueError(f"x + w = {x + w:.4f} exceeds 1 (bbox is [x, y, w, h] normalized 0-1)")
    if y + h > 1.0 + BBOX_TOLERANCE:
        raise ValueError(f"y + h = {y + h:.4f} exceeds 1 (bbox is [x, y, w, h] normalized 0-1)")
    return box


def _check_hhmm(value: str) -> str:
    match = _HHMM.fullmatch(value)
    if match is None:
        raise ValueError(f"time {value!r} must be HH:MM (quote it in YAML)")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if hours > 24 or minutes > 59 or (hours == 24 and minutes != 0):
        raise ValueError(f"time {value!r} is outside 00:00..24:00")
    return value


ZTimestamp = Annotated[datetime, BeforeValidator(parse_z_timestamp)]
OptionalZTimestamp = Annotated[datetime | None, BeforeValidator(_parse_optional_z_timestamp)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
PosFloat = Annotated[float, Field(gt=0.0)]
NonNegInt = Annotated[int, Field(ge=0)]
PosInt = Annotated[int, Field(gt=0)]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[^\s/?#]+$")]
"""Camera ids appear in URL paths and external ids: no whitespace, ``/``, ``?`` or ``#``."""
ZoneIdentifier = Annotated[str, Field(min_length=1, max_length=128)]
NonEmptyStr = Annotated[str, Field(min_length=1, max_length=256)]
NormBBox = Annotated[
    list[Unit], Field(min_length=4, max_length=4), AfterValidator(_check_bbox_extent)
]
FrameSize = Annotated[list[PosInt], Field(min_length=2, max_length=2)]
PolygonPoint = Annotated[list[Unit], Field(min_length=2, max_length=2)]
HhMm = Annotated[str, AfterValidator(_check_hhmm)]


class StrictModel(BaseModel):
    """Base of every validation model: unknown keys rejected, strict types, finite numbers."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


# --------------------------------------------------------------------------------------------
# 4.1 Events
# --------------------------------------------------------------------------------------------


class BoxModel(StrictModel):
    """One entry of ``details.boxes``."""

    track_id: NonNegInt
    label: BoxLabel
    bbox: NormBBox
    conf: Unit


class _DetailsBase(StrictModel):
    """Fields every ``details`` variant carries (contract 4.1)."""

    boxes: list[BoxModel]
    frame_size: FrameSize

    def _require_person_box(self) -> None:
        if not any(box.label == "person" for box in self.boxes):
            raise ValueError("boxes must include the box of the person involved (label 'person')")


class PpeViolationDetails(_DetailsBase):
    """``details`` of ``ppe_violation``."""

    missing_ppe: list[PpeCode] = Field(min_length=1)
    present_ppe: list[PpeCode]

    @model_validator(mode="after")
    def _person_box(self) -> Self:
        """The violating person must be among the boxes."""
        self._require_person_box()
        return self


class ZoneIntrusionDetails(_DetailsBase):
    """``details`` of ``zone_intrusion``."""

    duration_sec: NonNegFloat

    @model_validator(mode="after")
    def _person_box(self) -> Self:
        """The intruding person must be among the boxes."""
        self._require_person_box()
        return self


class VehicleProximityDetails(_DetailsBase):
    """``details`` of ``vehicle_proximity``."""

    distance_m: NonNegFloat
    person_track_id: NonNegInt
    vehicle_track_id: NonNegInt

    @model_validator(mode="after")
    def _person_and_vehicle_boxes(self) -> Self:
        """Both the person and the vehicle involved must be among the boxes."""
        self._require_person_box()
        if not any(box.label in VEHICLE_LABELS for box in self.boxes):
            raise ValueError(
                "boxes must include the box of the vehicle involved "
                f"(label one of {', '.join(VEHICLE_LABELS)})"
            )
        return self


class FallDetectedDetails(_DetailsBase):
    """``details`` of ``fall_detected``."""

    pose_conf: Unit

    @model_validator(mode="after")
    def _person_box(self) -> Self:
        """The fallen person must be among the boxes."""
        self._require_person_box()
        return self


class PersonImmobileDetails(_DetailsBase):
    """``details`` of ``person_immobile``."""

    immobile_sec: NonNegFloat

    @model_validator(mode="after")
    def _person_box(self) -> Self:
        """The immobile person must be among the boxes."""
        self._require_person_box()
        return self


class OvercrowdingDetails(_DetailsBase):
    """``details`` of ``overcrowding``."""

    count: NonNegInt
    max_persons: NonNegInt

    @model_validator(mode="after")
    def _person_box(self) -> Self:
        """The persons inside the zone must be among the boxes."""
        self._require_person_box()
        return self


class CustomDetails(_DetailsBase):
    """``details`` of ``custom`` (boxes may be empty: nobody needs to be involved)."""

    title: NonEmptyStr


def external_id_matches(external_id: str, camera_id: str, track_id: int, event_type: str) -> bool:
    """Check ``external_id`` against ``{camera_id}-<digits>-{track_id}-{type}`` (D-010).

    The seconds part is checked for digits only: the contract's own example pairs
    ``1762179667`` (2025-11-03) with ``started_at`` 2026-11-03.
    """
    pattern = f"{re.escape(camera_id)}-[0-9]+-{track_id}-{re.escape(event_type)}"
    return re.fullmatch(pattern, external_id) is not None


class _EventBase(StrictModel):
    """Envelope fields shared by every event type (contract 4.1)."""

    external_id: Annotated[str, Field(min_length=1, max_length=256)]
    camera_id: Identifier
    zone_id: ZoneIdentifier | None
    type: str
    severity: Severity | None
    confidence: Unit
    track_id: NonNegInt
    started_at: ZTimestamp
    ended_at: OptionalZTimestamp
    details: _DetailsBase

    @model_validator(mode="after")
    def _cross_field_rules(self) -> Self:
        """external_id pattern, nullable zone_id and the order of the timestamps."""
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


class PpeViolationEvent(_EventBase):
    """A ``ppe_violation`` event."""

    type: Literal["ppe_violation"]
    details: PpeViolationDetails


class ZoneIntrusionEvent(_EventBase):
    """A ``zone_intrusion`` event."""

    type: Literal["zone_intrusion"]
    details: ZoneIntrusionDetails


class VehicleProximityEvent(_EventBase):
    """A ``vehicle_proximity`` event; ``track_id`` is the person's track id (contract 4.1)."""

    type: Literal["vehicle_proximity"]
    details: VehicleProximityDetails

    @model_validator(mode="after")
    def _person_track(self) -> Self:
        """For vehicle_proximity the event's track_id is the person's track id."""
        if self.track_id != self.details.person_track_id:
            raise ValueError(
                f"track_id {self.track_id} must equal details.person_track_id "
                f"{self.details.person_track_id} (send the person's track_id)"
            )
        return self


class FallDetectedEvent(_EventBase):
    """A ``fall_detected`` event."""

    type: Literal["fall_detected"]
    details: FallDetectedDetails


class PersonImmobileEvent(_EventBase):
    """A ``person_immobile`` event."""

    type: Literal["person_immobile"]
    details: PersonImmobileDetails


class OvercrowdingEvent(_EventBase):
    """An ``overcrowding`` event."""

    type: Literal["overcrowding"]
    details: OvercrowdingDetails


class CustomEvent(_EventBase):
    """A ``custom`` event."""

    type: Literal["custom"]
    details: CustomDetails


AnyEvent = (
    PpeViolationEvent
    | ZoneIntrusionEvent
    | VehicleProximityEvent
    | FallDetectedEvent
    | PersonImmobileEvent
    | OvercrowdingEvent
    | CustomEvent
)
_EVENT_ADAPTER: Final[TypeAdapter[AnyEvent]] = TypeAdapter(
    Annotated[AnyEvent, Field(discriminator="type")]
)


class EventEndPatchModel(StrictModel):
    """Body of ``PATCH /api/v1/ingest/events/{external_id}``: exactly ``{"ended_at": ...}``."""

    ended_at: ZTimestamp


@dataclass(frozen=True, slots=True)
class ValidatedEvent:
    """A POSTed event whose payload and snapshot passed every contract check."""

    event: AnyEvent
    payload: dict[str, Any]
    """The payload exactly as received (decoded JSON object)."""
    snapshot_size: tuple[int, int]
    """``(width, height)`` of the snapshot JPEG."""


# --------------------------------------------------------------------------------------------
# 4.2 Heartbeat, 4.3 Statistics
# --------------------------------------------------------------------------------------------


class HeartbeatExtraModel(StrictModel):
    """``extra`` of a heartbeat: the contract's five keys plus the optional ``model_family``."""

    device: NonEmptyStr
    backend: NonEmptyStr
    queue_depth: NonNegInt
    frames_dropped: NonNegInt
    stream: NonEmptyStr
    model_family: NonEmptyStr | None = None
    """Optional detector family (DECISIONS.md D-020); when present it must be a string."""

    @model_validator(mode="after")
    def _model_family_not_null(self) -> Self:
        """``model_family`` may be omitted but not sent as null."""
        if "model_family" in self.model_fields_set and self.model_family is None:
            raise ValueError("extra.model_family must be a non-empty string when present")
        return self


class HeartbeatModel(StrictModel):
    """Body of ``POST /api/v1/ingest/heartbeat`` (contract 4.2)."""

    camera_id: Identifier
    fps: NonNegFloat
    model_name: NonEmptyStr
    model_version: NonEmptyStr
    status: Literal["ok", "degraded"]
    extra: HeartbeatExtraModel


class PpeStatModel(StrictModel):
    """Per-item counters of a statistics window."""

    checked: NonNegInt
    compliant: NonNegInt

    @model_validator(mode="after")
    def _compliant_within_checked(self) -> Self:
        """``compliant`` counts a subset of the ``checked`` tracks."""
        if self.compliant > self.checked:
            raise ValueError(f"compliant {self.compliant} exceeds checked {self.checked}")
        return self


class StatsModel(StrictModel):
    """Body of ``POST /api/v1/ingest/stats`` (contract 4.3, semantics of spec 5.11)."""

    camera_id: Identifier
    period_start: ZTimestamp
    period_end: ZTimestamp
    persons_total: NonNegInt
    persons_compliant: NonNegInt
    by_ppe: dict[PpeCode, PpeStatModel]

    @model_validator(mode="after")
    def _window_and_counts(self) -> Self:
        """Minute-aligned window, end after start and consistent counters."""
        for name, value in (("period_start", self.period_start), ("period_end", self.period_end)):
            if value.second or value.microsecond:
                raise ValueError(f"{name} must be aligned to a whole minute (seconds = 00)")
        if self.period_end <= self.period_start:
            raise ValueError("period_end must be after period_start")
        if self.persons_compliant > self.persons_total:
            raise ValueError(
                f"persons_compliant {self.persons_compliant} exceeds "
                f"persons_total {self.persons_total}"
            )
        for code, stat in self.by_ppe.items():
            if stat.checked > self.persons_total:
                raise ValueError(
                    f"by_ppe.{code}.checked {stat.checked} exceeds persons_total "
                    f"{self.persons_total} (checked tracks are persons inside PPE zones)"
                )
        return self


# --------------------------------------------------------------------------------------------
# 4.5 WebSocket messages (AI -> site)
# --------------------------------------------------------------------------------------------


class PersonObjectModel(StrictModel):
    """A person in a ``detections`` message."""

    track_id: NonNegInt
    label: Literal["person"]
    bbox: NormBBox
    conf: Unit
    ppe: dict[PpeCode, bool | None]
    zones: list[ZoneIdentifier]
    state: Literal["ok", "violation", "danger"]

    @model_validator(mode="after")
    def _ppe_only_inside_zones(self) -> Self:
        """``ppe`` holds only items required by the zones the person is in: none when in none."""
        if not self.zones and self.ppe:
            raise ValueError("ppe must be empty when the person is in no zone (zones is [])")
        return self


class VehicleObjectModel(StrictModel):
    """A vehicle in a ``detections`` message: no ``ppe``, ``zones`` or ``state``."""

    track_id: NonNegInt
    label: VehicleLabel
    bbox: NormBBox
    conf: Unit


DetectedObject = Annotated[PersonObjectModel | VehicleObjectModel, Field(discriminator="label")]


class DetectionsMessageModel(StrictModel):
    """``{"type": "detections", ...}``."""

    type: Literal["detections"]
    camera_id: Identifier
    ts: ZTimestamp
    frame_id: NonNegInt
    objects: list[DetectedObject]


class FrameMessageModel(StrictModel):
    """``{"type": "frame", ...}``; the JPEG itself is checked by :func:`analyse_ws_message`."""

    type: Literal["frame"]
    camera_id: Identifier
    ts: ZTimestamp
    jpeg_base64: Annotated[str, Field(min_length=1)]


class HeartbeatMessageModel(HeartbeatModel):
    """``{"type": "heartbeat", ...}``: the 4.2 payload plus ``type``."""

    type: Literal["heartbeat"]


AnyWsMessage = DetectionsMessageModel | FrameMessageModel | HeartbeatMessageModel
_WS_ADAPTER: Final[TypeAdapter[AnyWsMessage]] = TypeAdapter(
    Annotated[AnyWsMessage, Field(discriminator="type")]
)


# --------------------------------------------------------------------------------------------
# 4.4 Configuration (site -> AI): the mock's own data
# --------------------------------------------------------------------------------------------


class CameraInfoModel(StrictModel):
    """The ``camera`` object of a config response."""

    id: Identifier
    name: str | None = None
    stream_url: str | None = None
    stream_type: str | None = None


class ThresholdsModel(StrictModel):
    """Camera-level thresholds."""

    confirm_frames: PosInt | None = None
    cooldown_sec: NonNegFloat | None = None
    min_confidence: Unit | None = None


class ScheduleWindowModel(StrictModel):
    """A schedule window: ``days`` (1 = Monday ... 7 = Sunday) from ``from`` to ``to``."""

    days: list[Annotated[int, Field(ge=1, le=7)]] = Field(min_length=1)
    from_: HhMm = Field(alias="from")
    to: HhMm


class ZoneRulesModel(StrictModel):
    """Rule fields of a zone; null means the rule is not applied (or: use the thresholds)."""

    required_ppe: list[PpeCode] | None = None
    entry_forbidden: bool | None = None
    max_persons: NonNegInt | None = None
    min_vehicle_distance_m: PosFloat | None = None
    schedule: list[ScheduleWindowModel] | None = None
    confirm_frames: PosInt | None = None
    cooldown_sec: NonNegFloat | None = None
    severity_override: Severity | None = None


class ZoneModel(StrictModel):
    """A zone: normalized polygon, UI type hint and rules (``enabled`` per D-015, optional)."""

    id: ZoneIdentifier
    name: str = ""
    type: Literal["ppe", "danger", "vehicle", "counting"]
    polygon: list[PolygonPoint] = Field(min_length=3)
    rules: ZoneRulesModel = Field(default_factory=ZoneRulesModel)
    enabled: bool | None = None


class CameraConfigModel(StrictModel):
    """One camera configuration: the structure of ``GET /api/v1/ingest/config/{camera_id}``.

    ``version`` and ``updated_at`` are optional initial values; the mock owns them afterwards.
    """

    camera: CameraInfoModel
    frame_size_hint: FrameSize | None = None
    thresholds: ThresholdsModel = Field(default_factory=ThresholdsModel)
    ppe_types: list[PpeCode] = Field(default_factory=lambda: list(PPE_CODES))
    zones: list[ZoneModel]
    version: NonNegInt | None = None
    updated_at: OptionalZTimestamp = None

    @model_validator(mode="after")
    def _unique_zone_ids(self) -> Self:
        """Zone ids identify rule state on the AI side; they must be unique per camera."""
        ids = [zone.id for zone in self.zones]
        duplicates = sorted({zone_id for zone_id in ids if ids.count(zone_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate zone ids: {duplicates}")
        return self


class MockConfigFileModel(StrictModel):
    """Top level of ``tools/mock_config.yaml``: ``cameras: [<camera config>, ...]``."""

    cameras: list[CameraConfigModel] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_camera_ids(self) -> Self:
        """Camera ids must be unique."""
        ids = [camera.camera.id for camera in self.cameras]
        duplicates = sorted({camera_id for camera_id in ids if ids.count(camera_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate camera ids: {duplicates}")
        return self


def camera_config_to_contract(
    config: CameraConfigModel, version: int, updated_at: str
) -> dict[str, Any]:
    """Serialize a camera config into the contract's response structure (4.4).

    Every rule key is present (nulls included); a zone's ``enabled`` is emitted only when it was
    configured (D-015); ``version`` and ``updated_at`` are the mock's current values.
    """
    data = config.model_dump(mode="json", by_alias=True, exclude={"version", "updated_at"})
    for zone in data["zones"]:
        if zone.get("enabled") is None:
            zone.pop("enabled", None)
    data["version"] = version
    data["updated_at"] = updated_at
    return data


# --------------------------------------------------------------------------------------------
# Validation entry points
# --------------------------------------------------------------------------------------------


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Turn a pydantic error into readable ``"<location>: <message>"`` lines."""
    lines = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "<body>"
        lines.append(f"{location}: {error['msg']}")
    return lines


def _validate(validator: Callable[[object], _T], data: object, reason: str) -> _T:
    try:
        return validator(data)
    except ValidationError as exc:
        raise ContractError(reason, format_validation_errors(exc)) from exc


def _require_object(data: object, what: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ContractError(f"invalid {what}", [f"<body>: must be a JSON object, got {data!r:.80}"])
    return data


def _is_json_media_type(media_type: str) -> bool:
    return media_type == "application/json" or (
        media_type.startswith("application/") and media_type.endswith("+json")
    )


def media_type_of(content_type: str | None) -> str:
    """The lower-cased media type of a ``Content-Type`` header (parameters removed)."""
    return (content_type or "").split(";", 1)[0].strip().lower()


def parse_json_object(content_type: str | None, body: bytes, *, what: str) -> dict[str, Any]:
    """Decode a JSON request body that must be an object sent as ``application/json``.

    Raises:
        ContractError: Wrong ``Content-Type``, invalid JSON or not a JSON object.
    """
    if not _is_json_media_type(media_type_of(content_type)):
        raise ContractError(
            "Content-Type must be application/json",
            [f"Content-Type: {content_type or '<missing>'}, expected application/json"],
        )
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise ContractError(f"{what} is not valid JSON", [f"<body>: {exc}"]) from exc
    return _require_object(data, what)


def validate_event_payload(data: object) -> AnyEvent:
    """Validate an event payload (contract 4.1) and return the typed model.

    Raises:
        ContractError: With one line per problem.
    """
    _require_object(data, "event payload")
    return _validate(_EVENT_ADAPTER.validate_python, data, "invalid event payload")


def validate_patch(data: object) -> EventEndPatchModel:
    """Validate a PATCH body: exactly ``{"ended_at": "<timestamp>"}``."""
    _require_object(data, "PATCH body")
    return _validate(EventEndPatchModel.model_validate, data, "invalid PATCH body")


def validate_heartbeat(data: object) -> HeartbeatModel:
    """Validate a heartbeat body (contract 4.2)."""
    _require_object(data, "heartbeat")
    return _validate(HeartbeatModel.model_validate, data, "invalid heartbeat")


def validate_stats(data: object) -> StatsModel:
    """Validate a statistics body (contract 4.3)."""
    _require_object(data, "stats")
    return _validate(StatsModel.model_validate, data, "invalid stats")


def validate_ws_message(data: object) -> AnyWsMessage:
    """Validate a decoded WebSocket message of a known type (contract 4.5)."""
    _require_object(data, "WebSocket message")
    return _validate(_WS_ADAPTER.validate_python, data, "invalid WebSocket message")


def validate_camera_config(data: object) -> CameraConfigModel:
    """Validate a camera configuration object (contract 4.4 structure)."""
    _require_object(data, "camera config")
    return _validate(CameraConfigModel.model_validate, data, "invalid camera config")


def decode_base64(text: str, *, what: str) -> bytes:
    """Decode strict base64 (no whitespace, no data-URL prefix) or raise :class:`ContractError`."""
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError(f"{what} is not valid base64", [f"{what}: {exc}"]) from exc


def inspect_jpeg(data: bytes, *, max_side: int, what: str) -> tuple[int, int]:
    """Check that ``data`` is a decodable JPEG whose longest side is at most ``max_side``.

    Returns:
        ``(width, height)`` read with Pillow.

    Raises:
        ContractError: Not a JPEG (magic bytes ``FF D8 FF``), undecodable or too large.
    """
    if not data.startswith(JPEG_MAGIC):
        raise ContractError(
            f"{what} is not a JPEG", [f"{what}: magic bytes FF D8 FF missing ({len(data)} bytes)"]
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ContractError(f"{what} is not a decodable JPEG", [f"{what}: {exc}"]) from exc
    if image_format not in _JPEG_FORMATS:
        raise ContractError(f"{what} is not a JPEG", [f"{what}: decoded as {image_format}"])
    if max(width, height) > max_side:
        raise ContractError(
            f"{what} is too large",
            [f"{what}: {width}x{height}, the longest side must be <= {max_side} px"],
        )
    return width, height


def snapshot_geometry_problems(
    snapshot_size: tuple[int, int], frame_size: Sequence[int]
) -> list[str]:
    """Return problems when the snapshot cannot be the raw full frame of ``details.frame_size``.

    The snapshot is the raw frame, downscaled only when its longest side exceeds 1280 px, so its
    aspect ratio equals the original's and it is never larger than the original.
    """
    snap_w, snap_h = snapshot_size
    frame_w, frame_h = frame_size[0], frame_size[1]
    problems = []
    if snap_w > frame_w + 1 or snap_h > frame_h + 1:
        problems.append(
            f"snapshot: {snap_w}x{snap_h} is larger than details.frame_size "
            f"{frame_w}x{frame_h} (frame_size is the original frame size)"
        )
    expected = frame_w / frame_h
    actual = snap_w / snap_h
    if abs(actual - expected) > ASPECT_TOLERANCE * expected:
        problems.append(
            f"snapshot: aspect ratio {actual:.3f} ({snap_w}x{snap_h}) differs from "
            f"details.frame_size {expected:.3f} ({frame_w}x{frame_h}); the snapshot must be "
            "the whole raw frame"
        )
    return problems


def validate_event_submission(payload: object, snapshot: bytes | None) -> ValidatedEvent:
    """Validate a POSTed event: payload, snapshot JPEG and their consistency, all at once.

    Raises:
        ContractError: Listing every payload and snapshot problem found.
    """
    errors: list[str] = []
    event: AnyEvent | None = None
    snapshot_size: tuple[int, int] | None = None
    try:
        event = validate_event_payload(payload)
    except ContractError as exc:
        errors.extend(exc.errors)
    raw = payload if isinstance(payload, dict) else {}
    if snapshot is None:
        errors.append("snapshot: required (multipart file 'snapshot' or JSON 'snapshot_base64')")
    else:
        try:
            snapshot_size = inspect_jpeg(snapshot, max_side=MAX_SNAPSHOT_SIDE, what="snapshot")
        except ContractError as exc:
            errors.extend(exc.errors)
    if event is not None and snapshot_size is not None:
        errors.extend(snapshot_geometry_problems(snapshot_size, event.details.frame_size))
    if errors or event is None or snapshot_size is None:
        raise ContractError("invalid event", errors)
    return ValidatedEvent(event=event, payload=dict(raw), snapshot_size=snapshot_size)


# --------------------------------------------------------------------------------------------
# WebSocket message analysis
# --------------------------------------------------------------------------------------------


def excerpt_of(value: object, limit: int = _EXCERPT_CHARS) -> str:
    """Compact JSON (or text) excerpt of ``value`` with base64 blobs elided, at most ``limit``."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, bytes):
        text = f"<binary {len(value)} bytes>"
    else:
        try:
            text = json.dumps(_elide_blobs(value), separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _elide_blobs(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: (
                f"<{len(item)} base64 chars>"
                if key in ("jpeg_base64", "snapshot_base64") and isinstance(item, str)
                else _elide_blobs(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_elide_blobs(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class WsAnalysis:
    """Outcome of validating one WebSocket text or binary message (before rate limiting)."""

    kind: str
    """``detections``, ``frame``, ``heartbeat``, ``unknown`` (bad type) or ``malformed``."""
    camera_id: str | None
    message: AnyWsMessage | None
    """The validated model; None when invalid."""
    reason: str
    errors: tuple[str, ...]
    record: object
    """JSON-able form of the message for ``/debug/messages`` (base64 blobs elided)."""
    excerpt: str
    jpeg: bytes | None = None
    """Decoded JPEG of a valid ``frame`` message."""
    jpeg_size: tuple[int, int] | None = None

    @property
    def valid(self) -> bool:
        """True when the message passed every content check."""
        return self.message is not None and not self.errors


def _camera_of(data: Mapping[str, Any]) -> str | None:
    camera_id = data.get("camera_id")
    return camera_id if isinstance(camera_id, str) else None


def analyse_ws_message(text: str | None, data: bytes | None) -> WsAnalysis:
    """Validate one received WebSocket message.

    Args:
        text: The text frame, or None for a binary frame.
        data: The binary frame, or None for a text frame.

    Returns:
        The analysis; ``valid`` is False with readable ``errors`` for any contract violation:
        binary frames, invalid JSON, non-object JSON, missing or unknown ``type``, schema
        errors, and for ``frame`` messages invalid base64, non-JPEG data or more than 960 px.
    """
    if text is None:
        size = len(data or b"")
        return WsAnalysis(
            kind="malformed",
            camera_id=None,
            message=None,
            reason="binary WebSocket frames are not part of the contract",
            errors=(f"<message>: binary frame of {size} bytes (JSON text messages only)",),
            record=f"<binary {size} bytes>",
            excerpt=f"<binary {size} bytes>",
        )
    try:
        decoded = json.loads(text)
    except ValueError as exc:
        return WsAnalysis(
            kind="malformed",
            camera_id=None,
            message=None,
            reason="message is not valid JSON",
            errors=(f"<message>: {exc}",),
            record=excerpt_of(text),
            excerpt=excerpt_of(text),
        )
    if not isinstance(decoded, dict):
        return WsAnalysis(
            kind="malformed",
            camera_id=None,
            message=None,
            reason="message must be a JSON object",
            errors=(f"<message>: got {type(decoded).__name__}",),
            record=_elide_blobs(decoded),
            excerpt=excerpt_of(decoded),
        )
    record = _elide_blobs(decoded)
    camera_id = _camera_of(decoded)
    kind = decoded.get("type")
    if kind not in WS_MESSAGE_TYPES:
        return WsAnalysis(
            kind="unknown",
            camera_id=camera_id,
            message=None,
            reason="unknown message type",
            errors=(f"type: {kind!r} is not one of {', '.join(WS_MESSAGE_TYPES)}",),
            record=record,
            excerpt=excerpt_of(decoded),
        )
    kind_name = str(kind)
    try:
        message = validate_ws_message(decoded)
    except ContractError as exc:
        return WsAnalysis(
            kind=kind_name,
            camera_id=camera_id,
            message=None,
            reason=f"invalid {kind_name} message",
            errors=exc.errors,
            record=record,
            excerpt=excerpt_of(decoded),
        )
    if isinstance(message, FrameMessageModel):
        try:
            jpeg = decode_base64(message.jpeg_base64, what="jpeg_base64")
            jpeg_size = inspect_jpeg(jpeg, max_side=MAX_FRAME_SIDE, what="jpeg_base64")
        except ContractError as exc:
            return WsAnalysis(
                kind=kind_name,
                camera_id=camera_id,
                message=None,
                reason="invalid frame message",
                errors=exc.errors,
                record=record,
                excerpt=excerpt_of(decoded),
            )
        return WsAnalysis(
            kind=kind_name,
            camera_id=camera_id,
            message=message,
            reason="",
            errors=(),
            record=record,
            excerpt=excerpt_of(decoded),
            jpeg=jpeg,
            jpeg_size=jpeg_size,
        )
    return WsAnalysis(
        kind=kind_name,
        camera_id=camera_id,
        message=message,
        reason="",
        errors=(),
        record=record,
        excerpt=excerpt_of(decoded),
    )
