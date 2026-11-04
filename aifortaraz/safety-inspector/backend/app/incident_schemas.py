import json
import re
from datetime import datetime, timezone
from typing import Annotated, Literal, get_args

from pydantic import (
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from .schemas import Severity, StrictModel

Status = Literal["new", "acknowledged", "resolved", "false_positive"]

# Enumerations of the integration contract (docs/CONTRACT.md 4.1, 4.5).
PpeCode = Literal[
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
PPE_CODES = get_args(PpeCode)
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

# Strict numbers: no coercion from strings or booleans. Integers are bounded to the signed
# 64-bit range that the database can store, so oversized values are a 422, never a 500.
INT64_MAX = 2**63 - 1
Int64 = Annotated[int, Field(strict=True, ge=-(2**63), le=INT64_MAX)]
Count = Annotated[int, Field(strict=True, ge=0, le=INT64_MAX)]
UnitFloat = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
FrameDimension = Annotated[int, Field(strict=True, gt=0, le=16384)]
EARLIEST_EVENT = datetime(1970, 1, 1, tzinfo=timezone.utc)
LATEST_EVENT = datetime(3000, 1, 1, tzinfo=timezone.utc)


def utc(value: datetime):
    if value.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def contract_external_id(
    camera_id: str, started_at: datetime, track_id: int, kind: str
):
    """``{camera_id}-{unix_seconds_at_start}-{track_id}-{type}`` (contract 4.1)."""
    return f"{camera_id}-{int(started_at.timestamp())}-{track_id}-{kind}"


class Box(StrictModel):
    track_id: Int64 | None = None
    label: BoxLabel
    bbox: tuple[UnitFloat, UnitFloat, UnitFloat, UnitFloat]
    conf: UnitFloat

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value):
        x, y, width, height = value
        if width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
            raise ValueError("Bounding box must fit inside the frame")
        return value


class EventDetails(StrictModel):
    """Keys every details variant carries; unknown keys are rejected (contract 4.1)."""

    boxes: list[Box] = Field(min_length=1, max_length=200)
    frame_size: tuple[FrameDimension, FrameDimension]


class PpeViolationDetails(EventDetails):
    missing_ppe: list[PpeCode] = Field(min_length=1, max_length=100)
    present_ppe: list[PpeCode] = Field(max_length=100)


class ZoneIntrusionDetails(EventDetails):
    duration_sec: NonNegativeFloat


class VehicleProximityDetails(EventDetails):
    distance_m: NonNegativeFloat
    person_track_id: Int64
    vehicle_track_id: Int64


class FallDetectedDetails(EventDetails):
    pose_conf: UnitFloat


class PersonImmobileDetails(EventDetails):
    immobile_sec: NonNegativeFloat


class OvercrowdingDetails(EventDetails):
    count: Count
    max_persons: Count


class CustomDetails(EventDetails):
    title: str = Field(min_length=1, max_length=200)


DETAILS_BY_TYPE = {
    "ppe_violation": PpeViolationDetails,
    "zone_intrusion": ZoneIntrusionDetails,
    "vehicle_proximity": VehicleProximityDetails,
    "fall_detected": FallDetectedDetails,
    "person_immobile": PersonImmobileDetails,
    "overcrowding": OvercrowdingDetails,
    "custom": CustomDetails,
}


class EventIn(StrictModel):
    external_id: str = Field(min_length=1, max_length=250)
    camera_id: str = Field(min_length=1, max_length=100)
    zone_id: str | None = Field(default=None, min_length=1, max_length=100)
    type: EventType
    severity: Severity | None = None
    confidence: UnitFloat
    track_id: Int64
    started_at: datetime
    ended_at: datetime | None = None
    details: dict
    snapshot_base64: str | None = Field(default=None, repr=False)

    @field_validator("started_at", "ended_at")
    @classmethod
    def time_in_utc(cls, value):
        if value is None:
            return value
        value = utc(value)
        if not EARLIEST_EVENT <= value < LATEST_EVENT:
            raise ValueError("Timestamp must be between 1970 and 3000")
        return value

    @field_validator("details")
    @classmethod
    def typed_details(cls, value, info: ValidationInfo):
        if len(json.dumps(value, allow_nan=False)) > 128 * 1024:
            raise ValueError("Details exceed 128 KB")
        model = DETAILS_BY_TYPE.get(info.data.get("type"))
        if model is None:
            # An invalid type is reported on the type field itself.
            return value
        return model.model_validate(value).model_dump(mode="json")

    @model_validator(mode="after")
    def validate_event(self):
        if self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        pattern = (
            f"{re.escape(self.camera_id)}-[0-9]+-{self.track_id}-{re.escape(self.type)}"
        )
        if not re.fullmatch(pattern, self.external_id):
            raise ValidationError.from_exception_data(
                type(self).__name__,
                [
                    InitErrorDetails(
                        type=PydanticCustomError(
                            "external_id_format",
                            "external_id must be {expected}",
                            {
                                "expected": f"{self.camera_id}-<unix_seconds>-"
                                f"{self.track_id}-{self.type}"
                            },
                        ),
                        loc=("external_id",),
                        input=self.external_id,
                    )
                ],
            )
        return self


class EndEvent(StrictModel):
    ended_at: datetime

    @field_validator("ended_at")
    @classmethod
    def time_in_utc(cls, value):
        return utc(value)


class HeartbeatIn(StrictModel):
    camera_id: str = Field(min_length=1, max_length=100)
    fps: float = Field(ge=0, le=1000, allow_inf_nan=False)
    model_name: str = Field(max_length=100)
    model_version: str = Field(max_length=100)
    status: Literal["ok", "online", "offline", "degraded", "error"] = "ok"
    extra: dict = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def bounded_extra(cls, value):
        if len(json.dumps(value, allow_nan=False)) > 16 * 1024:
            raise ValueError("Heartbeat metadata exceeds 16 KB")
        return value


class PpeStat(StrictModel):
    checked: Count
    compliant: Count

    @model_validator(mode="after")
    def valid_counts(self):
        if self.compliant > self.checked:
            raise ValueError("Compliant count exceeds checked count")
        return self


class StatsIn(StrictModel):
    camera_id: str
    period_start: datetime
    period_end: datetime
    persons_total: Count
    persons_compliant: Count
    by_ppe: dict[str, PpeStat] = Field(default_factory=dict)

    @field_validator("period_start", "period_end")
    @classmethod
    def time_in_utc(cls, value):
        return utc(value)

    @model_validator(mode="after")
    def valid_counts(self):
        if (
            self.persons_compliant > self.persons_total
            or self.period_end <= self.period_start
            or len(self.by_ppe) > 100
        ):
            raise ValueError("Invalid period or compliance counts")
        return self


class AssignIn(StrictModel):
    user_id: str | None


class ResolveIn(StrictModel):
    note: str = Field(min_length=1, max_length=4000)


class FalsePositiveIn(StrictModel):
    reason: str = Field(min_length=1, max_length=4000)


class CommentIn(StrictModel):
    text: str = Field(min_length=1, max_length=4000)


class BulkIn(StrictModel):
    ids: list[str] = Field(min_length=1, max_length=100)
    action: Literal["acknowledge", "false_positive"]
    reason: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_bulk(self):
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("IDs must be unique")
        if self.action == "false_positive" and not self.reason:
            raise ValueError("False-positive reason is required")
        return self


class DetectionObject(Box):
    # null = unknown (contract 4.5); forwarded to browsers unchanged.
    ppe: dict[PpeCode, Annotated[bool, Field(strict=True)] | None] = Field(
        default_factory=dict
    )
    zones: list[str] = Field(default_factory=list, max_length=100)
    state: Literal["ok", "violation", "danger"] = "ok"


class DetectionsIn(StrictModel):
    type: Literal["detections"]
    camera_id: str
    ts: datetime
    frame_id: Int64
    objects: list[DetectionObject] = Field(max_length=200)

    @field_validator("ts")
    @classmethod
    def time_in_utc(cls, value):
        return utc(value)


class FrameIn(StrictModel):
    type: Literal["frame"]
    camera_id: str = Field(min_length=1, max_length=100)
    ts: datetime
    jpeg_base64: str = Field(max_length=7_000_000, repr=False)

    @field_validator("ts")
    @classmethod
    def time_in_utc(cls, value):
        return utc(value)


PageSize = Annotated[int, Field(ge=1, le=100)]
