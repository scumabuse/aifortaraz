import json
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .schemas import Coordinate, Severity, StrictModel

Status = Literal["new", "acknowledged", "resolved", "false_positive"]


def utc(value: datetime):
    if value.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(timezone.utc)


class Box(StrictModel):
    track_id: int | None = None
    label: str = Field(max_length=100)
    bbox: tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    conf: float = Field(ge=0, le=1, allow_inf_nan=False)

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value):
        x, y, width, height = value
        if width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
            raise ValueError("Bounding box must fit inside the frame")
        return value


class EventIn(StrictModel):
    external_id: str = Field(min_length=1, max_length=250)
    camera_id: str = Field(min_length=1, max_length=100)
    zone_id: str | None = Field(default=None, max_length=100)
    type: str = Field(min_length=1, max_length=50)
    severity: Severity | None = None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    track_id: int | None = None
    started_at: datetime
    ended_at: datetime | None = None
    details: dict = Field(default_factory=dict)
    snapshot_base64: str | None = Field(default=None, repr=False)

    @field_validator("started_at", "ended_at")
    @classmethod
    def time_in_utc(cls, value):
        return utc(value) if value else value

    @model_validator(mode="after")
    def validate_details(self):
        if self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if len(json.dumps(self.details, allow_nan=False)) > 128 * 1024:
            raise ValueError("Details exceed 128 KB")
        boxes = self.details.get("boxes", [])
        if not isinstance(boxes, list) or len(boxes) > 200:
            raise ValueError("At most 200 bounding boxes are allowed")
        for box in boxes:
            Box.model_validate(box)
        for key in ("missing_ppe", "present_ppe"):
            value = self.details.get(key, [])
            if (
                not isinstance(value, list)
                or len(value) > 100
                or not all(isinstance(code, str) and len(code) <= 50 for code in value)
            ):
                raise ValueError("PPE values must be arrays of codes")
        if "frame_size" in self.details:
            size = self.details["frame_size"]
            if (
                not isinstance(size, list)
                or len(size) != 2
                or not all(isinstance(v, int) and 0 < v <= 16384 for v in size)
            ):
                raise ValueError(
                    "frame_size must contain two positive integer dimensions"
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
    checked: int = Field(ge=0)
    compliant: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_counts(self):
        if self.compliant > self.checked:
            raise ValueError("Compliant count exceeds checked count")
        return self


class StatsIn(StrictModel):
    camera_id: str
    period_start: datetime
    period_end: datetime
    persons_total: int = Field(ge=0)
    persons_compliant: int = Field(ge=0)
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
    ppe: dict[str, bool] = Field(default_factory=dict)
    zones: list[str] = Field(default_factory=list, max_length=100)
    state: Literal["ok", "violation", "danger"] = "ok"


class DetectionsIn(StrictModel):
    type: Literal["detections"]
    camera_id: str
    ts: datetime
    frame_id: int
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
