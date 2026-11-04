from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    field_serializer,
)

Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Severity = Literal["low", "medium", "high", "critical"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CameraIn(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    area_id: str
    stream_type: Literal["rtsp", "mjpeg", "hls", "file", "webcam"]
    stream_url: str = Field(max_length=2048)
    preview_url: str | None = Field(default=None, max_length=2048)
    plan_x: Coordinate | None = None
    plan_y: Coordinate | None = None
    description: str | None = Field(default=None, max_length=4000)
    enabled: bool = True
    ai_enabled: bool = True

    @model_validator(mode="after")
    def validate_source(self):
        from urllib.parse import urlsplit

        if self.stream_type != "webcam" and not self.stream_url:
            raise ValueError("Stream URL is required")
        if self.stream_type == "rtsp" and urlsplit(self.stream_url).scheme not in (
            "rtsp",
            "rtsps",
        ):
            raise ValueError("RTSP URL must use rtsp:// or rtsps://")
        if (
            self.stream_type in ("mjpeg", "hls")
            and not (
                self.stream_url.startswith("/") and not self.stream_url.startswith("//")
            )
            and urlsplit(self.stream_url).scheme not in ("http", "https")
        ):
            raise ValueError("Stream URL must use http:// or https://")
        for value in [
            self.preview_url,
            self.stream_url if self.stream_type == "file" else None,
        ]:
            if (
                value
                and not (value.startswith("/") and not value.startswith("//"))
                and urlsplit(value).scheme not in ("http", "https")
            ):
                raise ValueError("URL must use http(s) or a local absolute path")
        for value in [self.stream_url, self.preview_url]:
            if value and any(char in value for char in ("\r", "\n", "\t")):
                raise ValueError("URL cannot contain control characters")
            if (
                value
                and urlsplit(value).scheme in ("http", "https", "rtsp", "rtsps")
                and not urlsplit(value).hostname
            ):
                raise ValueError("URL must contain a host")
        return self


class ScheduleEntry(StrictModel):
    days: list[Annotated[int, Field(ge=0, le=6)]] = Field(min_length=1, max_length=7)
    from_: str = Field(alias="from", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    to: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")

    @field_validator("days")
    @classmethod
    def unique_days(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Days must be unique")
        return value


class ConfigRules(StrictModel):
    required_ppe: list[str] = Field(default_factory=list, max_length=100)
    entry_forbidden: bool = False
    max_persons: Annotated[int, Field(ge=1, le=10000)] | None = None
    min_vehicle_distance_m: (
        Annotated[float, Field(gt=0, le=1000, allow_inf_nan=False)] | None
    ) = None
    schedule: list[ScheduleEntry] | None = Field(default=None, max_length=30)
    confirm_frames: Annotated[int, Field(ge=1, le=10000)] | None = None
    cooldown_sec: Annotated[int, Field(ge=0, le=86400)] | None = None
    severity_override: Severity | None = None


class ZoneRules(ConfigRules):
    regulation_ref: str | None = Field(default=None, max_length=500)
    responsible_user_id: str | None = None


class ZoneIn(StrictModel):
    id: UUID | None = None
    name: str = Field(min_length=1, max_length=200)
    color: str = Field(default="#65d9b2", pattern=r"^#[0-9a-fA-F]{6}$")
    zone_type: Literal["ppe", "danger", "vehicle", "counting"] = "ppe"
    polygon: list[tuple[Coordinate, Coordinate]] = Field(min_length=3, max_length=100)
    rules: ZoneRules = Field(default_factory=ZoneRules)
    enabled: bool = True
    sort_order: int = Field(default=0, ge=0, le=10000)

    @field_validator("polygon")
    @classmethod
    def valid_polygon(cls, points):
        if len(set(points)) != len(points):
            raise ValueError(
                "Polygon vertices must be unique; do not repeat the first point"
            )
        area = sum(
            a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])
        )
        if abs(area) < 1e-8:
            raise ValueError("Polygon must have non-zero area")

        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def on_segment(a, b, p):
            return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[
                1
            ] <= max(a[1], b[1])

        edges = list(zip(points, points[1:] + points[:1]))
        for i, (a, b) in enumerate(edges):
            for j, (c, d) in enumerate(edges[i + 1 :], i + 1):
                if j == i + 1 or (i == 0 and j == len(edges) - 1):
                    continue
                ab_c, ab_d, cd_a, cd_b = (
                    cross(a, b, c),
                    cross(a, b, d),
                    cross(c, d, a),
                    cross(c, d, b),
                )
                if (ab_c * ab_d < 0 and cd_a * cd_b < 0) or any(
                    [
                        abs(ab_c) < 1e-12 and on_segment(a, b, c),
                        abs(ab_d) < 1e-12 and on_segment(a, b, d),
                        abs(cd_a) < 1e-12 and on_segment(c, d, a),
                        abs(cd_b) < 1e-12 and on_segment(c, d, b),
                    ]
                ):
                    raise ValueError("Polygon cannot intersect itself")
        return points


class ZoneCreate(ZoneIn):
    camera_id: str


class ZoneOut(ZoneIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    camera_id: str
    updated_at: datetime

    @field_serializer("updated_at")
    def serialize_updated_at(self, value: datetime):
        return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


class CameraOut(CameraIn):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime
    area_name: str
    status: str
    fps: float | None
    model_name: str | None
    model_version: str | None
    last_heartbeat_at: str | None
    zones_count: int
    incidents_7d: int
    source: str | None = None
    relay_frame_age_sec: float | None = None


class ConfigCamera(BaseModel):
    id: str
    name: str
    stream_url: str
    stream_type: str


class Thresholds(StrictModel):
    confirm_frames: int = Field(ge=1, le=10000)
    cooldown_sec: int = Field(ge=0, le=86400)
    min_confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class ConfigZone(BaseModel):
    id: str
    name: str
    type: str
    polygon: list[tuple[Coordinate, Coordinate]]
    rules: ConfigRules


class IngestConfig(BaseModel):
    camera: ConfigCamera
    frame_size_hint: tuple[int, int]
    thresholds: Thresholds
    ppe_types: list[str]
    zones: list[ConfigZone]
    version: int
    updated_at: str


class KeyIn(StrictModel):
    name: str = Field(min_length=1, max_length=100)


class KeyOut(BaseModel):
    id: str
    name: str
    created_at: str
    revoked_at: str | None


class KeyCreated(KeyOut):
    key: str
