"""Pydantic models of the local service configuration (spec 7.2).

The YAML file, ``SAFETY_AI_<SECTION>__<KEY>`` environment variables and CLI flags are merged
by :mod:`safety_ai.config.loader` (precedence CLI > env > YAML > defaults) and validated here.
Zones and rules are never part of this file: they come from the backend config endpoint
(contract 4.4) or, for the demo, from a ``--zones`` file with the same structure.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from typing_extensions import Self

__all__ = [
    "AppSettings",
    "BackendSettings",
    "CameraSettings",
    "ModelSettings",
    "OutputSettings",
    "PathSettings",
    "PrivacySettings",
    "ProcessingSettings",
    "TelemetrySettings",
    "ThresholdSettings",
]

_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_DEVICE = re.compile(r"^(auto|cpu|mps|cuda|cuda:\d+|\d+)$")

Port = Annotated[int, Field(ge=1, le=65535)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class BackendSettings(_Section):
    """Website backend connection (contract 4)."""

    url: str = "http://localhost:8000"
    api_key: SecretStr | None = None
    ws: bool = True
    push_frames: bool = False
    timeout_s: float = Field(default=5.0, gt=0.0, le=120.0)
    enabled: bool = True
    """False runs without any backend (demo mode): events are printed and written to events/."""

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        """The backend URL must be http(s) without a trailing slash."""
        if not re.match(r"^https?://[^\s/]+", value):
            raise ValueError("backend.url must start with http:// or https://")
        return value.rstrip("/")

    @property
    def ws_url(self) -> str:
        """``ws://<host>/ws/ingest`` derived from the HTTP URL (api_key added by the client)."""
        if self.url.startswith("https://"):
            return "wss://" + self.url[len("https://") :] + "/ws/ingest"
        return "ws://" + self.url[len("http://") :] + "/ws/ingest"


class ModelSettings(_Section):
    """Model roles refer to registry names, never raw paths (spec 6.7)."""

    ppe: str = "ppe_s_latest"
    pose: str = "pose_n"
    vehicle: str = "same_as_ppe"
    backend: Literal["auto", "pt", "onnx", "openvino", "tensorrt"] = "auto"
    device: str = "auto"
    imgsz: int = Field(default=640, ge=128, le=1920)
    half: bool = True
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"
    class_conf: dict[str, Unit] = Field(default_factory=dict)
    """Optional per-class confidence overrides (spec 5.3)."""
    pose_imgsz: int = Field(default=320, ge=96, le=1280)
    registry: Path = Path("models/registry.json")

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        """Accept auto, cpu, mps, cuda, cuda:N or a bare GPU index."""
        if not _DEVICE.match(value):
            raise ValueError("models.device must be auto, cpu, mps, cuda, cuda:N or an index")
        return value

    @field_validator("imgsz", "pose_imgsz")
    @classmethod
    def _multiple_of_32(cls, value: int) -> int:
        """YOLO input sizes must be multiples of the model stride (32)."""
        if value % 32:
            raise ValueError("image size must be a multiple of 32")
        return value


class ThresholdSettings(_Section):
    """Detection and rule thresholds (local defaults; zone and backend values take precedence)."""

    conf: Unit = 0.25
    iou: Unit = 0.5
    confirm_frames: int = Field(default=8, ge=1, le=1000)
    cooldown_sec: float = Field(default=60.0, ge=0.0)
    violation_on: float = Field(default=-0.3, ge=-1.0, lt=0.0)
    present_on: float = Field(default=0.1, ge=0.0, le=1.0)
    track_lost_s: float = Field(default=3.0, gt=0.0)
    immobile_sec: float = Field(default=30.0, gt=0.0)
    min_confidence: Unit = 0.5
    """Used when the backend config does not send thresholds.min_confidence."""


class ProcessingSettings(_Section):
    """Real-time loop settings (spec 5.1, 5.9)."""

    target_fps: float = Field(default=12.0, gt=0.0, le=120.0)
    pose_every_n: int = Field(default=10, ge=1)
    detections_max_hz: float = Field(default=15.0, gt=0.0, le=15.0)


class OutputSettings(_Section):
    """MJPEG server, event artefacts and preview (spec 4.6, 5.10, 5.12, 5.13)."""

    mjpeg: bool = True
    mjpeg_port: Port = 8001
    mjpeg_fps: float = Field(default=12.0, gt=0.0, le=12.0)
    save_event_clips: bool = True
    events_dir: Path = Path("./events")
    events_disk_cap_gb: float = Field(default=2.0, gt=0.0)
    clip_pre_s: float = Field(default=5.0, ge=0.0, le=30.0)
    clip_post_s: float = Field(default=5.0, ge=0.0, le=30.0)
    preview: bool = False
    expose: bool = False
    """Bind the MJPEG and metrics servers to 0.0.0.0 instead of 127.0.0.1 (spec 9 Security)."""


class PrivacySettings(_Section):
    """Privacy options (spec 5.12, 9)."""

    blur_faces: bool = False


class TelemetrySettings(_Section):
    """Logging, metrics and health (spec 9)."""

    metrics_port: Port = 9100
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    log_file: Path | None = Path("logs/safety-ai.log")
    log_max_mb: int = Field(default=50, ge=1)
    log_backups: int = Field(default=5, ge=0)


class PathSettings(_Section):
    """Working directories (relative paths are resolved against the current directory)."""

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    cache_dir: Path = Path("cache")
    spool_dir: Path = Path("spool")
    logs_dir: Path = Path("logs")
    reports_dir: Path = Path("reports")
    calibration_dir: Path = Path("calibration")

    @field_validator("*")
    @classmethod
    def _safe_path(cls, value: Path) -> Path:
        """Reject paths with NUL bytes or empty names (spec 9: file paths are validated)."""
        text = str(value)
        if not text or "\x00" in text:
            raise ValueError("path must be a non-empty string without NUL bytes")
        return value


class CameraSettings(_Section):
    """One camera: id, source and the local timezone used for zone schedules."""

    id: str
    source: str | int | None = None
    """rtsp:// or http(s):// URL, video file, image directory, webcam index or fake:<yaml>.
    None means: use the stream_url from the backend config (contract 4.4)."""
    loop: bool = False
    fps: float | None = Field(default=None, gt=0.0, le=120.0)
    """Playback rate for image directories (and an override for files)."""
    timezone: str = "UTC"
    enabled: bool = True
    homography: list[list[float]] | None = None
    """Optional 3x3 image->ground-plane matrix (pixels -> metres) from ``safety-ai calibrate``."""
    zones_file: Path | None = None
    """Demo only: zones from a file with the contract's config structure instead of the backend."""

    @field_validator("id")
    @classmethod
    def _camera_id(cls, value: str) -> str:
        """Camera ids appear in URLs and external ids; keep them URL-safe."""
        if not _CAMERA_ID.match(value):
            raise ValueError("camera id must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
        return value

    @field_validator("timezone")
    @classmethod
    def _timezone(cls, value: str) -> str:
        """The timezone must be a valid IANA name (tzdata is a dependency for Windows)."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value

    @field_validator("homography")
    @classmethod
    def _homography(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        """A homography is a 3x3 matrix of finite numbers."""
        if value is None:
            return None
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("homography must be a 3x3 matrix")
        return value

    @field_validator("source")
    @classmethod
    def _source(cls, value: str | int | None) -> str | int | None:
        """Sources must be non-empty and free of NUL bytes; webcam indices are >= 0."""
        if isinstance(value, int):
            if value < 0:
                raise ValueError("webcam index must be >= 0")
            return value
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("source must be a non-empty string")
        return value


class AppSettings(_Section):
    """The complete validated service configuration."""

    backend: BackendSettings = Field(default_factory=BackendSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    cameras: list[CameraSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_cameras(self) -> Self:
        """Camera ids must be unique; MJPEG and metrics ports must differ."""
        ids = [c.id for c in self.cameras]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate camera ids: {duplicates}")
        if self.output.mjpeg and self.output.mjpeg_port == self.telemetry.metrics_port:
            raise ValueError("output.mjpeg_port and telemetry.metrics_port must differ")
        return self

    def camera(self, camera_id: str) -> CameraSettings:
        """Return the camera with ``camera_id`` or raise KeyError."""
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise KeyError(camera_id)
