from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from .database import Base


def new_id():
    return str(uuid4())


def utc_now():
    return datetime.now(timezone.utc)


def identifier():
    return Column(String(36), primary_key=True, default=new_id)


def timestamp():
    return Column(DateTime(timezone=True), default=utc_now, nullable=False)


class Area(Base):
    __tablename__ = "areas"
    id = identifier()
    name = Column(String(200), nullable=False)
    description = Column(Text)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('admin','engineer','master','viewer')"),
    )
    id = identifier()
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    full_name = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False)
    area_id = Column(ForeignKey("areas.id"))
    created_at = timestamp()


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (
        CheckConstraint("stream_type IN ('rtsp','mjpeg','hls','file','webcam')"),
        CheckConstraint("plan_x IS NULL OR (plan_x >= 0 AND plan_x <= 1)"),
        CheckConstraint("plan_y IS NULL OR (plan_y >= 0 AND plan_y <= 1)"),
    )
    id = identifier()
    name = Column(String(200), nullable=False)
    area_id = Column(ForeignKey("areas.id"), nullable=False)
    stream_type = Column(String(20), nullable=False)
    stream_url = Column(Text, nullable=False)
    preview_url = Column(Text)
    description = Column(Text)
    plan_x = Column(Float)
    plan_y = Column(Float)
    enabled = Column(Boolean, default=True, nullable=False)
    ai_enabled = Column(Boolean, default=True, nullable=False)
    created_at = timestamp()


class CameraStatus(Base):
    __tablename__ = "camera_status"
    __table_args__ = (CheckConstraint("status IN ('online','offline','degraded')"),)
    camera_id = Column(ForeignKey("cameras.id"), primary_key=True)
    status = Column(String(20), default="offline", nullable=False)
    fps = Column(Float)
    model_name = Column(String(100))
    model_version = Column(String(100))
    last_heartbeat_at = Column(DateTime(timezone=True))
    extra = Column(JSON, default=dict)


class Zone(Base):
    __tablename__ = "zones"
    __table_args__ = (
        CheckConstraint("zone_type IN ('ppe','danger','vehicle','counting')"),
    )
    id = identifier()
    camera_id = Column(ForeignKey("cameras.id"), nullable=False)
    name = Column(String(200), nullable=False)
    color = Column(String(20), nullable=False)
    zone_type = Column(String(20), nullable=False)
    polygon = Column(JSON, nullable=False)
    rules = Column(JSON, default=dict, nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    sort_order = Column(Integer, default=0, nullable=False)
    updated_at = timestamp()


class PpeType(Base):
    __tablename__ = "ppe_types"
    code = Column(String(50), primary_key=True)
    name_ru = Column(String(200), nullable=False)
    icon = Column(String(50))
    enabled = Column(Boolean, default=True, nullable=False)
    sort_order = Column(Integer, default=0, nullable=False)


class IncidentType(Base):
    __tablename__ = "incident_types"
    __table_args__ = (
        CheckConstraint("default_severity IN ('low','medium','high','critical')"),
    )
    code = Column(String(50), primary_key=True)
    name_ru = Column(String(200), nullable=False)
    default_severity = Column(String(20), nullable=False)
    color = Column(String(20), nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)


class Incident(Base):
    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_incidents_camera_started", "camera_id", "started_at"),
        Index("ix_incidents_status", "status"),
        Index("ix_incidents_type", "type_code"),
        CheckConstraint("status IN ('new','acknowledged','resolved','false_positive')"),
        CheckConstraint("severity IN ('low','medium','high','critical')"),
        CheckConstraint("source IN ('ai','mock','manual')"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)"),
    )
    id = identifier()
    external_id = Column(String(250), unique=True)
    source = Column(String(20), nullable=False)
    camera_id = Column(ForeignKey("cameras.id"), nullable=False)
    zone_id = Column(ForeignKey("zones.id"))
    type_code = Column(ForeignKey("incident_types.code"), nullable=False)
    severity = Column(String(20), nullable=False)
    status = Column(String(20), default="new", nullable=False)
    confidence = Column(Float)
    track_id = Column(Integer)
    person_ref = Column(String(100))
    started_at = Column(DateTime(timezone=True), nullable=False)
    ended_at = Column(DateTime(timezone=True))
    duration_sec = Column(Float)
    snapshot_path = Column(Text)
    thumbnail_path = Column(Text)
    clip_path = Column(Text)
    details = Column(JSON, default=dict, nullable=False)
    assignee_id = Column(ForeignKey("users.id"))
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_by = Column(ForeignKey("users.id"))
    resolved_at = Column(DateTime(timezone=True))
    resolved_by = Column(ForeignKey("users.id"))
    resolution_note = Column(Text)
    false_positive_reason = Column(Text)
    created_at = timestamp()


class IncidentEvent(Base):
    __tablename__ = "incident_events"
    __table_args__ = (
        Index("ix_incident_events_incident", "incident_id"),
        CheckConstraint(
            "event_type IN ('created','notified','acknowledged','assigned','commented','resolved','false_positive','reopened')"
        ),
    )
    id = identifier()
    incident_id = Column(ForeignKey("incidents.id"), nullable=False)
    event_type = Column(String(30), nullable=False)
    user_id = Column(ForeignKey("users.id"))
    payload = Column(JSON, default=dict, nullable=False)
    created_at = timestamp()


class IngestReceipt(Base):
    """Persist accepted external IDs, including IDs suppressed by a cooldown."""

    __tablename__ = "ingest_receipts"
    external_id = Column(String(250), primary_key=True)
    incident_id = Column(
        ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at = timestamp()


class Comment(Base):
    __tablename__ = "comments"
    id = identifier()
    incident_id = Column(ForeignKey("incidents.id"), nullable=False)
    user_id = Column(ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    created_at = timestamp()


class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (
        CheckConstraint("channel IN ('telegram','inapp')"),
        CheckConstraint("status IN ('sent','failed','skipped')"),
    )
    id = identifier()
    incident_id = Column(ForeignKey("incidents.id"), nullable=False)
    channel = Column(String(20), nullable=False)
    recipient = Column(String(200), nullable=False)
    status = Column(String(20), nullable=False)
    error = Column(Text)
    sent_at = Column(DateTime(timezone=True))
    read_at = Column(DateTime(timezone=True))


class NotificationSettings(Base):
    __tablename__ = "notification_settings"
    __table_args__ = (CheckConstraint("id = 1"),)
    id = Column(Integer, primary_key=True, default=1)
    telegram_bot_token = Column(Text)
    chat_ids = Column(JSON, default=list, nullable=False)
    rules = Column(JSON, default=dict, nullable=False)


class ComplianceStat(Base):
    __tablename__ = "compliance_stats"
    __table_args__ = (
        Index("ix_compliance_camera_period", "camera_id", "period_start"),
        CheckConstraint(
            "persons_compliant >= 0 AND persons_total >= persons_compliant"
        ),
    )
    id = identifier()
    camera_id = Column(ForeignKey("cameras.id"), nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    persons_total = Column(Integer, nullable=False)
    persons_compliant = Column(Integer, nullable=False)
    by_ppe = Column(JSON, default=dict, nullable=False)
    source = Column(String(20), default="ai", server_default="ai", nullable=False)


class Report(Base):
    __tablename__ = "reports"
    id = identifier()
    report_type = Column(String(30), nullable=False)
    period_from = Column(DateTime(timezone=True), nullable=False)
    period_to = Column(DateTime(timezone=True), nullable=False)
    params = Column(JSON, default=dict, nullable=False)
    file_path = Column(Text)
    created_by = Column(ForeignKey("users.id"), nullable=False)
    created_at = timestamp()


class FloorPlan(Base):
    __tablename__ = "floor_plans"
    id = identifier()
    area_id = Column(ForeignKey("areas.id"), unique=True, nullable=False)
    image_path = Column(Text, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = identifier()
    name = Column(String(100), nullable=False)
    key_hash = Column(Text, nullable=False)
    created_at = timestamp()
    revoked_at = Column(DateTime(timezone=True))


class Setting(Base):
    __tablename__ = "settings"
    key = Column(String(100), primary_key=True)
    value = Column(JSON, nullable=False)


class TelegramDelivery(Base):
    __tablename__ = "telegram_deliveries"
    __table_args__ = (
        Index("ix_telegram_due", "status", "next_attempt_at"),
        Index(
            "ix_telegram_cooldown", "camera_id", "type_code", "recipient", "created_at"
        ),
    )
    id = identifier()
    incident_id = Column(ForeignKey("incidents.id", ondelete="CASCADE"))
    camera_id = Column(String(36))
    type_code = Column(String(50))
    recipient = Column(String(32), nullable=False)
    label = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)
    attempts = Column(Integer, nullable=False, default=0)
    error = Column(Text)
    created_at = timestamp()
    next_attempt_at = Column(DateTime(timezone=True))
    sent_at = Column(DateTime(timezone=True))
