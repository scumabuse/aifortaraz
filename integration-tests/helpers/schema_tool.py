"""Export JSON Schemas of one side's contract models, or validate the canonical examples with them.

Runs INSIDE the virtual environment of the side it inspects (each side is imported in its own
interpreter so their dependencies never mix):

    <website venv python> schema_tool.py dump website
    <safety-ai venv python> schema_tool.py dump ai
    <python> schema_tool.py validate <website|ai> <examples dir>

``dump`` prints {"<message>": <JSON Schema>} for every contract message; ``validate`` prints
{"<example file>": "ok" | "<error>"}. Website models describe what the site ACCEPTS (validation
mode; for the config response: what it EMITS, serialization mode); AI models describe what the AI
EMITS (serialization mode; for the config: what it ACCEPTS, validation mode).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WEBSITE_BACKEND = ROOT / "aifortaraz" / "safety-inspector" / "backend"
AI_ROOT = ROOT / "safety-ai"

DETAIL_TYPES = ("ppe_violation", "zone_intrusion", "vehicle_proximity", "fall_detected", "person_immobile",
                "overcrowding", "custom")
DETAIL_CLASSES = ("PpeViolationDetails", "ZoneIntrusionDetails", "VehicleProximityDetails",
                  "FallDetectedDetails", "PersonImmobileDetails", "OvercrowdingDetails", "CustomDetails")


def website_models() -> dict[str, tuple[Any, str]]:
    """message -> (pydantic model, schema mode) for the website backend."""
    os.environ.setdefault("JWT_SECRET", "schema-tool-" + "x" * 40)
    os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="schema_tool_"))
    os.environ.setdefault("DATABASE_URL", "sqlite:///" + (Path(os.environ["DATA_DIR"]) / "db.sqlite").as_posix())
    sys.path.insert(0, str(WEBSITE_BACKEND))
    from app import incident_schemas as s  # noqa: PLC0415
    try:
        from app.schemas import IngestConfig  # noqa: PLC0415
    except ImportError:
        from app.stage_two import IngestConfig  # noqa: PLC0415
    models: dict[str, tuple[Any, str]] = {
        "event": (s.EventIn, "validation"),
        "patch": (s.EndEvent, "validation"),
        "heartbeat": (s.HeartbeatIn, "validation"),
        "stats": (s.StatsIn, "validation"),
        "ws_detections": (s.DetectionsIn, "validation"),
        "ws_frame": (s.FrameIn, "validation"),
        "ws_heartbeat": (s.HeartbeatIn, "validation"),
        "config": (IngestConfig, "serialization"),
    }
    for kind, cls in zip(DETAIL_TYPES, DETAIL_CLASSES, strict=True):
        models[f"details.{kind}"] = (getattr(s, cls), "validation")
    return models


def ai_models() -> dict[str, tuple[Any, str]]:
    """message -> (pydantic model, schema mode) for the AI module."""
    sys.path.insert(0, str(AI_ROOT))
    from safety_ai.backend import schemas as s  # noqa: PLC0415

    models: dict[str, tuple[Any, str]] = {
        "event": (s.EventPayload, "serialization"),
        "patch": (s.EventEndPatch, "serialization"),
        "heartbeat": (s.HeartbeatPayload, "serialization"),
        "stats": (s.StatsPayload, "serialization"),
        "ws_detections": (s.DetectionsMessage, "serialization"),
        "ws_frame": (s.FrameMessage, "serialization"),
        "ws_heartbeat": (s.HeartbeatMessage, "serialization"),
        "config": (s.CameraConfig, "validation"),
    }
    for kind in DETAIL_TYPES:
        models[f"details.{kind}"] = (s.DETAILS_BY_TYPE[kind], "serialization")
    return models


def models_for(side: str) -> dict[str, tuple[Any, str]]:
    """Model table of a side."""
    if side == "website":
        return website_models()
    if side == "ai":
        return ai_models()
    raise SystemExit(f"unknown side {side!r} (website or ai)")


def example_message(name: str) -> str:
    """Contract message of an example file name."""
    stem = Path(name).stem
    if stem.startswith("event_"):
        return "event"
    if stem.startswith("patch"):
        return "patch"
    return stem


def validate(side: str, examples: Path) -> dict[str, str]:
    """Validate every example with the side's model of its message."""
    models = models_for(side)
    results: dict[str, str] = {}
    for path in sorted(examples.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        message = example_message(path.name)
        model, _mode = models[message]
        payload = dict(data)
        if side == "website" and message == "ws_heartbeat":
            payload.pop("type", None)
        try:
            model.model_validate(payload)
            results[path.name] = "ok"
        except Exception as exc:  # noqa: BLE001 — every validation problem is reported as text
            results[path.name] = " | ".join(str(exc).splitlines()[:6])
    return results


def main() -> int:
    """CLI entry."""
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    command, side = sys.argv[1], sys.argv[2]
    if command == "dump":
        out = {name: model.model_json_schema(mode=mode) for name, (model, mode) in models_for(side).items()}
    elif command == "validate":
        out = validate(side, Path(sys.argv[3]))
    else:
        raise SystemExit(f"unknown command {command!r}")
    # Always UTF-8 bytes: the Windows console encoding (cp1251) would mangle Cyrillic descriptions.
    sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
