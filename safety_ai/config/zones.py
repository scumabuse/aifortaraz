"""Zones files for the demo and offline runs (``--zones``, ``cameras[].zones_file``; spec 7.2).

Zones and rules normally come from the backend config endpoint (contract 4.4). Without a backend
they come from a YAML or JSON file with the same structure as that endpoint's response for one
camera: ``camera`` (optional here), ``frame_size_hint``, ``thresholds``, ``ppe_types``, ``zones``,
``version`` and ``updated_at``. ``camera.id`` is always replaced by the camera the file is used
for, so one file can serve any camera.

Public API::

    load_zones_file(path: Path, camera_id: str) -> CameraConfig

Problems raise :class:`safety_ai.config.loader.ConfigError` with one readable line each, prefixed
with the file path (``tools/demo_zones.yaml: zones.0.polygon: polygon point [2, 2] is outside
0..1``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from safety_ai.backend.schemas import CameraConfig
from safety_ai.config.loader import (
    ConfigError,
    format_validation_error,
    parse_yaml_document,
    read_config_text,
)

__all__ = ["ZONES_SUFFIXES", "load_zones_file"]

ZONES_SUFFIXES: Final[tuple[str, ...]] = (".yaml", ".yml", ".json")
"""Accepted extensions of a zones file."""

_log = logging.getLogger(__name__)


class _DuplicateKeyError(ValueError):
    """A JSON object that defines the same key twice."""

    def __init__(self, key: str) -> None:
        """Remember the duplicated key."""
        super().__init__(key)
        self.key = key


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _parse_json(text: str, source: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            [f"{source}: JSON syntax error at line {exc.lineno}, column {exc.colno}: {exc.msg}"]
        ) from None
    except _DuplicateKeyError as exc:
        raise ConfigError([f"{source}: duplicate key {exc.key!r} in a JSON object"]) from None


def _check_zones_path(path: Path) -> None:
    if not path.exists():
        raise ConfigError([f"zones file not found: {path}"])
    if not path.is_file():
        raise ConfigError([f"zones file is not a regular file: {path}"])
    if path.suffix.lower() not in ZONES_SUFFIXES:
        raise ConfigError([f"zones file must have a .yaml, .yml or .json extension: {path}"])


def _camera_payload(document: Any, camera_id: str, source: str) -> dict[str, Any]:
    """Return the config response with ``camera.id`` set to ``camera_id``."""
    if document is None:
        document = {}
    if not isinstance(document, dict):
        kind = "a list" if isinstance(document, list) else f"a {type(document).__name__}"
        raise ConfigError(
            [
                f"{source}: the top level must be a mapping with the structure of one camera's "
                f"config response (camera, thresholds, ppe_types, zones, ...), not {kind}"
            ]
        )
    camera = document.get("camera")
    if camera is None:
        camera = {}
    elif not isinstance(camera, dict):
        raise ConfigError([f"{source}: camera: Input should be a mapping"])
    file_camera = camera.get("id")
    if file_camera is not None and file_camera != camera_id:
        _log.info("zones file %s names camera %r; using it for %r", source, file_camera, camera_id)
    return {**document, "camera": {**camera, "id": camera_id}}


def load_zones_file(path: Path, camera_id: str) -> CameraConfig:
    """Load the zones and rules of ``camera_id`` from a YAML or JSON file.

    Args:
        path: A ``.yaml``, ``.yml`` or ``.json`` file with the contract's config structure.
        camera_id: The camera the zones are for; it replaces ``camera.id`` from the file.

    Returns:
        The validated camera config (zones, rules, thresholds, PPE types).

    Raises:
        ConfigError: missing or unreadable file, a syntax error (with line and column) or invalid
            contents; every line is prefixed with the file path.
    """
    file = Path(path)
    _check_zones_path(file)
    text = read_config_text(file, what="zones file")
    source = str(file)
    if file.suffix.lower() == ".json":
        document = _parse_json(text, source)
    else:
        document = parse_yaml_document(text, source=source)
    payload = _camera_payload(document, camera_id, source)
    try:
        return CameraConfig.model_validate(payload)
    except ValidationError as exc:
        lines = format_validation_error(exc, model=CameraConfig)
        raise ConfigError([f"{source}: {line}" for line in lines]) from None
