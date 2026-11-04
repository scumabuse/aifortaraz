"""Tests of safety_ai.config.zones.load_zones_file (``--zones`` files, contract 4.4 structure)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from safety_ai.config import ConfigError, load_zones_file

CONTRACT_EXAMPLE: dict[str, Any] = {
    "camera": {
        "id": "cam-1",
        "name": "Press #2",
        "stream_url": "rtsp://192.168.1.10/live",
        "stream_type": "rtsp",
    },
    "frame_size_hint": [1280, 720],
    "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
    "ppe_types": [
        "helmet",
        "vest",
        "glasses",
        "gloves",
        "mask",
        "boots",
        "earmuffs",
        "harness",
        "face_shield",
    ],
    "zones": [
        {
            "id": "zone-press-2",
            "name": "Press #2",
            "type": "danger",
            "polygon": [[0.10, 0.20], [0.55, 0.18], [0.60, 0.80], [0.12, 0.85]],
            "rules": {
                "required_ppe": ["helmet", "glasses"],
                "entry_forbidden": False,
                "max_persons": 2,
                "min_vehicle_distance_m": None,
                "schedule": None,
                "confirm_frames": 8,
                "cooldown_sec": 60,
                "severity_override": None,
            },
        }
    ],
    "version": 42,
    "updated_at": "2026-11-03T12:00:00Z",
}

YAML_ZONES = """\
# Zones for the demo; the camera block is optional.
thresholds:
  confirm_frames: 6
ppe_types: [helmet, vest]
zones:
  - id: z-ppe
    name: Assembly
    type: ppe
    polygon: [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    rules:
      required_ppe: [helmet, vest]
      schedule:
        - days: [1, 2, 3, 4, 5]
          from: 08:00
          to: 20:00
  - id: z-danger
    polygon: [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0]]
    rules:
      entry_forbidden: true
"""


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def errors_of(path: Path, camera_id: str = "cam-1") -> list[str]:
    with pytest.raises(ConfigError) as info:
        load_zones_file(path, camera_id)
    return info.value.messages


def test_yaml_file_without_a_camera_block(tmp_path: Path) -> None:
    config = load_zones_file(write(tmp_path / "zones.yaml", YAML_ZONES), "cam-7")
    assert config.camera.id == "cam-7"
    assert config.thresholds.confirm_frames == 6
    assert config.ppe_types == ["helmet", "vest"]
    assert [zone.id for zone in config.zones] == ["z-ppe", "z-danger"]
    schedule = config.zones[0].rules.schedule
    assert schedule is not None
    assert (schedule[0].from_, schedule[0].to) == ("08:00", "20:00")
    assert config.zones[1].rules.entry_forbidden is True


def test_contract_example_as_json_with_the_camera_id_overridden(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write(tmp_path / "zones.json", json.dumps(CONTRACT_EXAMPLE))
    with caplog.at_level(logging.INFO, logger="safety_ai.config.zones"):
        config = load_zones_file(path, "cam-9")
    assert config.camera.id == "cam-9"
    assert config.camera.name == "Press #2"
    assert config.version == 42
    assert config.zones[0].rules.required_ppe == ["helmet", "glasses"]
    assert config.zones[0].rules.max_persons == 2
    assert "names camera 'cam-1'; using it for 'cam-9'" in caplog.text


def test_yml_extension_and_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "zones.yml"
    path.write_bytes(b"\xef\xbb\xbf" + YAML_ZONES.encode("utf-8"))
    assert len(load_zones_file(path, "cam-1").zones) == 2
    json_path = tmp_path / "zones.json"
    json_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(CONTRACT_EXAMPLE).encode("utf-8"))
    assert load_zones_file(json_path, "cam-1").zones[0].id == "zone-press-2"


def test_empty_file_means_no_zones(tmp_path: Path) -> None:
    config = load_zones_file(write(tmp_path / "zones.yaml", "# nothing yet\n"), "cam-1")
    assert config.camera.id == "cam-1"
    assert config.zones == []


def test_unknown_fields_are_ignored(tmp_path: Path) -> None:
    data = {**CONTRACT_EXAMPLE, "future_field": {"x": 1}}
    config = load_zones_file(write(tmp_path / "zones.json", json.dumps(data)), "cam-1")
    assert config.zones[0].id == "zone-press-2"


def test_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    assert errors_of(path) == [f"zones file not found: {path}"]


def test_directory_instead_of_a_file(tmp_path: Path) -> None:
    path = tmp_path / "zones.yaml"
    path.mkdir()
    assert errors_of(path) == [f"zones file is not a regular file: {path}"]


def test_wrong_extension(tmp_path: Path) -> None:
    path = write(tmp_path / "zones.txt", YAML_ZONES)
    assert errors_of(path) == [f"zones file must have a .yaml, .yml or .json extension: {path}"]


def test_yaml_syntax_error_with_line_and_column(tmp_path: Path) -> None:
    path = write(tmp_path / "zones.yaml", "zones:\n  - id: z1\n   polygon: []\n")
    assert errors_of(path) == [
        f"{path}: YAML syntax error at line 3, column 4: "
        "expected <block end>, but found '<block mapping start>'"
    ]


def test_json_syntax_error_with_line_and_column(tmp_path: Path) -> None:
    path = write(tmp_path / "zones.json", '{\n  "zones": [\n    {"id": }\n  ]\n}\n')
    assert errors_of(path) == [f"{path}: JSON syntax error at line 3, column 12: Expecting value"]


def test_duplicate_json_key(tmp_path: Path) -> None:
    path = write(tmp_path / "zones.json", '{"zones": [], "zones": []}')
    assert errors_of(path) == [f"{path}: duplicate key 'zones' in a JSON object"]


def test_top_level_list_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "zones.json", json.dumps([CONTRACT_EXAMPLE]))
    assert errors_of(path) == [
        f"{path}: the top level must be a mapping with the structure of one camera's config "
        "response (camera, thresholds, ppe_types, zones, ...), not a list"
    ]


def test_camera_block_must_be_a_mapping(tmp_path: Path) -> None:
    path = write(tmp_path / "zones.yaml", "camera: cam-1\nzones: []\n")
    assert errors_of(path) == [f"{path}: camera: Input should be a mapping"]


def test_validation_errors_are_prefixed_with_the_file(tmp_path: Path) -> None:
    path = write(
        tmp_path / "zones.yaml",
        "zones:\n"
        "  - id: z1\n"
        "    polygon: [[0, 0], [1, 0], [2, 2]]\n"
        "  - id: z2\n"
        "    polygon: [[0, 0], [1, 0], [1, 1]]\n"
        "    rules:\n"
        "      max_persons: -1\n"
        "      schedule: [{days: [8], from: '08:00', to: '09:00'}]\n",
    )
    assert errors_of(path) == [
        f"{path}: zones.0.polygon: polygon point [2, 2] is outside 0..1",
        f"{path}: zones.1.rules.max_persons: Input should be greater than or equal to 0",
        f"{path}: zones.1.rules.schedule.0.days.0: Input should be less than or equal to 7",
    ]


def test_duplicate_zone_ids_are_a_bare_message(tmp_path: Path) -> None:
    zone = {"id": "z1", "polygon": [[0, 0], [1, 0], [1, 1]]}
    path = write(tmp_path / "zones.json", json.dumps({"zones": [zone, zone]}))
    assert errors_of(path) == [f"{path}: duplicate zone ids: ['z1']"]


def test_invalid_camera_id_argument_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path / "zones.yaml", YAML_ZONES)
    assert errors_of(path, camera_id="bad id") == [
        f"{path}: camera.id: String should match pattern '^[A-Za-z0-9][A-Za-z0-9_.:-]*$'"
    ]
