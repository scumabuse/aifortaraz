"""Scenario model: readable errors, time, interpolation, stepping, configs, expectations."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from safety_ai.backend.schemas import PPE_CODE_VALUES
from safety_ai.capture.base import SourceError
from safety_ai.capture.scenario import (
    DEFAULT_CAMERA_ID,
    PPE_STATES,
    TRACK_LABELS,
    ExpectedAbsence,
    ExpectedEvent,
    Scenario,
    ScenarioError,
    load_scenario,
    parse_scenario,
    validation_messages,
)
from safety_ai.config.loader import parse_yaml_document
from safety_ai.inference.types import VEHICLE_LABELS
from safety_ai.logic.poses import POSE_NAMES, box_for_pose

FRAME_W, FRAME_H = 1280, 720
START = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)


def zone(zone_id: str = "z1", **rules: Any) -> dict[str, Any]:
    return {
        "id": zone_id,
        "name": f"Zone {zone_id}",
        "type": "ppe",
        "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        "rules": rules or {"required_ppe": ["helmet"]},
    }


def person(track_id: int = 1, *keyframes: dict[str, Any], **fields: Any) -> dict[str, Any]:
    frames = list(keyframes) or [{"t": 0, "feet": [0.5, 0.9], "height": 0.5}]
    return {"id": track_id, "label": "person", "keyframes": frames, **fields}


def vehicle(
    track_id: int = 5, label: str = "forklift", *keyframes: dict[str, Any]
) -> dict[str, Any]:
    frames = list(keyframes) or [{"t": 0, "box": [0.6, 0.2, 0.3, 0.3]}]
    return {"id": track_id, "label": label, "keyframes": frames}


def document(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "unit",
        "frame_size": [FRAME_W, FRAME_H],
        "fps": 10,
        "duration_s": 10,
        "start_utc": "2026-11-03T14:00:00Z",
        "config": {"zones": [zone()]},
        "tracks": [person(1), vehicle(5)],
    }
    data.update(overrides)
    return data


def event(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "ppe_violation",
        "zone": "z1",
        "track": 1,
        "start_s": 1.0,
        "start_tol_s": 0.5,
        "end_s": None,
    }
    data.update(overrides)
    return data


def problems(data: Any) -> list[str]:
    with pytest.raises(ScenarioError) as excinfo:
        parse_scenario(data, source="s.yaml")
    return excinfo.value.messages


def keyframe(data: dict[str, Any], track: int = 0, index: int = 0) -> dict[str, Any]:
    frame: dict[str, Any] = data["tracks"][track]["keyframes"][index]
    return frame


def write(tmp_path: Path, text: str, name: str = "scenario.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------
# Defaults and loading
# ---------------------------------------------------------------------------------------------


def test_a_minimal_scenario_gets_the_documented_defaults() -> None:
    scenario = parse_scenario(document())
    assert scenario.camera_id == DEFAULT_CAMERA_ID == "cam-1"
    assert scenario.timezone == "UTC"
    assert scenario.description == ""
    assert scenario.config_updates == []
    assert scenario.expected_events == []
    assert scenario.expected_absent == []
    assert scenario.start_utc == START
    assert scenario.n_frames == 100
    assert scenario.frame_period_s == pytest.approx(0.1)
    track = scenario.track(1)
    assert track.label == "person"
    assert track.conf == 0.9
    assert track.is_person
    assert not scenario.track(5).is_person


def test_constants_match_the_taxonomy_and_the_poses() -> None:
    assert tuple(TRACK_LABELS) == ("person", *VEHICLE_LABELS)
    assert PPE_STATES == ("present", "missing", "unknown")
    assert set(POSE_NAMES) == {
        "standing",
        "walking",
        "bending",
        "kneeling",
        "sitting",
        "falling",
        "lying",
    }


def test_load_scenario_reads_yaml_with_the_config_loader(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
name: from_yaml
frame_size: [640, 360]
fps: 12.5
duration_s: 2
start_utc: 2026-11-03T19:00:00+05:00
camera_id: cam-7
timezone: Asia/Yekaterinburg
config:
  zones:
    - id: z1
      polygon: [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]
      rules:
        required_ppe: [helmet]
        schedule:
          - {days: [2], from: 08:00, to: 20:00}
tracks:
  - id: 3
    keyframes:
      - {t: 0, box: [0.1, 0.1, 0.2, 0.5]}
""",
    )
    scenario = load_scenario(path)
    assert scenario.name == "from_yaml"
    assert scenario.camera_id == "cam-7"
    assert scenario.start_utc == START
    assert scenario.n_frames == 25
    window = scenario.config_at(0.0).zones[0].rules.schedule
    assert window is not None
    assert (window[0].from_, window[0].to) == ("08:00", "20:00")
    assert load_scenario(str(path)).name == "from_yaml"


def test_a_missing_file_is_a_readable_error(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(tmp_path / "nope.yaml")
    assert excinfo.value.messages == [f"scenario file not found: {tmp_path / 'nope.yaml'}"]


def test_yaml_syntax_errors_name_the_line_and_column(tmp_path: Path) -> None:
    path = write(tmp_path, "name: x\nfps: [10\nduration_s: 3\n")
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    (message,) = excinfo.value.messages
    assert message.startswith(f"{path}: YAML syntax error at line ")
    assert re.search(r"line \d+, column \d+", message)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "name: a\nname: b\n")
    with pytest.raises(ScenarioError, match="found duplicate key 'name'"):
        load_scenario(path)


@pytest.mark.parametrize(
    ("text", "kind"),
    [("", "an empty document"), ("- 1\n- 2\n", "a list"), ("just text\n", "a string")],
)
def test_the_top_level_must_be_a_mapping(tmp_path: Path, text: str, kind: str) -> None:
    path = write(tmp_path, text)
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    (message,) = excinfo.value.messages
    assert message.startswith(f"{path}: the top level must be a mapping")
    assert message.endswith(f"not {kind}")


def test_the_file_path_prefixes_every_message(tmp_path: Path) -> None:
    path = write(tmp_path, "name: x\nfps: 0\n")
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert len(excinfo.value.messages) >= 4
    assert all(line.startswith(f"{path}: ") for line in excinfo.value.messages)
    assert f"{path}: fps: Input should be greater than 0" in excinfo.value.messages
    assert f"{path}: frame_size: Field required" in excinfo.value.messages


def test_scenario_errors_are_source_errors_with_one_line_per_problem() -> None:
    error = ScenarioError(["a: one", "b: two"])
    assert isinstance(error, SourceError)
    assert isinstance(error, ValueError)
    assert error.messages == ["a: one", "b: two"]
    assert str(error) == "a: one\nb: two"
    assert ScenarioError("single").messages == ["single"]
    assert ScenarioError([]).messages == ["invalid scenario"]


def test_validation_messages_work_on_direct_model_validation() -> None:
    data = document()
    keyframe(data)["pose"] = "flying"
    with pytest.raises(ValidationError) as excinfo:
        Scenario.model_validate(data)
    assert validation_messages(excinfo.value) == [
        "tracks.0.keyframes.0.pose: unknown pose 'flying'; expected one of: "
        + ", ".join(POSE_NAMES)
    ]


# ---------------------------------------------------------------------------------------------
# Readable validation errors
# ---------------------------------------------------------------------------------------------

Mutation = Callable[[dict[str, Any]], None]


def _set(path: str, value: Any) -> Mutation:
    """Mutation setting a dotted path (digits index lists) in the document."""

    def apply(data: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        node: Any = data
        for part in parents:
            node = node[int(part)] if part.isdigit() else node[part]
        if last.isdigit():
            node[int(last)] = value
        else:
            node[last] = value

    return apply


def _delete(path: str) -> Mutation:
    def apply(data: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        node: Any = data
        for part in parents:
            node = node[int(part)] if part.isdigit() else node[part]
        del node[last]

    return apply


def _all(*mutations: Mutation) -> Mutation:
    def apply(data: dict[str, Any]) -> None:
        for mutation in mutations:
            mutation(data)

    return apply


ERROR_CASES: list[tuple[str, Mutation, str]] = [
    ("missing fps", _delete("fps"), "fps: Field required"),
    ("unknown top-level field", _set("fsp", 10), "fsp: unknown field"),
    ("zero fps", _set("fps", 0), "fps: Input should be greater than 0"),
    (
        "bad frame size",
        _set("frame_size", [0, 720]),
        "frame_size.0: Input should be greater than 0",
    ),
    ("naive start", _set("start_utc", "2026-11-03T14:00:00"), "start_utc: must include a timezone"),
    ("bad timezone", _set("timezone", "Mars/Base"), "timezone: unknown timezone 'Mars/Base'"),
    ("bad camera id", _set("camera_id", "cam 1"), "camera_id: String should match pattern"),
    (
        "keyframe typo",
        _set("tracks.0.keyframes.0.posee", "lying"),
        "tracks.0.keyframes.0.posee: unknown field",
    ),
    (
        "box and feet",
        _set("tracks.0.keyframes.0.box", [0.1, 0.1, 0.1, 0.1]),
        "tracks.0.keyframes.0: give either box [x, y, w, h] or feet [x, y] with height, not both",
    ),
    (
        "height without feet",
        _set("tracks.1.keyframes.0.height", 0.5),
        "tracks.1.keyframes.0.height: is only allowed together with feet",
    ),
    (
        "first feet keyframe without height",
        _delete("tracks.0.keyframes.0.height"),
        "tracks.0.keyframes.0.height: is required on the first feet keyframe",
    ),
    (
        "first keyframe without position",
        _set("tracks.0.keyframes", [{"t": 0, "pose": "lying"}]),
        "tracks.0.keyframes.0: the first keyframe needs a position",
    ),
    (
        "times not increasing",
        _set("tracks.0.keyframes", [{"t": 2, "box": [0.1, 0.1, 0.1, 0.4]}, {"t": 2}]),
        "tracks.0.keyframes.1.t: must be greater than the previous keyframe's t (2)",
    ),
    (
        "mixed positions",
        _set(
            "tracks.0.keyframes",
            [{"t": 0, "feet": [0.5, 0.9], "height": 0.4}, {"t": 1, "box": [0.1, 0.1, 0.1, 0.4]}],
        ),
        "tracks.0.keyframes.1: sets box but keyframes.0 sets feet; every keyframe of a track",
    ),
    (
        "box outside the frame",
        _set("tracks.1.keyframes.0.box", [0.9, 0.1, 0.2, 0.2]),
        "tracks.1.keyframes.0.box: box [x, y, w, h] must lie inside the frame with a positive size",
    ),
    (
        "box without size",
        _set("tracks.1.keyframes.0.box", [0.1, 0.1, 0.0, 0.2]),
        "tracks.1.keyframes.0.box: box [x, y, w, h] must lie inside the frame",
    ),
    (
        "feet outside 0..1",
        _set("tracks.0.keyframes.0.feet", [1.5, 0.9]),
        "tracks.0.keyframes.0.feet.0: Input should be less than or equal to 1",
    ),
    (
        "unknown pose",
        _set("tracks.0.keyframes.0.pose", "flying"),
        "tracks.0.keyframes.0.pose: unknown pose 'flying'; expected one of: standing, walking",
    ),
    (
        "unknown ppe state",
        _set("tracks.0.keyframes.0.ppe", {"helmet": "maybe"}),
        "tracks.0.keyframes.0.ppe.helmet: Input should be 'present', 'missing' or 'unknown'",
    ),
    (
        "unknown ppe code",
        _set("tracks.0.keyframes.0.ppe", {"helmett": "present"}),
        "tracks.0.keyframes.0.ppe.helmett",
    ),
    ("unknown label", _set("tracks.0.label", "bicycle"), "tracks.0.label: unknown label 'bicycle'"),
    ("bad confidence", _set("tracks.0.conf", 1.5), "tracks.0.conf: Input should be less than or"),
    (
        "vehicle with feet",
        _set("tracks.1.keyframes", [{"t": 0, "feet": [0.5, 0.9], "height": 0.3}]),
        "tracks.1.keyframes.0.feet: vehicles are placed with box [x, y, w, h]",
    ),
    (
        "vehicle with a pose",
        _set("tracks.1.keyframes.0.pose", "lying"),
        "tracks.1.keyframes.0.pose: is allowed for persons only",
    ),
    (
        "vehicle with ppe",
        _set("tracks.1.keyframes.0.ppe", {"helmet": "present"}),
        "tracks.1.keyframes.0.ppe: is allowed for persons only",
    ),
    (
        "duplicate track ids",
        _set("tracks.1.id", 1),
        "tracks.1.id: duplicate track id 1 (also tracks.0.id)",
    ),
    (
        "keyframe after the end",
        _set("tracks.1.keyframes", [{"t": 0, "box": [0.1, 0.1, 0.2, 0.2]}, {"t": 12}]),
        "tracks.1.keyframes.1.t: 12 is after the end of the scenario (duration_s 10)",
    ),
    (
        "config typo",
        _set("config.zones.0.rules.requried_ppe", ["helmet"]),
        "config.zones.0.rules.requried_ppe: unknown field",
    ),
    (
        "config top-level typo",
        _set("config.zone", []),
        "config.zone: unknown field",
    ),
    (
        "schedule typo",
        _set("config.zones.0.rules.schedule", [{"days": [1], "from": "08:00", "till": "09:00"}]),
        "config.zones.0.rules.schedule.0.till: unknown field",
    ),
    (
        "unknown required ppe code",
        _set("config.zones.0.rules.required_ppe", ["helmet", "helmett"]),
        "config.zones.0.rules.required_ppe.1: unknown PPE code 'helmett'; expected one of: helmet",
    ),
    (
        "unknown ppe type",
        _set("config.ppe_types", ["vest", "hat"]),
        "config.ppe_types.1: unknown PPE code 'hat'",
    ),
    (
        "camera id mismatch",
        _set("config.camera", {"id": "cam-9"}),
        "config.camera.id: 'cam-9' differs from camera_id 'cam-1'; leave it out",
    ),
    (
        "polygon outside",
        _set("config.zones.0.polygon", [[0, 0], [2, 0], [1, 1]]),
        "config.zones.0.polygon: polygon point [2, 0] is outside 0..1",
    ),
    (
        "duplicate zones",
        _set("config.zones", [zone("z1"), zone("z1")]),
        "config: duplicate zone ids: ['z1']",
    ),
    ("missing config", _delete("config"), "config: Field required"),
    (
        "update typo",
        _set("config_updates", [{"at_s": 5, "config": {"zones": [{**zone(), "rulez": {}}]}}]),
        "config_updates.0.config.zones.0.rulez: unknown field",
    ),
    (
        "update without config",
        _set("config_updates", [{"at_s": 5}]),
        "config_updates.0.config: Field required",
    ),
    (
        "updates out of order",
        _set("config_updates", [{"at_s": 5, "config": {}}, {"at_s": 4, "config": {}}]),
        "config_updates.1.at_s: must be greater than the previous update's at_s (5)",
    ),
    (
        "update after the end",
        _set("config_updates", [{"at_s": 10, "config": {}}]),
        "config_updates.0.at_s: 10 is not before the end of the scenario (duration_s 10)",
    ),
    (
        "unknown event type",
        _set("expected_events", [event(type="slip")]),
        "expected_events.0.type: Input should be 'ppe_violation'",
    ),
    (
        "event zone left out",
        _set("expected_events", [{k: v for k, v in event().items() if k != "zone"}]),
        "expected_events.0.zone: Field required",
    ),
    (
        "event end left out",
        _set("expected_events", [{k: v for k, v in event().items() if k != "end_s"}]),
        "expected_events.0.end_s: Field required",
    ),
    (
        "event end without tolerance",
        _set("expected_events", [event(end_s=3.0)]),
        "expected_events.0.end_tol_s: is required when end_s is set",
    ),
    (
        "event ends before it starts",
        _set("expected_events", [event(start_s=3.0, end_s=2.0, end_tol_s=0.5)]),
        "expected_events.0.end_s: must not be before start_s (3)",
    ),
    (
        "event details typo",
        _set("expected_events", [event(details={"missing": ["helmet"]})]),
        "expected_events.0.details.missing: is not a details field of ppe_violation (fields: "
        "missing_ppe, present_ppe, boxes, frame_size)",
    ),
    (
        "event zone unknown",
        _set("expected_events", [event(zone="nope")]),
        "expected_events.0.zone: unknown zone 'nope' (zones: z1)",
    ),
    (
        "event track unknown",
        _set("expected_events", [event(track=9)]),
        "expected_events.0.track: unknown track 9 (tracks: 1, 5)",
    ),
    (
        "event track is a vehicle",
        _set("expected_events", [event(track=5)]),
        "expected_events.0.track: track 5 is a forklift; events carry a person's track id",
    ),
    (
        "event starts after the end",
        _set("expected_events", [event(start_s=10.0)]),
        "expected_events.0.start_s: is not before the end of the scenario (duration_s 10)",
    ),
    (
        "event ends after the end",
        _set("expected_events", [event(end_s=11.0, end_tol_s=0.5)]),
        "expected_events.0.end_s: is after the end of the scenario (duration_s 10); use null",
    ),
    (
        "absence zone unknown",
        _set("expected_absent", [{"type": "zone_intrusion", "zone": "z9"}]),
        "expected_absent.0.zone: unknown zone 'z9' (zones: z1)",
    ),
    (
        "absence track unknown",
        _set("expected_absent", [{"type": "zone_intrusion", "track": 4}]),
        "expected_absent.0.track: unknown track 4 (tracks: 1, 5)",
    ),
    (
        "absence typo",
        _set("expected_absent", [{"type": "zone_intrusion", "tracks": 1}]),
        "expected_absent.0.tracks: unknown field",
    ),
]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [pytest.param(mutation, expected, id=name) for name, mutation, expected in ERROR_CASES],
)
def test_validation_errors_are_readable(mutation: Mutation, expected: str) -> None:
    data = document()
    mutation(data)
    lines = problems(data)
    assert any(line.startswith(f"s.yaml: {expected}") for line in lines), lines


def test_independent_problems_are_reported_together() -> None:
    data = document()
    _all(
        _delete("fps"),
        _set("tracks.0.keyframes.0.pose", "flying"),
        _set("tracks.1.keyframes.0.ppe", {"helmet": "present"}),
        _set("config.zones.0.rules.entry_forbiden", True),
        _set("expected_events", [event(end_s=3.0)]),
    )(data)
    lines = problems(data)
    assert lines == [
        "s.yaml: fps: Field required",
        "s.yaml: config.zones.0.rules.entry_forbiden: unknown field",
        "s.yaml: tracks.0.keyframes.0.pose: unknown pose 'flying'; expected one of: "
        + ", ".join(POSE_NAMES),
        "s.yaml: tracks.1.keyframes.0.ppe: is allowed for persons only",
        "s.yaml: expected_events.0.end_tol_s: is required when end_s is set",
    ]


def test_a_feet_keyframe_may_keep_the_previous_height_and_a_valid_box_may_touch_the_border() -> (
    None
):
    data = document()
    data["tracks"][0]["keyframes"].append({"t": 2, "feet": [0.6, 0.9]})
    data["tracks"][1]["keyframes"] = [{"t": 0, "box": [0.7, 0.6, 0.3, 0.4]}]
    scenario = parse_scenario(data)
    assert scenario.track(1).keyframes[1].height is None
    assert scenario.track_state(5, 0.0) is not None


# ---------------------------------------------------------------------------------------------
# Time model
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration", "fps", "frames"),
    [(10, 10, 100), (0.3, 10, 3), (1.05, 10, 11), (2, 12.5, 25), (0.01, 10, 1), (30, 12, 360)],
)
def test_n_frames_covers_the_duration(duration: float, fps: float, frames: int) -> None:
    scenario = parse_scenario(document(duration_s=duration, fps=fps, tracks=[]))
    assert scenario.n_frames == frames
    assert scenario.time_of(frames - 1) < duration


def test_time_of_maps_frame_ids_and_wraps_when_looping() -> None:
    scenario = parse_scenario(document(duration_s=3, fps=10))
    assert scenario.time_of(0) == 0.0
    assert scenario.time_of(25) == 2.5
    assert scenario.time_of(31) == pytest.approx(3.1), "past the end without loop: t keeps going"
    assert scenario.time_of(30, loop=True) == 0.0
    assert scenario.time_of(67, loop=True) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="frame_id must be >= 0"):
        scenario.time_of(-1)


def test_utc_and_scenario_seconds_convert_both_ways() -> None:
    scenario = parse_scenario(document(start_utc="2026-11-03T19:00:00+05:00"))
    assert scenario.start_utc == START
    assert scenario.start_utc.tzinfo is timezone.utc
    assert scenario.utc_at(1.5) == START + timedelta(seconds=1.5)
    assert scenario.seconds_at(START + timedelta(seconds=7.25)) == 7.25
    local = (START + timedelta(seconds=2)).astimezone(timezone(timedelta(hours=-3)))
    assert scenario.seconds_at(local) == 2.0


# ---------------------------------------------------------------------------------------------
# Tracks: interpolation and stepping
# ---------------------------------------------------------------------------------------------


def test_box_positions_interpolate_linearly_and_hold_after_the_last_keyframe() -> None:
    track = vehicle(
        5,
        "truck",
        {"t": 2, "box": [0.0, 0.2, 0.2, 0.2]},
        {"t": 6, "box": [0.4, 0.4, 0.4, 0.2]},
    )
    scenario = parse_scenario(document(tracks=[track]))
    assert scenario.track_state(5, 1.9) is None, "a track does not exist before its first keyframe"
    first = scenario.track_state(5, 2.0)
    assert first is not None
    assert first.box == pytest.approx((0.0, 0.2, 0.2, 0.2))
    middle = scenario.track_state(5, 3.0)
    assert middle is not None
    assert middle.box == pytest.approx((0.1, 0.25, 0.25, 0.2))
    assert middle.label == "truck"
    assert middle.pose is None
    assert middle.phase == 0.0
    assert dict(middle.ppe) == {}
    for t in (6.0, 9.9):
        later = scenario.track_state(5, t)
        assert later is not None
        assert later.box == pytest.approx((0.4, 0.4, 0.4, 0.2))


def test_feet_positions_interpolate_and_use_the_box_of_the_current_pose() -> None:
    track = person(
        1,
        {"t": 0, "feet": [0.2, 0.9], "height": 0.4},
        {"t": 4, "feet": [0.6, 0.8], "height": 0.6},
        {"t": 6, "feet": [0.8, 0.8], "pose": "lying"},
    )
    scenario = parse_scenario(document(tracks=[track]))
    state = scenario.track_state(1, 2.0)
    assert state is not None
    expected = box_for_pose(
        "standing", (0.4, 0.85), person_height_px=0.5 * FRAME_H, frame_w=FRAME_W, frame_h=FRAME_H
    )
    assert state.box == pytest.approx(expected)
    held_height = scenario.track_state(1, 5.0)
    assert held_height is not None
    assert held_height.box == pytest.approx(
        box_for_pose(
            "standing", (0.7, 0.8), person_height_px=0.6 * FRAME_H, frame_w=FRAME_W, frame_h=FRAME_H
        )
    )
    lying = scenario.track_state(1, 6.0)
    assert lying is not None
    x, y, w, h = lying.box
    assert (x + w / 2, y + h) == pytest.approx((0.8, 0.8)), "the feet point is kept"
    assert w * FRAME_W / (h * FRAME_H) > 3.5, "lying gives a wide, low box"


def test_pose_visibility_and_each_ppe_code_step_independently() -> None:
    track = person(
        1,
        {
            "t": 0,
            "feet": [0.5, 0.9],
            "height": 0.5,
            "ppe": {"helmet": "missing", "vest": "present"},
        },
        {"t": 2, "pose": "bending", "ppe": {"helmet": "present"}},
        {"t": 4, "visible": False},
        {"t": 5, "visible": True, "ppe": {"gloves": "missing", "vest": "unknown"}},
    )
    scenario = parse_scenario(document(tracks=[track]))

    def at(t: float) -> tuple[str | None, dict[str, str]] | None:
        state = scenario.track_state(1, t)
        return None if state is None else (state.pose, dict(state.ppe))

    before = at(1.9)
    assert before is not None
    assert before[0] == "standing"
    assert (before[1]["helmet"], before[1]["vest"], before[1]["gloves"]) == (
        "missing",
        "present",
        "unknown",
    )
    after = at(2.0)
    assert after is not None
    assert after[0] == "bending"
    assert (after[1]["helmet"], after[1]["vest"]) == ("present", "present")
    assert at(4.0) is None
    assert at(4.9) is None
    back = at(5.0)
    assert back is not None
    assert back[0] == "bending"
    assert (back[1]["helmet"], back[1]["vest"], back[1]["gloves"]) == (
        "present",
        "unknown",
        "missing",
    )
    assert set(back[1]) == set(PPE_CODE_VALUES)


def test_keyframe_times_tolerate_float_frame_times() -> None:
    track = person(1, {"t": 0, "box": [0.1, 0.1, 0.1, 0.5]}, {"t": 0.3, "visible": False})
    scenario = parse_scenario(document(tracks=[track]))
    assert scenario.track_state(1, scenario.time_of(2)) is not None
    assert scenario.time_of(3) == pytest.approx(0.3)
    assert scenario.track_state(1, scenario.time_of(3)) is None


def test_walking_phase_follows_time() -> None:
    track = person(1, {"t": 0, "feet": [0.5, 0.9], "height": 0.5, "pose": "walking"})
    scenario = parse_scenario(document(tracks=[track]))
    phases = []
    for t in (0.0, 0.25, 1.5, 2.75):
        state = scenario.track_state(1, t)
        assert state is not None
        phases.append(state.phase)
    assert phases == pytest.approx([0.0, 0.25, 0.5, 0.75])
    standing = parse_scenario(document()).track_state(1, 1.25)
    assert standing is not None
    assert standing.phase == 0.0


def test_boxes_are_clamped_and_tracks_outside_the_frame_are_not_detected() -> None:
    track = person(
        1,
        {"t": 0, "feet": [0.01, 0.95], "height": 0.5},
        {"t": 1, "feet": [0.5, 0.0]},
    )
    scenario = parse_scenario(document(tracks=[track]))
    border = scenario.track_state(1, 0.0)
    assert border is not None
    assert border.box[0] == 0.0, "clamped at the left border like a detector box"
    assert border.box[0] + border.box[2] <= 1.0
    assert scenario.track_state(1, 1.0) is None, "a box entirely above the frame is not detected"


def test_scene_lists_visible_tracks_in_id_order_and_unknown_ids_are_absent() -> None:
    tracks = [
        vehicle(9),
        person(2),
        person(5, {"t": 0, "box": [0.1, 0.1, 0.1, 0.4], "visible": False}),
    ]
    scenario = parse_scenario(document(tracks=tracks))
    assert [state.track_id for state in scenario.scene_at(1.0)] == [2, 9]
    assert scenario.track_state(77, 1.0) is None
    with pytest.raises(KeyError):
        scenario.track(77)
    assert scenario.track(5).keyframes[0].visible is False


# ---------------------------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------------------------


def test_config_at_switches_on_the_first_frame_at_or_after_each_update() -> None:
    updates = [
        {"at_s": 2.5, "config": {"zones": [zone("z2")]}},
        {"at_s": 6, "config": {"zones": []}},
    ]
    scenario = parse_scenario(document(config_updates=updates))
    assert scenario.config_update_times == (2.5, 6.0)
    base = scenario.config_at(0.0)
    assert [z.id for z in base.zones] == ["z1"]
    assert scenario.config_at(scenario.time_of(24)) is base
    second = scenario.config_at(scenario.time_of(25))
    assert [z.id for z in second.zones] == ["z2"]
    assert scenario.config_at(5.99) is second
    assert scenario.config_at(6.0).zones == []
    assert scenario.config_at(-1.0) is base
    assert scenario.zone_ids == frozenset({"z1", "z2"})


def test_configs_get_the_camera_id_frame_size_hint_and_increasing_versions() -> None:
    config = {"camera": {"id": "cam-4", "name": "Press 4"}, "version": 7, "zones": [zone()]}
    updates = [
        {"at_s": 1, "config": {"zones": []}},
        {"at_s": 2, "config": {"version": 20, "frame_size_hint": [640, 360]}},
        {"at_s": 3, "config": {}},
    ]
    scenario = parse_scenario(document(camera_id="cam-4", config=config, config_updates=updates))
    configs = [scenario.config_at(t) for t in (0, 1, 2, 3)]
    assert [c.version for c in configs] == [7, 8, 20, 21]
    assert all(c.camera.id == "cam-4" for c in configs)
    assert configs[0].camera.name == "Press 4"
    assert [c.frame_size_hint for c in configs] == [
        (1280, 720),
        (1280, 720),
        (640, 360),
        (1280, 720),
    ]
    assert configs[0].ppe_types == list(PPE_CODE_VALUES)
    assert parse_scenario(document()).config_at(0).version == 1


def test_the_config_is_validated_like_a_backend_response() -> None:
    config = {
        "thresholds": {"confirm_frames": 4, "cooldown_sec": 30, "min_confidence": 0.4},
        "ppe_types": ["helmet", "vest"],
        "zones": [
            {
                **zone("z1", required_ppe=["helmet", "glasses"], entry_forbidden=True),
                "enabled": False,
            }
        ],
        "updated_at": "2026-11-03T12:00:00Z",
    }
    scenario = parse_scenario(document(config=config))
    parsed = scenario.config_at(0)
    assert parsed.thresholds.confirm_frames == 4
    assert parsed.ppe_types == ["helmet", "vest"]
    assert parsed.zones[0].enabled is False
    assert parsed.zones[0].rules.required_ppe == ["helmet", "glasses"]
    assert parsed.zones[0].rules.entry_forbidden is True


# ---------------------------------------------------------------------------------------------
# Expectations
# ---------------------------------------------------------------------------------------------


def test_expected_event_helpers_match_keys_times_and_details() -> None:
    expected = ExpectedEvent.model_validate(
        event(
            start_s=2.0,
            start_tol_s=0.3,
            end_s=5.0,
            end_tol_s=0.5,
            details={
                "missing_ppe": ["helmet", "vest"],
                "present_ppe": [],
                "frame_size": [1280, 720],
            },
        )
    )
    assert expected.matches("ppe_violation", "z1", 1)
    assert not expected.matches("ppe_violation", "z2", 1)
    assert not expected.matches("ppe_violation", "z1", 2)
    assert not expected.matches("zone_intrusion", "z1", 1)
    assert expected.start_ok(1.7)
    assert expected.start_ok(2.3)
    assert not expected.start_ok(2.31)
    assert expected.end_ok(5.5)
    assert not expected.end_ok(5.51)
    assert not expected.end_ok(None)
    assert (
        expected.details_mismatches(
            {"missing_ppe": ["vest", "helmet"], "present_ppe": [], "frame_size": (1280, 720)}
        )
        == []
    )
    assert expected.details_mismatches({"missing_ppe": ["helmet"], "present_ppe": []}) == [
        "details.missing_ppe: expected ['helmet', 'vest'], got ['helmet']",
        "details.frame_size: missing (expected [1280, 720])",
    ]
    still_active = ExpectedEvent.model_validate(event())
    assert still_active.end_ok(None)
    assert not still_active.end_ok(9.0)


def test_details_numbers_match_within_a_micro_unit_and_booleans_are_not_numbers() -> None:
    expected = ExpectedEvent.model_validate(
        event(type="zone_intrusion", details={"duration_sec": 0.7, "boxes": [{"track_id": 1}]})
    )
    assert (
        expected.details_mismatches({"duration_sec": 0.7000004, "boxes": [{"track_id": 1}]}) == []
    )
    assert expected.details_mismatches({"duration_sec": 0.71, "boxes": [{"track_id": 1}]}) == [
        "details.duration_sec: expected 0.7, got 0.71"
    ]
    assert len(expected.details_mismatches({"duration_sec": 0.7, "boxes": [{"track_id": 2}]})) == 1
    flags = ExpectedEvent.model_validate(event(type="overcrowding", details={"count": 1}))
    assert flags.details_mismatches({"count": True}) == ["details.count: expected 1, got True"]


def test_expected_absence_matching() -> None:
    any_zone = ExpectedAbsence.model_validate({"type": "zone_intrusion"})
    assert any_zone.any_zone
    assert any_zone.matches("zone_intrusion", "z1", 3)
    assert any_zone.matches("zone_intrusion", None, 4)
    assert not any_zone.matches("overcrowding", "z1", 3)
    no_zone = ExpectedAbsence.model_validate({"type": "fall_detected", "zone": None})
    assert not no_zone.any_zone
    assert no_zone.matches("fall_detected", None, 1)
    assert not no_zone.matches("fall_detected", "z1", 1)
    narrow = ExpectedAbsence.model_validate({"type": "ppe_violation", "zone": "z1", "track": 2})
    assert narrow.matches("ppe_violation", "z1", 2)
    assert not narrow.matches("ppe_violation", "z1", 3)
    assert not narrow.matches("ppe_violation", "z2", 2)


def test_expectations_are_parsed_with_the_scenario() -> None:
    data = document(
        expected_events=[event(details={"missing_ppe": ["helmet"]})],
        expected_absent=[{"type": "zone_intrusion", "zone": None}, {"type": "overcrowding"}],
    )
    scenario = parse_scenario(data)
    assert scenario.expected_events[0].details == {"missing_ppe": ["helmet"]}
    assert scenario.expected_events[0].end_s is None
    assert not scenario.expected_absent[0].any_zone
    assert scenario.expected_absent[1].any_zone


def test_parsing_does_not_modify_the_input_document() -> None:
    data = document(config_updates=[{"at_s": 1, "config": {"zones": []}}])
    original = copy.deepcopy(data)
    parse_scenario(data)
    assert data == original


# ---------------------------------------------------------------------------------------------
# The format reference stays valid
# ---------------------------------------------------------------------------------------------


def test_the_readme_example_is_a_valid_scenario(repo_root: Path) -> None:
    readme = repo_root / "tests" / "scenarios" / "README.md"
    text = readme.read_text(encoding="utf-8")
    match = re.search(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
    assert match is not None, "the README must contain a complete YAML example"
    scenario = parse_scenario(parse_yaml_document(match.group(1), source=str(readme)))
    assert scenario.name == "example_ppe_and_forklift"
    assert scenario.n_frames == 300
    assert scenario.timezone == "Asia/Yekaterinburg"
    assert [z.id for z in scenario.config_at(0).zones] == ["zone-press", "zone-aisle"]
    assert scenario.config_at(24.0).zones[0].rules.required_ppe == ["vest"]
    walking = scenario.track_state(1, 1.0)
    assert walking is not None
    assert walking.pose == "walking"
    assert walking.ppe["helmet"] == "missing"
    assert scenario.track_state(1, 15.5) is None
    assert scenario.track_state(7, 17.9) is None
    assert scenario.track_state(7, 20.0) is not None
    assert scenario.expected_events[0].matches("ppe_violation", "zone-press", 1)
    assert len(scenario.expected_absent) == 3
