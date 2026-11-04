"""The demo configurations, their zones files and the demo video builder."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from safety_ai.config import load_settings, load_zones_file
from tools import make_demo_videos as videos


def _load(app_kit: Any, name: str, env: dict[str, str] | None = None) -> Any:
    return load_settings(app_kit.repo / "configs" / name, env=env or {}, dotenv_path=None)


def test_demo_integration_targets_the_website(app_kit: Any) -> None:
    settings = _load(app_kit, "demo_integration.yaml", {"SAFETY_AI_API_KEY": "ot_k"})
    assert settings.backend.url == "http://127.0.0.1:8000"
    assert settings.backend.api_key is not None
    assert settings.backend.api_key.get_secret_value() == "ot_k"
    assert settings.backend.ws is True and settings.backend.push_frames is False
    models = settings.models
    assert (models.ppe, models.imgsz, models.device) == ("ppe_cppe_latest", 640, "auto")
    assert settings.processing.target_fps == 10
    assert settings.output.expose is True and settings.output.mjpeg_port == 8001
    assert settings.telemetry.metrics_port == 9100
    assert [(c.id, c.source, c.loop) for c in settings.cameras] == [
        ("cam-1", "data/samples/cam1.mp4", True),
        ("cam-2", "data/samples/cam2.mp4", True),
    ]
    assert {c.timezone for c in settings.cameras} == {"Asia/Yekaterinburg"}
    text = (app_kit.repo / "configs" / "demo_integration.yaml").read_text(encoding="utf-8")
    assert "# source: 0" in text  # the commented webcam variant
    overridden = _load(
        app_kit,
        "demo_integration.yaml",
        {"SAFETY_AI_BACKEND_URL": "http://10.0.0.5:8000", "TZ": "Europe/Moscow"},
    )
    assert overridden.backend.url == "http://10.0.0.5:8000" and overridden.backend.api_key is None
    assert {c.timezone for c in overridden.cameras} == {"Europe/Moscow"}


def test_demo_file_targets_the_mock_backend(app_kit: Any) -> None:
    settings = _load(app_kit, "demo_file.yaml")
    assert settings.backend.url == "http://127.0.0.1:8010"
    assert settings.backend.api_key is not None
    assert settings.backend.api_key.get_secret_value() == "test-key"
    integration = _load(app_kit, "demo_integration.yaml")
    assert settings.cameras == integration.cameras and settings.models == integration.models
    assert settings.output == integration.output and settings.processing == integration.processing


@pytest.mark.parametrize(
    ("name", "zones"),
    [
        ("demo_zones_cam1.yaml", {"cam1-work-area": ["helmet", "vest"]}),
        ("demo_zones_cam2.yaml", {"cam2-bench": ["glasses", "gloves"]}),
    ],
)
def test_camera_zones_files_have_one_central_ppe_zone(
    app_kit: Any, name: str, zones: dict[str, list[str]]
) -> None:
    config = load_zones_file(app_kit.repo / "tools" / name, "cam-x")
    assert config.camera.id == "cam-x"
    assert {zone.id: zone.rules.required_ppe for zone in config.zones} == zones
    (zone,) = config.zones
    xs = [x for x, _ in zone.polygon]
    assert min(xs) < 0.2 and max(xs) > 0.8  # central, covering the image area


def test_the_demo_zones_cover_every_rule_kind(app_kit: Any) -> None:
    config = load_zones_file(app_kit.repo / "tools" / "demo_zones.yaml", "cam-1")
    rules = [zone.rules for zone in config.zones]
    assert len(config.zones) == 4
    assert any(r.required_ppe for r in rules)
    assert any(r.entry_forbidden for r in rules)
    assert any(r.max_persons is not None for r in rules)
    assert any(r.min_vehicle_distance_m is not None for r in rules)
    assert any(r.schedule for r in rules)


# ------------------------------------------------------------------------------------ demo videos


def _write_label(directory: Path, stem: str, rows: list[tuple[int, float]]) -> None:
    lines = [f"{cls} 0.5 0.5 0.2 {height}" for cls, height in rows]
    (directory / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_sh17(root: Path) -> Path:
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    layouts = {
        # protected: two persons, two helmets, a vest
        "a_protected": [(0, 0.8), (0, 0.6), (10, 0.1), (10, 0.1), (16, 0.2)],
        # unprotected: two persons, one helmet, a vest
        "b_mixed": [(0, 0.9), (0, 0.7), (10, 0.1), (16, 0.2)],
        # no work context (no helmet, no vest) - not a cam-1 image
        "c_street": [(0, 0.9), (12, 0.1)],
        # glasses and gloves with bare hands - a cam-2 image
        "d_lab": [(0, 0.8), (8, 0.05), (9, 0.05), (11, 0.05)],
        # persons too small
        "e_far": [(0, 0.1), (10, 0.02), (16, 0.03)],
    }
    for stem, rows in layouts.items():
        _write_label(labels, stem, rows)
        cv2.imwrite(str(images / f"{stem}.jpg"), np.full((853, 1280, 3), 120, np.uint8))
    return root


def test_sh17_selection_is_deterministic_and_follows_the_labels(tmp_path: Path) -> None:
    sh17 = _fake_sh17(tmp_path / "sh17")
    cam1 = videos.select_sh17(sh17, "cam1")
    assert [p.stem for p in cam1] == ["b_mixed", "a_protected"]  # alternating, unprotected first
    assert videos.select_sh17(sh17, "cam1") == cam1
    assert [p.stem for p in videos.select_sh17(sh17, "cam2")] == ["d_lab"]
    plan = videos.plan_cam1(sh17)
    assert [still.path.stem for still in plan.stills] == ["b_mixed", "a_protected"]
    assert [still.seconds for still in plan.stills] == [6.0, 7.0]
    fallback = videos.plan_cam1(None)
    assert {still.path.name for still in fallback.stills} == {"bus.jpg", "zidane.jpg"}
    assert sum(still.seconds for still in fallback.stills) == pytest.approx(90.0, abs=6.0)


def test_rendering_is_1280x720_at_12_fps_and_skipped_when_up_to_date(tmp_path: Path) -> None:
    still = tmp_path / "still.jpg"
    cv2.imwrite(str(still), np.random.default_rng(1).integers(0, 255, (600, 800, 3), np.uint8))
    plan = videos.VideoPlan("tiny", (videos.Still(still, 0.5), videos.Still(still, 0.5)), 7)
    out = tmp_path / "tiny.mp4"
    codec = videos.render_video(plan, out)
    capture = cv2.VideoCapture(str(out))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (
        1280,
        720,
    )
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0)
    capture.release()
    assert codec in ("avc1", "mp4v")
    assert not videos._up_to_date(plan, out)  # no manifest yet
    out.with_suffix(".json").write_text(json.dumps(plan.manifest()), encoding="utf-8")
    assert videos._up_to_date(plan, out)
    other = videos.VideoPlan("tiny", plan.stills, 8)
    assert not videos._up_to_date(other, out)


def test_the_pan_and_zoom_keeps_the_whole_still_visible() -> None:
    image = np.zeros((600, 900, 3), np.uint8)
    image[:, :, 2] = 255  # red still
    frames = list(videos._still_frames(image, 1.0, random.Random(3)))
    assert len(frames) == 12 and all(f.shape == (720, 1280, 3) for f in frames)
    middle = frames[0][300:420, 560:720]
    assert middle[:, :, 2].mean() > 200 and middle[:, :, 0].mean() < 60
