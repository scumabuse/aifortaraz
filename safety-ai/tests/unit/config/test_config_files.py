"""Tests of the shipped configuration files: configs/default.yaml and configs/example_rtsp.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from safety_ai.config import AppSettings, load_settings
from safety_ai.config.models import CameraSettings


def spec_example(repo_root: Path) -> str:
    """The YAML block of docs/SPEC.md section 7.2."""
    lines = (repo_root / "docs" / "SPEC.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("### 7.2."))
    fence = next(i for i in range(start, len(lines)) if lines[i].strip() == "```yaml")
    end = next(i for i in range(fence + 1, len(lines)) if lines[i].strip() == "```")
    return "\n".join(lines[fence + 1 : end]) + "\n"


def section_models() -> dict[str, type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {}
    for name, info in AppSettings.model_fields.items():
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            models[name] = annotation
    return models


def test_default_yaml_is_the_spec_example_verbatim(repo_root: Path) -> None:
    text = (repo_root / "configs" / "default.yaml").read_text(encoding="utf-8")
    assert text == spec_example(repo_root)


def test_default_yaml_loads_with_the_api_key_from_the_environment(repo_root: Path) -> None:
    settings = load_settings(
        repo_root / "configs" / "default.yaml",
        env={"SAFETY_AI_API_KEY": "key-from-env"},
        dotenv_path=None,
    )
    assert settings.backend.url == "http://localhost:8000"
    assert settings.backend.api_key is not None
    assert settings.backend.api_key.get_secret_value() == "key-from-env"
    assert settings.models.ppe == "ppe_s_latest"
    assert settings.thresholds.confirm_frames == 8
    assert settings.output.mjpeg_port == 8001
    assert settings.telemetry.metrics_port == 9100
    assert [(c.id, c.source, c.loop) for c in settings.cameras] == [
        ("cam-1", "data/samples/press.mp4", True),
        ("cam-2", "rtsp://user:pass@192.168.1.10/live", False),
    ]
    assert {c.timezone for c in settings.cameras} == {"Asia/Yekaterinburg"}


def test_default_yaml_loads_without_an_api_key(repo_root: Path) -> None:
    settings = load_settings(repo_root / "configs" / "default.yaml", env={}, dotenv_path=None)
    assert settings.backend.api_key is None


def test_example_rtsp_yaml_loads(repo_root: Path) -> None:
    settings = load_settings(
        repo_root / "configs" / "example_rtsp.yaml",
        env={"CAM1_PASSWORD": "pw", "SAFETY_AI_API_KEY": "k"},
        dotenv_path=None,
    )
    assert [camera.id for camera in settings.cameras] == ["cam-1", "cam-2"]
    assert all(str(camera.source).startswith("rtsp://") for camera in settings.cameras)
    assert settings.cameras[0].source == (
        "rtsp://viewer:pw@192.168.1.10:554/Streaming/Channels/101"
    )
    assert settings.backend.api_key is not None


def test_example_rtsp_documents_every_option(repo_root: Path) -> None:
    raw: dict[str, Any] = yaml.safe_load(
        (repo_root / "configs" / "example_rtsp.yaml").read_text(encoding="utf-8")
    )
    assert set(raw) == set(AppSettings.model_fields)
    for section, model in section_models().items():
        assert set(raw[section]) == set(model.model_fields), section
    cameras: list[dict[str, Any]] = raw["cameras"]
    assert set(cameras[0]) == set(CameraSettings.model_fields)


def test_example_rtsp_shows_the_defaults(repo_root: Path) -> None:
    settings = load_settings(repo_root / "configs" / "example_rtsp.yaml", env={}, dotenv_path=None)
    defaults = AppSettings()
    differing = {
        (section, name)
        for section in section_models()
        for name in type(getattr(defaults, section)).model_fields
        if getattr(getattr(settings, section), name) != getattr(getattr(defaults, section), name)
    }
    assert differing == {("models", "class_conf")}
    first = settings.cameras[0]
    camera_defaults = CameraSettings(id=first.id)
    assert (first.loop, first.fps, first.enabled, first.homography, first.zones_file) == (
        camera_defaults.loop,
        camera_defaults.fps,
        camera_defaults.enabled,
        camera_defaults.homography,
        camera_defaults.zones_file,
    )
