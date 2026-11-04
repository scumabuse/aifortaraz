"""Tests of safety_ai.config.loader: layering, ${VAR} expansion, env overrides, calibration, errors."""

from __future__ import annotations

import copy
import json
import os
import pickle
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from safety_ai.backend.schemas import CameraConfig
from safety_ai.config import (
    AppSettings,
    ConfigError,
    exit_on_config_error,
    format_validation_error,
    load_settings,
)
from safety_ai.config.loader import (
    CalibrationSidecar,
    calibration_path,
    parse_yaml_document,
    read_calibration_sidecar,
    write_calibration_sidecar,
)

MALFORMED = "malformed ${...} reference (write ${NAME} or ${NAME:-default}; $${ is a literal ${)"
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
MATRIX = [[0.01, 0.0, -3.0], [0.0, 0.04, -9.5], [0.0, 0.001, 1.0]]


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load(
    path: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    cli: dict[str, Any] | None = None,
    dotenv: Path | None = None,
) -> AppSettings:
    return load_settings(path, cli_overrides=cli, env=env or {}, dotenv_path=dotenv)


def errors_of(path: Path | None = None, **kwargs: Any) -> list[str]:
    with pytest.raises(ConfigError) as info:
        load(path, **kwargs)
    return info.value.messages


def posix(path: Path) -> str:
    return path.as_posix()


# ------------------------------------------------------------------------------------------------
# Layering
# ------------------------------------------------------------------------------------------------


def test_defaults_without_any_source() -> None:
    assert load().model_dump() == AppSettings().model_dump()


@pytest.mark.parametrize(
    ("use_yaml", "use_env", "use_cli", "expected"),
    [
        (False, False, False, "INFO"),
        (True, False, False, "WARNING"),
        (False, True, False, "ERROR"),
        (False, False, True, "DEBUG"),
        (True, True, False, "ERROR"),
        (True, False, True, "DEBUG"),
        (True, True, True, "DEBUG"),
    ],
)
def test_precedence_cli_over_env_over_yaml_over_defaults(
    tmp_path: Path, use_yaml: bool, use_env: bool, use_cli: bool, expected: str
) -> None:
    path = write(tmp_path / "c.yaml", "telemetry:\n  log_level: WARNING\n") if use_yaml else None
    env = {"SAFETY_AI_TELEMETRY__LOG_LEVEL": "ERROR"} if use_env else {}
    cli = {"telemetry": {"log_level": "DEBUG"}} if use_cli else None
    assert load(path, env=env, cli=cli).telemetry.log_level == expected


def test_layers_merge_per_setting(tmp_path: Path) -> None:
    path = write(
        tmp_path / "c.yaml",
        "telemetry:\n  log_level: WARNING\n  metrics_port: 9200\n  log_json: false\n",
    )
    settings = load(
        path,
        env={"SAFETY_AI_TELEMETRY__LOG_LEVEL": "ERROR"},
        cli={"telemetry": {"metrics_port": 9300}},
    )
    assert settings.telemetry.log_level == "ERROR"
    assert settings.telemetry.metrics_port == 9300
    assert settings.telemetry.log_json is False
    assert settings.telemetry.log_backups == 5


def test_empty_file_and_empty_sections_mean_defaults(tmp_path: Path) -> None:
    assert load(write(tmp_path / "empty.yaml", "")).model_dump() == AppSettings().model_dump()
    path = write(tmp_path / "c.yml", "backend:\n  # url: http://x\nprivacy:\ncameras:\n")
    assert load(path).model_dump() == AppSettings().model_dump()


def test_byte_order_mark_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "bom.yaml"
    path.write_bytes(b"\xef\xbb\xbftelemetry:\n  log_level: ERROR\n")
    assert load(path).telemetry.log_level == "ERROR"


def test_relative_config_path_is_resolved_against_the_current_directory(
    in_tmp_cwd: Path,
) -> None:
    write(in_tmp_cwd / "configs" / "c.yaml", "telemetry:\n  log_level: ERROR\n")
    assert load(Path("configs/c.yaml")).telemetry.log_level == "ERROR"


def test_os_environ_is_the_default_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("SAFETY_AI_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SAFETY_AI_TELEMETRY__LOG_LEVEL", "ERROR")
    assert load_settings(None, dotenv_path=None).telemetry.log_level == "ERROR"


# ------------------------------------------------------------------------------------------------
# ${VAR} expansion and .env
# ------------------------------------------------------------------------------------------------


def test_expansion_of_set_unset_empty_and_default_variables(tmp_path: Path) -> None:
    path = write(
        tmp_path / "c.yaml",
        "backend:\n"
        "  url: http://${HOST:-localhost}:${PORT:-8000}\n"
        "models:\n"
        "  ppe: ${MODEL}\n"
        "  pose: pre-${UNSET}-post\n"
        "cameras:\n"
        "  - id: cam-1\n"
        "    source: ${EMPTY:-fallback.mp4}\n"
        "  - id: cam-2\n"
        "    source: literal-$${NOT_EXPANDED}-$5-$$x\n",
    )
    settings = load(path, env={"HOST": "backend", "MODEL": "ppe_x", "EMPTY": ""})
    assert settings.backend.url == "http://backend:8000"
    assert settings.models.ppe == "ppe_x"
    assert settings.models.pose == "pre--post"
    assert settings.cameras[0].source == "fallback.mp4"
    assert settings.cameras[1].source == "literal-${NOT_EXPANDED}-$5-$$x"


def test_expanded_values_are_validated_by_their_setting_type(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  timeout_s: ${T:-7}\n  ws: ${WS:-false}\n")
    settings = load(path)
    assert settings.backend.timeout_s == 7.0
    assert settings.backend.ws is False


def test_unset_api_key_reference_means_no_key(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  api_key: ${SAFETY_AI_API_KEY}\n")
    assert load(path).backend.api_key is None


def test_keys_are_not_expanded(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "${SECTION}:\n  x: 1\n")
    assert errors_of(path, env={"SECTION": "backend"}) == ["${SECTION}: unknown setting"]


def test_malformed_reference_is_reported_with_its_setting(tmp_path: Path) -> None:
    path = write(
        tmp_path / "c.yaml",
        "models:\n  ppe: ${BROKEN\ncameras:\n  - id: cam-1\n    source: a${1X}b\n",
    )
    assert errors_of(path) == [f"models.ppe: {MALFORMED}", f"cameras.0.source: {MALFORMED}"]


def test_dotenv_provides_variables_for_expansion(tmp_path: Path) -> None:
    dotenv = write(tmp_path / ".env", "CAM_SOURCE=from-dotenv.mp4\nHOST=dotenv-host\n")
    path = write(
        tmp_path / "c.yaml",
        "backend:\n  url: http://${HOST}\ncameras:\n  - id: cam-1\n    source: ${CAM_SOURCE}\n",
    )
    settings = load(path, env={"HOST": "env-host"}, dotenv=dotenv)
    assert settings.cameras[0].source == "from-dotenv.mp4"
    assert settings.backend.url == "http://env-host"


def test_dotenv_overrides_apply_below_the_environment(tmp_path: Path) -> None:
    dotenv = write(
        tmp_path / ".env",
        "SAFETY_AI_TELEMETRY__LOG_LEVEL=ERROR\nSAFETY_AI_TELEMETRY__METRICS_PORT=9555\n",
    )
    settings = load(env={"SAFETY_AI_TELEMETRY__LOG_LEVEL": "DEBUG"}, dotenv=dotenv)
    assert settings.telemetry.log_level == "DEBUG"
    assert settings.telemetry.metrics_port == 9555


def test_broken_dotenv_override_is_ignored_when_the_environment_sets_it(tmp_path: Path) -> None:
    dotenv = write(tmp_path / ".env", "SAFETY_AI_CAMERAS=[{id: broken\n")
    settings = load(env={"SAFETY_AI_CAMERAS": "[{id: cam-1}]"}, dotenv=dotenv)
    assert [camera.id for camera in settings.cameras] == ["cam-1"]


def test_dotenv_api_key_and_url_are_used(tmp_path: Path) -> None:
    dotenv = write(
        tmp_path / ".env",
        "# comment\nSAFETY_AI_API_KEY=dotenv-key\nSAFETY_AI_BACKEND_URL=http://${HOST}:8000\n",
    )
    settings = load(env={"HOST": "site"}, dotenv=dotenv)
    assert settings.backend.api_key is not None
    assert settings.backend.api_key.get_secret_value() == "dotenv-key"
    assert settings.backend.url == "http://site:8000"


def test_malformed_reference_in_dotenv_is_reported(tmp_path: Path) -> None:
    dotenv = write(tmp_path / ".env", "GOOD=1\nBAD=${OOPS\n")
    assert errors_of(dotenv=dotenv) == [f"{dotenv}: BAD: {MALFORMED}"]


def test_dotenv_directory_or_missing_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / ".env").mkdir()
    assert load(dotenv=tmp_path / ".env").model_dump() == AppSettings().model_dump()
    assert load(dotenv=tmp_path / "missing.env").model_dump() == AppSettings().model_dump()


def test_dotenv_never_modifies_os_environ(tmp_path: Path) -> None:
    before = dict(os.environ)
    dotenv = write(tmp_path / ".env", "SAFETY_AI_UNIT_TEST_ONLY_VARIABLE=1\nSAFETY_AI_API_KEY=k\n")
    load(dotenv=dotenv)
    assert dict(os.environ) == before


def test_default_dotenv_path_is_in_the_current_directory(in_tmp_cwd: Path) -> None:
    write(in_tmp_cwd / ".env", "SAFETY_AI_TELEMETRY__LOG_LEVEL=ERROR\n")
    assert load_settings(None, env={}).telemetry.log_level == "ERROR"


# ------------------------------------------------------------------------------------------------
# Contract variables and nested overrides
# ------------------------------------------------------------------------------------------------


def test_contract_variables_set_backend_url_and_api_key(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  url: http://yaml:8000\n  api_key: yaml-key\n")
    settings = load(
        path,
        env={"SAFETY_AI_BACKEND_URL": "https://site.example/", "SAFETY_AI_API_KEY": "secret-1"},
    )
    assert settings.backend.url == "https://site.example"
    assert settings.backend.api_key is not None
    assert settings.backend.api_key.get_secret_value() == "secret-1"


def test_nested_form_wins_over_the_contract_variable(tmp_path: Path) -> None:
    settings = load(
        env={
            "SAFETY_AI_BACKEND_URL": "http://contract:8000",
            "SAFETY_AI_BACKEND__URL": "http://nested:8000",
        }
    )
    assert settings.backend.url == "http://nested:8000"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" k1 ", "k1"), ("0123", "0123"), ("12:30", "12:30"), ("true", "true"), ("1e5", "1e5")],
)
def test_api_key_text_is_kept_verbatim_but_stripped(raw: str, expected: str) -> None:
    for name in ("SAFETY_AI_API_KEY", "SAFETY_AI_BACKEND__API_KEY"):
        key = load(env={name: raw}).backend.api_key
        assert key is not None
        assert key.get_secret_value() == expected


@pytest.mark.parametrize("raw", ["", "   ", "null"])
def test_empty_api_key_means_no_key(tmp_path: Path, raw: str) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  api_key: yaml-key\n")
    assert load(path, env={"SAFETY_AI_API_KEY": raw}).backend.api_key is None


def test_nested_env_values_are_parsed_as_yaml() -> None:
    settings = load(
        env={
            "SAFETY_AI_BACKEND__WS": "false",
            "SAFETY_AI_OUTPUT__MJPEG_PORT": "8101",
            "SAFETY_AI_THRESHOLDS__CONF": "0.4",
            "SAFETY_AI_MODELS__CLASS_CONF": "{helmet: 0.4, vest: 0.3}",
            "SAFETY_AI_MODELS__CLASS_CONF__GLOVES": "0.2",
            "SAFETY_AI_OUTPUT__EVENTS_DIR": "2024",
        }
    )
    assert settings.backend.ws is False
    assert settings.output.mjpeg_port == 8101
    assert settings.thresholds.conf == pytest.approx(0.4)
    assert settings.models.class_conf == {"helmet": 0.4, "vest": 0.3, "gloves": 0.2}
    assert settings.output.events_dir == Path("2024")


def test_text_settings_keep_their_raw_text(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras:\n  - id: cam-1\n")
    settings = load(
        path,
        env={
            "SAFETY_AI_MODELS__DEVICE": "0",
            "SAFETY_AI_MODELS__PPE": "123",
            "SAFETY_AI_CAMERAS__0__SOURCE": "rtsp://h/live #1",
        },
    )
    assert settings.models.device == "0"
    assert settings.models.ppe == "123"
    assert settings.cameras[0].source == "rtsp://h/live #1"


def test_cameras_from_a_yaml_list_in_the_environment(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras:\n  - id: yaml-cam\n")
    settings = load(
        path, env={"SAFETY_AI_CAMERAS": '[{"id": "cam-9", "source": "a.mp4", "loop": true}]'}
    )
    assert [(c.id, c.source, c.loop) for c in settings.cameras] == [("cam-9", "a.mp4", True)]


def test_camera_items_are_overridden_by_index(tmp_path: Path) -> None:
    path = write(
        tmp_path / "c.yaml",
        "cameras:\n  - id: cam-1\n    source: a.mp4\n  - id: cam-2\n    source: b.mp4\n",
    )
    settings = load(
        path,
        env={"SAFETY_AI_CAMERAS__1__SOURCE": "rtsp://x/live", "SAFETY_AI_CAMERAS__0__SOURCE": "0"},
    )
    assert settings.cameras[0].source == 0
    assert settings.cameras[1].source == "rtsp://x/live"


def test_whole_section_is_applied_before_its_items() -> None:
    settings = load(
        env={
            "SAFETY_AI_CAMERAS__0__TIMEZONE": "Asia/Yekaterinburg",
            "SAFETY_AI_CAMERAS": "[{id: cam-1}]",
        }
    )
    assert settings.cameras[0].timezone == "Asia/Yekaterinburg"


def test_cameras_can_be_built_from_indexed_variables_in_numeric_order() -> None:
    env = {f"SAFETY_AI_CAMERAS__{index}__ID": f"cam-{index}" for index in range(11)}
    env["SAFETY_AI_CAMERAS__10__SOURCE"] = "ten.mp4"
    settings = load(env=env)
    assert [camera.id for camera in settings.cameras] == [f"cam-{index}" for index in range(11)]
    assert settings.cameras[10].source == "ten.mp4"


def test_nested_list_items_inside_a_camera() -> None:
    settings = load(
        env={
            "SAFETY_AI_CAMERAS__0__ID": "cam-1",
            "SAFETY_AI_CAMERAS__0__HOMOGRAPHY": json.dumps(IDENTITY),
            "SAFETY_AI_CAMERAS__0__HOMOGRAPHY__2__2": "2",
        }
    )
    assert settings.cameras[0].homography == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]


def test_empty_value_clears_an_optional_setting() -> None:
    assert load(env={"SAFETY_AI_TELEMETRY__LOG_FILE": ""}).telemetry.log_file is None


def test_variable_names_are_case_insensitive() -> None:
    settings = load(env={"safety_ai_telemetry__log_level": "ERROR", "Safety_Ai_Api_Key": "k"})
    assert settings.telemetry.log_level == "ERROR"
    assert settings.backend.api_key is not None


def test_unrelated_variables_are_ignored() -> None:
    settings = load(env={"SAFETY_AI_SOMETHING": "1", "SAFETY_AI_": "x", "OTHER__X": "2"})
    assert settings.model_dump() == AppSettings().model_dump()


def test_unknown_nested_setting_is_reported() -> None:
    assert errors_of(env={"SAFETY_AI_BACKEND__NONSENSE": "1"}) == [
        "backend.nonsense: unknown setting"
    ]


@pytest.mark.parametrize("name", ["SAFETY_AI_BACKEND__", "SAFETY_AI___URL", "SAFETY_AI_A____B"])
def test_malformed_override_names_are_reported(name: str) -> None:
    assert errors_of(env={name: "x"}) == [
        f"{name}: malformed override name (expected SAFETY_AI_<SECTION>__<KEY>)"
    ]


def test_override_index_out_of_range(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras:\n  - id: cam-1\n")
    assert errors_of(path, env={"SAFETY_AI_CAMERAS__5__SOURCE": "x"}) == [
        "SAFETY_AI_CAMERAS__5__SOURCE: cameras has 1 item(s); index 5 is out of range (use 0..1)"
    ]


def test_override_of_a_list_without_an_index(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras:\n  - id: cam-1\n")
    assert errors_of(path, env={"SAFETY_AI_CAMERAS__SOURCE": "x"}) == [
        "SAFETY_AI_CAMERAS__SOURCE: cameras is a list; address its items by index "
        "(SAFETY_AI_CAMERAS__0__...)"
    ]


def test_unparsable_override_value_is_reported() -> None:
    assert errors_of(env={"SAFETY_AI_CAMERAS": "[{id: a"}) == [
        "SAFETY_AI_CAMERAS: cannot parse the value: YAML syntax error at line 1, column 8: "
        "expected ',' or '}', but got '<stream end>'"
    ]


def test_invalid_override_value_is_reported_by_its_setting() -> None:
    assert errors_of(env={"SAFETY_AI_OUTPUT__MJPEG_PORT": "70000"}) == [
        "output.mjpeg_port: Input should be less than or equal to 65535"
    ]


# ------------------------------------------------------------------------------------------------
# CLI overrides
# ------------------------------------------------------------------------------------------------


def test_cli_dotted_keys_and_none_values(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  url: http://yaml:8000\n  timeout_s: 3\n")
    settings = load(
        path,
        cli={
            "telemetry.log_level": "DEBUG",
            "backend": {"url": None, "timeout_s": 9},
            "output.mjpeg_port": None,
            "models": {"class_conf.helmet": 0.45},
        },
    )
    assert settings.telemetry.log_level == "DEBUG"
    assert settings.backend.url == "http://yaml:8000"
    assert settings.backend.timeout_s == 9.0
    assert settings.output.mjpeg_port == 8001
    assert settings.models.class_conf == {"helmet": 0.45}


def test_cli_lists_replace_lower_layers(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras:\n  - id: cam-1\n  - id: cam-2\n")
    cli = {"cameras": [{"id": "cam-x", "source": "s.mp4"}]}
    original = copy.deepcopy(cli)
    settings = load(path, env={"SAFETY_AI_CAMERAS__0__SOURCE": "env.mp4"}, cli=cli)
    assert [(c.id, c.source) for c in settings.cameras] == [("cam-x", "s.mp4")]
    assert cli == original


def test_cli_wins_over_the_environment_for_the_api_key() -> None:
    settings = load(env={"SAFETY_AI_API_KEY": "env-key"}, cli={"backend": {"api_key": "cli-key"}})
    assert settings.backend.api_key is not None
    assert settings.backend.api_key.get_secret_value() == "cli-key"


def test_cli_empty_key_segment_is_reported() -> None:
    assert errors_of(cli={"backend..url": "http://x"}) == [
        "CLI override 'backend..url': empty key segment"
    ]


# ------------------------------------------------------------------------------------------------
# Calibration sidecars
# ------------------------------------------------------------------------------------------------


def calibrated_config(tmp_path: Path, cameras: str) -> Path:
    calibration = tmp_path / "cal"
    return write(
        tmp_path / "c.yaml", f"paths:\n  calibration_dir: {posix(calibration)}\ncameras:\n{cameras}"
    )


def test_calibration_sidecar_provides_the_homography(tmp_path: Path) -> None:
    sidecar = CalibrationSidecar.model_validate(
        {
            "homography": MATRIX,
            "frame_size": [1280, 720],
            "pixels": [[100, 600], [1180, 600], [900, 300], [380, 300]],
            "metres": [[0, 0], [10, 0], [10, 20], [0, 20]],
        }
    )
    written = write_calibration_sidecar(tmp_path / "cal", "cam-1", sidecar)
    assert written == tmp_path / "cal" / "cam-1.json"
    path = calibrated_config(tmp_path, "  - id: cam-1\n  - id: cam-2\n")
    settings = load(path)
    assert settings.cameras[0].homography == MATRIX
    assert settings.cameras[1].homography is None


def test_handwritten_sidecar_in_the_documented_format(tmp_path: Path) -> None:
    write(
        tmp_path / "cal" / "cam-1.json",
        json.dumps({"homography": MATRIX, "frame_size": [1280, 720], "pixels": [], "metres": []}),
    )
    settings = load(calibrated_config(tmp_path, "  - id: cam-1\n"))
    assert settings.cameras[0].homography == MATRIX


def test_inline_homography_wins_over_the_sidecar(tmp_path: Path) -> None:
    write(tmp_path / "cal" / "cam-1.json", json.dumps({"homography": MATRIX}))
    path = calibrated_config(tmp_path, f"  - id: cam-1\n    homography: {json.dumps(IDENTITY)}\n")
    assert load(path).cameras[0].homography == IDENTITY


def test_calibration_dir_from_the_environment(tmp_path: Path) -> None:
    write(tmp_path / "elsewhere" / "cam-1.json", json.dumps({"homography": MATRIX}))
    settings = load(
        env={
            "SAFETY_AI_PATHS__CALIBRATION_DIR": posix(tmp_path / "elsewhere"),
            "SAFETY_AI_CAMERAS": "[{id: cam-1}]",
        }
    )
    assert settings.cameras[0].homography == MATRIX


def test_invalid_sidecars_are_reported(tmp_path: Path) -> None:
    bad_shape = write(tmp_path / "cal" / "cam-1.json", json.dumps({"homography": [[1, 2, 3]]}))
    bad_json = write(tmp_path / "cal" / "cam-2.json", '{"homography": [1,\n')
    not_object = write(tmp_path / "cal" / "cam-3.json", "[1, 2]")
    path = calibrated_config(tmp_path, "  - id: cam-1\n  - id: cam-2\n  - id: cam-3\n")
    assert errors_of(path) == [
        f"{bad_shape}: homography.1: Field required",
        f"{bad_shape}: homography.2: Field required",
        f"{bad_json}: JSON syntax error at line 2, column 1: Expecting value",
        f"{not_object}: a calibration file must be a JSON object, not a list",
    ]


def test_non_finite_homography_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "s.json", '{"homography": [[1, 0, 0], [0, 1, 0], [0, 0, NaN]]}')
    with pytest.raises(ConfigError) as info:
        read_calibration_sidecar(path)
    assert info.value.messages == [f"{path}: homography.2.2: Input should be a finite number"]


def test_calibration_path_replaces_colons() -> None:
    assert calibration_path(Path("cal"), "site:cam-1") == Path("cal") / "site_cam-1.json"


def test_write_calibration_sidecar_overwrites_atomically(tmp_path: Path) -> None:
    first = CalibrationSidecar.model_validate({"homography": IDENTITY})
    second = CalibrationSidecar.model_validate({"homography": MATRIX, "frame_size": [640, 480]})
    write_calibration_sidecar(tmp_path, "cam-1", first)
    path = write_calibration_sidecar(tmp_path, "cam-1", second)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cam-1.json"]
    loaded = read_calibration_sidecar(path)
    assert loaded.matrix() == MATRIX
    assert loaded.frame_size == (640, 480)


# ------------------------------------------------------------------------------------------------
# Readable errors
# ------------------------------------------------------------------------------------------------


def test_unknown_key_error(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  url: http://x\n  nonsense: 1\n")
    assert errors_of(path) == ["backend.nonsense: unknown setting"]


def test_unknown_section_error(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "zones:\n  - id: z1\n")
    assert errors_of(path) == ["zones: unknown setting"]


def test_bad_timezone_error(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras:\n  - id: cam-1\n    timezone: Mars/Base\n")
    assert errors_of(path) == ["cameras.0.timezone: unknown timezone 'Mars/Base'"]


def test_out_of_range_port_error(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "output:\n  mjpeg_port: 70000\n")
    assert errors_of(path) == ["output.mjpeg_port: Input should be less than or equal to 65535"]


def test_every_problem_is_one_line(tmp_path: Path) -> None:
    path = write(
        tmp_path / "c.yaml",
        "backend:\n  nonsense: 1\noutput:\n  mjpeg_port: 0\n"
        "cameras:\n  - id: cam-1\n    timezone: Mars/Base\n  - source: x.mp4\n",
    )
    assert errors_of(path) == [
        "backend.nonsense: unknown setting",
        "output.mjpeg_port: Input should be greater than or equal to 1",
        "cameras.0.timezone: unknown timezone 'Mars/Base'",
        "cameras.1.id: Field required",
    ]


def test_union_alternatives_are_joined(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras:\n  - id: cam-1\n    source: [1, 2]\n")
    assert errors_of(path) == [
        "cameras.0.source: Input should be a valid string or Input should be a valid integer"
    ]


def test_model_level_error_is_a_bare_message(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras:\n  - id: cam-1\n  - id: cam-1\n")
    assert errors_of(path) == ["duplicate camera ids: ['cam-1']"]


def test_messages_never_contain_secret_values(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  api_key: [top-secret-value]\n")
    messages = errors_of(path)
    assert messages == ["backend.api_key: Input should be a valid string"]
    assert "top-secret-value" not in str(messages)


def test_missing_config_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    assert errors_of(path) == [f"config file not found: {path}"]


def test_config_path_that_is_a_directory(tmp_path: Path) -> None:
    path = tmp_path / "dir.yaml"
    path.mkdir()
    assert errors_of(path) == [f"config file is not a regular file: {path}"]


def test_config_file_with_a_wrong_extension(tmp_path: Path) -> None:
    path = write(tmp_path / "c.json", "{}")
    assert errors_of(path) == [f"config file must have a .yaml or .yml extension: {path}"]


def test_yaml_syntax_error_with_line_and_column(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  url: http://x\n   ws: true\n")
    assert errors_of(path) == [
        f"{path}: YAML syntax error at line 3, column 6: mapping values are not allowed here"
    ]


def test_unclosed_flow_sequence_error(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "cameras: [1, 2\n")
    assert errors_of(path) == [
        f"{path}: YAML syntax error at line 2, column 1: expected ',' or ']', but got '<stream end>'"
    ]


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  url: http://a\nbackend:\n  ws: false\n")
    assert errors_of(path) == [
        f"{path}: YAML syntax error at line 3, column 1: found duplicate key 'backend'"
    ]


def test_control_character_error(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "backend:\n  url: \x07\n")
    assert errors_of(path) == [
        f"{path}: YAML syntax error at line 2, column 8: unacceptable character #x0007 "
        "(special characters are not allowed)"
    ]


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "- a\n- b\n")
    assert errors_of(path) == [f"{path}: the top level must be a mapping of sections, not a list"]


def test_invalid_utf8_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_bytes(b"\xff\xfebackend: {}\n")
    assert errors_of(path) == [f"{path}: config file is not valid UTF-8 (byte 0)"]


def test_format_validation_error_infers_app_settings() -> None:
    with pytest.raises(ValidationError) as info:
        AppSettings.model_validate({"cameras": [{"id": "cam-1", "source": {"a": 1}}]})
    assert format_validation_error(info.value) == [
        "cameras.0.source: Input should be a valid string or Input should be a valid integer"
    ]


def test_format_validation_error_uses_aliases_of_other_models() -> None:
    config = {
        "camera": {"id": "cam-1"},
        "zones": [
            {
                "id": "z1",
                "polygon": [[0, 0], [1, 0], [1, 1]],
                "rules": {"schedule": [{"days": [1], "from": "25:00", "to": "26:00"}]},
            }
        ],
    }
    with pytest.raises(ValidationError) as info:
        CameraConfig.model_validate(config)
    assert format_validation_error(info.value, model=CameraConfig) == [
        "zones.0.rules.schedule.0.from: time '25:00' is out of range",
        "zones.0.rules.schedule.0.to: time '26:00' is out of range",
    ]


def test_format_validation_error_without_a_known_model_keeps_the_location() -> None:
    class Free(BaseModel):
        value: int | str

    with pytest.raises(ValidationError) as info:
        Free.model_validate({"value": [1]})
    assert format_validation_error(info.value) == [
        "value.int: Input should be a valid integer",
        "value.str: Input should be a valid string",
    ]


def test_exit_on_config_error_prints_and_exits_with_2(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        exit_on_config_error(ConfigError(["backend.nonsense: unknown setting", "second problem"]))
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "config error: backend.nonsense: unknown setting\nconfig error: second problem\n"
    )


def test_config_error_text_and_pickling() -> None:
    error = ConfigError(["a: one", "b: two"])
    assert str(error) == "a: one\nb: two"
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, ConfigError)
    assert restored.messages == ["a: one", "b: two"]
    assert ConfigError("single problem").messages == ["single problem"]
    assert ConfigError([]).messages == ["invalid configuration"]


# ------------------------------------------------------------------------------------------------
# YAML loader details
# ------------------------------------------------------------------------------------------------


def test_yaml_base60_numbers_stay_text() -> None:
    document = parse_yaml_document(
        "to: 20:00\nfrom: 8:30\nn: 12\nneg: -3\nf: 1.5\nh: 0x1F\nb: 0b101\ninf: .inf\nflag: yes\n",
        source="doc",
    )
    assert document == {
        "to": "20:00",
        "from": "8:30",
        "n": 12,
        "neg": -3,
        "f": 1.5,
        "h": 31,
        "b": 5,
        "inf": float("inf"),
        "flag": True,
    }


def test_yaml_merge_keys_are_not_duplicates() -> None:
    document = parse_yaml_document(
        "base: &base\n  a: 1\n  b: 2\nderived:\n  <<: *base\n  b: 3\n", source="doc"
    )
    assert document["derived"] == {"a": 1, "b": 3}


def test_yaml_multiple_documents_are_rejected() -> None:
    with pytest.raises(ConfigError) as info:
        parse_yaml_document("a: 1\n---\nb: 2\n", source="doc")
    assert info.value.messages == [
        "doc: YAML syntax error at line 2, column 1: "
        "expected a single document in the stream, but found another document"
    ]
