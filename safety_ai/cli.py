"""The ``safety-ai`` command line (spec 7.3; ARCHITECTURE 3.10).

Every command has ``--help``, none prompts, and the exit codes are 0 (success), 1 (runtime
failure) and 2 (invalid configuration or usage). Heavy modules (torch, Ultralytics, the training
stack) are imported inside the commands that need them, so ``--help`` and ``version`` stay fast.

Commands: ``run``, ``demo``, ``mock-backend``, ``data download|convert|merge|validate|stats``,
``train``, ``eval``, ``export``, ``bench``, ``calibrate``, ``capture``, ``autolabel``,
``selftest``, ``version``.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import platform
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, NoReturn, TypeVar

import typer

from safety_ai import __version__
from safety_ai.config.loader import ConfigError, exit_on_config_error, load_settings
from safety_ai.config.models import AppSettings

__all__ = ["app", "main"]

_log = logging.getLogger("safety_ai.cli")

EXIT_OK: Final[int] = 0
EXIT_FAILURE: Final[int] = 1
EXIT_USAGE: Final[int] = 2
_API_KEY_WARNING: Final[str] = (
    "warning: --api-key is visible in the shell history and the process list; prefer the "
    "SAFETY_AI_API_KEY environment variable or a .env file\n"
)
_DURATION: Final[re.Pattern[str]] = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|min|h)?\s*$")
_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

app = typer.Typer(
    name="safety-ai",
    help=(
        "AI module of the Digital Safety Inspector: real-time occupational-safety event "
        "detection on camera streams. Exit codes: 0 ok, 1 runtime failure, 2 configuration or "
        "usage error."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
data_app = typer.Typer(
    help="Dataset pipeline: download, convert, merge, validate and describe the training data.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")

_T = TypeVar("_T")


def _fail(message: str, code: int = EXIT_FAILURE) -> NoReturn:
    """Print ``error: <message>`` to stderr and exit with ``code``."""
    sys.stderr.write(f"error: {message}\n")
    sys.stderr.flush()
    raise typer.Exit(code)


def _settings(config: Path | None, overrides: dict[str, Any] | None = None) -> AppSettings:
    """Load the configuration; invalid settings print one line each and exit with 2."""
    try:
        return load_settings(config, overrides)
    except ConfigError as exc:
        exit_on_config_error(exc)


def _guard(action: Callable[[], _T]) -> _T:
    """Run a command body: configuration errors exit 2, Ctrl+C and failures exit 1."""
    try:
        return action()
    except typer.Exit:
        raise
    except ConfigError as exc:
        exit_on_config_error(exc)
    except KeyboardInterrupt:
        _fail("interrupted")
    except Exception as exc:
        _log.debug("command failed", exc_info=True)
        _fail(f"{type(exc).__name__}: {exc}")


def _basic_logging() -> None:
    """Human-readable INFO logging on stderr for the offline commands."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _split_list(text: str | None) -> list[str]:
    """``"a, b,c"`` -> ``["a", "b", "c"]`` (empty entries removed)."""
    if text is None:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


# --------------------------------------------------------------------------------------------
# run, demo, mock-backend, selftest, version
# --------------------------------------------------------------------------------------------


@app.command()
def run(
    *,
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML configuration file (e.g. configs/demo_file.yaml)."
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help=(
            "Quick start with one camera: rtsp:// or http(s):// URL, video file, image "
            "directory, webcam index or fake:<scenario.yaml>. Replaces the cameras of --config."
        ),
    ),
    camera: str = typer.Option("cam-1", "--camera", help="Camera id for --source."),
    loop: bool = typer.Option(False, "--loop", help="Repeat a --source file when it ends."),
    backend: str | None = typer.Option(
        None, "--backend", help="Website backend URL (default: config or SAFETY_AI_BACKEND_URL)."
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key (prefer SAFETY_AI_API_KEY: command-line arguments leak into the history).",
    ),
    show: bool = typer.Option(False, "--show", help="Open a preview window per camera."),
    expose: bool = typer.Option(
        False, "--expose", help="Bind the MJPEG and health servers to 0.0.0.0 (all interfaces)."
    ),
) -> None:
    """Run the full service: workers per camera, backend sync, spool, MJPEG and /health."""
    if config is None and source is None:
        _fail("give --config FILE or --source SOURCE (see safety-ai run --help)", EXIT_USAGE)
    if api_key is not None:
        sys.stderr.write(_API_KEY_WARNING)
    overrides: dict[str, Any] = {"backend.url": backend, "backend.api_key": api_key}
    if source is not None:
        overrides["cameras"] = [{"id": camera, "source": source, "loop": loop}]
    settings = _settings(config, overrides)
    if not any(cam.enabled for cam in settings.cameras):
        _fail("the configuration has no enabled camera (cameras: [...])", EXIT_USAGE)
    from safety_ai.app import run_service

    raise typer.Exit(_guard(lambda: run_service(settings, show=show, expose=expose)))


@app.command()
def demo(
    *,
    source: str = typer.Option(
        ..., "--source", help="Video file, image directory, URL, webcam index or fake:<yaml>."
    ),
    zones: Path | None = typer.Option(
        None, "--zones", help="Zones file with the contract's config structure."
    ),
    show: bool = typer.Option(False, "--show", help="Open the preview window."),
    mjpeg: bool = typer.Option(False, "--mjpeg", help="Serve the annotated MJPEG stream."),
    loop: bool = typer.Option(False, "--loop", help="Repeat the file or scenario."),
    duration: float | None = typer.Option(
        None, "--duration", min=0.1, help="Stop after this many seconds."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file for models, thresholds and output."
    ),
) -> None:
    """Offline demo without a backend: JSON-line events on stdout, events/ jpg+mp4, MJPEG."""
    settings = _settings(config)
    from safety_ai.demo import run_demo

    raise typer.Exit(
        _guard(
            lambda: run_demo(
                source,
                zones,
                show=show,
                mjpeg=mjpeg,
                loop=loop,
                settings=settings,
                duration_s=duration,
            )
        )
    )


@app.command("mock-backend")
def mock_backend(
    *,
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8010, "--port", min=0, max=65535, help="TCP port."),
    config: Path | None = typer.Option(
        None, "--config", help="Camera configuration YAML (default: tools/mock_config.yaml)."
    ),
    data_dir: Path = typer.Option(
        Path("mock_data"), "--data-dir", help="Where events.jsonl and snapshots/ go (emptied)."
    ),
) -> None:
    """The strict mock of the website backend (API key: SAFETY_AI_API_KEY, else test-key)."""
    from tools.mock_backend import DEFAULT_CONFIG_PATH, serve
    from tools.mock_state import ConfigFileError

    api_key = os.environ.get("SAFETY_AI_API_KEY") or "test-key"
    config_path = config if config is not None else DEFAULT_CONFIG_PATH
    try:
        serve(host, port, config_path, data_dir, api_key)
    except ConfigFileError as exc:
        _fail(str(exc), EXIT_USAGE)
    except OSError as exc:
        _fail(f"cannot serve on {host}:{port}: {exc.strerror or exc}")


@app.command()
def selftest(
    *,
    quick: bool = typer.Option(
        False, "--quick", help="Environment/registry and contract checks only."
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Check this backend (e.g. the website) instead of the mock."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key for --backend (default: SAFETY_AI_API_KEY)."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Models to check."),
) -> None:
    """Environment, registry and contract self-check; prints a pass/fail table."""
    if api_key is not None:
        sys.stderr.write(_API_KEY_WARNING)
    if backend is not None and not re.match(r"^https?://", backend):
        _fail("--backend must be an http:// or https:// URL", EXIT_USAGE)
    settings = _settings(config)
    key = api_key
    if backend is not None and key is None and settings.backend.api_key is not None:
        key = settings.backend.api_key.get_secret_value()
    from safety_ai.selftest import run_selftest

    logging.basicConfig(level=logging.WARNING)
    raise typer.Exit(
        _guard(
            lambda: run_selftest(
                quick, backend.rstrip("/") if backend else None, key, settings=settings
            )
        )
    )


def _runtime_versions() -> list[tuple[str, str]]:
    """Versions of the runtime stack (torch and Ultralytics are imported here)."""
    rows: list[tuple[str, str]] = []
    for module in ("ultralytics", "torch"):
        try:
            imported = importlib.import_module(module)
            rows.append((module, str(getattr(imported, "__version__", "unknown"))))
        except Exception as exc:
            rows.append((module, f"not available ({type(exc).__name__})"))
    return rows


@app.command()
def version(
    config: Path | None = typer.Option(None, "--config", "-c", help="Use models.device from it."),
) -> None:
    """Package, Python, Ultralytics and torch versions, the device and the backends."""
    settings = _settings(config)
    lines = [("safety-ai", __version__), ("python", platform.python_version())]
    lines.extend(_runtime_versions())
    try:
        from safety_ai.inference.backends import resolve_device

        device = resolve_device(settings.models.device)
    except Exception as exc:
        device = f"unavailable ({type(exc).__name__}: {exc})"
    lines.append(("device", f"{settings.models.device} -> {device}"))
    backends = ["pt"] + [
        name
        for name, module in (
            ("onnx", "onnxruntime"),
            ("openvino", "openvino"),
            ("tensorrt", "tensorrt"),
        )
        if importlib.util.find_spec(module) is not None
    ]
    lines.append(("backends", ", ".join(backends)))
    lines.append(("platform", f"{platform.system()} {platform.release()} ({platform.machine()})"))
    width = max(len(name) for name, _ in lines)
    for name, value in lines:
        typer.echo(f"{name.ljust(width)}  {value}")


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------

_SOURCES_HELP: Final[str] = "Comma-separated sources (sh17,css,hardhat,chv,forklift,harness; all)."


def _data_paths(data_dir: Path) -> Any:
    from safety_ai.training.pipeline import DataPaths

    return DataPaths(data_dir)


def _source_keys(sources: str | None) -> list[str]:
    from safety_ai.training.sources.catalog import parse_source_list

    try:
        return parse_source_list(sources)
    except (KeyError, ValueError) as exc:
        _fail(f"--sources: {exc}", EXIT_USAGE)


@data_app.command("download")
def data_download(
    sources: str | None = typer.Option(None, "--sources", help=_SOURCES_HELP),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data root."),
) -> None:
    """Download the datasets into data/raw/<source>/ (resumable) and log each attempt."""
    _basic_logging()
    from safety_ai.training.pipeline import run_download

    keys = _source_keys(sources)
    reports = _guard(lambda: run_download(keys, paths=_data_paths(data_dir)))
    for report in reports:
        typer.echo(f"{report.source:10s} {report.status:8s} {report.detail}")
    if reports and all(report.status in ("failed", "skipped") for report in reports):
        raise typer.Exit(EXIT_FAILURE)


@data_app.command("convert")
def data_convert(
    sources: str | None = typer.Option(None, "--sources", help=_SOURCES_HELP),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data root."),
) -> None:
    """Convert and remap the downloaded detection sources onto the unified taxonomy."""
    _basic_logging()
    from safety_ai.training.pipeline import run_convert

    keys = _source_keys(sources)
    summaries = _guard(lambda: run_convert(keys, paths=_data_paths(data_dir)))
    for key, summary in summaries.items():
        typer.echo(f"{key:10s} {summary.images} images, {summary.boxes_kept} boxes kept")


@data_app.command("merge")
def data_merge(
    sources: str | None = typer.Option(None, "--sources", help=_SOURCES_HELP),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data root."),
) -> None:
    """Validate, deduplicate, split and write data/merged (YOLO format)."""
    _basic_logging()
    from safety_ai.training.pipeline import run_merge

    keys = _source_keys(sources)
    summary = _guard(lambda: run_merge(keys, paths=_data_paths(data_dir)))
    typer.echo(f"merged {', '.join(summary.sources)}: splits {summary.splits}")


@data_app.command("validate")
def data_validate(
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data root."),
) -> None:
    """Re-check data/merged and store the result in data/merged/validation.json."""
    _basic_logging()
    from safety_ai.training.pipeline import run_validate

    report = _guard(lambda: run_validate(paths=_data_paths(data_dir)))
    typer.echo(f"validation {'passed' if report.ok else 'FAILED'}: images {report.images}")
    if not report.ok:
        for error in report.errors:
            typer.echo(f"  {error}", err=True)
        raise typer.Exit(EXIT_FAILURE)


@data_app.command("stats")
def data_stats(
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data root."),
    report: Path = typer.Option(Path("reports/data_stats.md"), "--report", help="Report file."),
) -> None:
    """Write reports/data_stats.md for data/merged."""
    _basic_logging()
    from safety_ai.training.pipeline import run_stats

    stats = _guard(lambda: run_stats(paths=_data_paths(data_dir), report_path=report))
    typer.echo(f"wrote {report} ({stats.total_images} images, {stats.total_instances} instances)")


# --------------------------------------------------------------------------------------------
# train, eval, export, bench
# --------------------------------------------------------------------------------------------


@app.command()
def train(
    *,
    base: str | None = typer.Option(
        None, "--base", help="Base weights (default: yolo26s.pt on CUDA, yolo26n.pt on CPU)."
    ),
    epochs: int | None = typer.Option(None, "--epochs", min=1, help="Epochs (default 80)."),
    imgsz: int | None = typer.Option(None, "--imgsz", min=32, help="Input size (default 640)."),
    time_box_min: float | None = typer.Option(
        None, "--time-box-min", min=1.0, help="Hard time box in minutes (default 180)."
    ),
    smoke: bool = typer.Option(
        False, "--smoke", help="CPU smoke preset: 2 epochs, imgsz 416, 500 images, <= 20 min."
    ),
    data: Path | None = typer.Option(
        None, "--data", help="Dataset data.yaml (default data/merged/data.yaml)."
    ),
    device: str | None = typer.Option(None, "--device", help="auto, cpu, cuda:N or mps."),
) -> None:
    """Fine-tune the PPE detector (Ultralytics API) and register the result."""
    _basic_logging()
    from safety_ai.training.train import TrainConfig, smoke_config
    from safety_ai.training.train import train as run_training

    overrides: dict[str, Any] = {
        key: value
        for key, value in (
            ("base", base),
            ("epochs", epochs),
            ("imgsz", imgsz),
            ("time_box_min", time_box_min),
            ("data", data),
            ("device", device),
        )
        if value is not None
    }
    config = smoke_config(**overrides) if smoke else TrainConfig(**overrides)
    result = _guard(lambda: run_training(config))
    typer.echo(
        f"trained {result.model_name}: {result.epochs_run}/{result.epochs_planned} epochs, "
        f"mAP50 {result.metrics.map50:.3f}; summary {result.summary_path}"
    )


@app.command("eval")
def evaluate_command(
    *,
    model: str = typer.Option(..., "--model", help="Registry name (e.g. ppe_n_latest) or weights."),
    data: Path | None = typer.Option(
        None, "--data", help="Dataset data.yaml (default data/merged/data.yaml)."
    ),
    device: str = typer.Option("auto", "--device", help="auto, cpu, cuda:N or mps."),
) -> None:
    """Evaluate a model and write reports/eval_<model>.md."""
    _basic_logging()
    from safety_ai.training.evaluate import evaluate
    from safety_ai.training.train import DEFAULT_DATA_YAML

    report = _guard(
        lambda: evaluate(model, data if data is not None else DEFAULT_DATA_YAML, device=device)
    )
    typer.echo(
        f"{report.model_name}: mAP50 {report.map50:.3f}, mAP50-95 {report.map50_95:.3f}; "
        f"report {report.report_path}"
    )


@app.command("export")
def export_command(
    *,
    model: str = typer.Option(..., "--model", help="Registry name (e.g. ppe_n_latest)."),
    formats: str = typer.Option(
        "onnx,openvino", "--formats", help="Comma-separated: onnx, openvino, tensorrt."
    ),
    imgsz: int | None = typer.Option(None, "--imgsz", min=32, help="Fixed input size."),
    device: str = typer.Option("cpu", "--device", help="Export device (TensorRT uses CUDA)."),
) -> None:
    """Export a registered model and register the artefacts."""
    _basic_logging()
    from safety_ai.training.export import export

    wanted = _split_list(formats)
    if not wanted:
        _fail("--formats needs at least one format", EXIT_USAGE)
    result = _guard(lambda: export(model, wanted, imgsz=imgsz, device=device))
    for artifact in result.artifacts:
        typer.echo(f"{artifact.format:9s} {artifact.path}")
    for skipped in result.skipped:
        typer.echo(f"{skipped.format:9s} skipped: {skipped.reason}")
    if not result.artifacts:
        raise typer.Exit(EXIT_FAILURE)


@app.command()
def bench(
    *,
    models: str | None = typer.Option(
        None, "--models", help="Comma-separated registry names (default: models.ppe, models.pose)."
    ),
    backends: str | None = typer.Option(
        None, "--backends", help="Comma-separated: pt, onnx, openvino, tensorrt (default: all)."
    ),
    seconds: float = typer.Option(20.0, "--seconds", min=1.0, help="Seconds per combination."),
    imgsz: int | None = typer.Option(None, "--imgsz", min=32, help="Inference size."),
    device: str | None = typer.Option(None, "--device", help="auto, cpu, cuda:N or mps."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Configuration file."),
) -> None:
    """Measure FPS, latency p50/p95 and peak RSS; results go to models/bench.json."""
    _basic_logging()
    settings = _settings(config)
    from safety_ai.inference.bench import bench as run_bench
    from safety_ai.inference.bench import default_bench_images, render_markdown
    from safety_ai.inference.registry import ModelRegistry

    names = _split_list(models) or [
        name for name in (settings.models.ppe, settings.models.pose) if name.lower() != "none"
    ]
    chosen = _split_list(backends) or None
    registry = ModelRegistry(settings.models.registry)
    report = _guard(
        lambda: run_bench(
            names,
            chosen,
            seconds,
            imgsz or settings.models.imgsz,
            device or settings.models.device,
            registry,
            default_bench_images(),
        )
    )
    typer.echo(render_markdown(report))


# --------------------------------------------------------------------------------------------
# calibrate, capture, autolabel
# --------------------------------------------------------------------------------------------


def _points(text: str, option: str) -> list[tuple[float, float]]:
    """``"x1,y1,...,x4,y4"`` -> four points; usage error otherwise."""
    try:
        values = [float(part) for part in _split_list(text)]
    except ValueError:
        _fail(f"{option} must be numbers: x1,y1,x2,y2,x3,y3,x4,y4", EXIT_USAGE)
    if len(values) != 8:
        _fail(f"{option} needs exactly 4 points (8 numbers), got {len(values)} numbers", EXIT_USAGE)
    return [(values[i], values[i + 1]) for i in range(0, 8, 2)]


@app.command()
def calibrate(
    *,
    camera: str = typer.Option(..., "--camera", help="Camera id (e.g. cam-1)."),
    pixels: str = typer.Option(..., "--pixels", help="Four image points: x1,y1,...,x4,y4."),
    metres: str = typer.Option(..., "--metres", help="The ground points in metres: X1,Y1,..."),
    frame_size: str | None = typer.Option(
        None, "--frame-size", help="Frame size the pixels were measured in, WxH (e.g. 1280x720)."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration (paths.calibration_dir)."
    ),
) -> None:
    """Compute the ground-plane homography and write calibration/<camera>.json."""
    settings = _settings(config)
    from safety_ai.config.loader import (
        CalibrationSidecar,
        write_calibration_sidecar,
    )
    from safety_ai.config.models import CameraSettings
    from safety_ai.logic.proximity import compute_homography

    try:
        CameraSettings(id=camera)
    except ValueError:
        _fail(f"--camera: invalid camera id {camera!r}", EXIT_USAGE)
    image_points = _points(pixels, "--pixels")
    ground_points = _points(metres, "--metres")
    size: tuple[int, int] | None = None
    if frame_size is not None:
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", frame_size)
        if match is None:
            _fail("--frame-size must be WxH, e.g. 1280x720", EXIT_USAGE)
        size = (int(match.group(1)), int(match.group(2)))
    try:
        matrix = compute_homography(image_points, ground_points)
    except ValueError as exc:
        _fail(f"cannot compute the homography: {exc}", EXIT_USAGE)
    sidecar = CalibrationSidecar.model_validate(
        {
            "homography": [[float(value) for value in row] for row in matrix],
            "frame_size": size,
            "pixels": [list(point) for point in image_points],
            "metres": [list(point) for point in ground_points],
        }
    )
    path = _guard(
        lambda: write_calibration_sidecar(settings.paths.calibration_dir, camera, sidecar)
    )
    typer.echo(f"wrote {path}")


def _seconds(text: str) -> float:
    """``2s``, ``500ms``, ``1.5``, ``1m``, ``1h`` -> seconds; usage error otherwise."""
    match = _DURATION.match(text)
    if match is None:
        _fail(f"--every: cannot read {text!r} (examples: 2s, 500ms, 1m)", EXIT_USAGE)
    value = float(match.group(1))
    unit = match.group(2) or "s"
    seconds = value * {"ms": 0.001, "s": 1.0, "m": 60.0, "min": 60.0, "h": 3600.0}[unit]
    if seconds <= 0:
        _fail("--every must be positive", EXIT_USAGE)
    return seconds


@app.command()
def capture(
    *,
    source: str = typer.Option(..., "--source", help="Video file or stream URL (or webcam index)."),
    every: str = typer.Option("2s", "--every", help="Interval between saved frames (2s, 500ms)."),
    out: Path = typer.Option(Path("data/venue"), "--out", help="Output directory."),
    max_frames: int | None = typer.Option(
        None, "--max-frames", min=1, help="Stop after this many frames."
    ),
) -> None:
    """Save a frame every N seconds from a source (venue images for labelling)."""
    _basic_logging()
    interval = _seconds(every)
    import cv2

    from safety_ai.capture.base import SourceError
    from safety_ai.capture.factory import open_source
    from safety_ai.capture.types import StreamState
    from safety_ai.clock import SystemClock
    from safety_ai.config.models import CameraSettings

    try:
        frames = open_source(CameraSettings(id="capture", source=source), SystemClock())
    except (SourceError, ValueError) as exc:
        _fail(str(exc), EXIT_USAGE)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    next_at: float | None = None

    def loop() -> None:
        nonlocal saved, next_at
        frames.start()
        while max_frames is None or saved < max_frames:
            frame = frames.read(1.0)
            if frame is None:
                if frames.state in (StreamState.ENDED, StreamState.FAILED):
                    return
                continue
            if next_at is not None and frame.mono < next_at:
                continue
            next_at = frame.mono + interval
            stamp = (
                frame.captured_at.strftime("%Y%m%dT%H%M%S")
                + f"{frame.captured_at.microsecond // 1000:03d}Z"
            )
            path = out / f"capture_{stamp}_{frame.frame_id:06d}.jpg"
            if not cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"cannot write {path}")
            saved += 1
            typer.echo(str(path))

    try:
        _guard(loop)
    finally:
        frames.stop()
    typer.echo(f"saved {saved} frame(s) to {out}")
    if saved == 0:
        raise typer.Exit(EXIT_FAILURE)


@app.command()
def autolabel(
    directory: Path = typer.Argument(..., help="Directory of images to label."),
    model: str = typer.Option(
        "ppe", "--model", help="Registry name, or ppe (models.ppe of the configuration)."
    ),
    conf: float = typer.Option(0.25, "--conf", min=0.0, max=1.0, help="Confidence threshold."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Configuration file."),
) -> None:
    """Pre-label images with a registered detector (YOLO labels in unified-taxonomy ids)."""
    _basic_logging()
    if not directory.is_dir():
        _fail(f"{directory} is not a directory", EXIT_USAGE)
    images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        _fail(f"no images in {directory}", EXIT_USAGE)
    settings = _settings(config)
    name = settings.models.ppe if model == "ppe" else model
    labelled = _guard(lambda: _autolabel(images, name, conf, settings))
    typer.echo(f"labelled {labelled} of {len(images)} image(s) with {name}")


def _autolabel(images: Sequence[Path], name: str, conf: float, settings: AppSettings) -> int:
    """Write one YOLO label file per image; returns how many images got at least one box."""
    import cv2

    from safety_ai.inference.backends import load_model, resolve_device
    from safety_ai.inference.detector import extract_boxes
    from safety_ai.inference.registry import ModelRegistry
    from safety_ai.inference.types import TAXONOMY
    from safety_ai.training.train import label_path

    registry = ModelRegistry(settings.models.registry)
    entry = registry.resolve(name)
    device = resolve_device(settings.models.device)
    loaded = load_model(entry, registry, backend="pt", device=device, task="detect")
    labels = loaded.labels
    with_boxes = 0
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            _log.warning("cannot read %s; skipped", image_path)
            continue
        results = loaded.model.predict(
            image, conf=conf, imgsz=settings.models.imgsz, device=device, verbose=False
        )
        lines: list[str] = []
        for box in extract_boxes(results[0] if results else None):
            label = labels[box.cls] if 0 <= box.cls < len(labels) else None
            if label is None or label not in TAXONOMY:
                continue
            x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in box.xyxyn)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(
                f"{TAXONOMY.index(label)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                f"{x2 - x1:.6f} {y2 - y1:.6f}"
            )
        target = label_path(image_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        with_boxes += bool(lines)
    names = "\n".join(f"  {index}: {label}" for index, label in enumerate(TAXONOMY))
    (images[0].parent / "autolabel_classes.yaml").write_text(
        f"# unified taxonomy class ids of the label files (written {time.strftime('%Y-%m-%d')})\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return with_boxes


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point; returns the exit code."""
    try:
        app(
            args=list(argv) if argv is not None else None,
            prog_name="safety-ai",
            standalone_mode=True,
        )
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return EXIT_OK
        if isinstance(code, int):
            return code
        sys.stderr.write(f"{code}\n")
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
