"""``safety-ai demo``: one camera in-process, no backend (ARCHITECTURE 2; spec 7.3, 5.12, 5.13).

The live-presentation fallback. :func:`run_demo` runs the real pipeline
(:class:`~safety_ai.worker.CameraPipeline`) on one source in the calling process - no worker
processes, no spool, no network:

* every event start, PPE change and end is printed to stdout as one JSON line (the start line
  is the contract payload, :class:`~safety_ai.sinks.JsonLinesSink`); logs go to stderr in the
  human console format;
* ``events/<external_id>.jpg`` (annotated) and ``events/<external_id>.mp4`` (5 s before and
  after) are written for every event (``output.save_event_clips``, ``output.events_dir``);
* ``--mjpeg`` serves ``http://127.0.0.1:<output.mjpeg_port>/stream/<camera>.mjpeg`` from a
  thread (``0.0.0.0`` with ``output.expose``);
* ``--show`` opens the preview window (``q`` quits; disabled with one warning without a display).

Zones come from ``--zones`` (the contract's config structure, :mod:`safety_ai.config.zones`),
else from the scenario of a ``fake:<scenario.yaml>`` source (its ``config_updates`` applied as
it plays), else there are none. The demo ends with the source (``--loop`` repeats files and
scenarios), after ``--duration`` seconds, with ``q`` in the preview or with Ctrl+C; every event
still ACTIVE is then ended (an ``ended`` line with reason ``shutdown``) and the clips finished.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Final, TextIO

from safety_ai.app import bind_host
from safety_ai.capture.base import SourceError
from safety_ai.capture.factory import open_source
from safety_ai.capture.scenario import load_scenario
from safety_ai.capture.types import FrameSource
from safety_ai.clock import SystemClock
from safety_ai.config.models import AppSettings, CameraSettings
from safety_ai.logic.engine import LogicEngine
from safety_ai.logic.ids import InMemoryIdAllocator
from safety_ai.output.event_recorder import EventRecorder
from safety_ai.output.mjpeg_server import MjpegServer
from safety_ai.output.preview import Preview
from safety_ai.sinks import (
    EventLogSink,
    JsonLinesSink,
    MjpegSink,
    PreviewSink,
    RecorderSink,
    probe_clip_codecs,
)
from safety_ai.telemetry.logging import configure_logging, get_logger
from safety_ai.worker import (
    CameraPipeline,
    CameraSetupError,
    PipelineHooks,
    PreviewControls,
    RunOutcome,
    build_models,
    fake_scenario_path,
    fake_source,
    logic_settings,
    select_config,
)

__all__ = ["DEFAULT_DEMO_CAMERA", "run_demo"]

DEFAULT_DEMO_CAMERA: Final[str] = "cam-1"
"""Camera id of the demo (a ``fake:`` scenario's own camera id wins)."""

_log = logging.getLogger(__name__)


def _demo_camera(
    source: str, zones_file: Path | None, loop: bool, settings: AppSettings, camera_id: str
) -> CameraSettings:
    """The single demo camera: the configured camera's timezone and calibration, if any."""
    base = settings.cameras[0] if settings.cameras else None
    return CameraSettings.model_validate(
        {
            "id": camera_id,
            "source": source,
            "loop": loop,
            "fps": base.fps if base is not None else None,
            "timezone": base.timezone if base is not None else "UTC",
            "homography": base.homography if base is not None else None,
            "zones_file": zones_file,
        }
    )


def run_demo(
    source: str,
    zones_file: Path | None,
    *,
    show: bool,
    mjpeg: bool,
    loop: bool,
    settings: AppSettings,
    duration_s: float | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run the offline demo on ``source`` (see the module docstring).

    Args:
        source: A video file, image directory, stream URL, webcam index or ``fake:<yaml>``.
        zones_file: Zones and rules in the contract's config structure (None: the scenario's
            zones for ``fake:`` sources, else none).
        show: Open the preview window.
        mjpeg: Serve the annotated stream over HTTP.
        loop: Repeat a file or scenario when it ends.
        settings: The configuration (models, thresholds, output, privacy, telemetry); its
            backend is never used.
        duration_s: Stop after this many seconds (None: run until the source ends or Ctrl+C).
        stdout: Where the JSON lines go (default ``sys.stdout``).

    Returns:
        0 when the demo ran, 1 when the source or the models could not be opened.

    Raises:
        ConfigError: The zones file or a scenario file is invalid (the CLI exits with 2).
    """
    api_key = settings.backend.api_key
    secrets = [api_key.get_secret_value()] if api_key is not None else []
    configure_logging(settings.telemetry, json_lines=False, secrets=secrets)
    os.environ.setdefault("YOLO_OFFLINE", "1")
    clock = SystemClock()
    try:
        scenario_path = fake_scenario_path(source)
        scenario = load_scenario(scenario_path) if scenario_path is not None else None
    except SourceError as exc:
        _log.error("demo source %s cannot be used: %s", source, exc)
        return 1
    camera_id = scenario.camera_id if scenario is not None else DEFAULT_DEMO_CAMERA
    camera = _demo_camera(source, zones_file, loop, settings, camera_id)
    demo_settings = settings.model_copy(
        update={
            "backend": settings.backend.model_copy(update={"enabled": False}),
            "cameras": [camera],
        }
    )
    log = get_logger(__name__, camera_id=camera.id)
    choice = select_config(camera, demo_settings, backend_config=None, scenario=scenario)
    try:
        frames: FrameSource = (
            fake_source(camera, scenario, clock)
            if scenario is not None
            else open_source(camera, clock)
        )
        detector, pose = build_models(demo_settings, camera, clock, scenario=scenario)
    except (SourceError, CameraSetupError) as exc:
        log.error("demo cannot start: %s", exc)
        return 1
    output = demo_settings.output
    controls = PreviewControls()
    out = stdout if stdout is not None else sys.stdout
    sinks: list[object] = [EventLogSink(camera.id)]
    recorder_sink: RecorderSink | None = None
    if output.save_event_clips:
        recorder = EventRecorder(
            output.events_dir,
            fps=demo_settings.processing.target_fps,
            pre_s=output.clip_pre_s,
            post_s=output.clip_post_s,
            disk_cap_bytes=int(output.events_disk_cap_gb * 1024**3),
            codec_preference=probe_clip_codecs(),
        )
        recorder_sink = RecorderSink(recorder, camera.id, close_timeout_s=10.0)
    sinks.append(
        JsonLinesSink(out, artefacts=recorder_sink.artefacts if recorder_sink is not None else None)
    )
    if recorder_sink is not None:
        sinks.append(recorder_sink)
    server: MjpegServer | None = None
    if mjpeg:
        host = bind_host(demo_settings, expose=False)
        server = MjpegServer(host, output.mjpeg_port, max_fps=output.mjpeg_fps)
        server.add_camera(camera.id)
        try:
            server.start()
        except OSError as exc:
            log.error(
                "cannot serve MJPEG on %s:%d (output.mjpeg_port): %s",
                host,
                output.mjpeg_port,
                exc.strerror or exc,
            )
            return 1
        log.info("MJPEG stream: %s", server.stream_url(camera.id))
        publish = server.publish
        sinks.append(MjpegSink(lambda jpeg: publish(camera.id, jpeg), max_fps=output.mjpeg_fps))
    if show or output.preview:
        preview = Preview(f"safety-ai demo {camera.id}", Path("screenshots"))
        sinks.append(PreviewSink(preview, controls.apply))
    engine = LogicEngine(
        camera.id,
        choice.config,
        logic_settings(demo_settings),
        timezone=camera.timezone,
        homography=camera.homography,
        id_allocator=InMemoryIdAllocator(),
    )
    stop = threading.Event()
    pipeline = CameraPipeline(
        camera.id,
        demo_settings,
        detector,
        pose,
        engine,
        sinks,
        clock,
        config_source=choice.source,
        scenario=scenario,
        scenario_loop=loop,
        hooks=PipelineHooks(),
        controls=controls,
    )
    log.info(
        "demo: source %s, zones from %s (%s), detector %s",
        source,
        choice.source,
        ", ".join(zone.id for zone in choice.config.zones) or "no zones",
        detector.info.name,
    )
    timer = threading.Timer(duration_s, stop.set) if duration_s is not None else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    outcome: RunOutcome = "stopped"
    try:
        outcome = pipeline.run(frames, stop)
    except KeyboardInterrupt:
        log.info("demo interrupted")
    finally:
        if timer is not None:
            timer.cancel()
        pipeline.shutdown()
        frames.stop()
        pipeline.close()
        if server is not None:
            server.stop()
    log.info(
        "demo finished (%s): %d frames, %d events started, %d ended",
        outcome,
        pipeline.frames_processed,
        pipeline.events_started,
        pipeline.events_ended,
    )
    return 1 if outcome == "failed" else 0
