"""Spawn-picklable fake worker targets of the supervisor tests.

Spawned children import this module by name (the ``fake_workers`` fixture puts its directory on
``sys.path``, which the children inherit). Every target has the signature of
:func:`safety_ai.worker.worker_main` and talks to the supervisor through the real IPC messages.
"""

from __future__ import annotations

import os
import queue
import sys
import time
from typing import Any

from safety_ai.ipc import BoundedOutbox, ConfigUpdate, OutboxMessage, Stop, WorkerSpec, WorkerStatus

STATUS_PERIOD_S = 0.05


def _status(spec: WorkerSpec, phase: str, **fields: Any) -> WorkerStatus:
    return WorkerStatus(
        camera_id=spec.camera_id,
        generation=spec.generation,
        pid=os.getpid(),
        phase=phase,  # type: ignore[arg-type]
        **fields,
    )


def crash_worker(spec: WorkerSpec, outbox: BoundedOutbox, control_q: Any, stop_event: Any) -> None:
    """Reports ``starting`` and dies with exit code 1 (a crash)."""
    outbox.put(_status(spec, "starting", error="simulated crash"))
    outbox.close()
    sys.exit(1)


def steady_worker(spec: WorkerSpec, outbox: BoundedOutbox, control_q: Any, stop_event: Any) -> None:
    """A healthy camera: connected, 10 fps; echoes configuration versions; stops on Stop."""
    version = spec.config.version if spec.config is not None else None
    started = time.monotonic()
    while not stop_event.is_set():
        stop = False
        while True:
            try:
                message = control_q.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, ConfigUpdate):
                version = message.config.version
            elif isinstance(message, Stop):
                stop = True
        if stop:
            break
        outbox.put(
            _status(
                spec,
                "running",
                stream_state="connected",
                processed_fps=10.0,
                target_fps=10.0,
                last_frame_age_s=0.05,
                frames_processed=int((time.monotonic() - started) * 10),
                config_version=version,
                config_source="backend" if version is not None else "none",
            )
        )
        time.sleep(STATUS_PERIOD_S)
    outbox.put(_status(spec, "stopped", config_version=version))
    outbox.close()
    sys.exit(0)


def mixed_worker(spec: WorkerSpec, outbox: BoundedOutbox, control_q: Any, stop_event: Any) -> None:
    """``crash_worker`` for cameras whose id starts with ``crash``, else ``steady_worker``."""
    if spec.camera_id.startswith("crash"):
        crash_worker(spec, outbox, control_q, stop_event)
    steady_worker(spec, outbox, control_q, stop_event)


def child_put(outbox: BoundedOutbox, message: OutboxMessage) -> None:
    """Put one message into the outbox from a spawned child, then exit."""
    outbox.put(message)
    outbox.close()


def ended_worker(spec: WorkerSpec, outbox: BoundedOutbox, control_q: Any, stop_event: Any) -> None:
    """A camera whose non-looping source ends after a short run (exit code EXIT_ENDED)."""
    from safety_ai.ipc import EXIT_ENDED  # noqa: PLC0415

    outbox.put(_status(spec, "running", stream_state="connected", processed_fps=10.0))
    time.sleep(0.2)
    outbox.put(_status(spec, "ended", stream_state="ended"))
    outbox.close()
    sys.exit(EXIT_ENDED)


ORPHAN_TRACK = 4242
"""Track id of the event the orphaning worker leaves open."""
ORPHAN_STARTED_AGO_S = 20
ORPHAN_ALIVE_AFTER_S = 7


def spool_open_event(spool_path: Any, camera_id: str, zone_id: str, now: Any) -> tuple[str, Any]:
    """Spool the POST of an ACTIVE event and its alive time, as a running worker does.

    Returns:
        The external id and the alive time recorded.
    """
    from datetime import timedelta  # noqa: PLC0415

    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from safety_ai.backend.schemas import EventPayload, make_external_id  # noqa: PLC0415
    from safety_ai.backend.spool import EventSpool  # noqa: PLC0415

    started = (now - timedelta(seconds=ORPHAN_STARTED_AGO_S)).replace(microsecond=0)
    alive = started + timedelta(seconds=ORPHAN_ALIVE_AFTER_S)
    external_id = make_external_id(camera_id, started, ORPHAN_TRACK, "ppe_violation")
    box = {"track_id": ORPHAN_TRACK, "label": "person", "bbox": [0.3, 0.3, 0.2, 0.6], "conf": 0.9}
    payload = EventPayload.model_validate(
        {
            "external_id": external_id,
            "camera_id": camera_id,
            "zone_id": zone_id,
            "type": "ppe_violation",
            "severity": None,
            "confidence": 0.9,
            "track_id": ORPHAN_TRACK,
            "started_at": started,
            "ended_at": None,
            "details": {
                "missing_ppe": ["helmet"],
                "present_ppe": [],
                "boxes": [box],
                "frame_size": [1280, 720],
            },
        }
    )
    ok, jpeg = cv2.imencode(".jpg", np.full((72, 128, 3), 80, np.uint8))
    assert ok
    spool = EventSpool(spool_path)
    try:
        spool.enqueue_post(payload, jpeg.tobytes())
        spool.mark_alive(camera_id, [external_id], alive)
    finally:
        spool.close()
    return external_id, alive


def orphaning_worker(
    spec: WorkerSpec, outbox: BoundedOutbox, control_q: Any, stop_event: Any
) -> None:
    """Generation 1 spools an ACTIVE event and is killed (no PATCH); later ones are steady."""
    if spec.generation > 1:
        steady_worker(spec, outbox, control_q, stop_event)
    from datetime import datetime, timezone  # noqa: PLC0415

    assert spec.config is not None and spec.spool_path is not None
    outbox.put(_status(spec, "running", stream_state="connected", processed_fps=10.0))
    spool_open_event(
        spec.spool_path, spec.camera_id, spec.config.zones[0].id, datetime.now(timezone.utc)
    )
    outbox.close()
    os._exit(9)


def hanging_worker(
    spec: WorkerSpec, outbox: BoundedOutbox, control_q: Any, stop_event: Any
) -> None:
    """Generation 1 hangs and ignores every stop; later generations are steady.

    A camera whose id starts with ``stuck-start`` hangs while starting (as in a model load that
    never returns); any other camera reports ``running`` for a moment, then falls silent.
    """
    if spec.generation > 1:
        steady_worker(spec, outbox, control_q, stop_event)
    outbox.put(_status(spec, "starting"))
    if not spec.camera_id.startswith("stuck-start"):
        for _ in range(5):
            outbox.put(_status(spec, "running", stream_state="connected", processed_fps=10.0))
            time.sleep(STATUS_PERIOD_S)
    while True:
        time.sleep(1.0)
