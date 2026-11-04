"""Fixtures of the output tests: frame results, synthetic JPEGs, MJPEG servers and publishers."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from safety_ai.inference.types import BBox, Detection, Keypoints
from safety_ai.logic.types import FrameResult, PersonView, VehicleView
from safety_ai.output.annotate import AnnotationInfo
from safety_ai.output.mjpeg_server import MjpegServer

UTC = datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc)

PersonFactory = Callable[..., PersonView]
ResultFactory = Callable[..., FrameResult]
JpegFactory = Callable[..., bytes]


@pytest.fixture
def info() -> AnnotationInfo:
    return AnnotationInfo(camera_id="cam-1", model_name="ppe_s_20261104", fps=11.8, utc=UTC)


@pytest.fixture
def make_person() -> PersonFactory:
    def factory(
        track_id: int = 1,
        bbox: BBox = (0.4, 0.3, 0.2, 0.6),
        *,
        state: str = "ok",
        ppe: Mapping[str, bool | None] | None = None,
        keypoints: Keypoints | None = None,
        items: tuple[Detection, ...] = (),
        active_events: tuple[str, ...] = (),
        conf: float = 0.9,
    ) -> PersonView:
        return PersonView(
            track_id=track_id,
            bbox=bbox,
            conf=conf,
            zones=(),
            ppe=dict(ppe or {}),
            evidence={},
            state=state,  # type: ignore[arg-type]
            items=items,
            keypoints=keypoints,
            active_events=active_events,
        )

    return factory


@pytest.fixture
def make_result() -> ResultFactory:
    def factory(
        persons: tuple[PersonView, ...] = (),
        vehicles: tuple[VehicleView, ...] = (),
        active_zone_ids: tuple[str, ...] = (),
        frame_size: tuple[int, int] = (640, 360),
    ) -> FrameResult:
        return FrameResult(
            frame_id=7,
            mono=100.0,
            utc=UTC,
            frame_size=frame_size,
            persons=persons,
            vehicles=vehicles,
            active_zone_ids=active_zone_ids,
        )

    return factory


def encode_test_jpeg(index: int, size: tuple[int, int] = (64, 48)) -> bytes:
    """A small JPEG whose content depends on ``index`` (distinct bytes per index)."""
    width, height = size
    image = np.full((height, width, 3), (index * 37) % 256, dtype=np.uint8)
    cv2.putText(image, str(index), (2, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buffer.tobytes()


@pytest.fixture
def jpeg_frame() -> JpegFactory:
    return encode_test_jpeg


class Publisher:
    """Publishes distinct JPEGs to an MJPEG server from a background thread."""

    def __init__(
        self,
        server: MjpegServer,
        camera_id: str,
        interval_s: float,
        frame: Callable[[int], bytes] = encode_test_jpeg,
    ) -> None:
        self.server = server
        self.camera_id = camera_id
        self.interval_s = interval_s
        self.frame = frame
        self.published: list[bytes] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="test-publisher", daemon=True)

    def start(self) -> Publisher:
        self.server.add_camera(self.camera_id)  # known (no 404) before the first frame
        self._thread.start()
        return self

    def _run(self) -> None:
        index = 0
        while not self._stop.is_set():
            jpeg = self.frame(index)
            self.server.publish(self.camera_id, jpeg)
            self.published.append(jpeg)
            index += 1
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


@pytest.fixture
def start_publisher() -> Iterator[Callable[..., Publisher]]:
    publishers: list[Publisher] = []

    def factory(
        server: MjpegServer,
        camera_id: str = "cam-1",
        interval_s: float = 0.005,
        frame: Callable[[int], bytes] = encode_test_jpeg,
    ) -> Publisher:
        publisher = Publisher(server, camera_id, interval_s, frame).start()
        publishers.append(publisher)
        return publisher

    yield factory
    for publisher in publishers:
        publisher.stop()


@pytest.fixture
def make_mjpeg_server() -> Iterator[Callable[..., MjpegServer]]:
    servers: list[MjpegServer] = []

    def factory(*, max_fps: float = 50.0, max_clients: int = 32) -> MjpegServer:
        server = MjpegServer("127.0.0.1", 0, max_fps=max_fps, max_clients=max_clients)
        server.start()
        servers.append(server)
        return server

    yield factory
    for server in servers:
        server.stop()
