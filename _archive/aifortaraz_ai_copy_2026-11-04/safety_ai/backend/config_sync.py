"""Camera configuration sync with the backend (contract 4.4, ARCHITECTURE 3.4).

:class:`ConfigSync` reads the zones and rules of every local camera from the backend at start,
every ``refresh_s`` seconds (60 s) and immediately on a ``config.updated`` WebSocket message, and
caches the last good configuration of each camera in ``<cache_dir>/config_<camera_id>.json``
(written atomically: temporary file + ``os.replace``).

- :meth:`ConfigSync.load_initial` asks the backend first (``GET /api/v1/ingest/config`` for all
  cameras, then ``GET /api/v1/ingest/config/{camera_id}`` for cameras missing from that list) and
  falls back to the cache per camera when the backend cannot provide it; it logs which source was
  used for every camera. Cameras missing from both are simply absent from the result.
- :meth:`ConfigSync.refresh` calls ``on_config`` only for cameras whose configuration changed
  (version or content).
- :meth:`ConfigSync.start` runs the periodic refresh in a daemon thread. It measures the interval
  on the injected clock and waits on an event in short slices, so :meth:`ConfigSync.stop` and
  :meth:`ConfigSync.request_refresh` interrupt the wait at once and a ``FakeClock`` can drive it.

Camera ids may contain characters that are invalid in Windows file names (``:``), so the cache
file name percent-encodes everything but URL-unreserved characters (``cam-1`` -> ``config_cam-1``).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Literal, Protocol

from safety_ai.backend.rest_client import ENDPOINT_CONFIG, BackendError
from safety_ai.backend.schemas import CameraConfig
from safety_ai.backend.spool import atomic_write_bytes, safe_file_component
from safety_ai.clock import Clock

__all__ = ["ConfigOrigin", "ConfigSource", "ConfigSync"]

_log = logging.getLogger(__name__)

ConfigOrigin = Literal["backend", "cache"]
"""Where a camera's current configuration came from."""

_ALL: Final[str] = "*"


class ConfigSource(Protocol):
    """What :class:`ConfigSync` needs from the REST client (:class:`RestClient` satisfies it)."""

    def get_config(self, camera_id: str) -> CameraConfig:
        """Fetch one camera's configuration; raises :class:`BackendError` on failure."""
        ...

    def get_all_configs(self) -> list[CameraConfig]:
        """Fetch every AI-enabled camera's configuration; raises :class:`BackendError`."""
        ...


class ConfigSync:
    """Keeps the configuration of the local cameras in step with the backend.

    Args:
        client: The REST client (anything implementing :class:`ConfigSource`).
        cache_dir: Directory of the ``config_<camera_id>.json`` cache files.
        camera_ids: The local cameras; configurations of other cameras are ignored.
        on_config: Called with a camera's new configuration whenever it changed.
        clock: Clock that measures the refresh interval.
        refresh_s: Seconds between two periodic refreshes.
        poll_s: Longest single wait of the background thread (the clock is re-read after it).

    Raises:
        ValueError: When ``refresh_s`` or ``poll_s`` is not positive.
    """

    def __init__(
        self,
        client: ConfigSource,
        cache_dir: Path,
        camera_ids: Sequence[str],
        *,
        on_config: Callable[[CameraConfig], None],
        clock: Clock,
        refresh_s: float = 60.0,
        poll_s: float = 1.0,
    ) -> None:
        """Store the settings; nothing is fetched until :meth:`load_initial` or a refresh."""
        if refresh_s <= 0:
            raise ValueError("refresh_s must be > 0")
        if poll_s <= 0:
            raise ValueError("poll_s must be > 0")
        self._client = client
        self._cache_dir = Path(cache_dir)
        self._camera_ids: tuple[str, ...] = tuple(dict.fromkeys(camera_ids))
        self._known = frozenset(self._camera_ids)
        self._on_config = on_config
        self._clock = clock
        self._refresh_s = refresh_s
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._refresh_lock = threading.RLock()
        self._current: dict[str, CameraConfig] = {}
        self._canonical: dict[str, str] = {}
        self._origin: dict[str, ConfigOrigin] = {}
        self._written: dict[str, str] = {}
        self._failures: dict[str, str] = {}
        self._pending: set[str] = set()
        self._pending_all = False
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ queries

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """The local cameras, in configuration order."""
        return self._camera_ids

    @property
    def running(self) -> bool:
        """True while the background refresh thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def cached(self, camera_id: str) -> CameraConfig | None:
        """The current (last good) configuration of ``camera_id``, or None."""
        with self._lock:
            return self._current.get(camera_id)

    def origin(self, camera_id: str) -> ConfigOrigin | None:
        """Where the current configuration of ``camera_id`` came from, or None."""
        with self._lock:
            return self._origin.get(camera_id)

    def cache_path(self, camera_id: str) -> Path:
        """``<cache_dir>/config_<camera_id>.json`` (the id percent-encoded where needed)."""
        return self._cache_dir / f"config_{safe_file_component(camera_id)}.json"

    def read_cache(self, camera_id: str) -> CameraConfig | None:
        """Read the cached configuration of ``camera_id``; None when missing or unusable."""
        path = self.cache_path(camera_id)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            _log.warning("cannot read the config cache %s: %s", path, exc)
            return None
        try:
            config = CameraConfig.model_validate_json(data)
        except ValueError as exc:
            _log.warning("ignoring the invalid config cache %s: %s", path, _first(exc))
            return None
        if config.camera.id != camera_id:
            _log.warning(
                "ignoring the config cache %s: it belongs to camera %r", path, config.camera.id
            )
            return None
        return config

    # ------------------------------------------------------------------ loading

    def load_initial(self) -> dict[str, CameraConfig]:
        """Load every camera's configuration: backend first, then the cache file.

        ``on_config`` is not called: the returned configurations are the starting point that
        later refreshes compare against.

        Returns:
            ``{camera_id: config}`` for the cameras that have a configuration.
        """
        with self._refresh_lock:
            found, reachable = self._fetch_all()
            result: dict[str, CameraConfig] = {}
            for camera_id in self._camera_ids:
                config = found.get(camera_id)
                origin: ConfigOrigin = "backend"
                if config is None:
                    config = self.read_cache(camera_id)
                    origin = "cache"
                if config is None:
                    _log.warning(
                        "camera %s: no configuration (backend %s, no usable cache at %s)",
                        camera_id,
                        "unreachable" if not reachable else "did not provide it",
                        self.cache_path(camera_id),
                        extra={"camera_id": camera_id},
                    )
                    continue
                self._apply(camera_id, config, origin)
                source = (
                    "the backend" if origin == "backend" else f"cache {self.cache_path(camera_id)}"
                )
                _log.info(
                    "camera %s: configuration version %d loaded from %s",
                    camera_id,
                    config.version,
                    source,
                    extra={"camera_id": camera_id},
                )
                result[camera_id] = config
            return result

    def refresh(self, camera_id: str | None = None) -> list[str]:
        """Re-read the configuration of one camera (None or an unknown id: of all cameras).

        Failures are logged and keep the current configuration.

        Returns:
            The ids of the cameras whose configuration changed (``on_config`` was called).
        """
        with self._refresh_lock:
            if camera_id is not None and camera_id in self._known:
                found: dict[str, CameraConfig] = {}
                try:
                    found[camera_id] = self._fetch_one(camera_id)
                except BackendError as exc:
                    self._note_failure(camera_id, exc)
            else:
                found, _ = self._fetch_all()
            changed: list[CameraConfig] = []
            for cam in self._camera_ids:
                config = found.get(cam)
                if config is not None and self._apply(cam, config, "backend"):
                    changed.append(config)
            for config in changed:
                self._notify(config)
            return [config.camera.id for config in changed]

    def request_refresh(self, camera_id: str | None = None) -> None:
        """Ask for an immediate refresh (``config.updated``); an unknown camera refreshes all.

        Returns at once while the background thread runs (it performs the refresh); without the
        thread the refresh runs synchronously in the caller's thread.
        """
        with self._lock:
            if camera_id is not None and camera_id in self._known:
                self._pending.add(camera_id)
            else:
                self._pending_all = True
        if self.running:
            self._wake.set()
            return
        self._process_requests()

    # ------------------------------------------------------------------ thread

    def start(self) -> None:
        """Start the background refresh thread (no-op while it is running)."""
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="config-sync", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the background thread, waiting at most ``timeout_s`` for it."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                _log.warning("config sync thread is still busy after %.1f s", timeout_s)
        self._thread = None

    def _run(self) -> None:
        next_due = self._clock.monotonic() + self._refresh_s
        while not self._stop.is_set():
            try:
                if self._has_requests():
                    if self._process_requests():
                        next_due = self._clock.monotonic() + self._refresh_s
                elif self._clock.monotonic() >= next_due:
                    next_due = self._clock.monotonic() + self._refresh_s
                    self.refresh(None)
            except Exception:
                _log.exception("configuration refresh failed")
            remaining = next_due - self._clock.monotonic()
            if self._stop.is_set() or self._has_requests() or remaining <= 0:
                continue
            self._wake.wait(min(self._poll_s, remaining))
            self._wake.clear()

    def _has_requests(self) -> bool:
        with self._lock:
            return self._pending_all or bool(self._pending)

    def _process_requests(self) -> bool:
        """Run the requested refreshes; True when all cameras were refreshed."""
        with self._lock:
            refresh_all = self._pending_all
            cameras = sorted(self._pending)
            self._pending_all = False
            self._pending.clear()
        if refresh_all:
            self.refresh(None)
            return True
        for camera_id in cameras:
            self.refresh(camera_id)
        return False

    # ------------------------------------------------------------------ internals

    def _fetch_all(self) -> tuple[dict[str, CameraConfig], bool]:
        """Fetch the configurations of all local cameras; returns them and reachability."""
        found: dict[str, CameraConfig] = {}
        try:
            configs = self._client.get_all_configs()
        except BackendError as exc:
            if exc.unreachable:
                self._note_failure(None, exc)
                return found, False
            _log.info("config list request failed (%s); requesting each camera instead", exc)
        else:
            for config in configs:
                camera_id = config.camera.id
                if camera_id in self._known and camera_id not in found:
                    found[camera_id] = config
                    self._note_success(camera_id)
        for camera_id in self._camera_ids:
            if camera_id in found:
                continue
            try:
                found[camera_id] = self._fetch_one(camera_id)
            except BackendError as exc:
                self._note_failure(camera_id, exc)
                if exc.unreachable:
                    return found, False
        return found, True

    def _fetch_one(self, camera_id: str) -> CameraConfig:
        """Fetch one camera's configuration; raises :class:`BackendError`."""
        config = self._client.get_config(camera_id)
        if config.camera.id != camera_id:
            raise BackendError(
                f"the backend answered the config request of camera {camera_id!r} with the "
                f"configuration of camera {config.camera.id!r}",
                endpoint=ENDPOINT_CONFIG,
                reason="invalid_response",
            )
        self._note_success(camera_id)
        return config

    def _apply(self, camera_id: str, config: CameraConfig, origin: ConfigOrigin) -> bool:
        """Make ``config`` current; returns True when it differs from the previous one."""
        canonical = config.model_dump_json(by_alias=True)
        with self._lock:
            changed = self._canonical.get(camera_id) != canonical
            self._current[camera_id] = config
            self._canonical[camera_id] = canonical
            self._origin[camera_id] = origin
            write = origin == "backend" and self._written.get(camera_id) != canonical
        if write:
            self._write_cache(camera_id, config, canonical)
        return changed

    def _write_cache(self, camera_id: str, config: CameraConfig, canonical: str) -> None:
        path = self.cache_path(camera_id)
        try:
            atomic_write_bytes(path, config.model_dump_json(by_alias=True, indent=2).encode())
        except OSError as exc:
            _log.warning("cannot write the config cache %s: %s", path, exc)
            return
        with self._lock:
            self._written[camera_id] = canonical

    def _notify(self, config: CameraConfig) -> None:
        camera_id = config.camera.id
        _log.info(
            "camera %s: configuration changed (version %d)",
            camera_id,
            config.version,
            extra={"camera_id": camera_id},
        )
        try:
            self._on_config(config)
        except Exception:
            _log.exception("on_config callback failed for camera %s", camera_id)

    def _note_failure(self, camera_id: str | None, exc: BackendError) -> None:
        """Log a failure at WARNING when it is new (or its cause changed), else at DEBUG."""
        target = camera_id if camera_id is not None else _ALL
        key = f"{exc.reason}:{exc.status_code}"
        with self._lock:
            repeated = self._failures.get(target) == key
            self._failures[target] = key
        subject = f"camera {camera_id}" if camera_id is not None else "all cameras"
        if repeated:
            _log.debug("configuration of %s still unavailable: %s", subject, exc)
        else:
            _log.warning("configuration of %s unavailable from the backend: %s", subject, exc)

    def _note_success(self, camera_id: str) -> None:
        with self._lock:
            recovered = self._failures.pop(camera_id, None) is not None
            recovered = self._failures.pop(_ALL, None) is not None or recovered
        if recovered:
            _log.info("configuration of camera %s is available from the backend again", camera_id)


def _first(exc: Exception) -> str:
    """First line of an exception message (pydantic errors are multi-line)."""
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__
