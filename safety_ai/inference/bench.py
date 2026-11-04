"""Benchmark core behind ``safety-ai bench`` (spec 6.6, 8.6).

For every requested model x available backend on one device, :func:`bench` runs 5 warm-up
inferences, then cycles the sample images for the requested number of seconds, timing every
inference end to end (pre-processing, model, post-processing and the copy of the boxes to the
host). It reports FPS (inferences / measured wall time), latency p50/p95 in milliseconds and the
peak resident set size (sampled after every inference) in MB.

By default every model x backend combination runs in its own freshly spawned process, so each
row describes one runtime alone: Ultralytics changes process-wide state while it runs (after an
ONNX or OpenVINO CPU model is set up it limits torch to 2 threads, which would slow every later
PyTorch row), and allocator caches and earlier models would inflate the later rows' RSS. The
peak RSS is therefore that of a process holding Python, torch, Ultralytics, the runtime and the
one model, which is what a camera worker needs. ``isolate=False`` measures in the calling process
instead (the test hooks ``loader``, ``timer`` and ``rss_mb`` need it); torch's thread count is
restored after each combination there.

Results are merged into ``models/bench.json``::

    {"created_at": "...Z", "host": {"cpu": ..., "gpu": ..., "ram_gb": ...},
     "results": [{"model", "backend", "device", "imgsz", "fps", "p50_ms", "p95_ms",
                  "peak_rss_mb", "n"}, ...]}

A new result replaces the stored one with the same model, backend and device; results for other
combinations are kept, except those of models that are no longer registered, which are dropped.
:func:`safety_ai.inference.backends.choose_backend` reads this file to pick the fastest backend.
The returned :class:`BenchReport` holds this run's results plus the combinations that were
skipped and why; :func:`load_bench` reads the merged file and :func:`render_markdown` renders
either as a Markdown table.

Public API: :func:`bench`, :func:`load_bench`, :func:`merge_results`, :func:`write_bench`,
:func:`render_markdown`, :func:`percentile`, :func:`host_info`, :func:`default_bench_images`,
:class:`BenchReport`, :class:`BenchResult`, :class:`BenchHost`, :class:`BenchSkip`.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
import multiprocessing
import platform
import sys
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from safety_ai.capture.types import ImageArray
from safety_ai.clock import Clock, SystemClock, format_utc_seconds
from safety_ai.inference.backends import (
    ModelLoader,
    available_backends,
    device_class,
    load_model,
    precision_kwargs,
    resolve_device,
    tensor_to_list,
)
from safety_ai.inference.registry import ModelRegistry, UtcTimestamp, atomic_write_text

__all__ = [
    "CHILD_OVERHEAD_S",
    "WARMUP_RUNS",
    "BenchHost",
    "BenchReport",
    "BenchResult",
    "BenchSkip",
    "bench",
    "default_bench_images",
    "host_info",
    "load_bench",
    "merge_results",
    "percentile",
    "render_markdown",
    "write_bench",
]

_LOG = logging.getLogger(__name__)

WARMUP_RUNS: Final[int] = 5
CHILD_OVERHEAD_S: Final[float] = 600.0
"""Time an isolated combination may take on top of ``seconds`` (start-up, loading, warm-up)."""
_JOIN_TIMEOUT_S: Final[float] = 30.0
_MB: Final[float] = float(1 << 20)
_BENCH_CONF: Final[float] = 0.25


class BenchHost(BaseModel):
    """The machine a benchmark ran on."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    cpu: str
    gpu: str | None = None
    ram_gb: float = Field(ge=0.0)


class BenchResult(BaseModel):
    """One model x backend x device measurement."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str
    """Registry entry name (aliases are resolved before measuring)."""
    backend: str
    device: str
    imgsz: int = Field(ge=1)
    fps: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    peak_rss_mb: float = Field(ge=0.0)
    n: int = Field(ge=1)
    """Number of measured inferences (warm-up excluded)."""

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity used when merging: ``(model, backend, device)``."""
        return (self.model, self.backend, self.device)


class BenchSkip(BaseModel):
    """A requested combination that was not measured, with the reason."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str
    backend: str
    reason: str


class BenchReport(BaseModel):
    """A benchmark run (or the merged ``bench.json`` when read with :func:`load_bench`)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    created_at: UtcTimestamp
    host: BenchHost
    results: list[BenchResult] = Field(default_factory=list)
    skipped: list[BenchSkip] = Field(default_factory=list)


def percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile (0..100) with linear interpolation between ranks.

    Raises:
        ValueError: For an empty sequence or ``q`` outside 0..100.
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within 0..100")
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * q / 100.0
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _cpu_name() -> str:
    """Human-readable CPU model (Ultralytics' probe, else the platform string)."""
    try:
        from ultralytics.utils.cpu import CPUInfo  # noqa: PLC0415  (lazy heavy import)

        return str(CPUInfo.name())
    except Exception:  # any probe failure falls back to the generic platform string
        return platform.processor() or platform.machine() or "unknown CPU"


def _gpu_name() -> str | None:
    """Name of CUDA device 0, or None without CUDA."""
    try:
        import torch  # noqa: PLC0415  (lazy heavy import)
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def host_info() -> BenchHost:
    """Describe this machine: CPU model, CUDA GPU (or None) and total RAM in GB."""
    ram_gb = psutil.virtual_memory().total / float(1 << 30)
    return BenchHost(cpu=_cpu_name(), gpu=_gpu_name(), ram_gb=round(ram_gb, 1))


def _process_rss_mb() -> float:
    """Resident set size of this process in MB."""
    return float(psutil.Process().memory_info().rss) / _MB


def default_bench_images() -> list[Path]:
    """The sample images shipped with Ultralytics (``bus.jpg``, ``zidane.jpg``)."""
    from ultralytics.utils import ASSETS  # noqa: PLC0415  (lazy heavy import)

    return [Path(ASSETS) / "bus.jpg", Path(ASSETS) / "zidane.jpg"]


def _as_image(image: Path | str | ImageArray) -> ImageArray:
    """Read an image path with OpenCV, or validate an in-memory BGR image."""
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if loaded is None:
            raise ValueError(f"cannot read benchmark image {image}")
        return np.ascontiguousarray(loaded, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"benchmark images must be HxWx3 BGR arrays, got {image.shape}")
    return image


def load_bench(path: Path) -> BenchReport | None:
    """Read ``bench.json``; None when it is missing or invalid (invalid files log a WARNING)."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except OSError as exc:
        _LOG.warning("cannot read %s: %s", path, exc)
        return None
    try:
        return BenchReport.model_validate(json.loads(text))
    except (ValueError, ValidationError) as exc:
        _LOG.warning("ignoring invalid benchmark file %s: %s", path, exc)
        return None


def merge_results(existing: Sequence[BenchResult], new: Sequence[BenchResult]) -> list[BenchResult]:
    """Replace stored results that share ``(model, backend, device)`` with new ones.

    The merged list is sorted by model, backend and device for stable files.
    """
    merged = {r.key: r for r in existing}
    merged.update({r.key: r for r in new})
    return [merged[k] for k in sorted(merged)]


def write_bench(
    path: Path, report: BenchReport, *, keep_models: Collection[str] | None = None
) -> BenchReport:
    """Merge ``report`` into the file at ``path`` atomically and return what was written.

    With ``keep_models`` (the registry's entry names), stored results of any other model are
    dropped, so the file never keeps numbers of models that no longer exist.
    """
    previous = load_bench(path)
    stored = previous.results if previous else []
    if keep_models is not None:
        dropped = sorted({r.model for r in stored if r.model not in keep_models})
        if dropped:
            _LOG.info("dropping benchmark results of unregistered models: %s", ", ".join(dropped))
        stored = [r for r in stored if r.model in keep_models]
    merged = BenchReport(
        created_at=report.created_at,
        host=report.host,
        results=merge_results(stored, report.results),
    )
    payload = merged.model_dump(mode="json", exclude={"skipped"})
    atomic_write_text(Path(path), json.dumps(payload, indent=2) + "\n")
    return merged


def _consume(results: Any) -> None:
    """Copy the boxes of every result to the host (forces GPU work to finish in the timing)."""
    for result in results or ():
        boxes = getattr(result, "boxes", None)
        data = getattr(boxes, "data", None) if boxes is not None else None
        if data is not None:
            tensor_to_list(data)


def _measure(
    model: Any,
    frames: Sequence[ImageArray],
    kwargs: dict[str, Any],
    *,
    seconds: float,
    warmup: int,
    timer: Callable[[], float],
    rss_mb: Callable[[], float],
) -> tuple[list[float], float, float]:
    """Warm up, then time inferences for ``seconds``; returns latencies (ms), wall s, peak MB."""
    peak = rss_mb()
    for index in range(warmup):
        _consume(model.predict(frames[index % len(frames)], **kwargs))
        peak = max(peak, rss_mb())
    latencies: list[float] = []
    start = timer()
    wall = 0.0
    index = 0
    while not latencies or wall < seconds:
        before = timer()
        _consume(model.predict(frames[index % len(frames)], **kwargs))
        after = timer()
        latencies.append(max(0.0, (after - before) * 1000.0))
        peak = max(peak, rss_mb())
        wall = after - start
        index += 1
    return latencies, wall, peak


def _release_device_memory(device: str) -> None:
    """Free cached GPU memory between models so each run starts from the same state."""
    gc.collect()
    if device_class(device) == "cuda":
        import torch  # noqa: PLC0415  (only reached on CUDA, where torch is installed)

        torch.cuda.empty_cache()


def _torch_threads() -> int | None:
    """torch's intra-op thread count, when torch is already imported (never imports it)."""
    torch = sys.modules.get("torch")
    return int(torch.get_num_threads()) if torch is not None else None


def _restore_torch_threads(count: int | None) -> None:
    """Undo a thread-count change made while a combination ran (see the module docstring)."""
    torch = sys.modules.get("torch")
    if count is not None and torch is not None and int(torch.get_num_threads()) != count:
        torch.set_num_threads(count)


@dataclass(frozen=True, slots=True)
class _Job:
    """One model x backend measurement; picklable, so it can run in a spawned process."""

    registry_path: Path
    model: str
    backend: str
    device: str
    task: str
    imgsz: int
    seconds: float
    warmup: int
    half: bool
    frames: tuple[ImageArray, ...]


def _measure_job(
    job: _Job,
    *,
    loader: ModelLoader | None,
    timer: Callable[[], float],
    rss_mb: Callable[[], float],
) -> BenchResult:
    """Load the job's model on its backend and measure it in this process."""
    registry = ModelRegistry(job.registry_path)
    entry = registry.resolve(job.model)
    loaded = load_model(
        entry, registry, backend=job.backend, device=job.device, task=job.task, loader=loader
    )
    kwargs: dict[str, Any] = {
        "imgsz": job.imgsz,
        "device": job.device,
        "conf": _BENCH_CONF,
        "verbose": False,
        **precision_kwargs(job.device, job.backend, job.half),
    }
    latencies, wall, peak = _measure(
        loaded.model,
        job.frames,
        kwargs,
        seconds=job.seconds,
        warmup=job.warmup,
        timer=timer,
        rss_mb=rss_mb,
    )
    del loaded
    _release_device_memory(job.device)
    return BenchResult(
        model=job.model,
        backend=job.backend,
        device=job.device,
        imgsz=job.imgsz,
        fps=round(len(latencies) / wall, 2) if wall > 0 else 0.0,
        p50_ms=round(percentile(latencies, 50), 2),
        p95_ms=round(percentile(latencies, 95), 2),
        peak_rss_mb=round(peak, 1),
        n=len(latencies),
    )


def _child_main(conn: Connection, job: _Job) -> None:
    """Entry point of a spawned benchmark process: measure ``job``, send the result or error."""
    try:
        try:
            result = _measure_job(job, loader=None, timer=time.perf_counter, rss_mb=_process_rss_mb)
            message: tuple[str, Any] = ("ok", result.model_dump(mode="json"))
        except Exception as exc:  # reported to the parent, which raises it there
            message = ("error", f"{type(exc).__name__}: {exc}")
        with contextlib.suppress(OSError):  # the parent stopped waiting and closed its end
            conn.send(message)
    finally:
        conn.close()


def _stop_process(process: Any) -> None:
    """Join a benchmark process, terminating it when it does not exit by itself."""
    process.join(timeout=_JOIN_TIMEOUT_S)
    if process.is_alive():
        process.terminate()
        process.join(timeout=_JOIN_TIMEOUT_S)


def _run_isolated(job: _Job, timeout_s: float) -> BenchResult:
    """Measure ``job`` in a freshly spawned process (spawn context on every platform).

    Raises:
        TimeoutError: When the process sends no result within ``timeout_s``.
        RuntimeError: When the measurement failed or the process died without a result.
    """
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_main,
        args=(sender, job),
        name=f"bench-{job.model}-{job.backend}",
        daemon=True,
    )
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout_s):
            raise TimeoutError(
                f"bench of {job.model}/{job.backend} sent no result within {timeout_s:.0f} s"
            )
        try:
            status, payload = receiver.recv()
        except EOFError:
            status, payload = "error", "the benchmark process exited without a result"
    finally:
        receiver.close()
        _stop_process(process)
    if status != "ok":
        raise RuntimeError(
            f"bench of {job.model}/{job.backend} on {job.device} failed "
            f"(exit code {process.exitcode}): {payload}"
        )
    return BenchResult.model_validate(payload)


def bench(  # noqa: PLR0917  (positional order fixed by the bench API contract)
    models: Sequence[str],
    backends: Sequence[str] | None,
    seconds: float,
    imgsz: int,
    device: str,
    registry: ModelRegistry,
    images: Sequence[Path | str | ImageArray],
    *,
    bench_path: Path | None = None,
    half: bool = True,
    warmup: int = WARMUP_RUNS,
    isolate: bool = True,
    loader: ModelLoader | None = None,
    timer: Callable[[], float] | None = None,
    rss_mb: Callable[[], float] | None = None,
    host: BenchHost | None = None,
    clock: Clock | None = None,
) -> BenchReport:
    """Measure every model x backend and merge the results into ``bench_path``.

    Args:
        models: Registry names or aliases; each is resolved to its entry name.
        backends: Backends to measure (None: every backend available for the model on the
            device). Requested but unavailable backends are reported in ``skipped``.
        seconds: Measured duration per combination (after the warm-up).
        imgsz: Inference size (a multiple of 32).
        device: ``auto``, ``cpu``, ``cuda:N`` or ``mps``; resolved once.
        registry: Where the models are registered.
        images: Sample image paths or BGR arrays, cycled in order.
        bench_path: Output file (default: ``bench.json`` next to the registry).
        half: FP16 on CUDA, as at runtime.
        warmup: Untimed inferences before measuring.
        isolate: Run each combination in its own spawned process (see the module docstring).
        loader: Model loader (test hook; needs ``isolate=False``).
        timer: High-resolution timer in seconds (test hook; needs ``isolate=False``).
        rss_mb: Resident-set-size probe in MB (test hook; needs ``isolate=False``).
        host: Host description (default: :func:`host_info`).
        clock: Source of ``created_at``.

    Returns:
        This run's results and skips; the file holds the merged results.

    Raises:
        ValueError: For non-positive ``seconds``, a bad ``imgsz``, no images, or test hooks
            combined with ``isolate``.
        TimeoutError: When an isolated combination hangs past ``seconds + CHILD_OVERHEAD_S``.
        RuntimeError: When an isolated combination fails (the message carries its error).
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    if not images:
        raise ValueError("at least one benchmark image is needed")
    if isolate and (loader is not None or timer is not None or rss_mb is not None):
        raise ValueError("loader, timer and rss_mb are in-process test hooks; pass isolate=False")
    frames = tuple(_as_image(image) for image in images)
    resolved = resolve_device(device)
    results: list[BenchResult] = []
    skipped: list[BenchSkip] = []
    for name in models:
        entry = registry.resolve(name)
        if entry.role == "fall":
            skipped.append(
                BenchSkip(model=entry.name, backend=entry.format, reason="not an image model")
            )
            continue
        available = available_backends(entry, resolved, root=registry.root)
        wanted = list(backends) if backends is not None else available
        for backend in wanted:
            if backend not in available:
                reason = f"not available on {resolved} (available: {', '.join(available)})"
                skipped.append(BenchSkip(model=entry.name, backend=backend, reason=reason))
                _LOG.warning("skipping %s/%s: %s", entry.name, backend, reason)
                continue
            job = _Job(
                registry_path=registry.path.absolute(),
                model=entry.name,
                backend=backend,
                device=resolved,
                task="pose" if entry.role == "pose" else "detect",
                imgsz=imgsz,
                seconds=seconds,
                warmup=warmup,
                half=half,
                frames=frames,
            )
            if isolate:
                result = _run_isolated(job, seconds + CHILD_OVERHEAD_S)
            else:
                threads = _torch_threads()
                try:
                    result = _measure_job(
                        job,
                        loader=loader,
                        timer=timer or time.perf_counter,
                        rss_mb=rss_mb or _process_rss_mb,
                    )
                finally:
                    _restore_torch_threads(threads)
            _LOG.info(
                "bench %s/%s on %s: %.1f FPS, p50 %.1f ms, p95 %.1f ms, peak RSS %.0f MB",
                result.model,
                result.backend,
                result.device,
                result.fps,
                result.p50_ms,
                result.p95_ms,
                result.peak_rss_mb,
            )
            results.append(result)
    report = BenchReport(
        created_at=(clock or SystemClock()).now_utc().replace(microsecond=0),
        host=host if host is not None else host_info(),
        results=results,
        skipped=skipped,
    )
    write_bench(
        bench_path if bench_path is not None else registry.path.with_name("bench.json"),
        report,
        keep_models=set(registry.names()),
    )
    return report


def render_markdown(report: BenchReport) -> str:
    """Render a report as a Markdown table (with a host line and the skipped combinations)."""
    host = report.host
    lines = [
        f"Host: {host.cpu}; GPU: {host.gpu or 'none'}; RAM {host.ram_gb:.1f} GB; "
        f"measured {format_utc_seconds(report.created_at)}",
        "",
        "| Model | Backend | Device | imgsz | FPS | p50 ms | p95 ms | Peak RSS MB | n |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r.model} | {r.backend} | {r.device} | {r.imgsz} | {r.fps:.1f} | {r.p50_ms:.1f} "
        f"| {r.p95_ms:.1f} | {r.peak_rss_mb:.0f} | {r.n} |"
        for r in report.results
    )
    if report.skipped:
        lines.append("")
        lines.append("Skipped:")
        lines.extend(f"- {s.model} / {s.backend}: {s.reason}" for s in report.skipped)
    return "\n".join(lines) + "\n"
