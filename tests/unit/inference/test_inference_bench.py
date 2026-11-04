"""Tests for the benchmark core with a fake timer, fake RSS probe and fake models."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from safety_ai.clock import FakeClock
from safety_ai.inference import backends, bench
from safety_ai.inference.bench import (
    BenchHost,
    BenchReport,
    BenchResult,
    host_info,
    load_bench,
    merge_results,
    percentile,
    render_markdown,
    write_bench,
)
from safety_ai.inference.registry import ModelRegistry, RegistryEntry

HOST = BenchHost(cpu="Test CPU", gpu=None, ram_gb=16.0)
T0 = datetime(2026, 11, 4, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    monkeypatch.setattr(backends, "_cuda_device_count", lambda: 0)
    monkeypatch.setattr(backends, "_mps_available", lambda: False)
    monkeypatch.setattr(backends, "_module_importable", lambda name: name == "onnxruntime")
    monkeypatch.setattr(backends, "_default_cfg_keys", lambda: frozenset({"quantize"}))


class FakeTimer:
    """Every model call takes ``step`` seconds (integer ticks, no float drift); reads are free."""

    def __init__(self, step: float) -> None:
        self.ticks = 0
        self.step = step

    def __call__(self) -> float:
        return self.ticks * self.step

    def advance(self) -> None:
        self.ticks += 1


class TimedModel:
    """A fake model whose predict advances the fake timer."""

    def __init__(self, timer: FakeTimer, names: dict[int, str], fake_result: Any) -> None:
        self.timer = timer
        self.names = names
        self.fake_result = fake_result
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def predict(self, source: Any, **kwargs: Any) -> list[Any]:
        self.calls.append((source, kwargs))
        self.timer.advance()
        return [self.fake_result([[0.1, 0.1, 0.2, 0.2, 0.9, 0]])]


@pytest.fixture
def registry(
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> ModelRegistry:
    entry = add_weights(
        tmp_registry,
        make_entry(
            name="vehicle_n", role="vehicle", variant="n", classes=["person"], path="models/v.pt"
        ),
    )
    onnx = tmp_registry.root / "models" / "v.onnx"
    onnx.write_bytes(b"onnx")
    tmp_registry.add_export("vehicle_n", "onnx", onnx, created_at=T0)
    add_weights(
        tmp_registry,
        make_entry(name="pose_n", role="pose", variant="n", classes=["person"], path="models/p.pt"),
    )
    add_weights(
        tmp_registry,
        make_entry(
            name="fall_lr",
            role="fall",
            format="joblib",
            classes=[],
            imgsz=0,
            variant=None,
            path="models/fall_lr.joblib",
        ),
    )
    assert entry.name == "vehicle_n"
    return tmp_registry


@pytest.fixture
def run(
    registry: ModelRegistry, fake_result: Callable[..., Any]
) -> Iterator[Callable[..., tuple[BenchReport, list[tuple[str, str]], list[TimedModel]]]]:
    def factory(
        models: list[str],
        backends_: list[str] | None,
        *,
        seconds: float = 0.3125,  # 10 steps of 1/32 s: exact in binary floating point
        step: float = 0.03125,
        rss: Callable[[], float] | None = None,
        images: list[Any] | None = None,
    ) -> tuple[BenchReport, list[tuple[str, str]], list[TimedModel]]:
        timer = FakeTimer(step)
        loads: list[tuple[str, str]] = []
        created: list[TimedModel] = []

        def loader(path: Path, task: str) -> TimedModel:
            loads.append((path.name, task))
            model = TimedModel(timer, {0: "person"}, fake_result)
            created.append(model)
            return model

        report = bench.bench(
            models,
            backends_,
            seconds,
            320,
            "cpu",
            registry,
            images if images is not None else [np.zeros((48, 64, 3), dtype=np.uint8)],
            isolate=False,
            loader=loader,
            timer=timer,
            rss_mb=rss or (lambda: 100.0),
            host=HOST,
            clock=FakeClock(T0),
        )
        return report, loads, created

    yield factory


# --------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------


def test_percentile_interpolates_linearly() -> None:
    values = [float(v) for v in range(1, 11)]
    assert percentile(values, 50) == pytest.approx(5.5)
    assert percentile(values, 95) == pytest.approx(9.55)
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 10.0
    assert percentile([3.0], 95) == 3.0
    assert percentile([5.0, 1.0], 50) == pytest.approx(3.0)


@pytest.mark.parametrize(("values", "q"), [([], 50), ([1.0], -1), ([1.0], 101)])
def test_percentile_rejects_bad_input(values: list[float], q: float) -> None:
    with pytest.raises(ValueError):
        percentile(values, q)


def _result(model: str, backend: str, device: str, fps: float) -> BenchResult:
    return BenchResult(
        model=model,
        backend=backend,
        device=device,
        imgsz=640,
        fps=fps,
        p50_ms=10.0,
        p95_ms=12.0,
        peak_rss_mb=500.0,
        n=10,
    )


def test_merge_replaces_the_same_combination_and_keeps_others() -> None:
    existing = [_result("b", "pt", "cpu", 1.0), _result("a", "pt", "cpu", 2.0)]
    new = [_result("a", "pt", "cpu", 3.0), _result("a", "onnx", "cpu", 4.0)]
    merged = merge_results(existing, new)
    assert [(r.model, r.backend, r.fps) for r in merged] == [
        ("a", "onnx", 4.0),
        ("a", "pt", 3.0),
        ("b", "pt", 1.0),
    ]


# --------------------------------------------------------------------------------------------
# bench()
# --------------------------------------------------------------------------------------------


def test_bench_measures_fps_latency_and_rss(
    run: Callable[..., Any], registry: ModelRegistry
) -> None:
    samples = iter([100.0, 110.0, 120.0, 150.0, 130.0] + [140.0] * 100)
    report, loads, created = run(["vehicle_n"], ["pt"], rss=lambda: next(samples))
    assert loads == [("v.pt", "detect")]
    [result] = report.results
    assert (result.model, result.backend, result.device, result.imgsz) == (
        "vehicle_n",
        "pt",
        "cpu",
        320,
    )
    assert result.n == 10
    assert result.fps == pytest.approx(32.0)
    assert result.p50_ms == pytest.approx(31.25)
    assert result.p95_ms == pytest.approx(31.25)
    assert result.peak_rss_mb == 150.0
    assert len(created[0].calls) == 5 + 10  # warm-up + measured
    kwargs = created[0].calls[0][1]
    assert kwargs == {"imgsz": 320, "device": "cpu", "conf": 0.25, "verbose": False}
    assert report.created_at == T0
    assert report.host == HOST
    stored = json.loads(registry.path.with_name("bench.json").read_text(encoding="utf-8"))
    assert stored["created_at"] == "2026-11-04T03:00:00Z"
    assert stored["host"] == {"cpu": "Test CPU", "gpu": None, "ram_gb": 16.0}
    assert stored["results"] == [
        {
            "model": "vehicle_n",
            "backend": "pt",
            "device": "cpu",
            "imgsz": 320,
            "fps": 32.0,
            "p50_ms": 31.25,
            "p95_ms": 31.25,
            "peak_rss_mb": 150.0,
            "n": 10,
        }
    ]
    assert "skipped" not in stored


def test_bench_runs_every_available_backend_and_resolves_aliases(run: Callable[..., Any]) -> None:
    report, loads, _ = run(["vehicle_latest", "pose_n"], None)
    assert [(r.model, r.backend) for r in report.results] == [
        ("vehicle_n", "pt"),
        ("vehicle_n", "onnx"),
        ("pose_n", "pt"),
    ]
    assert loads == [("v.pt", "detect"), ("v.onnx", "detect"), ("p.pt", "pose")]


def test_bench_skips_unavailable_backends_and_fall_models(
    run: Callable[..., Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.bench")
    report, loads, _ = run(["vehicle_n", "fall_lr"], ["openvino", "pt"])
    assert [(r.model, r.backend) for r in report.results] == [("vehicle_n", "pt")]
    assert [(s.model, s.backend) for s in report.skipped] == [
        ("vehicle_n", "openvino"),
        ("fall_lr", "joblib"),
    ]
    assert "not available" in report.skipped[0].reason
    assert loads == [("v.pt", "detect")]


def test_bench_merges_with_existing_results(
    run: Callable[..., Any], registry: ModelRegistry
) -> None:
    path = registry.path.with_name("bench.json")
    old = BenchReport(
        created_at=T0,
        host=HOST,
        results=[
            _result("vehicle_n", "pt", "cpu", 1.0),
            _result("vehicle_n", "pt", "cuda:0", 50.0),
        ],
    )
    write_bench(path, old)
    run(["vehicle_n"], ["pt"])
    merged = load_bench(path)
    assert merged is not None
    assert [(r.device, r.fps) for r in merged.results] == [("cpu", 32.0), ("cuda:0", 50.0)]


def test_bench_drops_stored_results_of_unregistered_models(
    run: Callable[..., Any], registry: ModelRegistry, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="safety_ai.inference.bench")
    path = registry.path.with_name("bench.json")
    old = BenchReport(
        created_at=T0,
        host=HOST,
        results=[_result("ppe_n_retired", "pt", "cpu", 9.0), _result("pose_n", "pt", "cpu", 7.0)],
    )
    write_bench(path, old)
    run(["vehicle_n"], ["pt"])
    merged = load_bench(path)
    assert merged is not None
    assert [(r.model, r.fps) for r in merged.results] == [("pose_n", 7.0), ("vehicle_n", 32.0)]
    assert "ppe_n_retired" in caplog.text


def test_write_bench_keeps_everything_without_a_model_filter(tmp_path: Path) -> None:
    path = tmp_path / "bench.json"
    write_bench(path, BenchReport(created_at=T0, host=HOST, results=[_result("a", "pt", "cpu", 1)]))
    merged = write_bench(
        path, BenchReport(created_at=T0, host=HOST, results=[_result("b", "pt", "cpu", 2)])
    )
    assert [r.model for r in merged.results] == ["a", "b"]


def test_bench_restores_torch_threads_changed_by_a_backend(
    registry: ModelRegistry, fake_result: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTorch:
        def __init__(self) -> None:
            self.threads = 6
            self.calls: list[int] = []

        def get_num_threads(self) -> int:
            return self.threads

        def set_num_threads(self, count: int) -> None:
            self.calls.append(count)
            self.threads = count

    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    timer = FakeTimer(0.05)

    class ThrottlingModel(TimedModel):
        def predict(self, source: Any, **kwargs: Any) -> list[Any]:
            torch.set_num_threads(2)  # what Ultralytics does after an ONNX/OpenVINO CPU setup
            return super().predict(source, **kwargs)

    bench.bench(
        ["vehicle_n"],
        ["pt", "onnx"],
        0.1,
        320,
        "cpu",
        registry,
        [np.zeros((8, 8, 3), dtype=np.uint8)],
        warmup=0,
        isolate=False,
        loader=lambda p, t: ThrottlingModel(timer, {0: "person"}, fake_result),
        timer=timer,
        rss_mb=lambda: 1.0,
        host=HOST,
        clock=FakeClock(T0),
    )
    assert torch.threads == 6
    assert torch.calls.count(6) == 2  # restored after each of the two combinations


@pytest.mark.parametrize("hook", ["loader", "timer", "rss_mb"])
def test_test_hooks_require_in_process_mode(registry: ModelRegistry, hook: str) -> None:
    hooks: dict[str, Any] = {"loader": lambda p, t: None, "timer": lambda: 0.0}
    hooks["rss_mb"] = lambda: 1.0
    with pytest.raises(ValueError, match="isolate=False"):
        bench.bench(
            ["vehicle_n"],
            ["pt"],
            1.0,
            320,
            "cpu",
            registry,
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            **{hook: hooks[hook]},
        )


# --------------------------------------------------------------------------------------------
# isolation (one spawned process per combination)
# --------------------------------------------------------------------------------------------


def _job(registry: ModelRegistry, model: str = "vehicle_n") -> Any:
    return bench._Job(
        registry_path=registry.path,
        model=model,
        backend="pt",
        device="cpu",
        task="detect",
        imgsz=320,
        seconds=0.1,
        warmup=0,
        half=False,
        frames=(np.zeros((8, 8, 3), dtype=np.uint8),),
    )


def test_a_failing_isolated_combination_raises_its_error_in_the_parent(
    registry: ModelRegistry,
) -> None:
    with pytest.raises(RuntimeError, match=r"RegistryError: model 'ghost_n' is not in"):
        bench._run_isolated(_job(registry, "ghost_n"), timeout_s=120.0)


class FakeProcess:
    """Stands in for a spawned process; records how it was stopped."""

    def __init__(self, *, alive_after_join: bool) -> None:
        self.alive = True
        self.alive_after_join = alive_after_join
        self.exitcode: int | None = None
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def join(self, timeout: float | None = None) -> None:
        self.events.append("join")
        if not self.alive_after_join:
            self.alive, self.exitcode = False, 0

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.events.append("terminate")
        self.alive, self.exitcode = False, -15


class FakeConnection:
    def __init__(self, *, ready: bool, message: Any = None) -> None:
        self.ready = ready
        self.message = message
        self.closed = False

    def poll(self, timeout: float) -> bool:
        return self.ready

    def recv(self) -> Any:
        if self.message is None:
            raise EOFError
        return self.message

    def close(self) -> None:
        self.closed = True


class FakeContext:
    def __init__(self, receiver: FakeConnection, process: FakeProcess) -> None:
        self.receiver = receiver
        self.process = process
        self.sender = FakeConnection(ready=False)

    def Pipe(self, duplex: bool = True) -> tuple[FakeConnection, FakeConnection]:
        assert duplex is False
        return self.receiver, self.sender

    def Process(self, **kwargs: Any) -> FakeProcess:
        assert kwargs["daemon"] is True
        assert kwargs["target"] is bench._child_main
        return self.process


@pytest.fixture
def fake_spawn(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeContext]:
    def install(receiver: FakeConnection, process: FakeProcess) -> FakeContext:
        context = FakeContext(receiver, process)
        monkeypatch.setattr(bench.multiprocessing, "get_context", lambda method: context)
        return context

    return install


def test_isolated_result_is_parsed(
    registry: ModelRegistry, fake_spawn: Callable[..., FakeContext]
) -> None:
    payload = _result("vehicle_n", "pt", "cpu", 12.5).model_dump(mode="json")
    context = fake_spawn(
        FakeConnection(ready=True, message=("ok", payload)), FakeProcess(alive_after_join=False)
    )
    result = bench._run_isolated(_job(registry), timeout_s=5.0)
    assert result == _result("vehicle_n", "pt", "cpu", 12.5)
    assert context.sender.closed and context.receiver.closed
    assert context.process.events == ["start", "join"]


def test_a_hanging_isolated_combination_times_out_and_is_terminated(
    registry: ModelRegistry, fake_spawn: Callable[..., FakeContext]
) -> None:
    context = fake_spawn(FakeConnection(ready=False), FakeProcess(alive_after_join=True))
    with pytest.raises(TimeoutError, match="no result within 5 s"):
        bench._run_isolated(_job(registry), timeout_s=5.0)
    assert context.process.events == ["start", "join", "terminate", "join"]
    assert context.receiver.closed


def test_an_isolated_process_dying_without_a_result_is_reported(
    registry: ModelRegistry, fake_spawn: Callable[..., FakeContext]
) -> None:
    fake_spawn(FakeConnection(ready=True), FakeProcess(alive_after_join=False))
    with pytest.raises(RuntimeError, match=r"exit code 0\): the benchmark process exited"):
        bench._run_isolated(_job(registry), timeout_s=5.0)


def test_child_main_reports_errors_and_survives_a_closed_parent(registry: ModelRegistry) -> None:
    class ClosedPipe:
        def __init__(self) -> None:
            self.sent: list[Any] = []
            self.closed = False

        def send(self, message: Any) -> None:
            self.sent.append(message)
            raise BrokenPipeError("the parent closed its end")

        def close(self) -> None:
            self.closed = True

    pipe = ClosedPipe()
    bench._child_main(pipe, _job(registry, "ghost_n"))  # type: ignore[arg-type]
    [(status, message)] = pipe.sent
    assert status == "error" and "ghost_n" in message
    assert pipe.closed


def test_bench_reads_image_paths(run: Callable[..., Any], tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    assert cv2.imwrite(str(image), np.full((40, 60, 3), 127, dtype=np.uint8))
    report, _, created = run(["vehicle_n"], ["pt"], images=[image, str(image)])
    assert report.results[0].n == 10
    frame = created[0].calls[0][0]
    assert frame.shape == (40, 60, 3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seconds": 0.0}, "seconds"),
        ({"images": []}, "image"),
        ({"images": [np.zeros((4, 4), dtype=np.uint8)]}, "HxWx3"),
    ],
)
def test_bench_rejects_bad_arguments(
    run: Callable[..., Any], kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run(["vehicle_n"], ["pt"], **kwargs)


def test_bench_rejects_unreadable_images_and_bad_imgsz(
    registry: ModelRegistry, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        bench.bench(["vehicle_n"], None, 1.0, 320, "cpu", registry, [tmp_path / "missing.jpg"])
    with pytest.raises(ValueError, match="imgsz"):
        bench.bench(["vehicle_n"], None, 1.0, 300, "cpu", registry, [tmp_path / "missing.jpg"])


def test_bench_with_an_explicit_output_path(
    registry: ModelRegistry, fake_result: Callable[..., Any], tmp_path: Path
) -> None:
    timer = FakeTimer(0.05)
    out = tmp_path / "custom" / "bench.json"
    report = bench.bench(
        ["vehicle_n"],
        ["pt"],
        0.1,
        320,
        "cpu",
        registry,
        [np.zeros((8, 8, 3), dtype=np.uint8)],
        bench_path=out,
        warmup=0,
        isolate=False,
        loader=lambda p, t: TimedModel(timer, {0: "person"}, fake_result),
        timer=timer,
        rss_mb=lambda: 1.0,
        host=HOST,
        clock=FakeClock(T0),
    )
    assert report.results[0].n == 2
    assert load_bench(out) is not None
    assert not registry.path.with_name("bench.json").exists()


# --------------------------------------------------------------------------------------------
# files, host and rendering
# --------------------------------------------------------------------------------------------


def test_load_bench_missing_and_invalid(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.bench")
    assert load_bench(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    assert load_bench(bad) is None
    bad.write_text(json.dumps({"created_at": "2026-11-04T03:00:00Z"}), encoding="utf-8")
    assert load_bench(bad) is None
    assert len(caplog.records) == 2
    folder = tmp_path / "folder.json"
    folder.mkdir()
    assert load_bench(folder) is None


def test_write_bench_ignores_an_invalid_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "bench.json"
    path.write_text("not json", encoding="utf-8")
    report = BenchReport(created_at=T0, host=HOST, results=[_result("m", "pt", "cpu", 5.0)])
    merged = write_bench(path, report)
    assert [r.model for r in merged.results] == ["m"]
    assert load_bench(path) == merged


def test_host_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "_cpu_name", lambda: "Some CPU")
    monkeypatch.setattr(bench, "_gpu_name", lambda: "Some GPU")
    info = host_info()
    assert (info.cpu, info.gpu) == ("Some CPU", "Some GPU")
    assert info.ram_gb > 0


def test_render_markdown() -> None:
    report = BenchReport(
        created_at=T0,
        host=HOST,
        results=[_result("vehicle_n", "pt", "cpu", 18.25)],
        skipped=[{"model": "vehicle_n", "backend": "tensorrt", "reason": "no CUDA"}],  # type: ignore[list-item]
    )
    text = render_markdown(report)
    lines = text.splitlines()
    assert lines[0] == "Host: Test CPU; GPU: none; RAM 16.0 GB; measured 2026-11-04T03:00:00Z"
    assert (
        lines[2] == "| Model | Backend | Device | imgsz | FPS | p50 ms | p95 ms | Peak RSS MB | n |"
    )
    assert lines[4] == "| vehicle_n | pt | cpu | 640 | 18.2 | 10.0 | 12.0 | 500 | 10 |"
    assert "- vehicle_n / tensorrt: no CUDA" in lines
    assert text.endswith("\n")
