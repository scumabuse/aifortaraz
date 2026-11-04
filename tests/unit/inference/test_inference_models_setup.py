"""Tests for model download and registration (httpx.MockTransport, fake loader; no network)."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest

from safety_ai.clock import FakeClock
from safety_ai.inference import models_setup
from safety_ai.inference.models_setup import (
    SH17_METRICS_SOURCE,
    SetupOutcome,
    check_models,
    download_file,
    ensure_weight,
    main,
    render_check_table,
    required_names,
    setup_models,
)
from safety_ai.inference.registry import ModelRegistry, RegistryError

T0 = datetime(2026, 11, 4, 3, 0, 0, tzinfo=timezone.utc)
ASSETS = "/ultralytics/assets/releases/download/v8.4.0/"
SH17 = "/ahmadmughees/SH17dataset/releases/download/v1/"


def _payload(name: str) -> bytes:
    """Distinct content (>= the 100 kB minimum) for each remote file."""
    return name.encode() + b"\x00" * 120_000


class Server:
    """A MockTransport 'GitHub' serving weights; paths in ``missing`` answer 404."""

    def __init__(self) -> None:
        self.missing: set[str] = set()
        self.requests: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        name = request.url.path.rsplit("/", 1)[-1]
        if request.url.path in self.missing or name in self.missing:
            return httpx.Response(404)
        return httpx.Response(200, content=_payload(name))

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)


class FakeModel:
    def __init__(self, names: dict[int, str], imgsz: Any = 640) -> None:
        self.names = names
        self.overrides = {"imgsz": imgsz}


@pytest.fixture
def server() -> Server:
    return Server()


@pytest.fixture
def client(server: Server) -> Iterator[httpx.Client]:
    http = server.client()
    yield http
    http.close()


@pytest.fixture
def loader(
    coco_names: dict[int, str], sh17_names: dict[int, str]
) -> Callable[[Path, str], FakeModel]:
    def load(path: Path, task: str) -> FakeModel:
        if path.name.startswith("ppe_"):
            assert task == "detect"
            return FakeModel(sh17_names)
        if "pose" in path.name:
            assert task == "pose"
            return FakeModel({0: "person"})
        assert task == "detect"
        return FakeModel(coco_names)

    return load


@pytest.fixture
def setup(
    tmp_registry: ModelRegistry,
    client: httpx.Client,
    loader: Callable[[Path, str], FakeModel],
) -> Callable[..., list[SetupOutcome]]:
    sleeps: list[float] = []

    def run(**overrides: Any) -> list[SetupOutcome]:
        options: dict[str, Any] = {
            "client": client,
            "loader": loader,
            "clock": FakeClock(T0),
            "sleep": sleeps.append,
        }
        options.update(overrides)
        return setup_models(tmp_registry, **options)

    run.sleeps = sleeps  # type: ignore[attr-defined]
    return run


def _statuses(outcomes: list[SetupOutcome]) -> dict[str, str]:
    return {o.name: o.status for o in outcomes}


# --------------------------------------------------------------------------------------------
# setup_models
# --------------------------------------------------------------------------------------------


def test_setup_registers_every_model(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry, server: Server
) -> None:
    outcomes = setup()
    assert _statuses(outcomes) == dict.fromkeys(
        ["vehicle_n", "vehicle_s", "pose_n", "ppe_n_sh17", "ppe_s_sh17"], "registered"
    )
    assert server.requests == [
        ASSETS + "yolo26n.pt",
        ASSETS + "yolo26s.pt",
        ASSETS + "yolo26n-pose.pt",
        SH17 + "yolo8n.pt",
        SH17 + "yolo8s.pt",
    ]
    models = tmp_registry.root / "models"
    assert sorted(p.name for p in models.iterdir()) == [
        "ppe_n_sh17.pt",
        "ppe_s_sh17.pt",
        "registry.json",
        "yolo26n-pose.pt",
        "yolo26n.pt",
        "yolo26s.pt",
    ]
    vehicle = tmp_registry.resolve("vehicle_n")
    assert (vehicle.role, vehicle.path, vehicle.format, vehicle.family, vehicle.variant) == (
        "vehicle",
        "models/yolo26n.pt",
        "pt",
        "yolo26",
        "n",
    )
    assert vehicle.classes == ["person", None, "vehicle", None, None, "vehicle", None, "vehicle"]
    assert vehicle.sha256 == hashlib.sha256(_payload("yolo26n.pt")).hexdigest()
    assert vehicle.created_at == T0
    assert vehicle.imgsz == 640
    assert "source" in vehicle.metrics
    assert tmp_registry.verify(vehicle)
    assert tmp_registry.resolve("vehicle_s").path == "models/yolo26s.pt"
    pose = tmp_registry.resolve("pose_n")
    assert (pose.role, pose.classes, pose.path) == ("pose", ["person"], "models/yolo26n-pose.pt")


def test_sh17_entries_map_classes_and_carry_reported_metrics(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    small = tmp_registry.resolve("ppe_s_latest")
    nano = tmp_registry.resolve("ppe_n_latest")
    assert (small.name, nano.name) == ("ppe_s_sh17", "ppe_n_sh17")
    assert (small.family, small.variant, small.role, small.path) == (
        "yolov8-sh17",
        "s",
        "ppe",
        "models/ppe_s_sh17.pt",
    )
    assert small.classes[:8] == [
        "person",
        "ear",
        "earmuffs",
        "face",
        "face_shield",
        "mask",
        "foot",
        "tools",
    ]
    assert small.classes[13] is None and small.classes[15] is None
    assert small.classes[16] == "vest"
    assert nano.metrics == {
        "source": SH17_METRICS_SOURCE,
        "units": "percent",
        "precision": 67.5,
        "recall": 53.6,
        "map50": 58.0,
        "map50_95": 36.6,
    }
    assert (small.metrics["precision"], small.metrics["recall"]) == (81.5, 55.7)
    assert (small.metrics["map50"], small.metrics["map50_95"]) == (63.7, 41.7)
    assert "SH17" in small.trained_on and "CC BY-NC-SA 4.0" in small.trained_on


def test_setup_is_idempotent(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry, server: Server
) -> None:
    setup()
    before = tmp_registry.path.read_bytes()
    server.requests.clear()
    outcomes = setup(clock=FakeClock(T0 + timedelta(days=3)))
    assert set(_statuses(outcomes).values()) == {"unchanged"}
    assert server.requests == []
    assert tmp_registry.path.read_bytes() == before


def test_measured_metrics_and_exports_survive_a_rerun(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    entry = tmp_registry.resolve("ppe_s_sh17")
    measured = entry.model_copy(
        update={"metrics": {"source": "measured: reports/e.md", "map50": 1}}
    )
    tmp_registry.add(measured)
    onnx = tmp_registry.root / "models" / "ppe_s_sh17.onnx"
    onnx.write_bytes(b"onnx")
    tmp_registry.add_export("ppe_s_sh17", "onnx", onnx, created_at=T0)
    setup()
    kept = tmp_registry.resolve("ppe_s_sh17")
    assert kept.metrics["source"] == "measured: reports/e.md"
    assert "onnx" in kept.exports


def test_corrupted_weights_are_downloaded_again(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry, server: Server
) -> None:
    setup()
    weights = tmp_registry.root / "models" / "yolo26n.pt"
    weights.write_bytes(b"corrupted")
    server.requests.clear()
    outcomes = setup()
    assert server.requests == [ASSETS + "yolo26n.pt"]
    assert _statuses(outcomes)["vehicle_n"] == "unchanged"
    assert weights.read_bytes() == _payload("yolo26n.pt")


def test_existing_files_without_records_are_adopted(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry, server: Server
) -> None:
    models = tmp_registry.root / "models"
    for name in ("yolo26n.pt", "yolo26s.pt", "yolo26n-pose.pt", "ppe_n_sh17.pt", "ppe_s_sh17.pt"):
        (models / name).write_bytes(b"local " + name.encode())
    outcomes = setup()
    assert server.requests == []
    assert set(_statuses(outcomes).values()) == {"registered"}
    assert tmp_registry.verify(tmp_registry.resolve("ppe_n_sh17"))


def test_yolo11_fallback_after_three_failed_downloads(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry, server: Server
) -> None:
    server.missing = {"yolo26n.pt", "yolo26s.pt", "yolo26n-pose.pt"}
    outcomes = setup(skip_sh17=True)
    assert set(_statuses(outcomes).values()) == {"registered"}
    assert server.requests.count(ASSETS + "yolo26n.pt") == 3
    assert setup.sleeps == [1.0, 2.0] * 3  # type: ignore[attr-defined]
    vehicle = tmp_registry.resolve("vehicle_n")
    assert (vehicle.family, vehicle.path) == ("yolo11", "models/yolo11n.pt")
    assert tmp_registry.resolve("pose_n").path == "models/yolo11n-pose.pt"


def test_yolo11_fallback_when_yolo26_does_not_load(
    setup: Callable[..., list[SetupOutcome]],
    tmp_registry: ModelRegistry,
    coco_names: dict[int, str],
) -> None:
    def old_ultralytics(path: Path, task: str) -> FakeModel:
        if path.name.startswith("yolo26"):
            raise ModuleNotFoundError("No module named 'ultralytics.nn.modules.block.C3k2'")
        return FakeModel({0: "person"} if task == "pose" else coco_names, imgsz=None)

    outcomes = setup(skip_sh17=True, loader=old_ultralytics)
    assert set(_statuses(outcomes).values()) == {"registered"}
    vehicle = tmp_registry.resolve("vehicle_n")
    assert (vehicle.family, vehicle.path, vehicle.imgsz) == ("yolo11", "models/yolo11n.pt", 640)


def test_models_failing_everything_are_reported(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry, server: Server
) -> None:
    server.missing = {
        "yolo26n.pt",
        "yolo11n.pt",
        "yolo26s.pt",
        "yolo11s.pt",
        "yolo26n-pose.pt",
        "yolo11n-pose.pt",
        "yolo8n.pt",
        "yolo8s.pt",
    }
    outcomes = setup()
    assert set(_statuses(outcomes).values()) == {"failed"}
    vehicle = next(o for o in outcomes if o.name == "vehicle_n")
    assert "yolo26n.pt: download failed 3 times" in vehicle.detail
    assert "yolo11n.pt: download failed 3 times" in vehicle.detail
    sh17 = next(o for o in outcomes if o.name == "ppe_s_sh17")
    assert "yolo8s.pt failed 3 times" in sh17.detail
    assert tmp_registry.entries() == ()


def test_coco_models_without_vehicle_classes_are_rejected(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    outcomes = setup(skip_sh17=True, loader=lambda path, task: FakeModel({0: "person", 1: "cat"}))
    statuses = _statuses(outcomes)
    assert statuses["vehicle_n"] == "failed"
    assert statuses["pose_n"] == "registered"
    detail = next(o.detail for o in outcomes if o.name == "vehicle_n")
    assert "lack" in detail


def test_sh17_names_must_match_the_taxonomy(
    setup: Callable[..., list[SetupOutcome]],
    tmp_registry: ModelRegistry,
    coco_names: dict[int, str],
) -> None:
    def renamed(path: Path, task: str) -> FakeModel:
        if path.name.startswith("ppe_"):
            return FakeModel({0: "person", 1: "earmuffs"})
        return FakeModel({0: "person"} if task == "pose" else coco_names)

    outcomes = setup(loader=renamed)
    failed = [o for o in outcomes if o.status == "failed"]
    assert [o.name for o in failed] == ["ppe_n_sh17", "ppe_s_sh17"]
    assert "differ from the SH17 names in datasets/taxonomy.yaml" in failed[0].detail
    assert tmp_registry.get("ppe_n_sh17") is None


def test_unloadable_sh17_weights_are_reported(
    setup: Callable[..., list[SetupOutcome]], coco_names: dict[int, str]
) -> None:
    def broken(path: Path, task: str) -> FakeModel:
        if path.name.startswith("ppe_"):
            raise RuntimeError("PytorchStreamReader failed reading zip archive")
        return FakeModel({0: "person"} if task == "pose" else coco_names)

    outcomes = setup(loader=broken)
    assert [o.status for o in outcomes if o.name.startswith("ppe_")] == ["failed", "failed"]
    assert "does not load" in next(o.detail for o in outcomes if o.name == "ppe_n_sh17")


def test_skip_sh17(setup: Callable[..., list[SetupOutcome]], server: Server) -> None:
    outcomes = setup(skip_sh17=True)
    assert [o.name for o in outcomes] == ["vehicle_n", "vehicle_s", "pose_n"]
    assert not any("SH17" in path for path in server.requests)


def test_setup_creates_its_own_client_when_none_is_given(
    tmp_registry: ModelRegistry,
    server: Server,
    loader: Callable[[Path, str], FakeModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx.Client] = []

    def new_client(timeout: httpx.Timeout) -> httpx.Client:
        http = server.client()
        created.append(http)
        return http

    monkeypatch.setattr(models_setup, "_new_client", new_client)
    outcomes = setup_models(tmp_registry, skip_sh17=True, loader=loader, clock=FakeClock(T0))
    assert len(outcomes) == 3
    assert len(created) == 1
    assert created[0].is_closed


# --------------------------------------------------------------------------------------------
# download_file / ensure_weight
# --------------------------------------------------------------------------------------------


def _client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_writes_atomically(tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "w.pt"
    with _client_for(lambda r: httpx.Response(200, content=b"x" * 200_000)) as http:
        assert download_file("https://example.test/w.pt", dest, client=http)
    assert dest.read_bytes() == b"x" * 200_000
    assert sorted(p.name for p in dest.parent.iterdir()) == ["w.pt"]


def test_download_retries_then_succeeds(tmp_path: Path) -> None:
    answers = iter([httpx.Response(503), httpx.Response(200, content=b"y" * 200_000)])
    sleeps: list[float] = []
    with _client_for(lambda r: next(answers)) as http:
        assert download_file(
            "https://example.test/w.pt", tmp_path / "w.pt", client=http, sleep=sleeps.append
        )
    assert sleeps == [1.0]


def test_download_rejects_truncated_and_tiny_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.models_setup")
    dest = tmp_path / "w.pt"
    dest.write_bytes(b"previous")
    truncated = httpx.Response(200, content=b"z" * 200_000, headers={"content-length": "300000"})
    with _client_for(lambda r: truncated) as http:
        assert not download_file(
            "https://example.test/w.pt", dest, client=http, sleep=lambda s: None
        )
    assert "truncated" in caplog.text
    with _client_for(lambda r: httpx.Response(200, content=b"tiny")) as http:
        assert not download_file(
            "https://example.test/w.pt", dest, client=http, sleep=lambda s: None
        )
    assert "only 4 bytes" in caplog.text
    assert dest.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.pt"]


def test_download_enforces_the_overall_deadline(tmp_path: Path) -> None:
    ticks = iter(range(0, 10_000, 500))
    with _client_for(lambda r: httpx.Response(200, content=b"x" * 200_000)) as http:
        ok = download_file(
            "https://example.test/w.pt",
            tmp_path / "w.pt",
            client=http,
            attempts=1,
            deadline_s=100.0,
            monotonic=lambda: float(next(ticks)),
        )
    assert not ok
    assert list(tmp_path.iterdir()) == []


def test_download_handles_transport_errors(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client_for(refuse) as http:
        assert not download_file(
            "https://example.test/w.pt", tmp_path / "w.pt", client=http, sleep=lambda s: None
        )


def test_download_validates_attempts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="attempts"):
        download_file("https://example.test/w.pt", tmp_path / "w.pt", attempts=0)


def test_download_creates_and_closes_its_own_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[httpx.Client] = []

    def new_client(timeout: httpx.Timeout) -> httpx.Client:
        http = _client_for(lambda r: httpx.Response(200, content=b"x" * 200_000))
        created.append(http)
        return http

    monkeypatch.setattr(models_setup, "_new_client", new_client)
    assert download_file("https://example.test/w.pt", tmp_path / "w.pt")
    assert created[0].is_closed


def test_ensure_weight(tmp_path: Path) -> None:
    dest = tmp_path / "w.pt"
    dest.write_bytes(b"good")
    digest = hashlib.sha256(b"good").hexdigest()
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=b"n" * 200_000)

    with _client_for(handler) as http:
        assert (
            ensure_weight("https://x.test/w.pt", dest, expected_sha256=digest, client=http)
            == "present"
        )
        assert (
            ensure_weight("https://x.test/w.pt", dest, expected_sha256=None, client=http)
            == "present"
        )
        assert requests == []
        assert (
            ensure_weight("https://x.test/w.pt", dest, expected_sha256="0" * 64, client=http)
            == "downloaded"
        )
    assert dest.read_bytes() == b"n" * 200_000
    with _client_for(lambda r: httpx.Response(404)) as http:
        status = ensure_weight(
            "https://x.test/v.pt",
            tmp_path / "v.pt",
            expected_sha256=None,
            client=http,
            sleep=lambda s: None,
        )
    assert status == "failed"


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


@pytest.fixture
def quiet_logging(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(models_setup.logging, "basicConfig", lambda **kw: calls.append(kw))
    return calls


def test_main_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_logging: list[dict[str, Any]]
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_setup(registry: ModelRegistry, **kwargs: Any) -> list[SetupOutcome]:
        seen.append({"registry": registry.path, **kwargs})
        return [SetupOutcome("vehicle_n", "registered", "models/yolo26n.pt", "ok")]

    monkeypatch.setattr(models_setup, "setup_models", fake_setup)
    registry_path = tmp_path / "models" / "registry.json"
    assert main(["--registry", str(registry_path), "--skip-sh17"]) == 0
    assert seen == [
        {"registry": registry_path, "models_dir": None, "skip_sh17": True, "skip_exports": False}
    ]
    assert quiet_logging and quiet_logging[0]["level"] == logging.INFO
    assert main(["--registry", str(registry_path), "--skip-exports"]) == 0
    assert seen[-1]["skip_exports"] is True

    def failing_setup(registry: ModelRegistry, **kwargs: Any) -> list[SetupOutcome]:
        return [
            SetupOutcome("vehicle_n", "unchanged", "models/yolo26n.pt", "ok"),
            SetupOutcome("ppe_s_sh17", "failed", None, "download failed"),
        ]

    monkeypatch.setattr(models_setup, "setup_models", failing_setup)
    assert main(["--registry", str(registry_path)]) == 1

    def broken_registry(registry: ModelRegistry, **kwargs: Any) -> list[SetupOutcome]:
        raise RegistryError("models/registry.json is not valid JSON")

    monkeypatch.setattr(models_setup, "setup_models", broken_registry)
    assert main(["--registry", str(registry_path), "--models-dir", str(tmp_path)]) == 1


def test_main_quiets_the_http_loggers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_logging: list[dict[str, Any]]
) -> None:
    for name in ("httpx", "httpcore"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
    monkeypatch.setattr(models_setup, "setup_models", lambda registry, **kwargs: [])
    assert main(["--registry", str(tmp_path / "models" / "registry.json")]) == 0
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--skip-sh17" in out and "--check" in out and "--skip-exports" in out


# --------------------------------------------------------------------------------------------
# export regeneration (make models re-exports what the registry lists, D-025)
# --------------------------------------------------------------------------------------------


class RecordingExporter:
    """An :data:`Exporter` that writes a fake artefact and registers it like ``export`` does."""

    def __init__(self, *, skip: str | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.skip = skip
        self.fail = fail

    def __call__(self, registry: ModelRegistry, name: str, fmt: str, imgsz: int) -> str | None:
        self.calls.append((name, fmt, imgsz))
        if self.fail:
            raise RuntimeError("onnx export crashed")
        if self.skip is not None:
            return self.skip
        target = registry.root / "models" / f"{name}.{fmt}.regenerated"
        target.write_bytes(f"{name} {fmt} {imgsz}".encode())
        registry.add_export(name, fmt, target, created_at=T0, imgsz=imgsz)  # type: ignore[arg-type]
        return None


def _with_exports(registry: ModelRegistry) -> None:
    """Register an ONNX (imgsz 320) and an OpenVINO export of ppe_s_sh17, both on disk."""
    models = registry.root / "models"
    (models / "ppe_s_sh17.onnx").write_bytes(b"onnx")
    registry.add_export("ppe_s_sh17", "onnx", models / "ppe_s_sh17.onnx", created_at=T0, imgsz=320)
    folder = models / "ppe_s_sh17_openvino_model"
    folder.mkdir()
    (folder / "ppe_s_sh17.xml").write_text("<net/>", encoding="utf-8")
    registry.add_export("ppe_s_sh17", "openvino", folder, created_at=T0)


def test_missing_exports_are_regenerated_at_their_recorded_size(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    _with_exports(tmp_registry)
    (tmp_registry.root / "models" / "ppe_s_sh17.onnx").unlink()
    exporter = RecordingExporter()
    outcomes = setup(exporter=exporter, runtime_available=lambda fmt: True)
    assert exporter.calls == [("ppe_s_sh17", "onnx", 320)]
    statuses = _statuses(outcomes)
    assert statuses["ppe_s_sh17:onnx"] == "exported"
    assert statuses["ppe_s_sh17:openvino"] == "unchanged"
    exported = next(o for o in outcomes if o.name == "ppe_s_sh17:onnx")
    assert exported.path == "models/ppe_s_sh17.onnx.regenerated"
    assert "imgsz 320" in exported.detail
    entry = tmp_registry.resolve("ppe_s_sh17")
    assert tmp_registry.verify(entry, "onnx")


def test_stale_exports_without_a_recorded_size_use_the_entry_size(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    _with_exports(tmp_registry)
    folder = tmp_registry.root / "models" / "ppe_s_sh17_openvino_model"
    (folder / "ppe_s_sh17.xml").write_text("<changed/>", encoding="utf-8")
    exporter = RecordingExporter()
    setup(exporter=exporter, runtime_available=lambda fmt: True)
    assert exporter.calls == [("ppe_s_sh17", "openvino", 640)]


def test_exports_whose_runtime_is_missing_are_skipped(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    _with_exports(tmp_registry)
    (tmp_registry.root / "models" / "ppe_s_sh17.onnx").unlink()
    exporter = RecordingExporter()
    outcomes = setup(exporter=exporter, runtime_available=lambda fmt: fmt != "onnx")
    assert exporter.calls == []
    skipped = next(o for o in outcomes if o.name == "ppe_s_sh17:onnx")
    assert skipped.status == "skipped"
    assert "onnxruntime is not installed" in skipped.detail


def test_exporter_skips_and_failures_are_reported(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    _with_exports(tmp_registry)
    (tmp_registry.root / "models" / "ppe_s_sh17.onnx").unlink()
    skipping = RecordingExporter(skip="TensorRT needs a CUDA GPU")
    outcomes = setup(exporter=skipping, runtime_available=lambda fmt: True)
    assert _statuses(outcomes)["ppe_s_sh17:onnx"] == "skipped"
    failing = RecordingExporter(fail=True)
    outcomes = setup(exporter=failing, runtime_available=lambda fmt: True)
    failed = next(o for o in outcomes if o.name == "ppe_s_sh17:onnx")
    assert failed.status == "failed"
    assert "RuntimeError: onnx export crashed" in failed.detail


def test_skip_exports_leaves_exports_alone(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    _with_exports(tmp_registry)
    (tmp_registry.root / "models" / "ppe_s_sh17.onnx").unlink()
    exporter = RecordingExporter()
    outcomes = setup(exporter=exporter, skip_exports=True)
    assert exporter.calls == []
    assert not any(":" in o.name for o in outcomes)


def test_main_fails_when_an_export_cannot_be_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_logging: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(
        models_setup,
        "setup_models",
        lambda registry, **kwargs: [
            SetupOutcome("pose_n", "unchanged", "models/yolo26n-pose.pt", "ok"),
            SetupOutcome("pose_n:onnx", "failed", "models/yolo26n-pose.onnx", "crashed"),
            SetupOutcome("pose_n:tensorrt", "skipped", "models/yolo26n-pose.engine", "no trt"),
        ],
    )
    assert main(["--registry", str(tmp_path / "models" / "registry.json")]) == 1


def test_default_exporter_runs_the_training_export(
    tmp_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stand-in for safety_ai.training.export (the real one imports torch and Ultralytics).
    calls: list[dict[str, Any]] = []
    answers = iter(
        [
            SimpleNamespace(artifacts=("m.onnx",), skipped=()),
            SimpleNamespace(artifacts=(), skipped=(SimpleNamespace(reason="no CUDA"),)),
            SimpleNamespace(artifacts=(), skipped=()),
        ]
    )

    def fake_export(name: str, formats: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append({"name": name, "formats": formats, **kwargs})
        return next(answers)

    fake_module = ModuleType("safety_ai.training.export")
    fake_module.export = fake_export  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safety_ai.training.export", fake_module)
    default = models_setup._export_with_ultralytics
    assert default(tmp_registry, "m", "onnx", 320) is None
    assert calls[0] == {
        "name": "m",
        "formats": ("onnx",),
        "imgsz": 320,
        "device": "cpu",
        "registry": tmp_registry,
    }
    assert default(tmp_registry, "m", "tensorrt", 640) == "no CUDA"
    with pytest.raises(RuntimeError, match="no openvino export of m was produced"):
        default(tmp_registry, "m", "openvino", 640)


# --------------------------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------------------------


def test_check_passes_for_a_complete_registry(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    _with_exports(tmp_registry)
    report = check_models(tmp_registry, required=required_names())
    assert report.passed
    assert [(r.name, r.format, r.status) for r in report.rows] == [
        ("vehicle_n", "pt", "ok"),
        ("vehicle_s", "pt", "ok"),
        ("pose_n", "pt", "ok"),
        ("ppe_n_sh17", "pt", "ok"),
        ("ppe_s_sh17", "pt", "ok"),
        ("ppe_s_sh17", "onnx", "ok"),
        ("ppe_s_sh17", "openvino", "ok"),
    ]
    onnx = report.rows[5]
    assert (onnx.artefact, onnx.path, onnx.size_bytes) == ("export", "models/ppe_s_sh17.onnx", 4)
    assert onnx.actual_sha256 == onnx.expected_sha256 == hashlib.sha256(b"onnx").hexdigest()
    folder = report.rows[6]
    assert folder.size_bytes == len("<net/>")
    assert report.count("ok") == 7


def test_check_reports_missing_and_mismatching_files(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    _with_exports(tmp_registry)
    models = tmp_registry.root / "models"
    (models / "yolo26s.pt").unlink()
    (models / "yolo26n.pt").write_bytes(b"tampered")
    (models / "ppe_s_sh17.onnx").unlink()
    report = check_models(tmp_registry, required=required_names(), runtime_available=lambda f: True)
    assert not report.passed
    by_key = {(r.name, r.format): r for r in report.rows}
    assert by_key[("vehicle_s", "pt")].status == "missing"
    assert by_key[("vehicle_s", "pt")].actual_sha256 is None
    tampered = by_key[("vehicle_n", "pt")]
    assert tampered.status == "mismatch"
    assert tampered.actual_sha256 == hashlib.sha256(b"tampered").hexdigest()
    assert "run python -m safety_ai.inference.models_setup" in tampered.detail
    assert by_key[("ppe_s_sh17", "onnx")].status == "missing"


def test_check_skips_unusable_exports_but_not_their_weights(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup()
    _with_exports(tmp_registry)
    (tmp_registry.root / "models" / "ppe_s_sh17.onnx").unlink()
    report = check_models(
        tmp_registry, required=required_names(), runtime_available=lambda f: False
    )
    assert report.passed
    onnx = next(r for r in report.rows if r.format == "onnx")
    assert onnx.status == "skipped" and onnx.usable
    assert "onnxruntime is not installed" in onnx.detail
    openvino = next(r for r in report.rows if r.format == "openvino")
    assert openvino.status == "ok"  # a valid export is reported as ok whatever the runtime
    (tmp_registry.root / "models" / "ppe_s_sh17.pt").unlink()
    report = check_models(
        tmp_registry, required=required_names(), runtime_available=lambda f: False
    )
    assert not report.passed


def test_check_requires_the_setup_entries(tmp_registry: ModelRegistry) -> None:
    report = check_models(tmp_registry, required=required_names(skip_sh17=True))
    assert not report.passed
    assert [(r.name, r.status) for r in report.rows] == [
        ("vehicle_n", "unregistered"),
        ("vehicle_s", "unregistered"),
        ("pose_n", "unregistered"),
    ]
    assert check_models(tmp_registry).passed  # nothing registered, nothing required
    assert required_names() == ("vehicle_n", "vehicle_s", "pose_n", "ppe_n_sh17", "ppe_s_sh17")


def test_render_check_table(
    setup: Callable[..., list[SetupOutcome]], tmp_registry: ModelRegistry
) -> None:
    setup(skip_sh17=True)
    (tmp_registry.root / "models" / "yolo26s.pt").unlink()
    report = check_models(tmp_registry, required=required_names())
    lines = render_check_table(report).splitlines()
    assert lines[0].split() == [
        "name",
        "role",
        "format",
        "path",
        "size",
        "sha256",
        "status",
        "detail",
    ]
    digest = hashlib.sha256(_payload("yolo26n.pt")).hexdigest()[:12]
    expected = ["vehicle_n", "vehicle", "pt", "models/yolo26n.pt", "0.1", "MB", digest, "ok"]
    assert lines[1].split() == expected
    status_column = lines[0].index("status")
    assert lines[1][status_column:] == "ok"
    detail_column = lines[0].index("detail")
    assert lines[2][status_column:detail_column].strip() == "missing"
    assert lines[2][detail_column:].startswith("file not found; run python -m")
    assert lines[4][status_column:detail_column].strip() == "unregistered"
    assert lines[4][detail_column:].startswith("required entry is not in the registry")
    assert lines[-1] == f"FAILED: {tmp_registry.path.as_posix()}: 2 ok, 0 skipped, 3 problem(s)"


def test_main_check_prints_the_table_and_sets_the_exit_code(
    setup: Callable[..., list[SetupOutcome]],
    tmp_registry: ModelRegistry,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup()

    def no_setup(registry: ModelRegistry, **kwargs: Any) -> list[SetupOutcome]:
        raise AssertionError("--check must not run the setup")

    monkeypatch.setattr(models_setup, "setup_models", no_setup)
    assert main(["--check", "--registry", str(tmp_registry.path)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("PASSED:")
    assert "ppe_s_sh17" in out
    (tmp_registry.root / "models" / "ppe_s_sh17.pt").unlink()
    assert main(["--check", "--registry", str(tmp_registry.path)]) == 1
    assert capsys.readouterr().out.splitlines()[-1].startswith("FAILED:")
    assert main(["--check", "--skip-sh17", "--registry", str(tmp_registry.path)]) == 1


def test_main_check_requires_the_sh17_entries_unless_skipped(
    setup: Callable[..., list[SetupOutcome]],
    tmp_registry: ModelRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(skip_sh17=True)
    assert main(["--check", "--registry", str(tmp_registry.path)]) == 1
    assert "ppe_n_sh17" in capsys.readouterr().out
    assert main(["--check", "--skip-sh17", "--registry", str(tmp_registry.path)]) == 0


def test_main_check_reports_an_invalid_registry(
    tmp_registry: ModelRegistry, capsys: pytest.CaptureFixture[str]
) -> None:
    tmp_registry.path.write_text("{not json", encoding="utf-8")
    assert main(["--check", "--registry", str(tmp_registry.path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model check failed" in captured.err and "not valid JSON" in captured.err


def test_check_imports_no_model_runtime(tmp_path: Path) -> None:
    code = (
        "import sys; from safety_ai.inference import models_setup as m; "
        f"rc = m.main(['--check', '--registry', {str(tmp_path / 'models' / 'registry.json')!r}]); "
        "heavy = [n for n in ('torch', 'ultralytics', 'cv2', 'onnxruntime', 'openvino') "
        "if n in sys.modules]; print(rc, heavy)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.stdout.strip().splitlines()[-1] == "1 []", done.stderr
