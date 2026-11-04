"""Tests for device resolution, backend probing/choice (fake bench.json) and model loading."""

from __future__ import annotations

import builtins
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from safety_ai.inference import backends
from safety_ai.inference.backends import (
    LoadedModel,
    artifact_path,
    available_backends,
    choose_backend,
    device_class,
    half_keyword,
    load_model,
    precision_kwargs,
    read_bench_fps,
    resolve_device,
    tensor_to_list,
)
from safety_ai.inference.registry import ExportEntry, ModelRegistry, RegistryEntry, RegistryError

T0 = datetime(2026, 11, 4, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def hardware(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Set the probed hardware.

    ``hardware(cuda=True, count=1, mps=False, modules=(...), providers=(...))``; ``providers``
    are the ONNX Runtime execution providers (default: the CPU one only).
    """

    def configure(
        *,
        cuda: bool = False,
        count: int = 1,
        mps: bool = False,
        modules: tuple[str, ...] = (),
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        monkeypatch.setattr(backends, "_cuda_available", lambda: cuda)
        monkeypatch.setattr(backends, "_cuda_device_count", lambda: count if cuda else 0)
        monkeypatch.setattr(backends, "_mps_available", lambda: mps)
        monkeypatch.setattr(backends, "_module_importable", lambda name: name in modules)
        monkeypatch.setattr(backends, "_onnx_providers", lambda: frozenset(providers))

    configure()
    return configure


def _exports(**paths: str) -> dict[str, ExportEntry]:
    return {fmt: ExportEntry(path=p, sha256="b" * 64, created_at=T0) for fmt, p in paths.items()}


@pytest.fixture
def exported_entry(tmp_path: Path, make_entry: Callable[..., RegistryEntry]) -> RegistryEntry:
    """An entry with onnx, openvino and tensorrt exports that exist under ``tmp_path``."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.onnx").write_bytes(b"onnx")
    (models / "m_openvino_model").mkdir()
    (models / "m_openvino_model" / "m.xml").write_text("x", encoding="utf-8")
    (models / "m.engine").write_bytes(b"engine")
    return make_entry(
        name="m",
        path="models/m.pt",
        exports=_exports(
            onnx="models/m.onnx", openvino="models/m_openvino_model", tensorrt="models/m.engine"
        ),
    )


# --------------------------------------------------------------------------------------------
# devices
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda:0"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(
    hardware: Callable[..., None], cuda: bool, mps: bool, expected: str
) -> None:
    hardware(cuda=cuda, mps=mps)
    assert resolve_device("auto") == expected
    assert resolve_device("") == expected
    assert resolve_device(" AUTO ") == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("cuda", "cuda:0"), ("cuda:1", "cuda:1"), ("1", "cuda:1"), ("0", "cuda:0")],
)
def test_explicit_cuda_devices(hardware: Callable[..., None], spec: str, expected: str) -> None:
    hardware(cuda=True, count=2)
    assert resolve_device(spec) == expected


def test_unavailable_devices_fall_back_with_a_warning(
    hardware: Callable[..., None], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.backends")
    hardware(cuda=False, mps=False)
    assert resolve_device("cuda:0") == "cpu"
    assert resolve_device("mps") == "cpu"
    hardware(cuda=True, count=1)
    assert resolve_device("cuda:3") == "cuda:0"
    assert resolve_device("cpu") == "cpu"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3


def test_mps_when_available(hardware: Callable[..., None]) -> None:
    hardware(mps=True)
    assert resolve_device("mps") == "mps"


@pytest.mark.parametrize("spec", ["gpu", "cuda:x", "tpu:0", "-1"])
def test_invalid_device_specs(hardware: Callable[..., None], spec: str) -> None:
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device(spec)


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", "cpu"), ("cuda:0", "cuda"), ("cuda", "cuda"), ("1", "cuda"), ("mps", "mps")],
)
def test_device_class(device: str, expected: str) -> None:
    assert device_class(device) == expected


# --------------------------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------------------------


def test_pt_is_always_available(
    hardware: Callable[..., None], make_entry: Callable[..., RegistryEntry], tmp_path: Path
) -> None:
    assert available_backends(make_entry(), "cpu", root=tmp_path) == ["pt"]


def test_exports_need_their_runtime(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(modules=("onnxruntime",))
    assert available_backends(exported_entry, "cpu", root=tmp_path) == ["pt", "onnx"]
    hardware(modules=("onnxruntime", "openvino", "tensorrt"))
    assert available_backends(exported_entry, "cpu", root=tmp_path) == ["pt", "onnx", "openvino"]


def test_tensorrt_needs_cuda_device_and_cuda(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(cuda=True, modules=("tensorrt",))
    assert available_backends(exported_entry, "cuda:0", root=tmp_path) == ["pt", "tensorrt"]
    assert available_backends(exported_entry, "cpu", root=tmp_path) == ["pt"]
    hardware(cuda=False, modules=("tensorrt",))
    assert available_backends(exported_entry, "cuda:0", root=tmp_path) == ["pt"]


def test_openvino_runs_only_on_the_cpu(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(cuda=True, mps=True, modules=("openvino",))
    assert available_backends(exported_entry, "cpu", root=tmp_path) == ["pt", "openvino"]
    assert available_backends(exported_entry, "cuda:0", root=tmp_path) == ["pt"]
    assert available_backends(exported_entry, "mps", root=tmp_path) == ["pt"]


def test_onnx_on_an_accelerator_needs_its_execution_provider(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(cuda=True, mps=True, modules=("onnxruntime",))
    assert available_backends(exported_entry, "cpu", root=tmp_path) == ["pt", "onnx"]
    assert available_backends(exported_entry, "cuda:0", root=tmp_path) == ["pt"]
    assert available_backends(exported_entry, "mps", root=tmp_path) == ["pt"]
    hardware(
        cuda=True,
        modules=("onnxruntime",),
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    assert available_backends(exported_entry, "cuda:0", root=tmp_path) == ["pt", "onnx"]
    assert available_backends(exported_entry, "mps", root=tmp_path) == ["pt"]
    hardware(mps=True, modules=("onnxruntime",), providers=("CoreMLExecutionProvider",))
    assert available_backends(exported_entry, "mps", root=tmp_path) == ["pt", "onnx"]


def test_onnx_providers_probe_reports_the_installed_runtime() -> None:
    providers = backends._onnx_providers()
    assert isinstance(providers, frozenset)
    assert all(isinstance(p, str) for p in providers)


def test_onnx_providers_probe_survives_a_broken_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def broken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            raise OSError("DLL load failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    assert backends._onnx_providers() == frozenset()


def test_missing_export_files_are_unavailable(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(modules=("onnxruntime", "openvino"))
    (tmp_path / "models" / "m.onnx").unlink()
    assert available_backends(exported_entry, "cpu", root=tmp_path) == ["pt", "openvino"]


def test_available_backends_defaults_to_the_current_directory(
    hardware: Callable[..., None],
    exported_entry: RegistryEntry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardware(modules=("onnxruntime",))
    monkeypatch.chdir(tmp_path)
    assert available_backends(exported_entry, "cpu") == ["pt", "onnx"]


def test_non_pt_primary_and_joblib_entries(
    hardware: Callable[..., None], make_entry: Callable[..., RegistryEntry], tmp_path: Path
) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "only.onnx").write_bytes(b"x")
    onnx_only = make_entry(name="only", format="onnx", path="models/only.onnx")
    assert available_backends(onnx_only, "cpu", root=tmp_path) == []
    hardware(modules=("onnxruntime",))
    assert available_backends(onnx_only, "cpu", root=tmp_path) == ["onnx"]
    assert (
        choose_backend("pt", onnx_only, "cpu", tmp_path / "none.json", root=tmp_path)[0] == "onnx"
    )
    fall = make_entry(
        name="fall_lr",
        role="fall",
        format="joblib",
        path="models/fall_lr.joblib",
        classes=[],
        imgsz=0,
    )
    assert available_backends(fall, "cpu", root=tmp_path) == []
    with pytest.raises(RegistryError, match="no image backend"):
        choose_backend("auto", fall, "cpu", tmp_path / "none.json", root=tmp_path)


def test_artifact_path(exported_entry: RegistryEntry, tmp_path: Path) -> None:
    assert artifact_path(exported_entry, "pt", tmp_path) == tmp_path / "models" / "m.pt"
    assert artifact_path(exported_entry, "onnx", tmp_path) == tmp_path / "models" / "m.onnx"
    assert artifact_path(exported_entry, "coreml", tmp_path) is None


# --------------------------------------------------------------------------------------------
# choice
# --------------------------------------------------------------------------------------------


def _bench(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"created_at": "2026-11-04T03:00:00Z", "results": rows}), "utf-8")
    return path


def _row(model: str, backend: str, device: str, fps: Any) -> dict[str, Any]:
    return {"model": model, "backend": backend, "device": device, "imgsz": 640, "fps": fps}


def test_explicit_available_backend_is_used(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(modules=("onnxruntime",))
    backend, path = choose_backend(
        "onnx", exported_entry, "cpu", tmp_path / "x.json", root=tmp_path
    )
    assert (backend, path) == ("onnx", tmp_path / "models" / "m.onnx")


def test_explicit_unavailable_backend_falls_back_to_pt_with_one_warning(
    hardware: Callable[..., None],
    exported_entry: RegistryEntry,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.backends")
    backend, path = choose_backend(
        "tensorrt", exported_entry, "cpu", tmp_path / "x.json", root=tmp_path
    )
    assert (backend, path) == ("pt", tmp_path / "models" / "m.pt")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "tensorrt" in warnings[0].getMessage()


@pytest.mark.parametrize(
    ("requested", "device", "hint"),
    [
        ("openvino", "cuda:0", "OpenVINO runs on the CPU"),
        ("onnx", "cuda:0", "no CUDAExecutionProvider"),
        ("onnx", "mps", "no CoreMLExecutionProvider"),
        ("tensorrt", "cpu", "TensorRT needs a CUDA device"),
    ],
)
def test_device_bound_backends_explain_the_fallback(
    hardware: Callable[..., None],
    exported_entry: RegistryEntry,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    requested: str,
    device: str,
    hint: str,
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.backends")
    hardware(cuda=True, mps=True, modules=("onnxruntime", "openvino", "tensorrt"))
    backend, _ = choose_backend(requested, exported_entry, device, tmp_path / "x", root=tmp_path)
    assert backend == "pt"
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert hint in warning.getMessage()


def test_runtime_without_device_constraint_gets_no_hint(
    hardware: Callable[..., None],
    exported_entry: RegistryEntry,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.backends")
    choose_backend("onnx", exported_entry, "cpu", tmp_path / "x", root=tmp_path)
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning.getMessage().endswith("(available: pt); using pt")


def test_auto_picks_the_fastest_measured_backend_per_device_class(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(cuda=True, modules=("onnxruntime", "openvino", "tensorrt"))
    bench = _bench(
        tmp_path,
        [
            _row("m", "pt", "cpu", 9.0),
            _row("m", "onnx", "cpu", 14.0),
            _row("m", "openvino", "cpu", 21.5),
            _row("m", "pt", "cuda:0", 48.0),
            _row("m", "tensorrt", "cuda:0", 95.0),
            _row("m", "openvino", "cuda:0", 22.0),
            _row("other", "onnx", "cpu", 500.0),
        ],
    )
    assert choose_backend("auto", exported_entry, "cpu", bench, root=tmp_path)[0] == "openvino"
    assert choose_backend("auto", exported_entry, "cuda:0", bench, root=tmp_path)[0] == "tensorrt"


def test_auto_ignores_measurements_of_unavailable_backends(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(modules=("onnxruntime",))
    bench = _bench(
        tmp_path,
        [
            _row("m", "openvino", "cpu", 50.0),
            _row("m", "onnx", "cpu", 12.0),
            _row("m", "pt", "cpu", 8),
        ],
    )
    assert choose_backend("auto", exported_entry, "cpu", bench, root=tmp_path)[0] == "onnx"


def test_auto_without_measurements_uses_pt(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(modules=("onnxruntime",))
    missing = tmp_path / "missing.json"
    assert choose_backend("auto", exported_entry, "cpu", missing, root=tmp_path)[0] == "pt"
    other_device = _bench(tmp_path, [_row("m", "onnx", "cuda:0", 99.0)])
    assert choose_backend("auto", exported_entry, "cpu", other_device, root=tmp_path)[0] == "pt"


def test_auto_ties_prefer_pt(
    hardware: Callable[..., None], exported_entry: RegistryEntry, tmp_path: Path
) -> None:
    hardware(modules=("onnxruntime",))
    bench = _bench(tmp_path, [_row("m", "onnx", "cpu", 10.0), _row("m", "pt", "cpu", 10.0)])
    assert choose_backend("auto", exported_entry, "cpu", bench, root=tmp_path)[0] == "pt"


def test_unknown_backend_name(exported_entry: RegistryEntry, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        choose_backend("coreml", exported_entry, "cpu", root=tmp_path)


def test_malformed_bench_files_are_ignored_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.backends")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert read_bench_fps(bad, "m", "cpu") == {}
    assert len(caplog.records) == 1
    bad.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert read_bench_fps(bad, "m", "cpu") == {}
    bad.write_text(json.dumps({"results": "nope"}), encoding="utf-8")
    assert read_bench_fps(bad, "m", "cpu") == {}


def test_bench_rows_are_validated_one_by_one(tmp_path: Path) -> None:
    rows: list[Any] = [
        "junk",
        _row("m", 3, "cpu", 10.0),
        _row("m", "onnx", None, 10.0),
        _row("m", "onnx", "cpu", True),
        _row("m", "onnx", "cpu", "fast"),
        _row("m", "onnx", "cpu", -1.0),
        _row("m", "onnx", "cpu", float("inf")),
        _row("m", "onnx", "cpu", 7.0),
        _row("m", "onnx", "cpu", 11.0),
        _row("m", "pt", "cuda:0", 30.0),
    ]
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"results": rows}).replace("Infinity", "1e999"), encoding="utf-8")
    assert read_bench_fps(path, "m", "cpu") == {"onnx": 11.0}
    assert read_bench_fps(path, "m", "cuda") == {"pt": 30.0}


# --------------------------------------------------------------------------------------------
# precision
# --------------------------------------------------------------------------------------------


def test_half_keyword_follows_the_installed_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_default_cfg_keys", lambda: frozenset({"quantize", "conf"}))
    assert half_keyword() == "quantize"
    monkeypatch.setattr(backends, "_default_cfg_keys", lambda: frozenset({"half", "conf"}))
    assert half_keyword() == "half"


def test_precision_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_default_cfg_keys", lambda: frozenset({"quantize"}))
    assert precision_kwargs("cuda:0", "pt", True) == {"quantize": 16}
    assert precision_kwargs("cuda:0", "tensorrt", True) == {"quantize": 16}
    assert precision_kwargs("cuda:0", "openvino", True) == {}
    assert precision_kwargs("cuda:0", "pt", False) == {}
    assert precision_kwargs("cpu", "pt", True) == {}
    assert precision_kwargs("mps", "pt", True) == {}
    monkeypatch.setattr(backends, "_default_cfg_keys", lambda: frozenset({"half"}))
    assert precision_kwargs("cuda:0", "pt", True) == {"half": True}


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------


class _Model:
    def __init__(self, names: dict[int, str] | None) -> None:
        if names is not None:
            self.names = names


def test_load_model_uses_the_chosen_backend(
    hardware: Callable[..., None],
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> None:
    entry = add_weights(tmp_registry, make_entry(classes=["person", "helmet"]))
    calls: list[tuple[Path, str]] = []

    def loader(path: Path, task: str) -> _Model:
        calls.append((path, task))
        return _Model({0: "person", 1: "helmet"})

    loaded = load_model(entry, tmp_registry, device="cpu", task="detect", loader=loader)
    assert isinstance(loaded, LoadedModel)
    assert loaded.backend == "pt"
    assert loaded.path == tmp_registry.root / "models" / "ppe_s_test.pt"
    assert calls == [(loaded.path, "detect")]
    assert loaded.labels == ("person", "helmet")
    info = loaded.info()
    assert (info.name, info.version, info.family, info.backend, info.device) == (
        "ppe_s_test",
        "2026-11-04",
        "yolo26",
        "pt",
        "cpu",
    )
    assert info.classes == ("person", "helmet")
    assert loaded.info(("person",)).classes == ("person",)


def test_load_model_checks_the_class_count(
    hardware: Callable[..., None],
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> None:
    entry = add_weights(tmp_registry, make_entry(classes=["person", "helmet"]))
    with pytest.raises(RegistryError, match="3 classes but the registry maps 2"):
        load_model(entry, tmp_registry, loader=lambda p, t: _Model({0: "a", 1: "b", 2: "c"}))
    assert load_model(entry, tmp_registry, loader=lambda p, t: _Model(None)).backend == "pt"


class _ExportedModel:
    """Like an exported Ultralytics model: ``names`` builds a predictor on the default device."""

    def __init__(self) -> None:
        self.overrides: dict[str, Any] = {"task": "detect"}
        self.devices_at_names: list[Any] = []

    @property
    def names(self) -> dict[int, str]:
        self.devices_at_names.append(self.overrides.get("device"))
        return {0: "person"}


def test_load_model_pins_the_device_before_reading_class_names(
    hardware: Callable[..., None],
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> None:
    entry = add_weights(tmp_registry, make_entry(classes=["person"]))
    model = _ExportedModel()
    loaded = load_model(entry, tmp_registry, device="cuda:0", loader=lambda p, t: model)
    assert loaded.model is model
    assert model.devices_at_names == ["cuda:0"]
    assert model.overrides == {"task": "detect", "device": "cuda:0"}


def test_pin_device_ignores_objects_without_overrides() -> None:
    plain = _Model({0: "person"})
    backends.pin_device(plain, "cpu")
    assert not hasattr(plain, "overrides")
    odd = _Model(None)
    odd.overrides = ("not", "a", "dict")  # type: ignore[attr-defined]
    backends.pin_device(odd, "cpu")
    assert odd.overrides == ("not", "a", "dict")  # type: ignore[attr-defined]


def test_load_model_missing_weights(
    hardware: Callable[..., None],
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_model(make_entry(), tmp_registry, loader=lambda p, t: _Model(None))


def test_load_model_warns_on_sha_mismatch(
    hardware: Callable[..., None],
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.backends")
    entry = add_weights(tmp_registry, make_entry(classes=["person"]))
    tmp_registry.resolve_path(entry).write_bytes(b"swapped weights")
    loaded = load_model(entry, tmp_registry, loader=lambda p, t: _Model({0: "person"}))
    assert loaded.backend == "pt"
    assert any("sha256" in r.getMessage() for r in caplog.records)
    caplog.clear()
    load_model(entry, tmp_registry, loader=lambda p, t: _Model({0: "person"}), verify=False)
    assert caplog.records == []


def test_tensor_to_list() -> None:
    assert tensor_to_list(np.array([[1.0, 2.0], [3.0, 4.0]])) == [[1.0, 2.0], [3.0, 4.0]]
    assert tensor_to_list(np.float32(2.5)) == [2.5]
    assert tensor_to_list((1, 2)) == [1, 2]
