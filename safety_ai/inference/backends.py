"""Device resolution, runtime-backend probing, backend choice and model loading.

Spec 5.3 (device ``auto`` -> cuda -> mps -> cpu, half precision on CUDA), 6.6 (pt, onnx,
openvino, tensorrt; the runtime picks the fastest backend measured in ``models/bench.json``
unless ``models.backend`` is set) and DECISIONS.md D-009 (the half-precision keyword is detected
from the installed Ultralytics instead of being hard-coded).

Heavy imports (torch, ultralytics, onnxruntime, openvino, tensorrt) happen lazily inside the
probes, so importing this module is cheap and side-effect free.

Public API:

* :func:`resolve_device`, :func:`device_class`
* :func:`available_backends`, :func:`artifact_path`, :func:`choose_backend`,
  :func:`read_bench_fps`
* :func:`half_keyword`, :func:`precision_kwargs`
* :class:`LoadedModel`, :func:`load_model`, :func:`load_yolo`, :data:`ModelLoader`,
  :func:`pin_device`, :func:`tensor_to_list`
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from safety_ai.inference.registry import (
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    class_names,
)
from safety_ai.inference.types import ModelInfo

__all__ = [
    "BACKENDS",
    "DEFAULT_BENCH_PATH",
    "BackendName",
    "DeviceClass",
    "LoadedModel",
    "ModelLoader",
    "artifact_path",
    "available_backends",
    "choose_backend",
    "device_class",
    "half_keyword",
    "load_model",
    "load_yolo",
    "pin_device",
    "precision_kwargs",
    "read_bench_fps",
    "resolve_device",
    "tensor_to_list",
]

_LOG = logging.getLogger(__name__)

BackendName = Literal["pt", "onnx", "openvino", "tensorrt"]
DeviceClass = Literal["cpu", "cuda", "mps"]

BACKENDS: Final[tuple[str, ...]] = ("pt", "onnx", "openvino", "tensorrt")
"""Runtime backends in preference order for ties."""
DEFAULT_BENCH_PATH: Final[Path] = Path("models/bench.json")
_RUNTIME_MODULES: Final[Mapping[str, str]] = {
    "onnx": "onnxruntime",
    "openvino": "openvino",
    "tensorrt": "tensorrt",
}
_ONNX_DEVICE_PROVIDERS: Final[Mapping[str, str]] = {
    "cuda": "CUDAExecutionProvider",
    "mps": "CoreMLExecutionProvider",
}
"""ONNX Runtime execution provider Ultralytics uses for each non-CPU device class."""
_CUDA_RE: Final = re.compile(r"^(?:cuda(?::(\d+))?|(\d+))$")
_HALF_FP16: Final[int] = 16

ModelLoader = Callable[[Path, str], Any]
"""``(weights path, task) -> model``; the default is :func:`load_yolo`."""


def _cuda_available() -> bool:
    """True when torch is installed and sees a CUDA device."""
    try:
        import torch  # noqa: PLC0415  (lazy: torch is heavy and optional for this probe)
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _cuda_device_count() -> int:
    """Number of visible CUDA devices (0 without torch or CUDA)."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return 0
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def _mps_available() -> bool:
    """True when torch reports an Apple Metal (MPS) device."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _module_importable(name: str) -> bool:
    """True when ``import name`` succeeds (a broken native install counts as unavailable)."""
    try:
        importlib.import_module(name)
    except Exception:  # ImportError, OSError (DLL load), RuntimeError from native init
        return False
    return True


def _onnx_providers() -> frozenset[str]:
    """Execution providers of the installed ONNX Runtime (empty when it does not import)."""
    try:
        import onnxruntime  # noqa: PLC0415  (lazy: optional native runtime)

        return frozenset(onnxruntime.get_available_providers())
    except Exception:  # ImportError, OSError (DLL load), RuntimeError from native init
        return frozenset()


def _default_cfg_keys() -> Collection[str]:
    """The argument names the installed Ultralytics accepts."""
    from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: PLC0415  (lazy heavy import)

    return frozenset(DEFAULT_CFG_DICT)


def resolve_device(spec: str) -> str:
    """Resolve a device spec to ``cpu``, ``mps`` or ``cuda:N``.

    ``auto`` (or empty) picks ``cuda:0``, then ``mps``, then ``cpu``. ``cuda``, ``cuda:N`` and a
    bare index ``N`` request a GPU; when CUDA (or MPS) is not available the device falls back to
    ``cpu`` and an out-of-range index to ``cuda:0``, each with one WARNING, so a configuration
    copied between machines still runs.

    Raises:
        ValueError: For an unrecognized spec.
    """
    text = spec.strip().lower()
    if text in ("", "auto"):
        if _cuda_available():
            return "cuda:0"
        if _mps_available():
            return "mps"
        return "cpu"
    if text == "cpu":
        return "cpu"
    if text == "mps":
        if _mps_available():
            return "mps"
        _LOG.warning("device 'mps' requested but MPS is not available; using cpu")
        return "cpu"
    match = _CUDA_RE.match(text)
    if match is None:
        raise ValueError(f"unknown device {spec!r}; expected auto, cpu, mps, cuda, cuda:N or N")
    index = int(match.group(1) or match.group(2) or 0)
    if not _cuda_available():
        _LOG.warning("device %r requested but CUDA is not available; using cpu", spec)
        return "cpu"
    count = _cuda_device_count()
    if index >= count:
        _LOG.warning(
            "device %r requested but only %d CUDA device(s) exist; using cuda:0", spec, count
        )
        return "cuda:0"
    return f"cuda:{index}"


def device_class(device: str) -> DeviceClass:
    """Classify a device string as ``cuda``, ``mps`` or ``cpu`` (bench results are per class)."""
    text = device.strip().lower()
    if text.startswith("cuda") or text.isdigit():
        return "cuda"
    if text == "mps":
        return "mps"
    return "cpu"


def artifact_path(entry: RegistryEntry, backend: str, root: Path) -> Path | None:
    """Return the path of ``backend``'s artefact of ``entry`` under ``root`` (None: not listed)."""
    if backend == entry.format:
        rel = entry.path
    else:
        export = next((e for fmt, e in entry.exports.items() if fmt == backend), None)
        if export is None:
            return None
        rel = export.path
    return root.joinpath(*rel.split("/"))


def available_backends(entry: RegistryEntry, device: str, *, root: Path | None = None) -> list[str]:
    """List the backends that can run ``entry`` on ``device``, in :data:`BACKENDS` order.

    A backend counts only when it really runs on ``device``, so measurements, the backend
    choice and the reported device never describe a run that silently happened elsewhere:

    * ``pt``: always, for entries whose primary format is ``pt``;
    * ``onnx``: when onnxruntime imports and the ONNX export exists; on a CUDA device only with
      ONNX Runtime's ``CUDAExecutionProvider`` and on MPS only with ``CoreMLExecutionProvider``
      (without them Ultralytics would fall back to the CPU after trying to pip-install
      ``onnxruntime-gpu``);
    * ``openvino``: when openvino imports, the OpenVINO export exists and ``device`` is the CPU
      (Ultralytics runs OpenVINO on the CPU whatever torch device is requested);
    * ``tensorrt``: when tensorrt imports, CUDA is available, ``device`` is a CUDA device and
      the ``.engine`` exists.

    ``root`` is the repository root the entry's paths are relative to (default: the current
    directory). Fall classifiers (``joblib``) have no image backend and yield an empty list.
    """
    base = Path.cwd() if root is None else Path(root)
    return list(_runnable_artifacts(entry, device, base))


def _runs_on(backend: str, dclass: DeviceClass) -> bool:
    """True when an importable ``backend`` runtime executes on devices of class ``dclass``."""
    if backend == "openvino":
        return dclass == "cpu"
    if backend == "tensorrt":
        return dclass == "cuda" and _cuda_available()
    if backend == "onnx" and dclass != "cpu":
        provider = _ONNX_DEVICE_PROVIDERS[dclass]
        return provider in _onnx_providers()
    return True


def _runnable_artifacts(entry: RegistryEntry, device: str, root: Path) -> dict[str, Path]:
    """Backend -> artefact path for every backend that can run ``entry`` on ``device``."""
    found: dict[str, Path] = {}
    dclass = device_class(device)
    for backend in BACKENDS:
        path = artifact_path(entry, backend, root)
        if path is None:
            continue
        if backend == "pt":
            found[backend] = path
            continue
        if not path.exists() or not _module_importable(_RUNTIME_MODULES[backend]):
            continue
        if _runs_on(backend, dclass):
            found[backend] = path
    return found


def read_bench_fps(bench_path: Path, model: str, dclass: str) -> dict[str, float]:
    """Return ``backend -> best FPS`` for ``model`` on devices of class ``dclass``.

    The file is read leniently: a missing or malformed ``bench.json`` (or malformed rows) never
    stops the runtime, it only means "no measurements" (malformed files log one WARNING).
    """
    try:
        raw = json.loads(Path(bench_path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        _LOG.warning("ignoring unreadable benchmark file %s: %s", bench_path, exc)
        return {}
    rows = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return {}
    best: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("model") != model:
            continue
        backend, device, fps = row.get("backend"), row.get("device"), row.get("fps")
        if not isinstance(backend, str) or not isinstance(device, str):
            continue
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            continue
        if not math.isfinite(fps) or fps <= 0 or device_class(device) != dclass:
            continue
        best[backend] = max(best.get(backend, 0.0), float(fps))
    return best


def choose_backend(
    requested: str,
    entry: RegistryEntry,
    device: str,
    bench_path: Path = DEFAULT_BENCH_PATH,
    *,
    root: Path | None = None,
) -> tuple[str, Path]:
    """Pick the backend for ``entry`` on ``device`` and return ``(backend, artefact path)``.

    * An explicit backend is used when available; otherwise ``pt`` is used and one WARNING is
      logged.
    * ``auto`` picks the available backend with the highest FPS measured for this model on this
      device class (cpu, cuda or mps) in ``bench_path``; without measurements it picks ``pt``.
      Equal FPS prefer the earlier backend of :data:`BACKENDS`.

    Raises:
        ValueError: For an unknown backend name.
        RegistryError: When the entry has no runnable backend at all (e.g. a joblib model).
    """
    if requested != "auto" and requested not in BACKENDS:
        raise ValueError(f"unknown backend {requested!r}; expected auto or one of {BACKENDS}")
    base = Path.cwd() if root is None else Path(root)
    runnable = _runnable_artifacts(entry, device, base)
    if not runnable:
        raise RegistryError(f"model {entry.name!r} (format {entry.format}) has no image backend")
    available = list(runnable)
    fallback = "pt" if "pt" in runnable else available[0]
    if requested == "auto":
        fps = read_bench_fps(bench_path, entry.name, device_class(device))
        measured = [b for b in available if b in fps]
        chosen = max(measured, key=lambda b: (fps[b], -BACKENDS.index(b))) if measured else fallback
    elif requested in runnable:
        chosen = requested
    else:
        _LOG.warning(
            "backend %r is not available for model %s on %s (available: %s)%s; using %s",
            requested,
            entry.name,
            device,
            ", ".join(available),
            _unavailable_hint(requested, device),
            fallback,
        )
        chosen = fallback
    return chosen, runnable[chosen]


def _unavailable_hint(backend: str, device: str) -> str:
    """Why ``backend`` cannot run on ``device`` when its runtime is device-bound, else ''."""
    dclass = device_class(device)
    if backend == "openvino" and dclass != "cpu":
        return "; OpenVINO runs on the CPU, set models.device: cpu to use it"
    if backend == "onnx" and dclass != "cpu":
        provider = _ONNX_DEVICE_PROVIDERS[dclass]
        return f"; ONNX Runtime has no {provider} here, set models.device: cpu to use it"
    if backend == "tensorrt" and dclass != "cuda":
        return "; TensorRT needs a CUDA device"
    return ""


def half_keyword() -> str:
    """Return the predict argument for FP16 accepted by the installed Ultralytics.

    Ultralytics 8.4 replaced ``half=True`` (now deprecated, it logs a warning) with
    ``quantize=16`` (D-009); older releases only know ``half``.
    """
    return "quantize" if "quantize" in _default_cfg_keys() else "half"


def precision_kwargs(device: str, backend: str, half: bool) -> dict[str, Any]:
    """Return the predict kwargs for half precision: FP16 on CUDA only.

    Nothing is passed on CPU/MPS, when ``half`` is off, or for OpenVINO (which runs on the CPU
    and would only round its inputs). ONNX and TensorRT take their precision from the artefact.
    """
    if not half or device_class(device) != "cuda" or backend == "openvino":
        return {}
    if half_keyword() == "quantize":
        return {"quantize": _HALF_FP16}
    return {"half": True}


def load_yolo(path: Path, task: str) -> Any:
    """Load Ultralytics ``YOLO`` weights (any exported format) for ``task``."""
    from ultralytics import YOLO  # noqa: PLC0415  (lazy: keeps ultralytics inside inference/)

    return YOLO(str(path), task=task)


def tensor_to_list(value: Any) -> list[Any]:
    """Convert a torch tensor, numpy array or sequence into (nested) Python lists.

    ``tolist()`` copies CUDA tensors to the host, so nothing tensor-like leaves inference/.
    """
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(value)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A registry entry loaded on a backend and device; ``model`` is the Ultralytics object."""

    entry: RegistryEntry
    backend: str
    path: Path
    device: str
    model: Any

    @property
    def labels(self) -> tuple[str | None, ...]:
        """Class index -> unified label (None = dropped), from the registry entry."""
        return tuple(self.entry.classes)

    def info(self, classes: tuple[str, ...] | None = None) -> ModelInfo:
        """Return the :class:`ModelInfo` of this model (``classes`` default: its labels)."""
        return ModelInfo(
            name=self.entry.name,
            version=self.entry.version,
            family=self.entry.family,
            backend=self.backend,
            device=self.device,
            classes=self.entry.labels if classes is None else classes,
        )


def pin_device(model: Any, device: str) -> None:
    """Make ``device`` the default device of an Ultralytics model object.

    Ultralytics builds a throw-away predictor on its own default device (CUDA when present) when
    ``model.names`` is read on an exported model before the first prediction; on a CUDA machine
    the ONNX backend then requires ``onnxruntime-gpu`` and tries to pip-install it. Setting
    ``overrides["device"]`` makes that predictor (and any call without ``device=``) use the
    device the model was loaded for. Objects without an ``overrides`` dict are left untouched.
    """
    overrides = getattr(model, "overrides", None)
    if isinstance(overrides, dict):
        overrides["device"] = device


def _check_class_count(entry: RegistryEntry, model: Any) -> None:
    """The registry's class mapping must cover exactly the model's classes."""
    names = getattr(model, "names", None)
    if names is None:
        return
    count = len(class_names(names))
    if count != len(entry.classes):
        raise RegistryError(
            f"model {entry.name!r}: the weights have {count} classes but the registry maps "
            f"{len(entry.classes)}; re-register the model"
        )


def load_model(
    entry: RegistryEntry,
    registry: ModelRegistry,
    *,
    backend: str = "auto",
    device: str = "cpu",
    task: str = "detect",
    loader: ModelLoader | None = None,
    bench_path: Path | None = None,
    verify: bool = True,
) -> LoadedModel:
    """Choose a backend for ``entry`` and load it.

    ``bench_path`` defaults to ``bench.json`` next to the registry file. With ``verify`` the
    weights' SHA-256 is checked against the registry; a mismatch logs a WARNING (the class count
    is always checked and a mismatch raises).

    Raises:
        ValueError: For an unknown backend name.
        RegistryError: No runnable backend, or a class-count mismatch.
        FileNotFoundError: When the weights are missing.
    """
    bench = bench_path if bench_path is not None else registry.path.with_name("bench.json")
    chosen, path = choose_backend(backend, entry, device, bench, root=registry.root)
    if not path.exists():
        raise FileNotFoundError(f"weights of {entry.name!r} not found at {path}")
    if verify and not registry.verify(entry, chosen):
        _LOG.warning("sha256 of %s does not match the registry entry %s", path, entry.name)
    model = (loader or load_yolo)(path, task)
    pin_device(model, device)
    _check_class_count(entry, model)
    _LOG.info("loaded model %s (%s) on %s from %s", entry.name, chosen, device, path)
    return LoadedModel(entry=entry, backend=chosen, path=path, device=device, model=model)
