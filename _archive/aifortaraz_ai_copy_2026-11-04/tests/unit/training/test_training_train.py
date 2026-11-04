"""Training: recipe defaults, smoke preset, dataset derivation, cache rule, time box, registration.

Ultralytics is replaced by a fake model whose ``train`` replays the trainer's callback sequence
(epoch start, batches, epoch end, validation, fit-epoch end) on a FakeClock, so the time box and
the post-training steps (copy, sha256, registry, summary) run deterministically in milliseconds.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from safety_ai.clock import FakeClock
from safety_ai.inference.registry import (
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    sha256_file,
)
from safety_ai.inference.types import TAXONOMY
from safety_ai.training.synthetic import generate_synthetic_dataset
from safety_ai.training.train import (
    DEFAULT_DATA_YAML,
    OVERSAMPLED_LIST,
    SUMMARY_NAME,
    DetectionMetrics,
    TimeBox,
    TrainConfig,
    base_candidates,
    choose_cache,
    default_model_name,
    default_workers,
    estimate_decoded_bytes,
    fits_another_epoch,
    gpu_notebook_commands,
    label_path,
    list_images,
    load_dataset_spec,
    parse_family_variant,
    prepare_dataset,
    reproduce_command,
    resolve_base,
    resolve_device,
    smoke_config,
    train,
    ultralytics_device,
    ultralytics_train_args,
    unique_model_name,
)

# ------------------------------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------------------------------


class FakeTrainer:
    """The attributes of an Ultralytics trainer that the time box and train() use."""

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.args = SimpleNamespace(**kwargs)
        self.epochs: int = kwargs.get("epochs", 3)
        self.epoch = 0
        self.start_epoch = 0
        self.stop = False
        save_dir = Path(kwargs.get("project", ".")) / kwargs.get("name", "run")
        self.best = save_dir / "weights" / "best.pt"
        self.last = save_dir / "weights" / "last.pt"
        self.scheduler = SimpleNamespace(last_epoch=-1)
        self.scheduler_resets = 0
        self.stopper = SimpleNamespace(best_epoch=0, patience=kwargs.get("patience", 15))

    def _setup_scheduler(self) -> None:
        self.scheduler_resets += 1
        self.scheduler = SimpleNamespace(last_epoch=-1)


class FakeYolo:
    """Replays Ultralytics' callback sequence; durations come from the FakeClock."""

    def __init__(
        self,
        weights: str,
        clock: FakeClock,
        *,
        setup_s: float = 5.0,
        batch_s: float = 10.0,
        batches: int = 3,
        val_s: float = 4.0,
        inflate_epochs_to: int | None = None,
    ) -> None:
        self.weights = weights
        self.clock = clock
        self.setup_s = setup_s
        self.batch_s = batch_s
        self.batches = batches
        self.val_s = val_s
        self.inflate_epochs_to = inflate_epochs_to
        self.callbacks: dict[str, list[Callable[[Any], None]]] = {}
        self.trainer: FakeTrainer | None = None
        self.kwargs: dict[str, Any] = {}

    def add_callback(self, event: str, func: Callable[[Any], None]) -> None:
        self.callbacks.setdefault(event, []).append(func)

    def _run(self, event: str) -> None:
        for func in self.callbacks.get(event, []):
            func(self.trainer)

    def train(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        self.clock.advance(self.setup_s)
        trainer = self.trainer = FakeTrainer(kwargs)
        self._run("on_train_start")
        epoch = 0
        while True:
            trainer.epoch = epoch
            self._run("on_train_epoch_start")
            for _ in range(self.batches):
                self.clock.advance(self.batch_s)
                self._run("on_train_batch_end")
                if trainer.stop:
                    break
            self._run("on_train_epoch_end")
            self.clock.advance(self.val_s)
            trainer.best.parent.mkdir(parents=True, exist_ok=True)
            trainer.best.write_bytes(f"best weights after epoch {epoch}".encode())
            trainer.last.write_bytes(f"last weights after epoch {epoch}".encode())
            if epoch + 1 >= trainer.epochs:
                trainer.stop = True
            if self.inflate_epochs_to is not None:
                trainer.epochs = self.inflate_epochs_to
            self._run("on_fit_epoch_end")
            if trainer.stop:
                break
            epoch += 1
        self.clock.advance(self.val_s)
        return SimpleNamespace(box=SimpleNamespace(map50=0.61, map=0.42, mp=0.7, mr=0.55))


def _box_trainer(epochs: int = 80, **extra: Any) -> FakeTrainer:
    return FakeTrainer({"epochs": epochs, "time": 3.0, "patience": 15, **extra})


def _write_image(path: Path, width: int = 40, height: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".jpg", np.full((height, width, 3), 127, np.uint8))
    assert ok
    path.write_bytes(data.tobytes())
    return path


def _mini_dataset(root: Path, *, train: int = 6, val: int = 2, test: int = 2) -> Path:
    for split, count in (("train", train), ("val", val), ("test", test)):
        for i in range(count):
            _write_image(root / "images" / split / f"{split}_{i}.jpg")
            label = root / "labels" / split / f"{split}_{i}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": list(TAXONOMY),
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


# ------------------------------------------------------------------------------------------
# Recipe and presets
# ------------------------------------------------------------------------------------------


def test_train_config_defaults_equal_the_spec_recipe() -> None:
    config = TrainConfig()
    assert config.recipe() == {
        "imgsz": 640,
        "epochs": 80,
        "patience": 15,
        "batch": -1,
        "optimizer": "auto",
        "cos_lr": True,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "flipud": 0.0,
        "degrees": 5.0,
        "scale": 0.5,
        "mixup": 0.1,
        "cache": "auto",
        "seed": 0,
        "deterministic": True,
    }
    assert config.base is None
    assert config.device == "auto"
    assert config.data == DEFAULT_DATA_YAML == Path("data/merged/data.yaml")
    assert config.project == Path("runs")
    assert config.time_box_min == 180.0
    assert config.workers is None
    assert config.use_oversampled
    assert config.registry_path == Path("models/registry.json")
    assert config.train_subset is None
    assert config.val_subset is None
    assert config.tag == ""


def test_workers_default_is_min_of_8_and_cpu_count() -> None:
    assert default_workers() == min(8, os.cpu_count() or 1)


def test_smoke_preset_matches_the_spec() -> None:
    config = smoke_config()
    assert (config.epochs, config.imgsz, config.device) == (2, 416, "cpu")
    assert config.train_subset == 500
    assert config.val_subset == 100
    assert config.time_box_min == 20.0
    assert config.batch == 16
    assert config.tag == "smoke"
    assert smoke_config(time_box_min=5.0, project=Path("x")).time_box_min == 5.0
    with pytest.raises(TypeError, match="unknown"):
        smoke_config(nonsense=1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"imgsz": 100},
        {"epochs": 0},
        {"batch": 0},
        {"batch": -2},
        {"time_box_min": 0},
        {"train_subset": 0},
        {"workers": -1},
        {"tag": "Smoke Run"},
        {"cache": "gpu"},
        {"patience": -1},
    ],
)
def test_train_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


def test_ultralytics_args_carry_the_recipe_and_the_time_budget(tmp_path: Path) -> None:
    config = TrainConfig(time_box_min=110)
    run_dir = tmp_path / "runs" / "ppe_s_20261104"
    args = ultralytics_train_args(
        config,
        data_yaml=tmp_path / "data.yaml",
        device="cuda:0",
        cache="ram",
        workers=8,
        run_dir=run_dir,
    )
    for key, value in config.recipe().items():
        if key != "cache":
            assert args[key] == value, key
    assert args["time"] == pytest.approx(110 / 60)
    assert args["cache"] == "ram"
    assert args["device"] == "0"
    assert args["workers"] == 8
    assert Path(args["project"]).is_absolute()
    assert Path(args["project"]) / args["name"] == run_dir.resolve()
    assert args["exist_ok"] is True
    assert args["data"] == str(tmp_path / "data.yaml")


# ------------------------------------------------------------------------------------------
# Device and base weights
# ------------------------------------------------------------------------------------------


def test_resolve_device() -> None:
    assert resolve_device("auto", cuda_available=True) == "cuda:0"
    assert resolve_device("auto", cuda_available=False) == "cpu"
    assert resolve_device("cuda") == "cuda:0"
    assert resolve_device("1") == "cuda:1"
    assert resolve_device("CPU") == "cpu"
    assert ultralytics_device("cuda:1") == "1"
    assert ultralytics_device("cpu") == "cpu"


def test_base_candidates_follow_the_device_and_fall_back_to_yolo11() -> None:
    assert base_candidates("cuda:0", yolo26_supported=True) == ["yolo26s.pt", "yolo11s.pt"]
    assert base_candidates("cpu", yolo26_supported=True) == ["yolo26n.pt", "yolo11n.pt"]
    assert base_candidates("mps", yolo26_supported=True) == ["yolo26n.pt", "yolo11n.pt"]
    assert base_candidates("cpu", yolo26_supported=False) == ["yolo11n.pt"]
    assert base_candidates("cpu")[0] == "yolo26n.pt"  # the installed Ultralytics ships YOLO26


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("yolo26s.pt", ("yolo26", "s")),
        ("yolo11n.pt", ("yolo11", "n")),
        ("weights/yolov8m.pt", ("yolov8", "m")),
        ("yolo26n-pose.pt", ("yolo26", "n")),
        ("ppe_s_20261104.pt", None),
        ("yolo26super.pt", None),
    ],
)
def test_parse_family_variant(name: str, expected: tuple[str, str] | None) -> None:
    assert parse_family_variant(name) == expected


def test_resolve_base_prefers_local_weights_and_registry_names(
    tmp_path: Path, in_tmp_cwd: Path
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26n.pt").write_bytes(b"w")
    local = resolve_base("yolo26n.pt", models_dir=models, registry=None)
    assert Path(local.weights) == models / "yolo26n.pt"
    assert (local.family, local.variant) == ("yolo26", "n")
    remote = resolve_base("yolo26s.pt", models_dir=models, registry=None)
    assert remote.weights == "yolo26s.pt"
    own = resolve_base(str(models / "ppe_s_20261104.pt"), models_dir=models, registry=None)
    assert (own.family, own.variant) == ("custom", "s")

    registry = ModelRegistry(models / "registry.json")
    weights = models / "ppe_s_20261104.pt"
    weights.write_bytes(b"trained")
    registry.add(_entry(registry, weights, name="ppe_s_20261104", variant="s", family="yolo26"))
    venue = resolve_base("ppe_s_latest", models_dir=models, registry=registry)
    assert Path(venue.weights).resolve() == weights.resolve()
    assert (venue.family, venue.variant) == ("yolo26", "s")


def _entry(registry: ModelRegistry, weights: Path, **fields: Any) -> RegistryEntry:
    data = {
        "role": "ppe",
        "path": registry.relative_path(weights),
        "format": "pt",
        "classes": list(TAXONOMY),
        "imgsz": 640,
        "trained_on": "test",
        "created_at": datetime(2026, 11, 4, tzinfo=timezone.utc),
        "sha256": sha256_file(weights),
        **fields,
    }
    return RegistryEntry.model_validate(data)


# ------------------------------------------------------------------------------------------
# Dataset description, oversampled list, subsets
# ------------------------------------------------------------------------------------------


def test_load_dataset_spec_resolves_splits_relative_to_the_yaml(tmp_path: Path) -> None:
    yaml_path = _mini_dataset(tmp_path / "ds")
    spec = load_dataset_spec(yaml_path)
    assert spec.names == TAXONOMY
    assert spec.root == (tmp_path / "ds").resolve()
    assert spec.sources("train") == ((tmp_path / "ds" / "images" / "train").resolve(),)
    assert len(list_images(spec.sources("train"))) == 6
    assert spec.sources("missing") == ()


def test_load_dataset_spec_accepts_a_path_key_lists_and_name_mappings(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _mini_dataset(root)
    listing = root / "val_list.txt"
    listing.write_text("./images/val/val_0.jpg\n\n./images/val/val_1.jpg\n", encoding="utf-8")
    other = tmp_path / "elsewhere" / "data.yaml"
    other.parent.mkdir()
    other.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": ["images/train", "images/test"],
                "val": "val_list.txt",
                "names": dict(enumerate(TAXONOMY)),
            }
        ),
        encoding="utf-8",
    )
    spec = load_dataset_spec(other)
    assert spec.root == root.resolve()
    assert len(list_images(spec.sources("train"))) == 8
    assert [p.name for p in list_images(spec.sources("val"))] == ["val_0.jpg", "val_1.jpg"]
    assert spec.names == TAXONOMY


def test_load_dataset_spec_requires_train_and_val(tmp_path: Path) -> None:
    bad = tmp_path / "data.yaml"
    bad.write_text("train: images/train\nnames: [a]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'val' is required"):
        load_dataset_spec(bad)


def test_label_path_replaces_the_last_images_folder() -> None:
    image = Path("data") / "images" / "x" / "images" / "train" / "a.jpg"
    assert label_path(image) == Path("data") / "images" / "x" / "labels" / "train" / "a.txt"


def test_oversampled_list_becomes_the_train_entry_of_a_derived_yaml(tmp_path: Path) -> None:
    yaml_path = _mini_dataset(tmp_path / "merged")
    lines = ["./images/train/train_0.jpg", "./images/train/train_1.jpg"] * 3
    lines += [f"./images/train/train_{i}.jpg" for i in range(2, 6)]
    (tmp_path / "merged" / OVERSAMPLED_LIST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    work = tmp_path / "run" / "dataset"

    prepared = prepare_dataset(load_dataset_spec(yaml_path), work)

    assert prepared.derived
    assert prepared.data_yaml == work / "data.yaml"
    derived = yaml.safe_load(prepared.data_yaml.read_text(encoding="utf-8"))
    list_file = (tmp_path / "merged" / OVERSAMPLED_LIST).resolve()
    assert derived["train"] == [list_file.as_posix()]
    assert derived["val"] == [(tmp_path / "merged" / "images" / "val").resolve().as_posix()]
    assert derived["test"] == [(tmp_path / "merged" / "images" / "test").resolve().as_posix()]
    assert derived["names"] == dict(enumerate(TAXONOMY))
    assert derived["nc"] == len(TAXONOMY)
    assert (prepared.train_entries, prepared.train_images, prepared.val_images) == (10, 6, 2)
    assert len(prepared.cache_candidates) == 10 + 2
    assert "10 entries in train_oversampled.txt" in prepared.description
    assert "val 2 images" in prepared.description
    assert "test 2 images" in prepared.description
    reread = load_dataset_spec(prepared.data_yaml)
    assert len(list_images(reread.sources("train"))) == 10
    assert len(list_images(reread.sources("train"), unique=True)) == 6


def test_without_an_oversampled_list_the_original_yaml_is_used(tmp_path: Path) -> None:
    yaml_path = _mini_dataset(tmp_path / "merged")
    prepared = prepare_dataset(load_dataset_spec(yaml_path), tmp_path / "work")
    assert not prepared.derived
    assert prepared.data_yaml == yaml_path.resolve()
    assert prepared.oversampled_list is None
    assert not (tmp_path / "work").exists()
    assert prepared.description.endswith("train 6 images, val 2 images, test 2 images")


def test_oversampled_list_is_ignored_when_disabled(tmp_path: Path) -> None:
    yaml_path = _mini_dataset(tmp_path / "merged")
    (tmp_path / "merged" / OVERSAMPLED_LIST).write_text("./images/train/train_0.jpg\n")
    prepared = prepare_dataset(load_dataset_spec(yaml_path), tmp_path / "w", use_oversampled=False)
    assert not prepared.derived


def test_oversampled_list_with_missing_files_is_rejected(tmp_path: Path) -> None:
    yaml_path = _mini_dataset(tmp_path / "merged")
    (tmp_path / "merged" / OVERSAMPLED_LIST).write_text("./images/train/gone.jpg\n")
    with pytest.raises(ValueError, match="missing files"):
        prepare_dataset(load_dataset_spec(yaml_path), tmp_path / "w")


def test_subsets_are_seeded_and_written_as_lists(tmp_path: Path) -> None:
    yaml_path = _mini_dataset(tmp_path / "merged", train=8, val=4)
    spec = load_dataset_spec(yaml_path)
    first = prepare_dataset(spec, tmp_path / "a", train_subset=3, val_subset=1, seed=4)
    second = prepare_dataset(spec, tmp_path / "b", train_subset=3, val_subset=1, seed=4)
    other = prepare_dataset(spec, tmp_path / "c", train_subset=3, val_subset=1, seed=5)
    a = (tmp_path / "a" / "train_subset.txt").read_text()
    assert a == (tmp_path / "b" / "train_subset.txt").read_text()
    assert len(a.splitlines()) == 3
    assert (first.train_images, first.train_total, first.val_images, first.val_total) == (
        3,
        8,
        1,
        4,
    )
    assert "train 3 of 8 images (seeded random subset)" in first.description
    assert "val 1 of 4 images (seeded random subset)" in first.description
    derived = yaml.safe_load(first.data_yaml.read_text(encoding="utf-8"))
    assert derived["train"] == [(tmp_path / "a" / "train_subset.txt").resolve().as_posix()]
    assert second.description == first.description
    assert other.train_images == 3
    assert (tmp_path / "c" / "train_subset.txt").read_text() != a


# ------------------------------------------------------------------------------------------
# RAM cache rule
# ------------------------------------------------------------------------------------------


def test_estimate_decoded_bytes_resizes_the_long_side_to_imgsz(tmp_path: Path) -> None:
    images = [_write_image(tmp_path / f"{i}.jpg", width=100, height=50) for i in range(3)]
    assert estimate_decoded_bytes(images, 64) == 3 * 64 * 32 * 3
    assert estimate_decoded_bytes(images * 2, 64) == 6 * 64 * 32 * 3
    assert estimate_decoded_bytes([], 640) == 0
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    assert estimate_decoded_bytes([broken], 64) == 64 * 64 * 3


def test_choose_cache_uses_ram_only_below_half_of_the_available_memory() -> None:
    gib = 1 << 30
    assert choose_cache("auto", 3 * gib, 8 * gib)[0] == "ram"
    value, note = choose_cache("auto", 4 * gib, 8 * gib)
    assert value is False
    assert "50 %" in note
    assert choose_cache("ram", 10 * gib, gib)[0] == "ram"
    assert choose_cache(True, 10 * gib, gib)[0] == "ram"
    assert choose_cache("disk", 0, 0)[0] == "disk"
    assert choose_cache(False, 0, 10 * gib)[0] is False


# ------------------------------------------------------------------------------------------
# Time box
# ------------------------------------------------------------------------------------------


def _epoch(
    box: TimeBox,
    trainer: FakeTrainer,
    clock: FakeClock,
    *,
    batches: int,
    batch_s: float,
    val_s: float,
) -> None:
    box.on_train_epoch_start(trainer)
    for _ in range(batches):
        clock.advance(batch_s)
        box.on_train_batch_end(trainer)
        if trainer.stop:
            break
    box.on_train_epoch_end(trainer)
    clock.advance(val_s)
    box.on_fit_epoch_end(trainer)


def test_fits_another_epoch() -> None:
    assert fits_another_epoch(100.0, 60.0, 40.0)
    assert not fits_another_epoch(99.9, 60.0, 40.0)


def test_time_box_stops_when_the_next_epoch_cannot_fit(fake_clock: FakeClock) -> None:
    box = TimeBox(600.0, max_epochs=80, clock=fake_clock)
    trainer = _box_trainer()
    box.on_train_start(trainer)
    epochs = 0
    while not trainer.stop:
        trainer.epoch = epochs
        _epoch(box, trainer, fake_clock, batches=10, batch_s=10.0, val_s=20.0)
        epochs += 1
    # 120 s per epoch, reserve 1.5 x 20 s: after 4 epochs 120 s remain < 120 + 30.
    assert epochs == 4
    assert box.epochs_completed == 4
    assert box.mean_epoch_s == pytest.approx(120.0)
    assert box.reserve_s == pytest.approx(30.0)
    assert box.stopped_by_box
    assert box.stop_reason is not None
    assert box.stop_reason.startswith("time box: 120 s left < mean epoch 120 s + reserve 30 s")
    assert box.elapsed_s + box.reserve_s <= box.budget_s


def test_time_box_stops_mid_epoch_when_the_final_validation_would_not_fit(
    fake_clock: FakeClock,
) -> None:
    box = TimeBox(100.0, max_epochs=80, clock=fake_clock)
    trainer = _box_trainer()
    box.on_train_start(trainer)
    box.on_train_epoch_start(trainer)
    stopped_after = 0
    for batch in range(1, 11):
        fake_clock.advance(20.0)
        box.on_train_batch_end(trainer)
        if trainer.stop:
            stopped_after = batch
            break
    assert stopped_after == 5  # 100 s elapsed: remaining 0 s <= reserve (5 % of the box)
    assert box.stopped_by_box
    assert "mid-epoch" in (box.stop_reason or "")


def test_time_box_stops_after_max_epochs(fake_clock: FakeClock) -> None:
    box = TimeBox(10_000.0, max_epochs=2, clock=fake_clock)
    trainer = _box_trainer(epochs=2)
    box.on_train_start(trainer)
    for epoch in range(2):
        trainer.epoch = epoch
        _epoch(box, trainer, fake_clock, batches=2, batch_s=1.0, val_s=1.0)
    assert trainer.stop
    assert box.stop_reason == "completed 2 epochs"
    assert not box.stopped_by_box


def test_time_box_caps_ultralytics_time_based_epoch_estimate(fake_clock: FakeClock) -> None:
    box = TimeBox(10_000.0, max_epochs=2, clock=fake_clock)
    trainer = _box_trainer(epochs=2)
    box.on_train_start(trainer)
    trainer.epoch = 0
    trainer.epochs = 300  # what Ultralytics' time mode would plan for a long box
    _epoch(box, trainer, fake_clock, batches=2, batch_s=1.0, val_s=1.0)
    assert trainer.epochs == 2
    assert trainer.args.epochs == 2
    assert trainer.scheduler_resets == 1
    assert trainer.scheduler.last_epoch == 0
    assert not trainer.stop


def test_time_box_tightens_ultralytics_time_to_the_end_of_the_box(fake_clock: FakeClock) -> None:
    box = TimeBox(3600.0, max_epochs=80, clock=fake_clock)
    trainer = _box_trainer()
    fake_clock.advance(300.0)  # dataset scan, caching, AutoBatch
    box.on_train_start(trainer)
    assert trainer.args.time == pytest.approx((3600.0 - 180.0 - 300.0) / 3600.0)
    trainer.epoch = 0
    _epoch(box, trainer, fake_clock, batches=5, batch_s=60.0, val_s=40.0)
    assert box.reserve_s == pytest.approx(60.0)
    assert trainer.args.time == pytest.approx((3600.0 - 60.0 - 300.0) / 3600.0)


def test_time_box_records_ultralytics_early_stopping(fake_clock: FakeClock) -> None:
    box = TimeBox(10_000.0, max_epochs=80, clock=fake_clock)
    trainer = _box_trainer()
    box.on_train_start(trainer)
    trainer.epoch = 19
    trainer.stopper = SimpleNamespace(best_epoch=5, patience=15)
    box.on_train_epoch_start(trainer)
    fake_clock.advance(10.0)
    box.on_train_epoch_end(trainer)
    trainer.stop = True
    box.on_fit_epoch_end(trainer)
    assert box.stop_reason == "early stopping: no improvement for 15 epochs (best epoch 5)"
    assert not box.stopped_by_box


def test_time_box_install_registers_every_callback() -> None:
    model = FakeYolo("w", FakeClock())
    box = TimeBox(60.0, max_epochs=1, clock=FakeClock())
    box.install(model)
    assert sorted(model.callbacks) == sorted(TimeBox.EVENTS)


def test_time_box_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        TimeBox(0.0, max_epochs=1, clock=FakeClock())
    with pytest.raises(ValueError):
        TimeBox(10.0, max_epochs=0, clock=FakeClock())


# ------------------------------------------------------------------------------------------
# Names, commands, metrics
# ------------------------------------------------------------------------------------------


def test_model_names(tmp_path: Path) -> None:
    day = datetime(2026, 11, 4, 23, 59, tzinfo=timezone.utc)
    assert default_model_name("s", day) == "ppe_s_20261104"
    assert default_model_name("n", day, "smoke") == "ppe_n_smoke_20261104"
    (tmp_path / "runs" / "ppe_s_20261104").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "ppe_s_20261104_2.pt").write_bytes(b"x")
    name = unique_model_name(
        "ppe_s_20261104",
        project=tmp_path / "runs",
        models_dir=tmp_path / "models",
        taken=["ppe_s_20261104_3"],
    )
    assert name == "ppe_s_20261104_4"


def test_reproduce_and_notebook_commands() -> None:
    assert (
        reproduce_command(TrainConfig(), base="yolo26s.pt")
        == "safety-ai train --base yolo26s.pt --epochs 80 --imgsz 640 --time-box-min 180"
    )
    assert reproduce_command(TrainConfig(time_box_min=110.5), base="yolo26s.pt").endswith(
        "--time-box-min 110.5"
    )
    assert (
        reproduce_command(smoke_config(), base="yolo26n.pt")
        == "safety-ai train --smoke --base yolo26n.pt --epochs 2 --imgsz 416 --time-box-min 20"
    )
    assert reproduce_command(TrainConfig(data=Path("data/venue/data.yaml"), epochs=10)).endswith(
        "--epochs 10 --imgsz 640 --time-box-min 180 --data data/venue/data.yaml"
    )
    assert reproduce_command(TrainConfig()).startswith("safety-ai train --base yolo26s.pt")
    notebook = gpu_notebook_commands(TrainConfig(), base="yolo26s.pt")
    assert "!pip install -q -e ." in notebook.splitlines()
    assert notebook.splitlines()[-1] == "!" + reproduce_command(TrainConfig(), base="yolo26s.pt")


def test_detection_metrics_from_ultralytics_values() -> None:
    from_box = DetectionMetrics.from_ultralytics(
        SimpleNamespace(box=SimpleNamespace(map50=0.5, map=0.3, mp=0.6, mr=0.4))
    )
    assert from_box == DetectionMetrics(0.5, 0.3, 0.6, 0.4)
    from_dict = DetectionMetrics.from_ultralytics(
        {
            "metrics/precision(B)": 0.6,
            "metrics/recall(B)": 0.4,
            "metrics/mAP50(B)": 0.5,
            "metrics/mAP50-95(B)": 0.3,
        }
    )
    assert from_dict == from_box
    assert DetectionMetrics.from_ultralytics(None) is None
    registry_metrics = from_box.as_registry_metrics("measured: runs/x")
    assert registry_metrics["source"] == "measured: runs/x"
    assert registry_metrics["mAP50"] == 0.5


# ------------------------------------------------------------------------------------------
# Full run with a fake Ultralytics model
# ------------------------------------------------------------------------------------------


@pytest.fixture
def synthetic_yaml(tmp_path: Path) -> Path:
    return generate_synthetic_dataset(tmp_path / "synthetic", n_images=20, seed=0)


def _config(tmp_path: Path, data: Path, **overrides: Any) -> TrainConfig:
    values: dict[str, Any] = {
        "data": data,
        "device": "cpu",
        "epochs": 3,
        "time_box_min": 10.0,
        "project": tmp_path / "runs",
        "models_dir": tmp_path / "models",
        "registry_path": tmp_path / "models" / "registry.json",
        **overrides,
    }
    return TrainConfig(**values)


def test_train_registers_the_model_after_a_fake_run(
    tmp_path: Path, synthetic_yaml: Path, fake_clock: FakeClock, in_tmp_cwd: Path
) -> None:
    loaded: list[str] = []
    models: list[FakeYolo] = []

    def factory(weights: str) -> FakeYolo:
        loaded.append(weights)
        models.append(FakeYolo(weights, fake_clock))
        return models[-1]

    config = _config(tmp_path, synthetic_yaml)
    result = train(config, clock=fake_clock, model_factory=factory, available_ram=1 << 40)

    assert loaded == ["yolo26n.pt"]
    assert result.model_name == "ppe_n_20261103"
    assert result.model_path == (tmp_path / "models" / "ppe_n_20261103.pt").resolve()
    assert result.model_path.read_bytes() == b"best weights after epoch 2"
    assert result.sha256 == sha256_file(result.model_path)
    assert (result.epochs_run, result.epochs_planned) == (3, 3)
    assert result.stop_reason == "completed the 3 epochs Ultralytics planned"
    assert result.metrics == DetectionMetrics(0.61, 0.42, 0.7, 0.55)
    assert result.time_used_s == pytest.approx(5.0 + 3 * (30.0 + 4.0) + 4.0)
    assert result.command == "safety-ai train --base yolo26n.pt --epochs 3 --imgsz 640 " + (
        f"--time-box-min 10 --data {synthetic_yaml.as_posix()}"
    )

    kwargs = models[0].kwargs
    assert kwargs["time"] == pytest.approx(10.0 / 60.0)
    assert kwargs["cache"] == "ram"
    assert kwargs["device"] == "cpu"
    assert Path(kwargs["project"]) / kwargs["name"] == result.run_dir
    assert kwargs["data"] == str(synthetic_yaml.resolve())

    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    entry = registry.resolve("ppe_n_latest")
    assert entry.name == "ppe_n_20261103"
    assert (entry.role, entry.format, entry.variant, entry.family) == ("ppe", "pt", "n", "yolo26")
    assert entry.base == "yolo26n.pt"
    assert entry.classes == list(TAXONOMY)
    assert entry.imgsz == 640
    assert entry.path == "models/ppe_n_20261103.pt"
    assert registry.verify(entry)
    assert entry.metrics["source"] == "measured: runs/ppe_n_20261103"
    assert entry.metrics["mAP50"] == 0.61
    assert entry.metrics["mAP50-95"] == 0.42
    assert entry.metrics["precision"] == 0.7
    assert entry.metrics["recall"] == 0.55
    assert "train 16 images, val 2 images, test 2 images" in entry.trained_on
    assert entry.created_at == datetime(2026, 11, 3, 14, 1, 51, tzinfo=timezone.utc)

    summary = (result.run_dir / SUMMARY_NAME).read_text(encoding="utf-8")
    assert summary.startswith("# Training summary — ppe_n_20261103")
    assert "| 0.6100 | 0.4200 | 0.7000 | 0.5500 |" in summary
    assert result.command in summary
    assert "3 run, at most 3 planned" in summary
    assert "registered as ppe_n_20261103" in summary


def test_train_falls_back_to_yolo11_when_yolo26_cannot_be_loaded(
    tmp_path: Path, synthetic_yaml: Path, fake_clock: FakeClock, in_tmp_cwd: Path
) -> None:
    def factory(weights: str) -> FakeYolo:
        if weights.startswith("yolo26"):
            raise FileNotFoundError("download failed")
        return FakeYolo(weights, fake_clock)

    result = train(_config(tmp_path, synthetic_yaml), clock=fake_clock, model_factory=factory)
    assert (result.base, result.family, result.variant) == ("yolo11n.pt", "yolo11", "n")


def test_train_reports_every_failed_base(tmp_path: Path, synthetic_yaml: Path) -> None:
    def factory(weights: str) -> FakeYolo:
        raise OSError(f"cannot load {weights}")

    with pytest.raises(RuntimeError, match=r"yolo26n\.pt.*yolo11n\.pt"):
        train(_config(tmp_path, synthetic_yaml), model_factory=factory)


def test_time_box_stops_a_fake_run_and_caps_inflated_epochs(
    tmp_path: Path, synthetic_yaml: Path, fake_clock: FakeClock, in_tmp_cwd: Path
) -> None:
    def factory(weights: str) -> FakeYolo:
        return FakeYolo(weights, fake_clock, batch_s=40.0, val_s=20.0, inflate_epochs_to=500)

    config = _config(tmp_path, synthetic_yaml, epochs=80, time_box_min=10.0)
    result = train(config, clock=fake_clock, model_factory=factory)
    # 5 s setup, 140 s per epoch, reserve 30 s: four epochs end at 565 s; 35 s < 170 s remain.
    assert result.epochs_run == 4
    assert result.stop_reason.startswith("time box:")
    assert result.time_used_s <= result.time_box_s


def test_smoke_runs_are_registered_without_a_variant(
    tmp_path: Path, synthetic_yaml: Path, fake_clock: FakeClock, in_tmp_cwd: Path
) -> None:
    config = smoke_config(
        data=synthetic_yaml,
        project=tmp_path / "runs",
        models_dir=tmp_path / "models",
        registry_path=tmp_path / "models" / "registry.json",
        imgsz=128,
    )
    result = train(config, clock=fake_clock, model_factory=lambda w: FakeYolo(w, fake_clock))
    assert result.model_name == "ppe_n_smoke_20261103"
    assert result.epochs_run == 2
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    entry = registry.resolve("ppe_n_smoke_20261103")
    assert entry.variant is None
    with pytest.raises(RegistryError):
        registry.resolve("ppe_n_latest")
    assert "train 16 images, val 2 images, test 2 images" in entry.trained_on
    assert len((result.run_dir / "dataset" / "train_subset.txt").read_text().splitlines()) == 16
    assert result.command.startswith("safety-ai train --smoke --base yolo26n.pt --epochs 2")


def test_a_second_run_on_the_same_day_gets_a_new_name(
    tmp_path: Path, synthetic_yaml: Path, fake_clock: FakeClock, in_tmp_cwd: Path
) -> None:
    config = _config(tmp_path, synthetic_yaml, epochs=1)
    first = train(config, clock=fake_clock, model_factory=lambda w: FakeYolo(w, fake_clock))
    second = train(config, clock=fake_clock, model_factory=lambda w: FakeYolo(w, fake_clock))
    assert (first.model_name, second.model_name) == ("ppe_n_20261103", "ppe_n_20261103_2")
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    assert registry.names() == ("ppe_n_20261103", "ppe_n_20261103_2")
    assert registry.resolve("ppe_n_latest").name == "ppe_n_20261103_2"


def test_train_rejects_datasets_with_other_classes(tmp_path: Path) -> None:
    yaml_path = _mini_dataset(tmp_path / "ds")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["names"] = ["person", "helmet"]
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unified taxonomy"):
        train(_config(tmp_path, yaml_path), model_factory=lambda w: FakeYolo(w, FakeClock()))


def test_unregistered_runs_still_copy_the_weights(
    tmp_path: Path, synthetic_yaml: Path, fake_clock: FakeClock, in_tmp_cwd: Path
) -> None:
    config = _config(tmp_path, synthetic_yaml, epochs=1, register=False, cache=False)
    result = train(config, clock=fake_clock, model_factory=lambda w: FakeYolo(w, fake_clock))
    assert result.model_path.is_file()
    assert not result.registered
    assert result.cache == "off (disabled by configuration)"
    assert not (tmp_path / "models" / "registry.json").exists()
