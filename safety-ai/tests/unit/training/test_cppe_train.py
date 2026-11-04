"""Construction-PPE run machinery: run specs, naming, resume, divergence, guard, chain.

Ultralytics is replaced by fakes: the fake trainer replays the callback sequence of
``BaseTrainer._do_train`` (including the extra ``on_fit_epoch_end`` of the final validation) on a
FakeClock, and the fake run processes of the chain are scripted, so every test runs in
milliseconds without a GPU or a dataset.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from safety_ai.clock import FakeClock
from safety_ai.inference.registry import ModelRegistry
from safety_ai.training import remap
from safety_ai.training.progress import (
    BLOCK_END,
    BLOCK_START,
    render_progress_block,
    replace_block,
    write_progress_block,
)
from safety_ai.training.train import (
    CHAIN_STATUS,
    CPPE_ALIAS,
    EXIT_DIVERGED,
    GATE_FILE,
    STUCK_EPOCHS,
    CppeContext,
    CppeHyper,
    CppeRunGuard,
    CppeRunSpec,
    StatusStore,
    best_row,
    cap_cuda_memory,
    checkpoint_state,
    closed_attempts,
    cppe_model_name,
    cppe_run_spec,
    cppe_run_specs,
    cppe_workers,
    is_diverged,
    read_results_csv,
    run_chain,
    run_cppe,
)

T0 = datetime(2026, 11, 4, 18, 0, 0, tzinfo=timezone.utc)
HEADER = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),"
    "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,"
    "val/dfl_loss,lr/pg0,lr/pg1,lr/pg2"
)


# ------------------------------------------------------------------------------------------
# Run specs and naming
# ------------------------------------------------------------------------------------------


def test_run_specs_follow_spec_6_2() -> None:
    specs = cppe_run_specs()
    assert list(specs) == ["r0", "r1", "r2", "r3"]
    r0, r1, r2, r3 = (specs[k] for k in ("r0", "r1", "r2", "r3"))
    assert (r0.base, r0.epochs, r0.imgsz, r0.fraction, r0.batch) == ("yolo26n.pt", 1, 320, 0.1, 8)
    assert r0.time_box_s == pytest.approx(600.0)
    assert r0.data.as_posix() == "data/construction-ppe-unified/data.yaml"
    assert (r1.base, r1.epochs, r1.imgsz, r1.patience, r1.batch) == ("yolo26n.pt", 60, 640, 20, -1)
    assert r1.time_box_s == pytest.approx(2700.0)
    assert r1.data.as_posix() == "datasets/construction-ppe.yaml"
    assert r1.model_classes == remap.SOURCE_NAMES
    assert (r2.base, r2.epochs, r2.imgsz, r2.patience, r2.batch, r2.cache) == (
        "yolo26s.pt",
        150,
        640,
        30,
        -1,
        "disk",
    )
    assert r2.time_box_s == pytest.approx(9000.0)
    assert r2.data.as_posix() == "data/construction-ppe-unified/data_oversampled.yaml"
    assert (r3.base, r3.freeze, r3.time_box_s) == ("yolo26s.pt", 10, pytest.approx(4500.0))
    total_gpu_h = sum(s.time_box_h for s in (r1, r2, r3))
    assert total_gpu_h <= 5.0


def test_train_args_carry_the_common_settings_and_the_6_3_recipe(tmp_path: Path) -> None:
    args = cppe_run_spec("r2").train_args(
        project=tmp_path / "runs", name="ppe_cppe_s_20261104_r2", workers=8, device="0"
    )
    assert args["seed"] == 0 and args["deterministic"] is True and args["plots"] is True
    assert args["save_period"] == 10 and args["exist_ok"] is True and args["amp"] is True
    assert args["time"] == 2.5 and args["epochs"] == 150 and args["workers"] == 8
    assert Path(args["project"]).is_absolute() and args["name"] == "ppe_cppe_s_20261104_r2"
    assert "freeze" not in args
    expected = {
        "optimizer": "auto",
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "close_mosaic": 15,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "erasing": 0.3,
        "label_smoothing": 0.0,
        "nbs": 64,
    }
    assert {k: args[k] for k in expected} == expected == CppeHyper().as_args()
    assert "lr0" not in args
    r3 = cppe_run_spec("R3").train_args(project=tmp_path, name="x", workers=2, device="cpu")
    assert r3["freeze"] == 10
    restart = cppe_run_spec("r1").train_args(
        project=tmp_path, name="x", workers=2, device="0", overrides={"lr0": 0.0003, "amp": False}
    )
    assert restart["lr0"] == 0.0003 and restart["amp"] is False


def test_unknown_runs_and_bad_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown run"):
        cppe_run_spec("r9")
    base = cppe_run_spec("r0")
    with pytest.raises(ValueError, match="imgsz"):
        CppeRunSpec(**{**_spec_fields(base), "imgsz": 300})
    with pytest.raises(ValueError, match="fraction"):
        CppeRunSpec(**{**_spec_fields(base), "fraction": 0.0})
    with pytest.raises(ValueError, match="class_map lacks"):
        CppeRunSpec(**{**_spec_fields(base), "class_map": {}})


def _spec_fields(spec: CppeRunSpec) -> dict[str, Any]:
    return {name: getattr(spec, name) for name in spec.__dataclass_fields__}


def test_model_names_follow_t007() -> None:
    assert cppe_model_name(cppe_run_spec("r0"), T0) == "ppe_cppe_n_20261104_r0"
    assert cppe_model_name(cppe_run_spec("r2"), T0) == "ppe_cppe_s_20261104_r2"
    assert cppe_model_name(cppe_run_spec("r3"), T0) != cppe_model_name(cppe_run_spec("r2"), T0)


def test_registry_classes_map_the_release_and_the_unified_names() -> None:
    assert cppe_run_spec("r1").registry_classes[-1] is None  # no_vest
    assert cppe_run_spec("r1").registry_classes[4] == "glasses"  # goggles
    assert cppe_run_spec("r2").registry_classes == list(remap.UNIFIED_NAMES)


# ------------------------------------------------------------------------------------------
# Resume detection, results.csv, divergence
# ------------------------------------------------------------------------------------------


def test_checkpoint_state_distinguishes_resumable_finished_and_broken(tmp_path: Path) -> None:
    assert checkpoint_state(tmp_path / "missing.pt") == "missing"
    resumable = tmp_path / "last.pt"
    torch.save({"epoch": 4, "optimizer": {"state": {}}}, resumable)
    assert checkpoint_state(resumable) == "resumable"
    finished = tmp_path / "done.pt"
    torch.save({"epoch": -1, "optimizer": None}, finished)
    assert checkpoint_state(finished) == "finished"
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a checkpoint")
    assert checkpoint_state(broken) == "unreadable"


def _write_results(path: Path, rows: list[tuple[float, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER] + [",".join(f"{v:g}" for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(epoch: int, map50: float, map5095: float) -> tuple[float, ...]:
    return (
        epoch,
        10.0 * epoch,
        1.2,
        2.3,
        0.0,
        0.5,
        0.4,
        map50,
        map5095,
        1.1,
        2.0,
        0.0,
        1e-3,
        1e-3,
        1e-3,
    )


def test_results_csv_parsing_and_best_row(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    assert read_results_csv(path) == []
    _write_results(path, [_row(1, 0.2, 0.1), _row(2, 0.4, 0.3), _row(3, 0.35, 0.25)])
    rows = read_results_csv(path)
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3]
    best = best_row(rows)
    assert best is not None and best["epoch"] == 2


@pytest.mark.parametrize(
    ("history", "losses", "diverged"),
    [
        ([0.0] * STUCK_EPOCHS, [1.0, 2.0], True),
        ([0.0] * (STUCK_EPOCHS - 1), [1.0], False),
        ([0.0] * 9 + [0.01], [1.0], False),
        ([0.2], [float("nan"), 1.0], True),
        ([0.2], [1.0, float("inf")], True),
        ([], [], False),
    ],
)
def test_divergence_rules(history: list[float], losses: list[float], diverged: bool) -> None:
    assert (is_diverged(history, losses) is not None) is diverged


def test_open_attempts_count_their_last_reported_time() -> None:
    entry = {
        "attempts": [{"kind": "start", "seconds": 100.0}, {"kind": "resume", "seconds": None}],
        "latest": {"elapsed_attempt_s": 42.0},
    }
    assert [a["seconds"] for a in closed_attempts(entry)] == [100.0, 42.0]
    assert closed_attempts({}) == []


# ------------------------------------------------------------------------------------------
# The run guard on a fake trainer
# ------------------------------------------------------------------------------------------


class FakeTrainer:
    def __init__(self, epochs: int, clock: FakeClock) -> None:
        self.epochs = epochs
        self.epoch = 0
        self.start_epoch = 0
        self.stop = False
        self.args = SimpleNamespace(epochs=epochs, optimizer="AdamW", lr0=0.000667, time=0.75)
        self.amp = False
        self.batch_size = 16
        self.device = "cuda:0"
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        self.tloss: Any = {"box_loss": 1.0, "cls_loss": 2.0, "l1_loss": 0.0}
        self.metrics: dict[str, float] = {}
        self.scheduler = SimpleNamespace(last_epoch=-1)
        self.scheduler_resets = 0
        self.stopper = SimpleNamespace(best_epoch=0, patience=0)
        self.clock = clock

    def _setup_scheduler(self) -> None:
        self.scheduler_resets += 1
        self.scheduler = SimpleNamespace(last_epoch=-1)


def _replay(
    guard: CppeRunGuard,
    trainer: FakeTrainer,
    clock: FakeClock,
    *,
    epoch_s: float,
    replan_to: int | None = None,
    maps: Callable[[int], float] = lambda e: 0.1 + 0.01 * e,
    nan_at: int | None = None,
) -> int:
    """Replay BaseTrainer's loop; return the number of epochs run."""
    guard.on_train_start(trainer)
    epoch = 0
    while True:
        trainer.epoch = epoch
        for _ in range(4):
            clock.advance(epoch_s / 4)
            guard.on_train_batch_end(trainer)
            if trainer.stop:
                break
        trainer.metrics = {
            "metrics/mAP50(B)": maps(epoch) * 1.5,
            "metrics/mAP50-95(B)": maps(epoch),
        }
        box = float("nan") if nan_at == epoch else 1.0
        trainer.tloss = {
            "box_loss": torch.tensor(box),
            "cls_loss": torch.tensor(2.0),
            "l1_loss": torch.tensor(0.0),
        }
        final_epoch = epoch + 1 >= trainer.epochs
        trainer.stop = trainer.stop or final_epoch
        if replan_to is not None:
            trainer.epochs = trainer.args.epochs = replan_to
            trainer.stop = trainer.stop or epoch >= trainer.epochs
        guard.on_fit_epoch_end(trainer)
        if trainer.stop:
            break
        epoch += 1
    trainer.metrics = {"metrics/mAP50(B)": 0.9, "metrics/mAP50-95(B)": 0.6}
    trainer.epoch += 1
    guard.on_fit_epoch_end(trainer)  # final validation of best.pt
    trainer.epoch -= 1
    return epoch + 1


def test_guard_caps_the_time_mode_epoch_replan(fake_clock: FakeClock) -> None:
    trainer = FakeTrainer(60, fake_clock)
    states: list[dict[str, Any]] = []
    guard = CppeRunGuard(budget_s=2700, max_epochs=60, clock=fake_clock, on_state=states.append)
    ran = _replay(guard, trainer, fake_clock, epoch_s=10.0, replan_to=270)
    assert ran == 60
    assert trainer.epochs == 60 and trainer.args.epochs == 60
    assert trainer.scheduler_resets >= 1
    assert guard.epochs_this_attempt == 60
    assert guard.final == {"map50": 0.9, "map50_95": 0.6}
    assert states[0]["setup"] == {
        "optimizer": "AdamW",
        "lr0": 0.000667,
        "amp": False,
        "batch": 16,
        "start_epoch": 0,
        "device": "cuda:0",
    }
    last = states[-2]
    assert last["epoch"] == 60 and last["epochs_planned"] == 60
    assert last["losses"] == [1.0, 2.0, 0.0] and last["map50_95"] == pytest.approx(0.69)


def test_guard_keeps_the_smoke_run_at_one_epoch(fake_clock: FakeClock) -> None:
    trainer = FakeTrainer(1, fake_clock)
    guard = CppeRunGuard(budget_s=600, max_epochs=1, clock=fake_clock)
    assert _replay(guard, trainer, fake_clock, epoch_s=5.0, replan_to=120) == 1
    assert trainer.epochs == 1


def test_guard_enforces_the_cumulative_time_box(fake_clock: FakeClock) -> None:
    trainer = FakeTrainer(60, fake_clock)
    guard = CppeRunGuard(
        budget_s=95.0, max_epochs=60, clock=fake_clock, used_before_s=2605.0, time_box_s=2700.0
    )
    ran = _replay(guard, trainer, fake_clock, epoch_s=40.0)
    assert ran == 3  # stopped mid-epoch 3 when the remaining 95 s ran out
    assert guard.stop_reason is not None and "time box" in guard.stop_reason


def test_guard_stops_a_diverged_run(fake_clock: FakeClock) -> None:
    trainer = FakeTrainer(60, fake_clock)
    guard = CppeRunGuard(budget_s=10_000, max_epochs=60, clock=fake_clock)
    ran = _replay(guard, trainer, fake_clock, epoch_s=5.0, maps=lambda e: 0.0)
    assert ran == STUCK_EPOCHS
    assert guard.divergence is not None and "stuck" in guard.divergence
    nan_trainer = FakeTrainer(60, fake_clock)
    nan_guard = CppeRunGuard(budget_s=10_000, max_epochs=60, clock=fake_clock)
    assert _replay(nan_guard, nan_trainer, fake_clock, epoch_s=5.0, nan_at=2) == 3
    assert nan_guard.divergence is not None and "NaN" in nan_guard.divergence


def test_guard_counts_the_history_of_earlier_attempts(fake_clock: FakeClock) -> None:
    trainer = FakeTrainer(60, fake_clock)
    guard = CppeRunGuard(
        budget_s=10_000, max_epochs=60, clock=fake_clock, history=[0.0] * (STUCK_EPOCHS - 1)
    )
    assert _replay(guard, trainer, fake_clock, epoch_s=5.0, maps=lambda e: 0.0) == 1
    assert guard.divergence is not None


# ------------------------------------------------------------------------------------------
# Status store and PROGRESS.md block
# ------------------------------------------------------------------------------------------


def test_status_store_merges_and_survives_a_stale_lock(tmp_path: Path) -> None:
    store = StatusStore(tmp_path / "chain.json", timeout_s=0.2, sleep=lambda s: None)
    assert store.read() == {"state": "idle", "runs": {}}
    store.update_run("r1", name="ppe_cppe_n_20261104_r1", state="running")
    store.update_run("r1", latest={"epoch": 3})
    assert store.run("r1") == {
        "name": "ppe_cppe_n_20261104_r1",
        "state": "running",
        "latest": {"epoch": 3},
    }
    lock = tmp_path / "chain.json.lock"
    lock.write_text("123", encoding="utf-8")
    with pytest.raises(TimeoutError):
        store.update_run("r1", state="finished")
    old = lock.stat().st_mtime - 1000
    os.utime(lock, (old, old))
    store.update_run("r1", state="finished")
    assert store.run("r1")["state"] == "finished" and not lock.exists()


def test_progress_block_replaces_only_its_markers() -> None:
    text = (
        f"# Progress\r\n\r\nother\r\n\r\n## X\r\n\r\n{BLOCK_START}\r\nold\r\n"
        f"{BLOCK_END}\r\ntail\r\n"
    )
    updated = replace_block(text, "new line 1\nnew line 2")
    assert updated == (
        f"# Progress\r\n\r\nother\r\n\r\n## X\r\n\r\n{BLOCK_START}\r\nnew line 1\r\nnew line 2"
        f"\r\n{BLOCK_END}\r\ntail\r\n"
    )
    appended = replace_block("# Progress\n", "body")
    assert appended.endswith(f"{BLOCK_START}\nbody\n{BLOCK_END}\n")
    assert "## PPE training on Construction-PPE" in appended
    with pytest.raises(ValueError):
        replace_block(f"{BLOCK_START}\n{BLOCK_START}\n{BLOCK_END}\n", "x")
    with pytest.raises(ValueError):
        replace_block(f"{BLOCK_END}\n{BLOCK_START}\n", "x")


def test_progress_writer_rereads_the_file_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "PROGRESS.md"
    write_progress_block(path, "first")
    content = path.read_text(encoding="utf-8")
    path.write_text("# edited by someone else\n\n" + content, encoding="utf-8")
    write_progress_block(path, "second")
    final = path.read_text(encoding="utf-8")
    assert final.startswith("# edited by someone else")
    assert "second" in final and "first" not in final


# ------------------------------------------------------------------------------------------
# run_cppe with a fake Ultralytics
# ------------------------------------------------------------------------------------------


class FakeYolo:
    """Plays one Ultralytics training: writes weights and results.csv, replays callbacks."""

    calls: list[dict[str, Any]]

    def __init__(self, weights: str, clock: FakeClock, script: dict[str, Any]) -> None:
        self.weights = weights
        self.clock = clock
        self.script = script
        self.callbacks: dict[str, list[Callable[[Any], None]]] = {}
        self.names = dict(enumerate(script.get("names", remap.UNIFIED_NAMES)))

    def add_callback(self, event: str, func: Callable[[Any], None]) -> None:
        self.callbacks.setdefault(event, []).append(func)

    def _run(self, event: str, trainer: Any) -> None:
        for func in self.callbacks.get(event, []):
            func(trainer)

    def train(self, **kwargs: Any) -> Any:
        self.script.setdefault("calls", []).append({"weights": self.weights, **kwargs})
        if kwargs.get("resume"):
            run_dir = Path(self.weights).parent.parent
            epochs = self.script["epochs"]
        else:
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            epochs = kwargs["epochs"]
        if self.script.get("crash"):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        trainer = FakeTrainer(epochs, self.clock)
        maps = self.script.get("maps", lambda e: 0.1 + 0.01 * e)
        guard_epochs = 0
        epoch = 0
        for event in ("on_train_start",):
            self._run(event, trainer)
        rows = read_results_csv(run_dir / "results.csv")
        epoch = len(rows)
        while True:
            trainer.epoch = epoch
            self.clock.advance(30.0)
            self._run("on_train_batch_end", trainer)
            trainer.metrics = {
                "metrics/mAP50(B)": maps(epoch) * 1.5,
                "metrics/mAP50-95(B)": maps(epoch),
            }
            rows_text = _row(epoch + 1, maps(epoch) * 1.5, maps(epoch))
            _append_row(run_dir / "results.csv", rows_text)
            trainer.stop = trainer.stop or epoch + 1 >= trainer.epochs
            self._run("on_fit_epoch_end", trainer)
            guard_epochs += 1
            weights = run_dir / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "optimizer": {"s": 1}}, weights / "last.pt")
            torch.save({"epoch": epoch, "optimizer": {"s": 1}}, weights / "best.pt")
            if trainer.stop:
                break
            epoch += 1
        torch.save({"epoch": -1, "optimizer": None}, run_dir / "weights" / "last.pt")
        torch.save({"epoch": -1, "optimizer": None}, run_dir / "weights" / "best.pt")
        for name in ("results.png", "confusion_matrix.png", "BoxPR_curve.png", "BoxF1_curve.png"):
            (run_dir / name).write_bytes(b"png")
        return SimpleNamespace(box=SimpleNamespace(map50=0.55, map=0.33, mp=0.6, mr=0.5))


def _append_row(path: Path, row: tuple[float, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(",".join(f"{v:g}" for v in row) + "\n")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "registry.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "PROGRESS.md").write_text("# Progress\n", encoding="utf-8")
    return tmp_path


def _context(root: Path, clock: FakeClock, script: dict[str, Any]) -> CppeContext:
    return CppeContext.create(
        root,
        clock=clock,
        registry=ModelRegistry(root / "models" / "registry.json"),
        factory=lambda weights: FakeYolo(weights, clock, script),
    )


def test_run_cppe_trains_finalizes_and_never_retrains(project: Path, fake_clock: FakeClock) -> None:
    script: dict[str, Any] = {}
    context = _context(project, fake_clock, script)
    assert run_cppe("r0", context) == 0
    status = StatusStore(project / CHAIN_STATUS).run("r0")
    name = status["name"]
    assert name == "ppe_cppe_n_20261103_r0" and status["state"] == "finished"
    assert status["final"]["mAP50"] == 0.55 and status["final"]["mAP50-95"] == 0.33
    call = script["calls"][0]
    assert call["epochs"] == 1 and call["imgsz"] == 320 and call["fraction"] == 0.1
    assert call["weights"] == "yolo26n.pt"  # no local copy in the test project
    assert (project / "models" / f"{name}.pt").is_file()
    class_map = json.loads((project / "models" / f"{name}_class_map.json").read_text())
    assert class_map == {n: n for n in remap.UNIFIED_NAMES}
    report = project / "reports" / "runs" / name
    assert {p.name for p in report.iterdir()} >= {
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "summary.md",
    }
    entry = ModelRegistry(project / "models" / "registry.json").get(name)
    assert entry is not None and entry.exclude_from_latest and entry.aliases == [CPPE_ALIAS]
    assert entry.training is not None and entry.training["run"] == "r0"
    assert entry.training["epochs_run"] == 1 and entry.metrics["split"] == "val"
    assert entry.classes == list(remap.UNIFIED_NAMES) and entry.imgsz == 320
    # never retrain a finished run
    assert run_cppe("r0", context) == 0
    assert len(script["calls"]) == 1


def test_run_cppe_resumes_an_unfinished_checkpoint(project: Path, fake_clock: FakeClock) -> None:
    script: dict[str, Any] = {"epochs": 3}
    context = _context(project, fake_clock, script)
    name = "ppe_cppe_n_20261103_r1"
    run_dir = project / "runs" / name
    (run_dir / "weights").mkdir(parents=True)
    torch.save({"epoch": 1, "optimizer": {"s": 1}}, run_dir / "weights" / "last.pt")
    _append_row(run_dir / "results.csv", _row(1, 0.3, 0.2))
    _append_row(run_dir / "results.csv", _row(2, 0.4, 0.25))
    StatusStore(project / CHAIN_STATUS).update_run(
        "r1",
        name=name,
        state="crashed",
        attempts=[{"kind": "start", "seconds": 600.0}],
    )
    script["names"] = remap.SOURCE_NAMES
    assert run_cppe("r1", context) == 0
    call = script["calls"][0]
    assert call["resume"] is True and call["weights"].endswith("last.pt")
    status = StatusStore(project / CHAIN_STATUS).run("r1")
    assert [a["kind"] for a in status["attempts"]] == ["start", "resume"]
    assert status["time_used_s"] == pytest.approx(600.0 + status["attempts"][1]["seconds"])
    entry = ModelRegistry(project / "models" / "registry.json").get(name)
    assert entry is not None and entry.classes[-1] is None and entry.aliases == []


def test_run_cppe_restarts_a_diverged_run_with_half_lr_and_no_amp(
    project: Path, fake_clock: FakeClock
) -> None:
    script: dict[str, Any] = {"maps": lambda e: 0.0}
    context = _context(project, fake_clock, script)
    assert run_cppe("r2", context) == EXIT_DIVERGED
    status = StatusStore(project / CHAIN_STATUS)
    entry = status.run("r2")
    assert entry["state"] == "diverged" and "stuck" in entry["divergence"]
    status.update_run("r2", restart_pending=True)
    script["maps"] = lambda e: 0.2
    assert run_cppe("r2", context) == 0
    restart = script["calls"][1]
    assert restart["lr0"] == pytest.approx(0.000667 / 2) and restart["amp"] is False
    assert restart["optimizer"] == "AdamW"
    assert (project / "runs" / f"{entry['name']}_diverged").is_dir()
    final = status.run("r2")
    assert final["state"] == "finished" and final["restarted_for_divergence"] is True


def test_run_cppe_records_a_crash_with_its_traceback(project: Path, fake_clock: FakeClock) -> None:
    script: dict[str, Any] = {"crash": True}
    context = _context(project, fake_clock, script)
    assert run_cppe("r3", context) == 1
    entry = StatusStore(project / CHAIN_STATUS).run("r3")
    assert entry["state"] == "crashed"
    traceback_file = project / entry["traceback"]
    assert "illegal memory access" in traceback_file.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------
# The chain
# ------------------------------------------------------------------------------------------


class FakeProcess:
    def __init__(self, pid: int, polls: int, code: int, on_exit: Callable[[], None]) -> None:
        self.pid = pid
        self._polls = polls
        self._code = code
        self._on_exit = on_exit

    def poll(self) -> int | None:
        if self._polls > 0:
            self._polls -= 1
            return None
        if self._on_exit is not None:
            self._on_exit()
            self._on_exit = None
        return self._code


def test_chain_skips_finished_runs_waits_for_the_gate_and_restarts_divergence(
    project: Path, fake_clock: FakeClock
) -> None:
    store = StatusStore(project / CHAIN_STATUS)
    store.update_run("r1", name="ppe_cppe_n_20261103_r1", state="finished")
    launched: list[str] = []
    outcomes = {"r2": [(EXIT_DIVERGED, "diverged"), (0, "finished")], "r3": [(0, "finished")]}

    def launch(run_id: str, root: Path) -> FakeProcess:
        launched.append(run_id)
        code, state = outcomes[run_id].pop(0)

        def finish() -> None:
            values: dict[str, Any] = {"state": state}
            if state == "diverged":
                values["divergence"] = "mAP50-95 stuck at 0"
            store.update_run(run_id, **values)

        return FakeProcess(1000 + len(launched), 2, code, finish)

    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            (project / GATE_FILE).parent.mkdir(parents=True, exist_ok=True)
            (project / GATE_FILE).write_text("{}", encoding="utf-8")

    context = _context(project, fake_clock, {})
    assert run_chain(context, launch=launch, sleep=sleep, poll_s=60.0) == 0
    assert launched == ["r2", "r2", "r3"]
    assert sleeps[:3] == [60.0, 60.0, 60.0]
    status = store.read()
    assert status["state"] == "finished" and status["current_run"] is None
    assert status["runs"]["r1"]["state"] == "finished"
    assert status["runs"]["r2"]["state"] == "finished"
    assert status["runs"]["r2"]["restart_pending"] is True  # the child clears it when it starts
    progress = (project / "PROGRESS.md").read_text(encoding="utf-8")
    assert BLOCK_START in progress and "state **finished**" in progress


def test_chain_marks_a_second_divergence_failed_and_continues(
    project: Path, fake_clock: FakeClock
) -> None:
    store = StatusStore(project / CHAIN_STATUS)
    (project / GATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    (project / GATE_FILE).write_text("{}", encoding="utf-8")
    store.update_run("r2", restarted_for_divergence=True)

    def launch(run_id: str, root: Path) -> FakeProcess:
        code = EXIT_DIVERGED if run_id == "r2" else 0

        def finish() -> None:
            store.update_run(run_id, state="diverged" if run_id == "r2" else "finished")

        return FakeProcess(7, 0, code, finish)

    context = _context(project, fake_clock, {})
    assert run_chain(context, runs=("r2", "r3"), launch=launch, sleep=lambda s: None) == 0
    status = store.read()
    assert status["runs"]["r2"]["state"] == "failed"
    assert status["runs"]["r3"]["state"] == "finished"


def test_chain_resumes_crashes_then_stops_with_the_traceback(
    project: Path, fake_clock: FakeClock
) -> None:
    store = StatusStore(project / CHAIN_STATUS)
    launches: list[str] = []

    def launch(run_id: str, root: Path) -> FakeProcess:
        launches.append(run_id)

        def crash() -> None:
            store.update_run(
                run_id, state="crashed", traceback="reports/cppe/logs/r1_traceback_1.txt"
            )

        return FakeProcess(9, 0, 1, crash)

    context = _context(project, fake_clock, {})
    assert run_chain(context, runs=("r1",), launch=launch, sleep=lambda s: None) == 1
    assert launches == ["r1", "r1", "r1"]
    status = store.read()
    assert status["state"] == "crashed" and status["current_run"] == "r1"
    assert status["traceback"] == "reports/cppe/logs/r1_traceback_1.txt"


def test_progress_rows_show_epoch_losses_and_eta() -> None:
    block = render_progress_block(
        updated_at="2026-11-04 18:00",
        notes="**Stage:** S3",
        chain={
            "state": "running",
            "current_run": "r1",
            "pid": 42,
            "log": "reports/cppe/logs/chain.log",
            "runs": {
                "r1": {
                    "state": "running",
                    "name": "ppe_cppe_n_20261104_r1",
                    "best_map50_95": None,
                    "latest": {
                        "epoch": 12,
                        "epochs_planned": 60,
                        "losses": [1.2, 2.3, 0.0],
                        "map50": 0.41,
                        "map50_95": 0.22,
                        "elapsed": "6.0 min",
                        "eta": "24.0 min",
                    },
                }
            },
        },
    )
    assert "| r1 | running | ppe_cppe_n_20261104_r1 | 12/60 | 1.200 / 2.300 / 0.000 |" in block
    assert "0.4100 | 0.2200" in block and "24.0 min" in block
    assert not math.isnan(len(block))


def test_worker_count_defaults_to_three_and_honours_the_override() -> None:
    assert cppe_workers({}) == min(3, os.cpu_count() or 1)
    assert cppe_workers({"SAFETY_AI_CPPE_WORKERS": "1"}) == 1
    assert cppe_workers({"SAFETY_AI_CPPE_WORKERS": "64"}) == min(8, os.cpu_count() or 1)
    assert cppe_workers({"SAFETY_AI_CPPE_WORKERS": "many"}) == min(3, os.cpu_count() or 1)


def test_cuda_memory_cap_uses_the_free_share(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, int]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=0: (4 << 30, 6 << 30))
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda f, d=0: calls.append((f, d))
    )
    assert cap_cuda_memory() == {"free_gib": 4.0, "total_gib": 6.0, "fraction": 0.633}
    assert calls == [(pytest.approx(0.95 * 4 / 6), 0)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cap_cuda_memory() is None
