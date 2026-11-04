"""Typer commands of the Construction-PPE training pipeline (docs/TRAINING_SPEC.md 8.3).

Runnable as ``python -m safety_ai.training.cli_cppe <command>``; :func:`register` mounts the same
commands into another Typer application (``safety_ai/cli.py`` calls ``register(app)`` to expose
them as ``safety-ai cppe ...``).

Commands: ``env``; ``data download|verify|prepare|remap|all``; ``train --run r0..r3``;
``chain`` (R1 -> R2 -> R3, ``--detach`` starts it in the background); ``status``; ``progress``;
``gate``; ``eval``; ``export``; ``predict``; ``smoke`` (R0 end to end).
"""

# ruff: noqa: B008 - typer.Option() defaults are the Typer idiom (as in safety_ai/cli.py)
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from safety_ai.training import cppe_data, cppe_prepare
from safety_ai.training.environment import write_environment_report

__all__ = ["app", "data_app", "main", "register"]

app = typer.Typer(
    name="cppe",
    help="Construction-PPE PPE-detector pipeline: data, training runs, evaluation, export.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Construction-PPE dataset: download, verify, prepare/remap.")
app.add_typer(data_app, name="data")

_PROG = "python -m safety_ai.training.cli_cppe"


def _paths(root: Path) -> cppe_data.CppePaths:
    return cppe_data.CppePaths(root.resolve())


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("env")
def env_command(
    out: Path = typer.Option(Path("reports/environment.md"), help="Markdown report path."),
) -> None:
    """Detect GPU/CPU/RAM and the training mode (spec 6.1); write reports/environment.md."""
    env = write_environment_report(out, command=f"{_PROG} env --out {out.as_posix()}")
    typer.echo(f"mode {env.mode} ({env.device}): {env.reason}")


@data_app.command("download")
def data_download(
    root: Path = typer.Option(Path("."), help="Project root."),
    force: bool = typer.Option(False, help="Download again even when the data is present."),
    attempts: int = typer.Option(3, min=1, help="Attempts of Ultralytics' own download."),
    archive: Path | None = typer.Option(
        None, help="Archive the present dataset came from (recorded with its sha256)."
    ),
    retry_every: float = typer.Option(
        0.0, help="Seconds between retries while only synthetic data exists (0 = no retry)."
    ),
) -> None:
    """Obtain the release in the spec 4.3 order; skip when present and valid."""
    paths = _paths(root)
    if retry_every > 0:
        report = cppe_data.acquire_until(
            paths,
            retry_every_s=retry_every,
            acquire=lambda p: cppe_data.acquire_dataset(
                p, attempts=attempts, force=force, archive=archive
            ),
        )
    else:
        report = cppe_data.acquire_dataset(paths, attempts=attempts, force=force, archive=archive)
    _echo_json(report.to_json())
    if not report.real_data:
        raise typer.Exit(1)


@data_app.command("verify")
def data_verify(root: Path = typer.Option(Path("."), help="Project root.")) -> None:
    """Verify counts, label lines, images and the archive manifest (spec 4.4)."""
    paths = _paths(root)
    report = cppe_data.verify_release(paths.source)
    summary = report.summary()
    cppe_data.write_json(paths.cppe_reports / "source_verification.json", summary)
    _echo_json({k: summary[k] for k in ("ok", "counts", "problem_counts", "differences")})
    if not report.ok:
        raise typer.Exit(1)


def _prepare(root: Path, force_split: bool, command: str) -> None:
    paths = _paths(root)
    result = cppe_prepare.run_s1(
        paths,
        commands=(f"{_PROG} data download", f"{_PROG} data verify", command),
        force_split=force_split,
    )
    _echo_json(
        {
            "ok": result.ok,
            "source": result.source.counts,
            "unified": result.unified.counts,
            "identity_violations": result.unified.identity,
            "test_images": len(result.split_payload["test_from_train"]),
            "val_excluded": len(result.split_payload.get("val_excluded", [])),
            "near_pairs": len(result.near.pairs),
            "repeat_factors": result.unified.summary["repeat_factors"],
            "oversampled_entries": result.unified.summary["oversampled_entries"],
            "report": result.report.as_posix(),
        }
    )
    if not result.ok:
        raise typer.Exit(1)


@data_app.command("prepare")
def data_prepare(
    root: Path = typer.Option(Path("."), help="Project root."),
    force_split: bool = typer.Option(False, help="Re-carve the committed test split."),
) -> None:
    """Statistics, near-duplicates, test carve, samples, unified copy, manifest, data_stats.md."""
    _prepare(root, force_split, f"{_PROG} data prepare")


@data_app.command("remap")
def data_remap(
    root: Path = typer.Option(Path("."), help="Project root."),
    force_split: bool = typer.Option(False, help="Re-carve the committed test split."),
) -> None:
    """Build the unified copy (same as ``data prepare``: the copy needs its checks)."""
    _prepare(root, force_split, f"{_PROG} data remap")


@data_app.command("all")
def data_all(
    root: Path = typer.Option(Path("."), help="Project root."),
    archive: Path | None = typer.Option(None, help="Archive the present dataset came from."),
) -> None:
    """``data download`` then ``data prepare``."""
    paths = _paths(root)
    report = cppe_data.acquire_dataset(paths, archive=archive)
    if not report.real_data:
        _echo_json(report.to_json())
        raise typer.Exit(1)
    _prepare(root, False, f"{_PROG} data all")


@app.command("train")
def train_command(
    run: str = typer.Option(..., "--run", help="r0, r1, r2 or r3 (spec 6.2)."),
    fresh: bool = typer.Option(False, help="Start over instead of resuming last.pt."),
) -> None:
    """Train one run in this process (resumes an unfinished run, never retrains a finished one)."""
    from safety_ai.training.train import run_cppe  # noqa: PLC0415 - imports torch

    raise typer.Exit(run_cppe(run, fresh=fresh))


@app.command("finalize")
def finalize_command(
    run: str = typer.Option(..., "--run", help="The stopped run (e.g. r1)."),
    reason: str = typer.Option("stopped from outside", help="Stop reason for the registry."),
) -> None:
    """Finalize a run stopped from outside: weights, reports, class map, registry, alias."""
    from safety_ai.training.train import finalize_stopped_run  # noqa: PLC0415 - imports torch

    raise typer.Exit(finalize_stopped_run(run, reason=reason))


@app.command("chain")
def chain_command(
    detach: bool = typer.Option(False, help="Start the chain in the background and return."),
) -> None:
    """Run R1 -> R2 -> R3 (R2/R3 wait for reports/cppe/gate_r2.json)."""
    from safety_ai.training.train import run_chain, start_detached_chain  # noqa: PLC0415

    if detach:
        typer.echo(f"chain started detached, pid {start_detached_chain(Path.cwd())}")
        return
    raise typer.Exit(run_chain())


@app.command("status")
def status_command() -> None:
    """Print reports/cppe/chain.json (state, current run, per-run progress)."""
    path = Path("reports/cppe/chain.json")
    typer.echo(path.read_text(encoding="utf-8") if path.is_file() else '{"state": "idle"}')


@app.command("eval")
def eval_command(
    *,
    model: str = typer.Option(..., "--model", help="Registry name, alias or weights path."),
    data: Path = typer.Option(
        Path("data/construction-ppe-unified/data.yaml"), "--data", help="Dataset YAML."
    ),
    split: str = typer.Option("val", "--split", help="val or test (test: production only)."),
    imgsz: int = typer.Option(640, "--imgsz", min=32, help="Input size (spec 7.1: 640)."),
    device: str = typer.Option("auto", "--device", help="auto, cpu, cuda:N."),
    speed_images: int = typer.Option(50, "--speed-images", min=1, help="Batch-1 timings."),
) -> None:
    """Spec 7.1 evaluation into reports/eval/<model>/ (per class, bare group, pairs, speed)."""
    from safety_ai.training.evaluate import evaluate_cppe  # noqa: PLC0415 - imports torch

    command = (
        f"{_PROG} eval --model {model} --data {data.as_posix()} --split {split} "
        f"--imgsz {imgsz} --device {device} --speed-images {speed_images}"
    )
    report = evaluate_cppe(
        model,
        data,
        split=split,
        imgsz=imgsz,
        device=device,
        speed_images=speed_images,
        command=command,
    )
    _echo_json(
        {
            "model": report.model_name,
            "overall": report.overall,
            "bare_group": report.bare_group,
            "report": (report.report_dir / "report.md").as_posix(),
        }
    )


def register(target: typer.Typer, name: str = "cppe") -> None:
    """Mount these commands into ``target`` as the group ``name`` (``safety-ai cppe ...``)."""
    target.add_typer(app, name=name)


def main(argv: list[str] | None = None) -> int:
    """Run the Typer app; return the exit code."""
    try:
        app(args=argv, prog_name=_PROG)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
