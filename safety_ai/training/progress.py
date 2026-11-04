"""The Construction-PPE block of PROGRESS.md (DECISIONS.md T-009).

PROGRESS.md is shared with the integration work, so the training pipeline owns exactly one
marker-delimited block::

    <!-- cppe-progress:start -->
    ...
    <!-- cppe-progress:end -->

:func:`write_progress_block` re-reads the file immediately before every write, replaces only the
text between the markers (appending a new section with the markers when they are missing), keeps
the file's newline style and writes through a temporary file plus ``os.replace``. The block is
rendered by :func:`render_progress_block` from the stage notes (``reports/cppe/progress_notes.md``,
edited by hand at each stage) and the run-chain status (``reports/cppe/chain.json``), so the chain
and the stage notes never overwrite each other.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

__all__ = [
    "BLOCK_END",
    "BLOCK_START",
    "SECTION_TITLE",
    "render_progress_block",
    "replace_block",
    "write_progress_block",
]

BLOCK_START: Final[str] = "<!-- cppe-progress:start -->"
BLOCK_END: Final[str] = "<!-- cppe-progress:end -->"
SECTION_TITLE: Final[str] = "## PPE training on Construction-PPE (docs/TRAINING_SPEC.md)"
_REPLACE_ATTEMPTS: Final[int] = 10


def replace_block(text: str, block: str) -> str:
    """Return ``text`` with the marker block replaced by ``block`` (or appended as a section).

    ``block`` is the content between the markers (without them). Line endings of the result
    follow ``text`` (CRLF when the file uses CRLF).

    Raises:
        ValueError: When the markers appear in the wrong order or more than once.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    body = block.replace("\r\n", "\n").strip("\n").replace("\n", newline)
    starts, ends = text.count(BLOCK_START), text.count(BLOCK_END)
    if starts == 0 and ends == 0:
        prefix = text if text.endswith(newline) or not text else text + newline
        section = newline.join(["", SECTION_TITLE, "", BLOCK_START, body, BLOCK_END, ""])
        return prefix + section
    if starts != 1 or ends != 1:
        raise ValueError(f"expected one {BLOCK_START} and one {BLOCK_END}, found {starts}/{ends}")
    head, rest = text.split(BLOCK_START, 1)
    if BLOCK_END not in rest:
        raise ValueError(f"{BLOCK_END} comes before {BLOCK_START}")
    _, tail = rest.split(BLOCK_END, 1)
    return f"{head}{BLOCK_START}{newline}{body}{newline}{BLOCK_END}{tail}"


def write_progress_block(path: Path, block: str) -> None:
    """Re-read ``path``, replace only the marker block and write it back atomically."""
    target = Path(path)
    text = target.read_bytes().decode("utf-8") if target.exists() else ""
    updated = replace_block(text, block)
    if updated == text:
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(updated.encode("utf-8"))
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


def _cell(value: Any) -> str:
    """Render a table cell (dash for None)."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "/")


def _run_rows(runs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows = [
        "| run | state | name | epoch | box / cls / dfl loss | mAP50 | mAP50-95 | best mAP50-95 "
        "| elapsed | ETA |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run_id, run in runs.items():
        latest: Mapping[str, Any] = run.get("latest") or {}
        losses = latest.get("losses")
        loss_text = " / ".join(f"{v:.3f}" for v in losses) if losses else "-"
        epoch = latest.get("epoch")
        planned = latest.get("epochs_planned")
        epoch_text = f"{epoch}/{planned}" if epoch is not None and planned else _cell(epoch)
        rows.append(
            f"| {run_id} | {_cell(run.get('state'))} | {_cell(run.get('name'))} | {epoch_text} "
            f"| {loss_text} | {_cell(latest.get('map50'))} | {_cell(latest.get('map50_95'))} "
            f"| {_cell(run.get('best_map50_95'))} | {_cell(latest.get('elapsed'))} "
            f"| {_cell(latest.get('eta'))} |"
        )
    return rows


def render_progress_block(
    *,
    updated_at: str,
    notes: str,
    chain: Mapping[str, Any] | None,
    commands: Sequence[str] = (),
) -> str:
    """Render the block: timestamp, stage notes, run-chain table, how to check the chain."""
    lines = [f"Updated {updated_at} (UTC) by the Construction-PPE pipeline.", ""]
    if notes.strip():
        lines += [notes.strip(), ""]
    if chain:
        state = chain.get("state", "-")
        current = chain.get("current_run") or "-"
        lines += [
            f"**Run chain** (`reports/cppe/chain.json`): state **{state}**, current run "
            f"{current}, pid {chain.get('pid', '-')}, log `{chain.get('log', '-')}`.",
            "",
            *_run_rows(chain.get("runs") or {}),
            "",
        ]
    if commands:
        lines += ["Check the chain:", "", *(f"- `{c}`" for c in commands), ""]
    return "\n".join(lines).rstrip("\n")
