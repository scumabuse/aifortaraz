"""``safety-ai selftest``: the contract self-check against the strict mock and its table."""

from __future__ import annotations

import io
from pathlib import Path

from safety_ai.selftest import (
    CheckResult,
    SelftestReport,
    _field_errors,
    render_table,
    run_selftest,
)
from tools.mock_backend import MockBackendServer

CONTRACT_CHECKS = (
    "config list",
    "camera config",
    "event multipart",
    "event json snapshot_base64",
    "event repeated",
    "event patch",
    "patch unknown",
    "re-POST with ended_at",
    "heartbeat (supported_ppe)",
    "stats",
    "websocket detections + heartbeat",
    "mock violations",
)


def _rows(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        for name in CONTRACT_CHECKS:
            if f"  {name}  " in line or line.rstrip().endswith(f"  {name}"):
                rows[name] = line
    return rows


def test_the_quick_selftest_passes_against_the_in_process_mock() -> None:
    out = io.StringIO()
    assert run_selftest(quick=True, out=out) == 0
    text = out.getvalue()
    rows = _rows(text)
    assert set(rows) == set(CONTRACT_CHECKS)
    assert all(row.startswith("PASS") for row in rows.values()), text
    assert "201" in rows["event multipart"] and "200" in rows["event repeated"]
    assert "404" in rows["patch unknown"]
    assert "selftest PASSED" in text and "models  " not in text  # quick: no model loading


def test_a_refused_api_key_fails_the_selftest_readably(tmp_path: Path) -> None:
    out = io.StringIO()
    with MockBackendServer(port=0, data_dir=tmp_path / "mock", api_key="right-key") as mock:
        code = run_selftest(quick=True, backend_url=mock.base_url, api_key="wrong-key", out=out)
    text = out.getvalue()
    assert code == 1
    assert "401" in _rows(text)["config list"]
    assert "rejected the API key" in text and "selftest FAILED" in text
    assert "wrong-key" not in text and "right-key" not in text


def test_a_422_fails_the_report_and_lists_the_field_errors() -> None:
    report = SelftestReport()
    report.add(CheckResult("contract", "stats", True, "POST /api/v1/ingest/stats"))
    report.add(
        CheckResult(
            "contract",
            "event multipart",
            False,
            "POST /api/v1/ingest/events",
            "201",
            "422",
            ("body.zone_id: unknown zone",),
        )
    )
    text = render_table(report)
    assert not report.passed
    assert "    event multipart: body.zone_id: unknown zone" in text
    assert text.splitlines()[-1] == "selftest FAILED: 1 passed, 1 failed"


def test_field_errors_are_read_from_the_website_and_the_mock_bodies() -> None:
    website = (
        '{"error": {"code": "validation_error", "message": "invalid payload", "details": '
        '{"fields": [{"loc": ["body", "zone_id"], "msg": "unknown zone"}]}}}'
    )
    assert "body.zone_id: unknown zone" in _field_errors(website)
    mock = '{"detail": "contract violation", "errors": ["details.boxes: Field required"]}'
    assert "details.boxes: Field required" in _field_errors(mock)
    assert _field_errors("plain text") == ("plain text",)
