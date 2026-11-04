"""docs/CONTRACT.md is the integration contract (integration matrix row 111).

The single source of truth is the integration root's ``contract/CONTRACT.md``; this repository
keeps a byte-identical copy (the root test compares the hashes of all three copies). These tests
check that the copy here is that contract and not the module's earlier one (section 4 of
docs/SPEC.md, which it supersedes).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "docs" / "CONTRACT.md"
INTEGRATION_CONTRACT = REPO.parent / "contract" / "CONTRACT.md"

HEADINGS = (
    "# 4. The integration contract (single source of truth)",
    "## 4.1. Events (AI → site)",
    "## 4.2. Heartbeat (AI → site), every 5 s per camera",
    "## 4.3. Statistics (AI → site), every 60 s per camera, minute-aligned",
    "## 4.4. Configuration (site → AI)",
    "## 4.5. WebSocket AI → site (/ws/ingest)",
    "## 4.6. WebSocket site → browser (/ws/live)",
    "## 4.7. Video",
    "## 4.8. Durability and errors",
)


def _contract() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def test_contract_doc_has_exactly_the_sections_of_the_integration_contract() -> None:
    headings = [line for line in _contract().splitlines() if line.startswith("#")]
    assert headings == list(HEADINGS)


def test_contract_doc_names_every_endpoint_and_channel() -> None:
    contract = _contract()
    for item in (
        "Base path /api/v1",
        "X-API-Key: <key>",
        "POST /api/v1/ingest/events",
        "PATCH /api/v1/ingest/events/{external_id}",
        "POST /api/v1/ingest/heartbeat",
        "POST /api/v1/ingest/stats",
        "GET /api/v1/ingest/config/{camera_id}",
        "GET /api/v1/ingest/config (all cameras with AI enabled)",
        "ws://<host>/ws/ingest?api_key=<key>",
        "ws://<host>/ws/live?token=<JWT>",
        "GET /api/v1/stream/{camera_id}.mjpeg",
        "http://<ai-host>:8001/stream/{camera_id}.mjpeg",
    ):
        assert item in contract, item


def test_contract_doc_carries_the_integration_rules() -> None:
    contract = _contract()
    for rule in (
        "extra.supported_ppe",
        '"extra": {…free-form…}',
        'Error bodies {"error": {"code", "message", "details"}}',
        "null → zone severity_override → type default",
        '[{"days": [1..7], "from": "HH:MM", "to": "HH:MM"}]',
        "(1 = Monday)",
        "external_id = {camera_id}-{unix_seconds_at_start}-{track_id}-{type}",
        "422 means a client bug (logged, saved, fails selftest); 429 honours Retry-After",
    ):
        assert rule in contract, rule


def test_contract_doc_is_a_byte_identical_copy_of_the_integration_contract() -> None:
    data = CONTRACT.read_bytes()
    assert b"\r\n" not in data  # LF only, exactly as the root copy is committed
    assert data.endswith(b"\n")
    # Inside the integration checkout the root holds the master copy: the bytes must match.
    if INTEGRATION_CONTRACT.is_file():
        root = hashlib.sha256(INTEGRATION_CONTRACT.read_bytes()).hexdigest()
        assert hashlib.sha256(data).hexdigest() == root
