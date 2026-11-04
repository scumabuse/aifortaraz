"""docs/CONTRACT.md must contain section 4 of docs/SPEC.md verbatim (spec 4)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _section_4(spec_text: str) -> str:
    start = spec_text.index("## 4. Integration contract with the website")
    end = spec_text.index("## 5. Functional requirements")
    return spec_text[start:end].rstrip()


def test_contract_doc_is_verbatim_copy_of_spec_section_4() -> None:
    spec = (REPO / "docs" / "SPEC.md").read_text(encoding="utf-8")
    contract = (REPO / "docs" / "CONTRACT.md").read_text(encoding="utf-8")
    assert _section_4(spec) in contract


def test_contract_doc_mentions_every_endpoint() -> None:
    contract = (REPO / "docs" / "CONTRACT.md").read_text(encoding="utf-8")
    for endpoint in (
        "POST /api/v1/ingest/events",
        "PATCH /api/v1/ingest/events/{external_id}",
        "POST /api/v1/ingest/heartbeat",
        "POST /api/v1/ingest/stats",
        "GET /api/v1/ingest/config/{camera_id}",
        "ws://<host>/ws/ingest?api_key=<key>",
    ):
        assert endpoint in contract
