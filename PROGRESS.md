# Integration progress

Updated after every phase and at least every 30 minutes (integration spec 3). On restart read this
file, DECISIONS.md, CHECKLIST.md and INTEGRATION_INVENTORY.md and continue from the first unfinished item.

## Status — 2026-11-04 20:35 (UTC+5) — Phase A done, Phase B (fixing) starting

### Done
- Root repository created (contract/CONTRACT.md = spec section 4 verbatim).
- Environment facts collected (Windows 11, Python 3.12.10, Node 26.13 / npm 11.20, make 4.4.1, GTX 1660 SUPER, no Docker, no webcam, ports 8000/5173/8080/8001/9100/8010 free, no demo videos on either side).
- Duplicate resolved: the stale AI-module copy at the top of aifortaraz/ archived to `_archive/aifortaraz_ai_copy_2026-11-04/` (aifortaraz `83a018c`, I-001).

- INTEGRATION_INVENTORY.md: website and AI maps with file:line, baselines (website green: 58 + 8 tests, lint, build; AI: 2596 tests green, mypy green, ruff red), real website responses (`reports/inventory/website_responses.md`, captured by `scripts/inventory_probe.py`), conformance matrix of 111 rows (56 OK, 55 to fix: website 27, AI 26, both 2).

### In progress
- Phase B: website fixes (validation, schedule days, null ppe, KPI, seeds/key/env, WS robustness, supported_ppe, labels, badges/filter) and AI fixes (integration layer: CLI, worker, sinks, supervisor, selftest; heartbeat extra + supported_ppe; config null tolerance; 401 handling; severity null; mock port 8010; demo_integration.yaml; demo videos).

### Next commands
- Website: `cd aifortaraz/safety-inspector/backend && ../.venv/Scripts/python -m pytest -q`; `npm --prefix aifortaraz/safety-inspector/frontend test && npm --prefix aifortaraz/safety-inspector/frontend run lint && npm --prefix aifortaraz/safety-inspector/frontend run build`.
- AI: `cd safety-ai && make check`.

### Open problems
- The AI module has no CLI/worker/supervisor yet (I-002), so "safety-ai run against the mock" cannot be executed in Phase A; recorded as MISSING on AI.
