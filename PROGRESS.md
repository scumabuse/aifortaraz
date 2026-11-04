# Integration progress

Updated after every phase and at least every 30 minutes (integration spec 3). On restart read this
file, DECISIONS.md, CHECKLIST.md and INTEGRATION_INVENTORY.md and continue from the first unfinished item.

## Status — 2026-11-04 23:20 (UTC+5) — Phase B done, 8.1–8.2 green, Phase C/D/E in progress

### Done since 20:35
- Phase B website: b2a1994 (frontend), 2fbd88e (backend), 680c63b (oversized snapshot → 422 with field errors).
- Phase B AI: 1f0f9ae (logic core, fakes), 5a4eca1 (contract conformance: schemas, clients, spool, config sync, strict mock on 8010), 5a8e336 (`safety-ai run`: supervisor, workers, sinks, demo, selftest, CLI), bd23ac1 (README, D-038..D-043, lap dependency). AI-side evidence: 2850 unit/property tests and the e2e run green (`reports/inventory/ai_messages_after.md`).
- 8.1 contract tests: 5 passed. 8.2 flow tests: 16/16 passed (`reports/runtime/flows_run1.log`: 13 passed, 3 failed on test defects; fixed in f54a30e — zones with incident history are disabled instead of deleted, and the detections test waits for `ok` after the violation; `reports/runtime/flows_run2.log`: the 3 passed).

### In progress
- Phase C wiring checks, 8.3 real-model smoke with screenshots, 8.4 performance budget, `scripts/verify_run.py` (CPU only: the GPU is busy with the Construction-PPE training chain of safety-ai, see safety-ai/docs/TRAINING_SPEC.md).
- Phase E root documentation (README, docs/INTEGRATION.md, TROUBLESHOOTING.md, DEMO_SCRIPT.md) and the inventory's commit column.

### Next
- Verification Protocol (10.2) ×3 after the training work in safety-ai is committed (both tasks change safety-ai; the passes must see the final tree).

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
