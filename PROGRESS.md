# Integration progress

Updated after every phase and at least every 30 minutes (integration spec 3). On restart read this
file, DECISIONS.md, CHECKLIST.md and INTEGRATION_INVENTORY.md and continue from the first unfinished item.

## Status — 2026-11-04 19:45 (UTC+5) — Phase A (discovery and inventory) in progress

### Done
- Root repository created (contract/CONTRACT.md = spec section 4 verbatim).
- Environment facts collected (Windows 11, Python 3.12.10, Node 26.13 / npm 11.20, make 4.4.1, GTX 1660 SUPER, no Docker, no webcam, ports 8000/5173/8080/8001/9100/8010 free, no demo videos on either side).
- Duplicate resolved: the stale AI-module copy at the top of aifortaraz/ archived to `_archive/aifortaraz_ai_copy_2026-11-04/` (aifortaraz `83a018c`, I-001).

### In progress
- Website and AI inventories (file:line maps and the conformance matrix), website dependency install, baselines.

### Next commands
- `aifortaraz/safety-inspector/.venv/Scripts/python -m pytest -q` (in backend/), `npm --prefix aifortaraz/safety-inspector/frontend test`, `... run lint`, `... run build`.
- `safety-ai/.venv/Scripts/python -m pytest -q -p no:cacheprovider tests/unit tests/property -m "not e2e and not slow"`.

### Open problems
- The AI module has no CLI/worker/supervisor yet (I-002), so "safety-ai run against the mock" cannot be executed in Phase A; recorded as MISSING on AI.
