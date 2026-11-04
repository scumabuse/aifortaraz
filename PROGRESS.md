# Progress

Updated after every milestone and at least every 30 minutes (spec 3). On restart: read this file,
DECISIONS.md and CHECKLIST.md, then continue from the first unticked checklist item.

## Status — 2026-11-04 03:35 (local, UTC+5)

### Done
- Environment surveyed (DECISIONS.md D-004): Windows 11, Python 3.12.10, GTX 1660 SUPER 6 GB (CUDA OK), no Docker/WSL, GNU make available.
- `.venv` created; installed torch 2.14.1+cu126, torchvision 0.29.1, ultralytics 8.4.201, OpenCV 5.0.0, onnxruntime 1.31.1, OpenVINO 2026.4.1 and the rest of the stack.
- YOLO26 weights load; yolo26n 640: 46.7 ms CPU / 21.5 ms GPU on bus.jpg (D-008).
- Dataset reachability probed (D-005, D-006, D-007): SH17 labels via range requests + Pexels images; Construction Site Safety (Kaggle, 228 MB, CC BY 4.0); Hard Hat Detection (Kaggle, 1.3 GB, CC0); CHV (Google Drive); UR Fall / Le2i Kaggle mirrors; no harness dataset found; Roboflow needs a key.
- Specification saved verbatim to docs/SPEC.md; DECISIONS.md D-001…D-026; CHECKLIST.md with every 10.2 item unticked.
- Core shared types written: `safety_ai/clock.py`, `inference/types.py`, `capture/types.py`, `backend/schemas.py` (round-trips every contract example), `logic/types.py`, `config/models.py`.

### In progress — Wave A (04:00): M0 remainder + module work of M1–M5 in parallel
Ten parallel engineers (workflow `safety-ai-wave-a`, run wf_63c0f098-793), disjoint file ownership, each with unit tests and ruff/mypy/pytest gates on its files:
A1 config loader + logging + Makefile + requirements.lock + quality tools + configs; A2 backend client (REST, SQLite spool + sender, WebSocket, config sync); A3 strict mock backend + mock_config.yaml; A4 capture sources; A5 inference (registry, backends, detector + tracking, pose, models_setup, bench core); A6 logic core (geometry, zones, association, smoothing, proximity); A7 fall/immobile + synthetic poses; A8 outputs (annotate, MJPEG, event recorder, preview) + metrics/health; A9 dataset pipeline + real downloads; A10 synthetic dataset + train/eval/export.

### Plan after Wave A
- Wave B: logic rules/state machines/tracks/stats/engine; fall classifier on UR Fall/Le2i; scenario model + FakeSource/FakeDetector/FakePose; worker/app/demo/CLI integration. GPU training (yolo26s then yolo26n, ≤ 3 h total) starts in the background as soon as data/merged and train.py exist.
- Wave C: 14 scenarios + runner, integration tests (8.4), e2e + synthetic video + verify_run + demo configs + selftest, Docker/compose + docs.
- Wave D: adversarial review and fixes, then the Verification Protocol ×3.

### Next (exact commands)
- `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit tests/property` once Wave A returns.
- `make lint typecheck test`.

### Open problems
- Docker cannot be built on this machine (D-004 fallback).
- No harness dataset reachable yet; forklift data on Kaggle is small (search continues in M5).
