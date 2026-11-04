# Progress

Updated after every milestone and at least every 30 minutes (spec 3). On restart: read this file,
DECISIONS.md and CHECKLIST.md, then continue from the first unticked checklist item.

## Status — 2026-11-04 11:40 (local, UTC+5)

### Done
- Environment surveyed (DECISIONS.md D-004): Windows 11, Python 3.12.10, GTX 1660 SUPER 6 GB (CUDA OK), no Docker/WSL, GNU make available.
- `.venv` created; installed torch 2.14.1+cu126, torchvision 0.29.1, ultralytics 8.4.201, OpenCV 5.0.0, onnxruntime 1.31.1, OpenVINO 2026.4.1 and the rest of the stack.
- YOLO26 weights load; yolo26n 640: 46.7 ms CPU / 21.5 ms GPU on bus.jpg (D-008).
- Dataset reachability probed (D-005, D-006, D-007): SH17 labels via range requests + Pexels images; Construction Site Safety (Kaggle, 228 MB, CC BY 4.0); Hard Hat Detection (Kaggle, 1.3 GB, CC0); CHV (Google Drive); UR Fall / Le2i Kaggle mirrors; no harness dataset found; Roboflow needs a key.
- Specification saved verbatim to docs/SPEC.md; DECISIONS.md D-001…D-026; CHECKLIST.md with every 10.2 item unticked.
- Core shared types written: `safety_ai/clock.py`, `inference/types.py`, `capture/types.py`, `backend/schemas.py` (round-trips every contract example), `logic/types.py`, `config/models.py`.

### Wave A result (03:55–04:40, committed 03a9d75)
- 10 parallel engineers; the account's 5-hour usage limit stopped 9 of them before they returned, but their files were on disk: ~56k lines, 2248 unit/property tests passing, 7 failing (fall ×3, mock rate limiter ×1, Makefile tests ×3).
- Real downloads done: SH17 8099/8099 images (Pexels 1280 px + 1 archive fallback), CSS 2801, harness (Kaggle manoj223/safety-harness, 566, MIT), CHV 1330, UR Fall 70 videos; Hard Hat partial (9067 files); no forklift, no Le2i yet.
- Lead fixes after the restart (11:20): lint/mypy of the shared modules (schemas default_factory typing, PEP 604 unions, long docstrings).

### In progress — Wave B (11:35, workflow `safety-ai-wave-b`, run wf_893e4e43-b50)
Dependency-chained, ≤ 5 agents at once to stay inside the usage limit:
- B-data: finish the data pipeline, build data/merged + data.yaml + train_oversampled.txt + reports/data_stats.md, then write `data/merged/READY`.
- B-engine: tracks, rule state machines, stats aggregator, id allocator, LogicEngine (ARCHITECTURE 3.8/4).
- B-fix: the 7 failing tests, repo-wide ruff/format, Makefile/lock checks.
- B-models: models_setup (weights + registry), `--check`, e2e inference/training tests, ONNX/OpenVINO exports + verification, CPU bench.
- B-fakes: scenario YAML model, FakeSource/FakeDetector/FakePose.
- then B-worker (worker, sinks, IPC, demo, CLI) → B-app (supervisor, run service); B-fallclf (UR Fall classifier).
- GPU training driver (background): waits for `data/merged/READY`, then trains yolo26n (70 min) and yolo26s (110 min) through `safety_ai.training.train.train` (log in the session scratchpad, results in runs/gpu_training_driver.json).

### Next (exact commands)
- After Wave B: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit tests/property` and `make lint typecheck`.
- Wave C: 14 scenarios + runner, integration tests (8.4), e2e + synthetic video + verify_run + demo configs + selftest, docs + tools/check_docker.py.

### Open problems
- Docker cannot be built on this machine (D-004 fallback).
- Forklift and Le2i not downloaded yet (B-data retries for ≤ 15 min, otherwise skipped and documented); vehicles then come from the COCO vehicle_n model (spec 5.3 fallback).
- The account usage limit can stop agents mid-task again; their files survive and the next wave resumes from them.
