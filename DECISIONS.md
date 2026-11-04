# Integration decisions

Decisions taken while integrating the website (`aifortaraz/safety-inspector`) and the AI module
(`safety-ai`), each with the reason and the rejected alternatives (integration spec 3). Ids are stable.

## I-001 The top-level AI-module copy in aifortaraz/ is archived
- **Decision:** `aifortaraz/` contained, next to the website in `safety-inspector/`, the files of the AI module (`safety_ai/`, `configs/`, `datasets/`, `docker/`, `docs/`, `tests/`, `tools/`, `pyproject.toml`, `requirements.lock`, `Makefile`, `README.md`, `DECISIONS.md`, `PROGRESS.md`, `CHECKLIST.md`, `docker-compose.yml`, dot-files). Their git tree hashes are identical to `safety-ai` at commit `3faef81` (the state pushed to GitHub at 11:45), the website commit `91e7b9b` touched only `safety-inspector/`, and nothing in the website imports them. They were moved to `_archive/aifortaraz_ai_copy_2026-11-04/` (aifortaraz commit `83a018c`); git history keeps them too. A short top-level README now points to `safety-inspector/`.
- **Rejected:** keeping two copies of the AI module (they would drift); deleting without an archive (spec 2 forbids irreversible deletion).

## I-002 The AI module's integration layer is completed as part of the integration
- **Decision:** the canonical AI module (`safety-ai/`) was built in two interrupted waves; on disk it has the contract client, spool, mock backend, capture, inference, the logic engine and the fakes, but no `safety_ai/cli.py`, `worker.py`, `app.py`, `demo.py`, `sinks.py`, `ipc.py` or `selftest.py`, so `safety-ai run` does not exist. These modules are part of the AI module's own specification (its docs/ARCHITECTURE.md 3.10) and are required to integrate at all, so they are implemented now (with tests) rather than treated as new features.
- **Rejected:** driving the website with the website team's `scripts/fake_ai_client.py` instead of the real AI module (would not satisfy the mission).

## I-003 Models: no retraining, the SH17 nano detector is the demo model
- **Decision:** the registry contains `ppe_n_sh17` / `ppe_s_sh17` (the SH17 authors' YOLOv8 weights, mapped to the unified taxonomy), `vehicle_n` / `vehicle_s` (COCO yolo26n/s, car/truck/bus → vehicle), `pose_n` (yolo26n-pose) and ONNX/OpenVINO exports of `ppe_n_sh17` and `pose_n`. No detector trained on Construction-PPE (or on the merged dataset) exists: the planned GPU training never started because its session ended. The integration spec forbids retraining, so `configs/demo_integration.yaml` uses `ppe_n_sh17` (nano, CPU-friendly) with `vehicle_n` for vehicles.
- **Rejected:** training now (explicit non-goal).

## I-004 No Docker on this machine
- **Decision:** Docker is not installed (and WSL is absent, so it cannot be installed unattended). The root `docker-compose.yml` is written and reviewed and validated statically, but `make up` is not executed here; this is recorded in CHECKLIST.md and FINAL_REPORT.md as the spec allows.

## I-005 No webcam on this machine
- **Decision:** no camera device is present (PnP query, device not opened). The webcam variant of cam-1 is documented and commented in `configs/demo_integration.yaml`; all runs use files and fake sources.

## I-006 No pushes to GitHub
- **Decision:** `safety-ai/` and `aifortaraz/` both have `origin = github.com/scumabuse/aifortaraz` with diverged histories; the integration spec asks for local commits only, so nothing is pushed.
