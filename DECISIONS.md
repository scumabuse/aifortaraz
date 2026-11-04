# Decisions

Every non-obvious decision made while building safety-ai, with the reason and the alternatives
that were rejected (spec 3). Ids are stable; later decisions may supersede earlier ones explicitly.

## D-001 Repository root
- **Decision:** the repository is `safety-ai/` inside the provided (empty) working folder, laid out as in spec 7.1.
- **Reason:** spec 7.1 names the root `safety-ai/`.
- **Rejected:** using the working folder itself as the root (would not match the documented layout).

## D-002 Truncated specification
- **Decision:** the specification was received with sections 2–10.2; the chat input limit cut it inside 10.2 after "The CPU Docker image builds", and section 1 and 10.3 were not received. The received text is kept verbatim in `docs/SPEC.md`. The rest of the Definition of Done is reconstructed from the requirements stated elsewhere in the spec (CUDA image and compose from 7.4, soak from 9, placeholders from 3, shutdown from 9, docs from 2, selftest from 7.3, three clean passes from 3) and listed in CHECKLIST.md, marked "(reconstructed)".
- **Reason:** spec 3 forbids stopping to ask; every reconstructed item is a requirement the received text already states.
- **Rejected:** ignoring the missing items (would silently reduce scope).

## D-003 Verification Protocol (reconstructed spec 10.3)
- **Decision:** the protocol is the ordered list below; `safety-ai selftest --quick` runs steps 1–2, `safety-ai selftest` runs all steps.
  1. Environment and packaging: `safety-ai version`, `safety-ai --help` lists all 13 commands, every registry entry exists with a matching sha256 and loads.
  2. Contract self-check: an in-process strict mock backend receives one message of every outgoing type over real HTTP/WebSocket; zero violations, zero HTTP 422.
  3. Static checks: `ruff check`, `ruff format --check`, `mypy`.
  4. Unit and property tests; line coverage of `safety_ai/logic` and `safety_ai/backend` ≥ 90 %.
  5. Scenario suite (14 scenarios of 8.3).
  6. Integration tests (8.4).
  7. End-to-end 90 s run: mock backend + `safety-ai run --config configs/demo_file.yaml`, verified by `tools/verify_run.py`.
  8. Demo run: `safety-ai demo` prints JSON-line events, writes a jpg and an mp4 per event; MJPEG serves 10 frames (`tools/fetch_mjpeg.py`).
  9. Models: evaluation report present, ONNX export verified against `.pt` (mean IoU ≥ 0.9 on 20 images).
  10. Benchmarks: `models/bench.json` refreshed, targets checked.
  11. Docker: CPU and CUDA images build (or the documented fallback of D-004 when Docker is unavailable).
  12. Soak: 30-minute run, RSS growth < 10 % after minute 5, no reconnect storm, no dropped events.
  13. Hygiene: no placeholders (TODO, FIXME, NotImplementedError, bare `pass` bodies, skip/xfail), every documented command re-run, git tree clean.
- **Reason:** these are exactly the checks the received spec requires before completion.
- **Rejected:** a shorter protocol without the soak (spec 9 makes the soak part of the protocol).

## D-004 Build machine and the Docker fallback
- **Decision:** the build machine is Windows 11, Python 3.12.10, NVIDIA GTX 1660 SUPER (6 GB, CC 7.5, driver 610.88), i5-10400F (6C/12T), 16 GB RAM. Docker and WSL are not installed and installing them needs administrator rights and a reboot, so images cannot be built here. Fallback: (1) `tools/check_docker.py` statically validates both Dockerfiles and the compose file (every COPY source exists and is not excluded by `.dockerignore`, the entrypoint/CMD use real CLI commands, ports and health URLs match the configuration); downloading and executing the hadolint binary was refused by this environment's safety policy, so no external linter binary is used; (2) `requirements.lock` is resolved for linux/py3.11 by uv and every pinned Linux wheel is downloaded (not executed) for that platform; (3) the install steps of the Dockerfile (`pip install --no-deps -r requirements.lock`, `pip install --no-deps .`) are executed in a clean virtual environment on this machine. FINAL_REPORT.md states that `docker build` itself was not executed.
- **Reason:** spec 3: after the primary way is impossible, use the fallback and record it.
- **Rejected:** claiming the images build without having built them.

## D-005 Credentials and dataset access
- **Decision:** there is no `.env`, no Kaggle and no Roboflow credential on this machine. Kaggle's public download endpoint works anonymously (`/api/v1/datasets/download/<owner>/<slug>` redirects to a signed archive URL), so Kaggle mirrors are used. Roboflow Universe needs an API key, so Roboflow-only datasets fall back to Kaggle mirrors or are skipped and documented.
- **Reason:** spec 6.2 access rules.
- **Rejected:** scraping Roboflow without a key.

## D-006 SH17 acquisition without the 14 GB archive
- **Decision:** the SH17 Kaggle archive is 14.1 GB (full-resolution images, up to 8192×5462). Labels, `train_files.txt`, `val_files.txt` and per-image `meta-data/*.json` are read from the archive with HTTP range requests (the non-image tail is 13.4 MB); images are downloaded from the Pexels CDN listed in the official `data/list_of_all_urls.csv` at 1280 px width (≈200 kB each). YOLO labels are normalized, so the resize does not change them; every image's aspect ratio is checked against the original width/height in `meta-data` and mismatches are dropped. A missing Pexels image falls back to extracting that single image from the archive by range request.
- **Reason:** the 15 GB raw-data cap (spec 3) and download time; the GitHub repository ships only the URL list and the download script, not the labels.
- **Rejected:** downloading the whole archive (≈14 GB, leaves no room for other datasets).

## D-007 SH17 class names
- **Decision:** the class list was verified against the dataset's own `sh17.yaml`: `0 person, 1 ear, 2 ear-mufs, 3 face, 4 face-guard, 5 face-mask, 6 foot, 7 tool, 8 glasses, 9 gloves, 10 helmet, 11 hands, 12 head, 13 medical-suit, 14 shoes, 15 safety-suit, 16 safety-vest`. The spec's names `earmuffs`, `face-mask-medical` and `tools` are `ear-mufs`, `face-mask` and `tool` in the dataset; the mapping in `datasets/taxonomy.yaml` uses the dataset's names (ear-mufs → earmuffs, face-mask → mask, tool → tools, face-guard → face_shield, shoes → boots, safety-vest → vest, medical-suit and safety-suit → drop, the rest 1:1).
- **Reason:** spec 6.2: verify every class list against the dataset's own data.yaml before mapping.

## D-008 Detector family
- **Decision:** ultralytics 8.4.201 is installed and `yolo26n.pt`, `yolo26n-pose.pt` and `yolo26s.pt` load, so the YOLO26 family is used; the runtime still probes at startup and falls back to YOLO11, and reports the family in `heartbeat.extra.model_family`.
- **Measured:** yolo26n at 640 on `bus.jpg`: 46.7 ms median on the CPU (PyTorch), 21.5 ms on the GTX 1660 SUPER.

## D-009 Ultralytics half-precision flag
- **Decision:** ultralytics 8.4 deprecates `half=` in favour of `quantize=`; the inference wrapper detects the supported keyword at runtime instead of hard-coding one.

## D-010 Inconsistent external_id in the contract example
- **Decision:** the contract example pairs `external_id` `cam-1-1762179667-17-ppe_violation` (1762179667 = 2025-11-03T14:21:07Z) with `started_at` 2026-11-03T14:21:07Z. We implement the field rule literally: the seconds part is always `floor(started_at)` (unit-tested). Schemas and the mock check the pattern `{camera_id}-<digits>-{track_id}-{type}`, so the contract's own example remains valid.
- **Rejected:** an exact seconds check in the mock (would reject the contract example).

## D-011 Timestamp precision
- **Decision:** `started_at`, `ended_at`, `period_start`, `period_end` are serialized with whole seconds (`2026-11-03T14:21:07Z`), matching the examples and the seconds in `external_id`; WebSocket `ts` values carry milliseconds (`...07.250Z`). Both are ISO 8601 UTC with a trailing Z.

## D-012 Response of GET /api/v1/ingest/config
- **Decision:** the contract does not show the all-cameras response. The literal reading is a JSON array of per-camera config objects; the client also accepts an object wrapping the array under `cameras`, `configs`, `items` or `data`. The mock returns an array.

## D-013 track_id of PPE item boxes
- **Decision:** item and bare-part boxes in `details.boxes` carry the owning person's `track_id`. The contract shows only integer track ids; `null` could be rejected by a strict backend, and the owner id is the true association.

## D-014 external_id collisions
- **Decision:** when `{camera}-{seconds}-{track}-{type}` was already used (same track in two zones or with two vehicles within one second, or a restart reusing a track id in the same second), `started_at` is moved forward by one second until the id is unique (checked against this process and the spool). The id therefore always matches the reported `started_at`.
- **Rejected:** adding a suffix (would change the contract format).

## D-015 Zone `enabled`
- **Decision:** the contract zone object has no `enabled` field although 4.4 says "every enabled zone". The parser accepts an optional `enabled` (default true).

## D-016 `ppe_types`
- **Decision:** a zone's `required_ppe` is intersected with the camera's `ppe_types` (an empty list means all types), so the site can disable an item type per camera.

## D-017 `thresholds.min_confidence`
- **Decision:** persons and vehicles below the backend's `min_confidence` (fallback: local `thresholds.min_confidence`) are not passed to the rules; PPE items and bare parts are filtered only by the inference threshold (`thresholds.conf` and per-class overrides), because item confidences are naturally lower and filtering them would turn PRESENT into MISSING.

## D-018 Severity
- **Decision:** `severity` is `null` for every type except `fall_detected` and `person_immobile`, which are `critical` (spec 5.9). `severity_override` is applied by the backend, not by this module.

## D-019 Strict outgoing, lenient incoming
- **Decision:** outgoing models use `extra="forbid"`; incoming config and server messages ignore unknown fields and drop unknown PPE codes with a warning, so a newer website does not take the AI offline.

## D-020 Heartbeat extra
- **Decision:** `heartbeat.extra` carries `device`, `backend`, `queue_depth`, `frames_dropped`, `stream` (as in the contract) plus `model_family` (spec 5.3 requires the family choice in the heartbeat extra).

## D-021 OpenCV packages
- **Decision:** ultralytics depends on `opencv-python`; the package depends on `opencv-python-headless` (spec 9). In a plain `pip install` both end up installed and the headless binary wins, so the preview window auto-disables with a one-line warning when the OpenCV build has no GUI. The `preview` extra plus `pip install --force-reinstall --no-deps opencv-python` enables the window; Docker installs with `--no-deps` from `requirements.lock`, which excludes `opencv-python`.

## D-022 Licence of this package
- **Decision:** AGPL-3.0, because the package links Ultralytics (AGPL-3.0). SH17 is CC BY-NC-SA 4.0 and each dataset keeps its own licence (spec 6.8).

## D-023 `tools/` is packaged
- **Decision:** `tools/` (mock backend, verifiers) is a package included in the wheel so `safety-ai mock-backend` also works from a non-editable install.

## D-024 Independent mock validation; event-time cooldown in the mock
- **Decision:** the strict mock backend validates with its own models written from docs/CONTRACT.md, not with the client's schemas, so a bug in the client schemas cannot hide itself. The mock's 60 s cooldown (camera+zone+type+track) compares event `started_at` values, not wall-clock arrival, so scenario tests driven by a fake clock are deterministic.

## D-025 Committed weights
- **Decision:** `.pt` weights ≤ 50 MB and `models/registry.json` are committed so the demo works offline from a fresh clone; ONNX/OpenVINO/TensorRT exports are git-ignored and regenerated by `make models`.

## D-026 Training plan on this machine
- **Decision:** a CUDA GPU is available, so a real fine-tune runs: yolo26s (the spec's CUDA base) and yolo26n (the CPU-demo model, spec 8.6 targets the nano model on a CPU laptop), sharing the 3-hour CUDA time box (about 110 + 70 minutes).
- **Rejected:** only the small model (too slow for the CPU laptop target); only the nano model (ignores the spec's CUDA recipe).

## D-027 Zone schedules
- **Decision:** `schedule: null` and `schedule: []` both mean "always active"; a window with `to <= from` runs past midnight and belongs to the day it starts on; `24:00` is accepted as an end time. When a zone leaves its schedule, memberships reset and its ACTIVE events end (reason `config_changed`).
- **Rejected:** `[]` = never active (a zone that can never be active is not a meaningful configuration).

## D-028 ended_at
- **Decision:** `ended_at` is the UTC time of the frame on which the end condition was confirmed, except for track loss, where it is the UTC time the track was last seen (the violation was observed until then).

## D-029 ppe_violation across several zones
- **Decision:** the item smoothing uses the smallest `confirm_frames` among the zones requiring the item; the event's `zone_id` is the first zone (config order) the person is in that requires a violated item; its cooldown applies to the track key. The rule needs N = 1 frame of "some item in VIOLATION" because the item smoothing already applied `confirm_frames`.

## D-030 End of person_immobile
- **Decision:** the spec defines the start of person_immobile but not its end; it ends when the posture is not horizontal for 2 s (as fall_detected), when the centroid moves ≥ 2 % of the frame width from where the stillness began, or when the track is unseen for 10 s.

## D-031 PPE values in the detections stream
- **Decision:** for each required item: `false` while the item is in VIOLATION, `true` when its smoothed score is ≥ `present_on`, `null` otherwise (unknown or not yet decided).

# PPE training on Construction-PPE

Decisions of the Construction-PPE training work (docs/TRAINING_SPEC.md, received 2026-11-04). Ids
T-xxx are independent of the integration decisions D-xxx above; neither renumbers the other.

## T-001 Use the release that is actually published (November 2026)
- **Decision:** the download URL of spec 4.2/4.3 now serves the November 2026 release of Ultralytics Construction-PPE, and ultralytics 8.4.201 ships its YAML (`ultralytics/cfg/datasets/construction-ppe.yaml`, pinned verbatim in `datasets/construction-ppe.yaml`). That release is used, and every verification checks its published facts: 1,000 images = 800 train + 200 val, no test split, 12 classes, 14,863 boxes (README.md of the archive and the docs page fetched 2026-11-04).
- **Differences from spec 4.1/4.2** (also in `reports/data_stats.md`): 1,000 instead of 1,416 images; split 800/200/none instead of 1,132/143/141; archive 647,666,037 bytes instead of about 178 MB; 12 classes `0 helmet, 1 gloves, 2 vest, 3 boots, 4 goggles, 5 Person, 6 no_helmet, 7 no_goggle, 8 no_gloves, 9 no_boots, 10 forklift, 11 no_vest` instead of 11 (no `none` class; `forklift` and `no_vest` are new; every id from Person on shifts by one); licence CC BY 4.0 instead of AGPL-3.0; citation `@dataset{Ultralytics_Construction_PPE_2026, author = "{Ultralytics team}", title = {Ultralytics Construction-PPE Dataset}, month = {November}, year = {2026}, license = {CC-BY-4.0}, url = {https://docs.ultralytics.com/datasets/detect/construction-ppe}, publisher = {Ultralytics}}` instead of Dalvi et al. v1.0.0 (January 2025). The archive also ships `README.md`, `LICENSE.txt` (CC BY 4.0) and `manifest.jsonl` (file, split, sha256 and annotation count per image); its README says the images "combine human-collected and generated images". The docs page publishes no mAP numbers, so R1 has no published number to compare with.
- **Reason:** spec 3 forbids stopping; the published data is the only data there is, and the spec's own instruction is to verify the facts and record differences.
- **Rejected:** hunting for the January 2025 release elsewhere (no official mirror; a third-party copy would not be verifiable).

## T-002 Held-out test split carved from the official train split
- **Decision:** the release has no test split, so a deterministic 100-image test split is carved out of the official train split (seed 0), stratified so that each class's instance share in test is close to train, with near-duplicate clusters (pHash Hamming distance <= 4, mirrored hash included) kept on one side, so the test split is leakage-free by construction (asserted by a test on the committed pHashes). The official val split (200) is never touched. Unified copy: train 700 / val 200 / test 100. R1 trains on the official YAML (800/200). Val images that near-duplicate a train image are excluded from metric computation through a deduplicated val list when there are any, and full-val numbers are reported too. The carved file names are committed in `datasets/construction-ppe-split.json`, so every rebuild (and the notebook) uses the same split.
- **Reason:** spec 7.1 needs a test split that is evaluated once; carving it from train keeps the official val comparable with anyone else's numbers on this release.
- **Rejected:** splitting val (halves the only official evaluation split); no test split (spec 7.1 and 9.2 require one).

## T-003 Class mapping of the November 2026 release
- **Decision:** helmet->helmet, gloves->gloves, vest->vest, boots->boots, goggles->glasses, Person->person, no_helmet->head, no_goggle->face, no_gloves->hands, no_boots->foot, forklift->forklift, no_vest->dropped. Unified training ids: `0 person, 1 head, 2 face, 3 hands, 4 foot, 5 helmet, 6 vest, 7 glasses, 8 gloves, 9 boots` (spec 5.1) plus `10 forklift` (the runtime taxonomy has `forklift` and vehicle proximity can use it; it has 275 train instances, above the 50-instance minimum set for keeping it). The spec-5.2 `none` inspection becomes a `no_vest` inspection (counts per split, mean box size, position relative to Person boxes, 12 annotated examples in `reports/samples/no_vest/`); default decision: drop it, because neither the taxonomy nor the web contract has a torso class and the runtime decides vest MISSING from a tracked person without a vest. The evidence is in `reports/data_stats.md`.
- **Rejected:** mapping no_vest to `person` (duplicate, smaller person boxes would teach a wrong person extent).

## T-004 GPU mode; AMP requested, FP32 in practice
- **Decision:** GPU mode (GTX 1660 SUPER, 6143 MiB = 6.0 GiB total VRAM meets the >= 6 GB rule; `reports/environment.md`). Every run passes `amp=True` as specified; Ultralytics' `check_amp` disables AMP on GTX 16xx cards, so training runs in FP32. What actually happened is read from each run's log and recorded in its run report.
- **Rejected:** forcing AMP (Ultralytics refuses it on this card family because of NaN losses).

## T-005 Cache and workers
- **Decision:** `cache="disk"` for R2/R3 (spec 6.2: RAM 15.9 GiB < 16 GiB), no cache for R0/R1 (the spec sets none for them); `workers=min(8, cpu_count)=8`.

## T-006 R3 variant
- **Decision:** R3 = `yolo26s.pt` with `freeze=10`, same data and settings as R2.
- **Reason:** `yolo26m` would not meet the CPU >= 8 FPS acceptance (spec 7.5) and is slow on a 6 GB card.

## T-007 Model names
- **Decision:** `ppe_cppe_<variant>_<YYYYMMDD of the run start>_<run>`, e.g. `ppe_cppe_n_20261104_r0`, `ppe_cppe_s_20261105_r2`; the name is fixed when a run first starts and kept across resumes (stored in `reports/cppe/chain.json` and next to the run).
- **Reason:** the plain spec name `ppe_cppe_<variant>_<YYYYMMDD>` collides between runs of the same variant on the same day (R2 and R3 are both `s`).

## T-008 Registry: explicit aliases and exclusion from the computed aliases
- **Decision:** `safety_ai/inference/registry.py` gains two optional, backward-compatible entry fields: `aliases` (explicit alias names, unique across the registry, never equal to an entry name, resolved before the computed aliases) and `exclude_from_latest` (keeps an entry out of `<role>_latest` and `<role>_<variant>_latest`). Both are written to the JSON only when set, and an export's `imgsz` only when known, so the five existing entries keep their exact JSON (tested by a byte-for-byte round trip of `models/registry.json`). Every Construction-PPE entry sets `exclude_from_latest`, so `ppe_n_latest` still resolves to `ppe_n_sh17` (the integration demo and tests rely on it). `ModelRegistry.set_alias(alias, name)` moves an explicit alias in one atomic write. `ppe_cppe_latest` is assigned to the production model by the evaluation stage (S4); until then it points at R0 provisionally so the pipeline can be exercised.

## T-009 Tracking of the training work
- **Decision:** this section of DECISIONS.md (T-xxx); in PROGRESS.md a section whose marker block `<!-- cppe-progress:start -->...<!-- cppe-progress:end -->` is rewritten by the pipeline (`safety_ai/training/progress.py`: re-read before every write, only the block replaced, atomic write) from `reports/cppe/progress_notes.md` and `reports/cppe/chain.json`; in CHECKLIST.md the section "PPE training — Definition of Done (docs/TRAINING_SPEC.md 9.2)" with the 12 items of spec 9.2; FINAL_REPORT.md at the end.

## T-010 Acquisition: resumed curl fallback
- **Decision:** step 1 of spec 4.3 (`check_det_dataset("construction-ppe.yaml")`) ran at about 0.15 MB/s from ultralytics.com and into the previous datasets_dir (Ultralytics captures `DATASETS_DIR` at import time, before `settings.update`), so it was stopped; step 2 (`curl -L` from the GitHub release URL to `data/raw/construction-ppe.zip`) stalled at 93,990,912 bytes. That partial file was copied to `data/raw/cppe_t1/construction-ppe.zip` and resumed with an HTTP range request (`curl -L -C -`, finished in 64 s). The archive is exactly 647,666,037 bytes, every member passes its CRC (`zipfile.testzip()` returned None), sha256 `e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232`; it was extracted (path-traversal checked) to `data/construction-ppe/`. The pipeline's own download code (`safety_ai/training/cppe_data.py`) implements spec 4.3 in order: the settings update first and `check_det_dataset` in a child process (so `DATASETS_DIR` is read after the update), 3 attempts, then the curl fallback (resumable), then a user-provided `data/raw/construction-ppe.zip`, then the synthetic 12-class dataset plus a retry every 15 minutes; when the dataset is present and verifies, it skips the download. Ultralytics settings: `datasets_dir=<safety-ai>/data`, `runs_dir=<safety-ai>/runs`, `sync`, `wandb`, `comet`, `mlflow`, `clearml` false; the key `hub` does not exist in 8.4.201 and is skipped and logged.
