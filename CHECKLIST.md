# Checklist — Definition of Done (spec 10.2)

Every item is ticked only with the command that proves it and the key lines of its output.
Items marked *(reconstructed)* complete the part of 10.2 that was cut off in the received
specification (DECISIONS.md D-002).

## Definition of Done

- [ ] pip install -e .[dev] in a fresh virtual environment succeeds; safety-ai --help lists every command of section 7.3; safety-ai version prints versions and the selected device.
- [ ] make check is green: lint, format, types, unit, property, scenario, integration, end-to-end smoke; coverage of logic/ and backend/ ≥ 90 %.
- [ ] safety-ai mock-backend plus safety-ai run --config configs/demo_file.yaml run for 90 s: heartbeats every 5 s, detections streaming, stats at the minute boundary, zero ERROR lines, /health returns 200.
- [ ] All 14 scenarios of section 8.3 pass with the exact expected event streams and no unexpected events.
- [ ] Idempotency, cooldown and spool recovery tests pass; stopping the mock backend for 20 s loses no events and creates no duplicates.
- [ ] Every outgoing message type was validated by the strict mock backend during the end-to-end run with zero schema violations (/debug/messages).
- [ ] safety-ai demo --source <file> --zones tools/demo_zones.yaml prints events as JSON lines and writes a jpg and an mp4 per event.
- [ ] http://127.0.0.1:8001/stream/cam-1.mjpeg serves annotated frames (10 frames fetched by tools/fetch_mjpeg.py).
- [ ] Training pipeline proven: either a real fine-tune with metrics in reports/eval_*.md, or the CPU smoke run on the synthetic dataset plus documented download attempts for every real dataset; models/registry.json valid; ONNX export verified against the .pt backend (mean IoU ≥ 0.9 on 20 images).
- [ ] safety-ai bench results recorded in models/bench.json and FINAL_REPORT.md; targets met or every deviation explained with the tuning that was tried.
- [ ] The CPU Docker image builds
- [ ] The CUDA Docker image builds and docker-compose.yml starts mock-backend and safety-ai with healthchecks on /health *(reconstructed, spec 7.4)*
- [ ] A 30-minute soak run shows RSS growth < 10 % after the first 5 minutes, no reconnect storms and no dropped events *(reconstructed, spec 9)*
- [ ] SIGINT/SIGTERM shut everything down within 5 s after PATCHing active events and flushing the spool for up to 2 s *(reconstructed, spec 9)*
- [ ] No placeholders anywhere: no TODO, FIXME, NotImplementedError, bare pass bodies, skipped or xfail tests *(reconstructed, spec 3)*
- [ ] safety-ai selftest and safety-ai selftest --quick pass and print a pass/fail table *(reconstructed, spec 7.3)*
- [ ] Documentation complete — README.md, docs/ARCHITECTURE.md, docs/CONTRACT.md (section 4 verbatim), docs/TRAINING.md, docs/DEMO.md, DECISIONS.md, PROGRESS.md, CHECKLIST.md, FINAL_REPORT.md — and every documented command was run *(reconstructed, spec 2)*
- [ ] The Verification Protocol passed three consecutive times with zero findings; reports/make_check_final.txt saved *(reconstructed, spec 3, 8.7)*
- [ ] Git history has a commit per milestone and the working tree is clean at the end *(reconstructed, spec 3)*

## Milestones (spec 10.1)

- [ ] M0 Bootstrap
- [ ] M1 Contract
- [ ] M2 Capture and inference
- [ ] M3 Logic
- [ ] M4 Outputs
- [ ] M5 Data and training
- [ ] M6 Packaging
- [ ] M7 Verification
- [ ] M8 Final polish

## PPE training — Definition of Done (docs/TRAINING_SPEC.md 9.2)

Adapted to the published release by DECISIONS.md T-001/T-002 (1,000 images, 800 train / 200 val, a
100-image test split carved from train, 12 classes, no `none` class). Every item is ticked only with
the command that proves it and the key lines of its output.

- [ ] Dataset downloaded and verified: 800 / 200 images with labels (unified copy 700 / 200 / 100), all label lines valid, reports/data_stats.md written, 24 annotated samples inspected and described.
- [ ] Unified copy built with the mapping of 5.1 (T-003); instance counts match source minus dropped classes (asserted by a test); no_vest decision (the spec's none decision, T-003) recorded with evidence; oversampling list built and reported.
- [ ] R0 smoke passed end to end; R2 completed within its time box (and R1/R3 on GPU); results.csv, plots and best.pt saved; registry entries valid JSON with sha256.
- [ ] Val report with per-class metrics, bare-class group, pairwise confusions, speed; test report for the production model only; thresholds.json with F1-optimal per-class values.
- [ ] Error analysis with 100 dumped examples and at most five recommendations, at least one applied.
- [ ] predict.py outputs the contract-shaped JSON for 12 test images; images inspected.
- [ ] ONNX export verified (parity >= 0.9 mean IoU, counts +-1); OpenVINO / TensorRT exported where installable; bench.json written; CPU >= 8 FPS at 640 or at 512 with the reason.
- [ ] Integration: safety-ai make check and e2e smoke green with models.ppe: ppe_cppe_latest.
- [ ] notebooks/train_colab.ipynb executes in smoke mode via nbconvert; its remap cell is identical to remap.py (test).
- [ ] docs/TRAINING.md, MODEL_CARD.md (dataset, licence, citation, metrics, intended use, limitations), README updated; every documented command executed during verification.
- [ ] grep -rn "TODO\|FIXME\|NotImplementedError\|xfail\|skip(" <package> tests/ returns nothing.
- [ ] DECISIONS.md complete; FINAL_REPORT.md written; PROGRESS.md says DONE with the final commit hash.
